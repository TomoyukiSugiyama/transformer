//! 行優先 (row-major) の連続メモリで行列を表現する `Matrix`。
//!
//! 既存コードは `Vec<Vec<f32>>` (jagged) を使っているが、こちらは flat 表現で
//! メモリ局所性が良く、 BLAS 互換 (`row_stride = cols, col_stride = 1`) でもある。
//! `from_jagged` / `to_jagged` で相互変換できる。
//!
//! 主要 API:
//! - 構築: `zeros`, `from_jagged`, `from_flat`
//! - 形状: `rows`, `cols`, `shape`
//! - アクセス: `row`, `row_mut`, `get`, `set`, `data`, `data_mut`
//! - 演算: `matmul` (BLAS), `matmul_naive` (rayon), `transpose`, `add_in_place`,
//!         `softmax_rows_in_place`, `split_columns`, `concat_columns` ほか
//!
//! `matmul` は `matrixmultiply::sgemm` (SIMD + キャッシュタイリングされた pure-Rust BLAS)
//! に委譲する。 `matmul_naive` は rayon で行並列化した素朴な i-k-j 実装で、 数値検証や
//! ベンチ用に保持してある。 その他の要素演算 (transpose, add, softmax) は rayon で並列化。

#![allow(dead_code)]

use rayon::prelude::*;

/// macOS では Apple Accelerate Framework (内部で AMX を活用する CBLAS) を使う。
/// 他 OS では `matrixmultiply` クレート (pure Rust SIMD カーネル) にフォールバック。
#[cfg(target_os = "macos")]
mod accelerate {
    use std::os::raw::{c_float, c_int};

    pub const CBLAS_ROW_MAJOR: c_int = 101;
    pub const CBLAS_NO_TRANS: c_int = 111;
    pub const CBLAS_TRANS: c_int = 112;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn cblas_sgemm(
            layout: c_int,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            b: *const c_float,
            ldb: c_int,
            beta: c_float,
            c: *mut c_float,
            ldc: c_int,
        );
    }
}

#[derive(Clone, Debug)]
pub struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Default for Matrix {
    /// 形状 (0, 0)、 内部 `Vec<f32>` も空の "プレースホルダ" Matrix。
    /// `std::mem::take(&mut field)` で一時的に所有権を取り出してから戻す
    /// パターンで利用する (perf-alloc 系の buffer reuse で多用)。
    fn default() -> Self {
        Self {
            data: Vec::new(),
            rows: 0,
            cols: 0,
        }
    }
}

impl Matrix {
    /// 全要素 0 で初期化した (rows, cols) 行列。
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0f32; rows * cols],
            rows,
            cols,
        }
    }

    /// 既にある flat data (row-major) から Matrix を構築。
    /// `data.len() != rows * cols` のとき panic。
    pub fn from_flat(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "from_flat: data length {} != rows*cols {}",
            data.len(),
            rows * cols
        );
        Self { data, rows, cols }
    }

    /// jagged 表現 `Vec<Vec<f32>>` から flat に変換。
    /// 各行の長さが揃っていない場合は panic。
    pub fn from_jagged(rows: &[Vec<f32>]) -> Self {
        assert!(!rows.is_empty(), "from_jagged: empty input");
        let cols = rows[0].len();
        assert!(
            rows.iter().all(|r| r.len() == cols),
            "from_jagged: inconsistent row lengths"
        );
        let mut data = Vec::with_capacity(rows.len() * cols);
        for row in rows {
            data.extend_from_slice(row);
        }
        Self {
            data,
            rows: rows.len(),
            cols,
        }
    }

    /// flat → jagged 変換（既存 API との橋渡し用）。
    pub fn to_jagged(&self) -> Vec<Vec<f32>> {
        self.data
            .chunks(self.cols)
            .map(|row| row.to_vec())
            .collect()
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// `i` 行目への参照。
    pub fn row(&self, i: usize) -> &[f32] {
        let start = i * self.cols;
        &self.data[start..start + self.cols]
    }

    /// `i` 行目への可変参照。
    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        let start = i * self.cols;
        &mut self.data[start..start + self.cols]
    }

    /// 要素アクセス。
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.cols + j]
    }

    /// 要素設定。
    pub fn set(&mut self, i: usize, j: usize, v: f32) {
        self.data[i * self.cols + j] = v;
    }

    /// 行列積: `self (m, k) @ other (k, n) → (m, n)`。
    /// macOS では Apple Accelerate Framework の `cblas_sgemm` (内部で AMX を活用)、
    /// それ以外では `matrixmultiply::sgemm` (pure-Rust SIMD カーネル) を使う。
    /// row-major 表現なので leading dimension は `cols` (= row stride)。
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "matmul shape mismatch: ({}, {}) × ({}, {})",
            self.rows, self.cols, other.rows, other.cols
        );
        let m = self.rows;
        let n = other.cols;
        let mut out = Matrix::zeros(m, n);
        sgemm_general(self, false, other, false, 1.0, 0.0, &mut out);
        out
    }

    /// `self^T @ other`。 `self` (k, m), `other` (k, n) → `(m, n)`。
    /// `cache_x.transpose().matmul(&dl_dy)` のような **転置行列を一旦アロケーション** していた
    /// パターンを、 BLAS の trans フラグだけで処理して **transpose のメモリコピーを完全に省く**。
    ///
    /// Phase 7-3 高速化で導入: backward 系の `*.transpose().matmul(_)` を一掃する。
    pub fn matmul_t1(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.rows, other.rows,
            "matmul_t1 shape mismatch: ({}, {})^T × ({}, {})",
            self.rows, self.cols, other.rows, other.cols
        );
        let m = self.cols; // self^T の行数 = self の列数
        let n = other.cols;
        let mut out = Matrix::zeros(m, n);
        sgemm_general(self, true, other, false, 1.0, 0.0, &mut out);
        out
    }

    /// `self @ other^T`。 `self` (m, k), `other` (n, k) → `(m, n)`。
    /// `dl_dz @ w.transpose()` のような **転置行列を一旦アロケーション** していたパターンを、
    /// BLAS の trans フラグだけで処理して **transpose のメモリコピーを完全に省く**。
    ///
    /// Phase 7-3 高速化で導入: backward 系の `*.matmul(&w.transpose())` を一掃する。
    pub fn matmul_t2(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.cols,
            "matmul_t2 shape mismatch: ({}, {}) × ({}, {})^T",
            self.rows, self.cols, other.rows, other.cols
        );
        let m = self.rows;
        let n = other.rows; // other^T の列数 = other の行数
        let mut out = Matrix::zeros(m, n);
        sgemm_general(self, false, other, true, 1.0, 0.0, &mut out);
        out
    }

    // ---------- Phase 7-4: `_into` 系 API ----------
    // forward/backward を回すたびに `Matrix::zeros(...)` で何 MB も確保していた箱所を、
    // 既存 buffer に **直接書き込む** ことで毎ステップのアロケを削減する。
    // sgemm の場合 BLAS が beta=0 で C 全体を書き換えるので、 buffer の中身は事前ゼロ初期化
    // 不要 (resize 時に 0 充填されているのでそのまま使える)。

    /// shape を確保し直すユーティリティ。 既存容量で足りる場合は再アロケしない。
    /// `_into` 系を呼ぶ前に呼んで形を合わせるのに使う。
    pub fn ensure_shape(&mut self, rows: usize, cols: usize) {
        let needed = rows * cols;
        if self.data.len() != needed {
            self.data.resize(needed, 0.0);
        }
        self.rows = rows;
        self.cols = cols;
    }

    /// `out = self @ other` を **既存 buffer に書き込む** matmul。
    /// `out.shape() == (self.rows, other.cols)` であること必須 (満たさない場合 panic)。
    /// 中身は完全に上書きされるのでゼロ初期化済みである必要はない。
    pub fn matmul_into(&self, other: &Matrix, out: &mut Matrix) {
        assert_eq!(
            self.cols, other.rows,
            "matmul_into shape mismatch: ({}, {}) × ({}, {})",
            self.rows, self.cols, other.rows, other.cols
        );
        sgemm_general(self, false, other, false, 1.0, 0.0, out);
    }

    /// `out = self^T @ other` を既存 buffer に書き込む。
    pub fn matmul_t1_into(&self, other: &Matrix, out: &mut Matrix) {
        assert_eq!(
            self.rows, other.rows,
            "matmul_t1_into shape mismatch: ({}, {})^T × ({}, {})",
            self.rows, self.cols, other.rows, other.cols
        );
        sgemm_general(self, true, other, false, 1.0, 0.0, out);
    }

    /// `out = self @ other^T` を既存 buffer に書き込む。
    pub fn matmul_t2_into(&self, other: &Matrix, out: &mut Matrix) {
        assert_eq!(
            self.cols, other.cols,
            "matmul_t2_into shape mismatch: ({}, {}) × ({}, {})^T",
            self.rows, self.cols, other.rows, other.cols
        );
        sgemm_general(self, false, other, true, 1.0, 0.0, out);
    }

    /// `out += self @ other` を計算する **fused multiply-add 版**。 BLAS の beta=1 経路で
    /// 1 度の sgemm に圧縮される (旧来は temp = matmul → add_in_place の 2 段階)。
    /// 勾配累積 (`grad_w += x^T @ dl_dy`) に最適。
    pub fn matmul_add_into(&self, other: &Matrix, out: &mut Matrix) {
        assert_eq!(
            self.cols, other.rows,
            "matmul_add_into shape mismatch: ({}, {}) × ({}, {})",
            self.rows, self.cols, other.rows, other.cols
        );
        sgemm_general(self, false, other, false, 1.0, 1.0, out);
    }

    /// `out += self^T @ other`
    pub fn matmul_t1_add_into(&self, other: &Matrix, out: &mut Matrix) {
        assert_eq!(
            self.rows, other.rows,
            "matmul_t1_add_into shape mismatch: ({}, {})^T × ({}, {})",
            self.rows, self.cols, other.rows, other.cols
        );
        sgemm_general(self, true, other, false, 1.0, 1.0, out);
    }

    /// `out += self @ other^T`
    pub fn matmul_t2_add_into(&self, other: &Matrix, out: &mut Matrix) {
        assert_eq!(
            self.cols, other.cols,
            "matmul_t2_add_into shape mismatch: ({}, {}) × ({}, {})^T",
            self.rows, self.cols, other.rows, other.cols
        );
        sgemm_general(self, false, other, true, 1.0, 1.0, out);
    }

    /// `out = self^T` を既存 buffer に書き込む。 `out.shape() == (self.cols, self.rows)` 必須。
    pub fn transpose_into(&self, out: &mut Matrix) {
        let m = self.rows;
        let n = self.cols;
        assert_eq!(
            out.shape(),
            (n, m),
            "transpose_into shape mismatch: out={:?} expected ({n}, {m})",
            out.shape()
        );
        out.data
            .par_chunks_mut(m)
            .enumerate()
            .for_each(|(j, out_row)| {
                for i in 0..m {
                    out_row[i] = self.data[i * n + j];
                }
            });
    }

    /// `split_columns` の destination 版。 `out.len() == n` (= 分割数) で、
    /// 各 `out[h]` の shape は `(self.rows, self.cols / n)` でなければならない。
    /// MHA の split_heads で head 用 buffer を毎回 alloc するのを抑える。
    pub fn split_columns_into(&self, out: &mut [Matrix]) {
        let n = out.len();
        assert!(n > 0, "split_columns_into: empty out");
        assert!(
            self.cols % n == 0,
            "split_columns_into: cols {} not divisible by {}",
            self.cols,
            n
        );
        let block = self.cols / n;
        for (h, dst) in out.iter_mut().enumerate() {
            assert_eq!(
                dst.shape(),
                (self.rows, block),
                "split_columns_into: out[{h}] shape mismatch"
            );
            for i in 0..self.rows {
                let src_start = i * self.cols + h * block;
                let dst_start = i * block;
                dst.data[dst_start..dst_start + block]
                    .copy_from_slice(&self.data[src_start..src_start + block]);
            }
        }
    }

    /// `concat_columns` の destination 版。 `parts.iter()` の shape はすべて等しく、
    /// `out.shape() == (rows, block * parts.len())` でなければならない。
    pub fn concat_columns_into(parts: &[&Matrix], out: &mut Matrix) {
        assert!(!parts.is_empty(), "concat_columns_into: empty parts");
        let rows = parts[0].rows;
        let block = parts[0].cols;
        assert!(
            parts.iter().all(|p| p.rows == rows && p.cols == block),
            "concat_columns_into: inconsistent shapes"
        );
        let n = parts.len();
        let cols = block * n;
        assert_eq!(
            out.shape(),
            (rows, cols),
            "concat_columns_into: out shape mismatch: expected ({rows}, {cols}) got {:?}",
            out.shape()
        );
        for (h, part) in parts.iter().enumerate() {
            for i in 0..rows {
                let dst = i * cols + h * block;
                let src = i * block;
                out.data[dst..dst + block].copy_from_slice(&part.data[src..src + block]);
            }
        }
    }

    /// rayon 並列の素朴な i-k-j matmul。 BLAS との数値比較やベンチ用に残してある。
    pub fn matmul_naive(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "matmul_naive shape mismatch: ({}, {}) × ({}, {})",
            self.rows, self.cols, other.rows, other.cols
        );
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;
        let a_data = &self.data;
        let b_data = &other.data;
        let mut out = Matrix::zeros(m, n);

        out.data
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, c_row)| {
                let a_row_start = i * k;
                for l in 0..k {
                    let aik = a_data[a_row_start + l];
                    let b_row_start = l * n;
                    for j in 0..n {
                        c_row[j] += aik * b_data[b_row_start + j];
                    }
                }
            });
        out
    }

    /// 転置: `(m, n) → (n, m)`。出力行ごとに rayon で並列化。
    pub fn transpose(&self) -> Matrix {
        let m = self.rows;
        let n = self.cols;
        let mut out = Matrix::zeros(n, m);
        out.data
            .par_chunks_mut(m)
            .enumerate()
            .for_each(|(j, out_row)| {
                for i in 0..m {
                    out_row[i] = self.data[i * n + j];
                }
            });
        out
    }

    /// in-place 加算: `self += other`。要素ごとに rayon 並列化。
    pub fn add_in_place(&mut self, other: &Matrix) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "add_in_place shape mismatch: {:?} vs {:?}",
            self.shape(),
            other.shape()
        );
        self.data
            .par_iter_mut()
            .zip(other.data.par_iter())
            .for_each(|(a, b)| *a += *b);
    }

    /// 各行に同じ bias ベクトルを in-place で加算: `self[i, j] += bias[j]`。
    pub fn add_row_bias_in_place(&mut self, bias: &[f32]) {
        assert_eq!(
            bias.len(),
            self.cols,
            "add_row_bias_in_place: bias len {} != cols {}",
            bias.len(),
            self.cols
        );
        let cols = self.cols;
        for row in self.data.chunks_mut(cols) {
            for (v, b) in row.iter_mut().zip(bias.iter()) {
                *v += *b;
            }
        }
    }

    /// 各行を縦に集計（列ごとの合計）して返す。
    /// dL/d_bias = Σ_t dL/d_z[t] のようなパターンで使う。
    pub fn sum_rows_into_cols(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.cols];
        for row in self.data.chunks(self.cols) {
            for (o, v) in out.iter_mut().zip(row.iter()) {
                *o += *v;
            }
        }
        out
    }

    /// 要素ごとに `f` を適用した新しい行列を返す。
    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Matrix {
        let data: Vec<f32> = self.data.iter().map(|&v| f(v)).collect();
        Self::from_flat(data, self.rows, self.cols)
    }

    /// 同形 2 行列の要素ごと演算（new = f(self, other)）。
    /// dl_dz1 = dl_da ⊙ GELU'(z1) のようなパターンで使う。
    pub fn elementwise_with<F: Fn(f32, f32) -> f32>(&self, other: &Matrix, f: F) -> Matrix {
        assert_eq!(
            self.shape(),
            other.shape(),
            "elementwise_with shape mismatch: {:?} vs {:?}",
            self.shape(),
            other.shape()
        );
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| f(a, b))
            .collect();
        Self::from_flat(data, self.rows, self.cols)
    }

    /// 行ごとの数値安定 softmax を in-place で適用。
    pub fn softmax_rows_in_place(&mut self) {
        let cols = self.cols;
        self.data.par_chunks_mut(cols).for_each(|row| {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|x| (x - max).exp()).sum();
            for x in row.iter_mut() {
                *x = ((*x) - max).exp() / sum;
            }
        });
    }

    /// 列方向に等分して n 個のサブ行列に分割する。
    /// MHA の split_heads（seq, d_model）→ n_heads 個の (seq, d_head) で使う。
    /// `cols % n != 0` のとき panic。
    pub fn split_columns(&self, n: usize) -> Vec<Matrix> {
        assert!(
            self.cols % n == 0,
            "split_columns: cols {} not divisible by {}",
            self.cols,
            n
        );
        let block = self.cols / n;
        (0..n)
            .map(|h| {
                let mut data = Vec::with_capacity(self.rows * block);
                for i in 0..self.rows {
                    let start = i * self.cols + h * block;
                    data.extend_from_slice(&self.data[start..start + block]);
                }
                Matrix::from_flat(data, self.rows, block)
            })
            .collect()
    }

    /// `split_columns` の逆。 各行ごとに横に並べて 1 つの行列にまとめる。
    /// Phase 7-3 では `&[Matrix]` のまま受け取れる API を維持し、 owned Vec も &[&Matrix] も
    /// 受け付けたいシーンでは `concat_columns_refs` を使う。
    pub fn concat_columns(parts: &[Matrix]) -> Matrix {
        assert!(!parts.is_empty(), "concat_columns: empty input");
        let rows = parts[0].rows;
        let block = parts[0].cols;
        assert!(
            parts.iter().all(|p| p.rows == rows && p.cols == block),
            "concat_columns: inconsistent shapes"
        );
        let n = parts.len();
        let cols = block * n;
        let mut data = vec![0.0f32; rows * cols];
        for (h, part) in parts.iter().enumerate() {
            for i in 0..rows {
                let dst = i * cols + h * block;
                let src = i * block;
                data[dst..dst + block].copy_from_slice(&part.data[src..src + block]);
            }
        }
        Matrix::from_flat(data, rows, cols)
    }
}

/// `out = alpha * op(A) @ op(B) + beta * out` を計算する内部 sgemm ヘルパ。
///
/// `trans_a` / `trans_b` で各オペランドを転置するかを指定する。 転置は **メタデータだけ** で
/// 切り替わり、 メモリ上の transpose コピーは発生しない (BLAS の trans フラグを使う)。
/// `beta=1.0` を指定すれば「既存の out に加算」される (= fused matmul-add で勾配累積に最適)。
///
/// 期待形状:
/// - `op(A)` = `(m, k)`、 `op(B)` = `(k, n)`、 `out` = `(m, n)`
/// - row-major 表現
fn sgemm_general(
    a: &Matrix,
    trans_a: bool,
    b: &Matrix,
    trans_b: bool,
    alpha: f32,
    beta: f32,
    out: &mut Matrix,
) {
    let (m, k) = if trans_a {
        (a.cols, a.rows)
    } else {
        (a.rows, a.cols)
    };
    let (k_b, n) = if trans_b {
        (b.cols, b.rows)
    } else {
        (b.rows, b.cols)
    };
    assert_eq!(
        k, k_b,
        "sgemm_general inner-dim mismatch: op(A)=({m}, {k}) × op(B)=({k_b}, {n})"
    );
    assert_eq!(
        out.shape(),
        (m, n),
        "sgemm_general out shape mismatch: expected ({m}, {n}) got {:?}",
        out.shape()
    );

    // leading dimension は **転置の有無に関わらず元の `cols`** であることに注意。
    // BLAS は op(X) ではなく X 自身の lda を使う。
    let lda = a.cols as i32;
    let ldb = b.cols as i32;
    let ldc = n as i32;

    #[cfg(target_os = "macos")]
    // SAFETY: out は (m, n) を確保済みで a/b とメモリ非共有。
    unsafe {
        accelerate::cblas_sgemm(
            accelerate::CBLAS_ROW_MAJOR,
            if trans_a {
                accelerate::CBLAS_TRANS
            } else {
                accelerate::CBLAS_NO_TRANS
            },
            if trans_b {
                accelerate::CBLAS_TRANS
            } else {
                accelerate::CBLAS_NO_TRANS
            },
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a.data.as_ptr(),
            lda,
            b.data.as_ptr(),
            ldb,
            beta,
            out.data.as_mut_ptr(),
            ldc,
        );
    }

    #[cfg(not(target_os = "macos"))]
    // SAFETY: 同上。 matrixmultiply は明示的な trans フラグを持たないので、 row/col stride
    // を入れ替えることで論理的な転置を表現する。
    // 通常 (no trans):    rsa = cols, csa = 1
    // 転置 (trans):       rsa = 1,    csa = cols    ← 行と列の役割を入れ替え
    unsafe {
        let (rsa, csa) = if trans_a {
            (1isize, a.cols as isize)
        } else {
            (a.cols as isize, 1isize)
        };
        let (rsb, csb) = if trans_b {
            (1isize, b.cols as isize)
        } else {
            (b.cols as isize, 1isize)
        };
        matrixmultiply::sgemm(
            m,
            k,
            n,
            alpha,
            a.data.as_ptr(),
            rsa,
            csa,
            b.data.as_ptr(),
            rsb,
            csb,
            beta,
            out.data.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn zeros_has_correct_shape() {
        let m = Matrix::zeros(3, 4);
        assert_eq!(m.shape(), (3, 4));
        assert!(m.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn jagged_roundtrip_preserves_values() {
        let jagged = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let mat = Matrix::from_jagged(&jagged);
        assert_eq!(mat.shape(), (2, 3));
        assert_eq!(mat.to_jagged(), jagged);
    }

    #[test]
    fn row_access_returns_correct_slice() {
        let mat = Matrix::from_jagged(&[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        assert_eq!(mat.row(0), &[1.0, 2.0]);
        assert_eq!(mat.row(2), &[5.0, 6.0]);
        assert_eq!(mat.get(1, 1), 4.0);
    }

    #[test]
    fn matmul_2x3_times_3x2_gives_correct_result() {
        // [[1,2,3],     [[7,8],          [[58, 64],
        //  [4,5,6]] @   [9,10],       =   [139,154]]
        //              [11,12]]
        let a = Matrix::from_jagged(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let b = Matrix::from_jagged(&[vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]]);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), (2, 2));
        assert!(approx_eq(c.get(0, 0), 58.0));
        assert!(approx_eq(c.get(0, 1), 64.0));
        assert!(approx_eq(c.get(1, 0), 139.0));
        assert!(approx_eq(c.get(1, 1), 154.0));
    }

    #[test]
    fn matmul_matches_naive_jagged_implementation() {
        let a_j = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let b_j = vec![
            vec![7.0, 8.0, 9.0, 10.0],
            vec![11.0, 12.0, 13.0, 14.0],
            vec![15.0, 16.0, 17.0, 18.0],
        ];
        // Naive O(mnk) jagged
        let m = a_j.len();
        let k = a_j[0].len();
        let n = b_j[0].len();
        let mut expected = vec![vec![0.0f32; n]; m];
        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    expected[i][j] += a_j[i][l] * b_j[l][j];
                }
            }
        }
        let actual = Matrix::from_jagged(&a_j).matmul(&Matrix::from_jagged(&b_j));
        assert_eq!(actual.to_jagged(), expected);
    }

    #[test]
    fn transpose_swaps_rows_and_cols() {
        let mat = Matrix::from_jagged(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let t = mat.transpose();
        assert_eq!(t.shape(), (3, 2));
        assert_eq!(t.to_jagged(), vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
    }

    #[test]
    fn double_transpose_is_identity() {
        let mat = Matrix::from_jagged(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let tt = mat.transpose().transpose();
        assert_eq!(tt.to_jagged(), mat.to_jagged());
    }

    #[test]
    fn add_in_place_sums_elementwise() {
        let mut a = Matrix::from_jagged(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::from_jagged(&[vec![10.0, 20.0], vec![30.0, 40.0]]);
        a.add_in_place(&b);
        assert_eq!(a.to_jagged(), vec![vec![11.0, 22.0], vec![33.0, 44.0]]);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn matmul_panics_on_shape_mismatch() {
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(4, 5);
        let _ = a.matmul(&b);
    }

    /// BLAS 版 matmul と naive 版 matmul が浮動小数誤差の範囲で一致することを確認。
    /// AdamW やレイヤー実装が naive を前提にしていたので、 ここで等価性を保証しておく。
    #[test]
    fn matmul_blas_matches_naive_for_random_matrices() {
        use rand::{RngExt, SeedableRng, rngs::SmallRng};
        let mut rng = SmallRng::seed_from_u64(123);
        let cases: &[(usize, usize, usize)] = &[
            (1, 1, 1),
            (3, 5, 7),
            (16, 32, 8),
            (64, 128, 256),
            (128, 256, 256),
        ];
        for &(m, k, n) in cases {
            let a_data: Vec<f32> = (0..m * k).map(|_| rng.random_range(-1.0..1.0)).collect();
            let b_data: Vec<f32> = (0..k * n).map(|_| rng.random_range(-1.0..1.0)).collect();
            let a = Matrix::from_flat(a_data, m, k);
            let b = Matrix::from_flat(b_data, k, n);
            let c_blas = a.matmul(&b);
            let c_naive = a.matmul_naive(&b);
            assert_eq!(c_blas.shape(), c_naive.shape());
            // 行優先 dot 累積差は ε * k 程度
            let tol = 1e-3 * k as f32;
            for (x, y) in c_blas.data().iter().zip(c_naive.data().iter()) {
                assert!(
                    (x - y).abs() <= tol,
                    "shape=({m},{k},{n}) blas={x} naive={y} diff={}",
                    (x - y).abs()
                );
            }
        }
    }

    /// Phase 7-3: 転置版 matmul_t1 (= self^T @ other) が `transpose().matmul(_)` と
    /// 数値的に一致することを確認。 BLAS の trans フラグ経路の正当性検証。
    #[test]
    fn matmul_t1_matches_transpose_then_matmul() {
        use rand::{RngExt, SeedableRng, rngs::SmallRng};
        let mut rng = SmallRng::seed_from_u64(42);
        let cases: &[(usize, usize, usize)] = &[
            (1, 1, 1),
            (3, 5, 7),
            (16, 32, 8),
            (64, 128, 256),
        ];
        for &(k, m, n) in cases {
            // self: (k, m), other: (k, n)  → self^T @ other = (m, n)
            let a_data: Vec<f32> = (0..k * m).map(|_| rng.random_range(-1.0..1.0)).collect();
            let b_data: Vec<f32> = (0..k * n).map(|_| rng.random_range(-1.0..1.0)).collect();
            let a = Matrix::from_flat(a_data, k, m);
            let b = Matrix::from_flat(b_data, k, n);
            let expected = a.transpose().matmul(&b);
            let actual = a.matmul_t1(&b);
            assert_eq!(actual.shape(), expected.shape());
            let tol = 1e-3 * k as f32;
            for (x, y) in actual.data().iter().zip(expected.data().iter()) {
                assert!(
                    (x - y).abs() <= tol,
                    "matmul_t1 mismatch ({k},{m},{n}): {x} vs {y}"
                );
            }
        }
    }

    /// Phase 7-3: 転置版 matmul_t2 (= self @ other^T) が `self.matmul(&other.transpose())` と
    /// 数値的に一致することを確認。
    #[test]
    fn matmul_t2_matches_matmul_then_transpose() {
        use rand::{RngExt, SeedableRng, rngs::SmallRng};
        let mut rng = SmallRng::seed_from_u64(43);
        let cases: &[(usize, usize, usize)] = &[
            (1, 1, 1),
            (3, 5, 7),
            (16, 32, 8),
            (64, 128, 256),
        ];
        for &(m, k, n) in cases {
            // self: (m, k), other: (n, k)  → self @ other^T = (m, n)
            let a_data: Vec<f32> = (0..m * k).map(|_| rng.random_range(-1.0..1.0)).collect();
            let b_data: Vec<f32> = (0..n * k).map(|_| rng.random_range(-1.0..1.0)).collect();
            let a = Matrix::from_flat(a_data, m, k);
            let b = Matrix::from_flat(b_data, n, k);
            let expected = a.matmul(&b.transpose());
            let actual = a.matmul_t2(&b);
            assert_eq!(actual.shape(), expected.shape());
            let tol = 1e-3 * k as f32;
            for (x, y) in actual.data().iter().zip(expected.data().iter()) {
                assert!(
                    (x - y).abs() <= tol,
                    "matmul_t2 mismatch ({m},{k},{n}): {x} vs {y}"
                );
            }
        }
    }

    /// matmul_t1 と matmul_t2 の組み合わせ: A^T @ B^T = (B @ A)^T
    /// (BLAS の double-trans が壊れていないかの sanity check)
    #[test]
    fn matmul_t1_t2_roundtrip_identity() {
        let a = Matrix::from_jagged(&[
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]); // (2, 3)
        let b = Matrix::from_jagged(&[
            vec![7.0, 8.0],
            vec![9.0, 10.0],
        ]); // (2, 2)
        // a^T (3, 2) @ b (2, 2) = (3, 2)
        let direct = a.matmul_t1(&b);
        // (b^T @ a)^T = ((2,2) @ (2,3))^T = (2,3)^T = (3,2)
        let via_t2 = b.matmul_t1(&a).transpose();
        for (x, y) in direct.data().iter().zip(via_t2.data().iter()) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    #[test]
    fn from_flat_constructs_with_given_shape() {
        let m = Matrix::from_flat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        assert_eq!(m.shape(), (2, 3));
        assert_eq!(m.row(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn add_row_bias_adds_to_each_row() {
        let mut m = Matrix::from_jagged(&[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        m.add_row_bias_in_place(&[10.0, 100.0]);
        assert_eq!(
            m.to_jagged(),
            vec![vec![11.0, 102.0], vec![13.0, 104.0], vec![15.0, 106.0]]
        );
    }

    #[test]
    fn sum_rows_into_cols_returns_column_sums() {
        let m = Matrix::from_jagged(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(m.sum_rows_into_cols(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn map_applies_function_to_each_element() {
        let m = Matrix::from_jagged(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let m2 = m.map(|v| v * 2.0);
        assert_eq!(m2.to_jagged(), vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
    }

    #[test]
    fn elementwise_with_combines_two_matrices() {
        let a = Matrix::from_jagged(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::from_jagged(&[vec![10.0, 20.0], vec![30.0, 40.0]]);
        let r = a.elementwise_with(&b, |x, y| x * y);
        assert_eq!(r.to_jagged(), vec![vec![10.0, 40.0], vec![90.0, 160.0]]);
    }

    #[test]
    fn softmax_rows_in_place_sums_to_one_per_row() {
        let mut m = Matrix::from_jagged(&[vec![1.0, 2.0, 3.0], vec![3.0, 1.0, 2.0]]);
        m.softmax_rows_in_place();
        for i in 0..m.rows() {
            let s: f32 = m.row(i).iter().sum();
            assert!((s - 1.0).abs() < 1e-5);
        }
    }

    /// Phase 7-4: `_into` 系が non-`_into` 版と同じ結果を返すことを確認。
    #[test]
    fn matmul_into_matches_matmul() {
        let a = Matrix::from_jagged(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let b = Matrix::from_jagged(&[vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]]);
        let expected = a.matmul(&b);
        let mut actual = Matrix::zeros(2, 2);
        a.matmul_into(&b, &mut actual);
        assert_eq!(actual.to_jagged(), expected.to_jagged());
    }

    #[test]
    fn matmul_t1_into_matches_matmul_t1() {
        let a = Matrix::from_jagged(&[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]); // (3, 2)
        let b = Matrix::from_jagged(&[vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 12.0], vec![13.0, 14.0, 15.0]]); // (3, 3)
        let expected = a.matmul_t1(&b); // (2, 3)
        let mut actual = Matrix::zeros(2, 3);
        a.matmul_t1_into(&b, &mut actual);
        assert_eq!(actual.to_jagged(), expected.to_jagged());
    }

    #[test]
    fn matmul_t2_into_matches_matmul_t2() {
        let a = Matrix::from_jagged(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]); // (2, 3)
        let b = Matrix::from_jagged(&[vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 12.0]]); // (2, 3)
        let expected = a.matmul_t2(&b); // (2, 2)
        let mut actual = Matrix::zeros(2, 2);
        a.matmul_t2_into(&b, &mut actual);
        assert_eq!(actual.to_jagged(), expected.to_jagged());
    }

    #[test]
    fn transpose_into_matches_transpose() {
        let a = Matrix::from_jagged(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let expected = a.transpose();
        let mut actual = Matrix::zeros(3, 2);
        a.transpose_into(&mut actual);
        assert_eq!(actual.to_jagged(), expected.to_jagged());
    }

    #[test]
    fn split_columns_into_matches_split_columns() {
        let m = Matrix::from_jagged(&[
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ]);
        let expected = m.split_columns(3);
        let mut actual: Vec<Matrix> = (0..3).map(|_| Matrix::zeros(2, 2)).collect();
        m.split_columns_into(&mut actual);
        for (e, a) in expected.iter().zip(actual.iter()) {
            assert_eq!(e.to_jagged(), a.to_jagged());
        }
    }

    #[test]
    fn concat_columns_into_matches_concat_columns() {
        let a = Matrix::from_jagged(&[vec![1.0, 2.0], vec![7.0, 8.0]]);
        let b = Matrix::from_jagged(&[vec![3.0, 4.0], vec![9.0, 10.0]]);
        let c = Matrix::from_jagged(&[vec![5.0, 6.0], vec![11.0, 12.0]]);
        let expected = Matrix::concat_columns(&[a.clone(), b.clone(), c.clone()]);
        let mut actual = Matrix::zeros(2, 6);
        Matrix::concat_columns_into(&[&a, &b, &c], &mut actual);
        assert_eq!(actual.to_jagged(), expected.to_jagged());
    }

    #[test]
    fn matmul_add_into_accumulates_correctly() {
        let a = Matrix::from_jagged(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::from_jagged(&[vec![5.0, 6.0], vec![7.0, 8.0]]);
        let mut acc = Matrix::from_jagged(&[vec![100.0, 200.0], vec![300.0, 400.0]]);
        // expected = acc + a @ b
        let prod = a.matmul(&b);
        let expected: Vec<f32> = acc
            .data()
            .iter()
            .zip(prod.data().iter())
            .map(|(x, y)| x + y)
            .collect();
        a.matmul_add_into(&b, &mut acc);
        for (x, y) in acc.data().iter().zip(expected.iter()) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn matmul_t1_add_into_accumulates_correctly() {
        let a = Matrix::from_jagged(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::from_jagged(&[vec![5.0, 6.0], vec![7.0, 8.0]]);
        let mut acc = Matrix::from_jagged(&[vec![10.0, 20.0], vec![30.0, 40.0]]);
        let prod = a.matmul_t1(&b); // (2, 2)
        let expected: Vec<f32> = acc
            .data()
            .iter()
            .zip(prod.data().iter())
            .map(|(x, y)| x + y)
            .collect();
        a.matmul_t1_add_into(&b, &mut acc);
        for (x, y) in acc.data().iter().zip(expected.iter()) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn matmul_t2_add_into_accumulates_correctly() {
        let a = Matrix::from_jagged(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::from_jagged(&[vec![5.0, 6.0], vec![7.0, 8.0]]);
        let mut acc = Matrix::from_jagged(&[vec![10.0, 20.0], vec![30.0, 40.0]]);
        let prod = a.matmul_t2(&b);
        let expected: Vec<f32> = acc
            .data()
            .iter()
            .zip(prod.data().iter())
            .map(|(x, y)| x + y)
            .collect();
        a.matmul_t2_add_into(&b, &mut acc);
        for (x, y) in acc.data().iter().zip(expected.iter()) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn ensure_shape_resizes_correctly() {
        let mut m = Matrix::zeros(2, 3);
        m.ensure_shape(4, 5);
        assert_eq!(m.shape(), (4, 5));
        assert_eq!(m.data().len(), 20);
        m.ensure_shape(2, 2);
        assert_eq!(m.shape(), (2, 2));
        assert_eq!(m.data().len(), 4);
    }

    #[test]
    fn split_then_concat_columns_is_identity() {
        let m = Matrix::from_jagged(&[
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ]);
        let parts = m.split_columns(3);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].to_jagged(), vec![vec![1.0, 2.0], vec![7.0, 8.0]]);
        assert_eq!(parts[1].to_jagged(), vec![vec![3.0, 4.0], vec![9.0, 10.0]]);
        assert_eq!(parts[2].to_jagged(), vec![vec![5.0, 6.0], vec![11.0, 12.0]]);

        let recombined = Matrix::concat_columns(&parts);
        assert_eq!(recombined.to_jagged(), m.to_jagged());
    }
}
