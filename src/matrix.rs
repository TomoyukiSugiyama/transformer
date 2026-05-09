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
        let k = self.cols;
        let n = other.cols;
        let mut out = Matrix::zeros(m, n);

        #[cfg(target_os = "macos")]
        // SAFETY: out は (m, n) を確保済みで self/other とメモリ非共有。
        // row-major かつ no-transpose、 leading dim = cols (= 行内連続の row stride)。
        unsafe {
            accelerate::cblas_sgemm(
                accelerate::CBLAS_ROW_MAJOR,
                accelerate::CBLAS_NO_TRANS,
                accelerate::CBLAS_NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                self.data.as_ptr(),
                k as i32,
                other.data.as_ptr(),
                n as i32,
                0.0,
                out.data.as_mut_ptr(),
                n as i32,
            );
        }

        #[cfg(not(target_os = "macos"))]
        // SAFETY: 同上。 matrixmultiply の row/col stride 表現に合わせる。
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                self.data.as_ptr(),
                self.cols as isize,
                1,
                other.data.as_ptr(),
                other.cols as isize,
                1,
                0.0,
                out.data.as_mut_ptr(),
                n as isize,
                1,
            );
        }

        out
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

    /// `split_columns` の逆。各行ごとに横に並べて 1 つの行列にまとめる。
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
