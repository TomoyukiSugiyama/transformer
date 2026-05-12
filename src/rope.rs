use crate::matrix::Matrix;

/// Rotary Position Embedding (Su et al. 2021, RoFormer)。
///
/// 位置 `m` におけるトークンの head ベクトル `(x_0, x_1, ..., x_{d_head-1})` を、
/// 2 次元ペア `(x_{2i}, x_{2i+1})` ごとに角度 `m * θ_i` だけ回転する。
/// `θ_i = base^(-2i/d_head)`。 学習可能パラメータなし (cos / sin テーブルは固定)。
///
/// 適用は **Q, K にのみ** 行い (V には掛けない)、 attention の内積が
/// `(m - n)` のみに依存するという相対位置不変性を実現する。
pub struct Rope {
    cos_table: Vec<Vec<f32>>, // (max_len, d_head/2)
    sin_table: Vec<Vec<f32>>, // (max_len, d_head/2)
    max_len: usize,
    d_head: usize,
}

impl Rope {
    pub fn new(max_len: usize, d_head: usize, base: f32) -> Self {
        assert!(d_head % 2 == 0, "d_head must be even for RoPE");
        let half = d_head / 2;
        let mut cos_table = vec![vec![0.0; half]; max_len];
        let mut sin_table = vec![vec![0.0; half]; max_len];
        // f64 で precompute して f32 に落とす (小数誤差を最小化)
        for pos in 0..max_len {
            for i in 0..half {
                let theta = (base as f64).powf(-2.0 * i as f64 / d_head as f64);
                let angle = pos as f64 * theta;
                cos_table[pos][i] = angle.cos() as f32;
                sin_table[pos][i] = angle.sin() as f32;
            }
        }
        Self {
            cos_table,
            sin_table,
            max_len,
            d_head,
        }
    }

    pub fn d_head(&self) -> usize {
        self.d_head
    }

    #[allow(dead_code)]
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// 順方向の回転を in-place で適用する (forward)。
    /// 入力 `x` の shape は `(seq, d_head)`。
    pub fn apply_in_place(&self, x: &mut Matrix) {
        let seq = x.rows();
        assert_eq!(x.cols(), self.d_head, "RoPE: x.cols mismatch");
        assert!(
            seq <= self.max_len,
            "RoPE: seq {} exceeds max_len {}",
            seq,
            self.max_len
        );
        let half = self.d_head / 2;
        for pos in 0..seq {
            let cos_row = &self.cos_table[pos];
            let sin_row = &self.sin_table[pos];
            let row = x.row_mut(pos);
            for i in 0..half {
                let x0 = row[2 * i];
                let x1 = row[2 * i + 1];
                let c = cos_row[i];
                let s = sin_row[i];
                row[2 * i] = c * x0 - s * x1;
                row[2 * i + 1] = s * x0 + c * x1;
            }
        }
    }

    /// 単一行 (1 token 分の head ベクトル) を **指定位置の角度** で in-place 回転する。
    /// KV cache を使った逐次推論で、 新規 token 1 つだけを 「位置 `pos` に置いた」 として
    /// 回転するために使う。 通常の `apply_in_place` は行 index = 位置と仮定するので、
    /// 過去 cache (位置 0..cur_len) と並べた瞬間の新規 token (位置 cur_len) には使えない。
    pub fn apply_at_position(&self, row: &mut [f32], pos: usize) {
        assert_eq!(row.len(), self.d_head, "RoPE: row.len mismatch");
        assert!(
            pos < self.max_len,
            "RoPE: pos {} exceeds max_len {}",
            pos,
            self.max_len
        );
        let half = self.d_head / 2;
        let cos_row = &self.cos_table[pos];
        let sin_row = &self.sin_table[pos];
        for i in 0..half {
            let x0 = row[2 * i];
            let x1 = row[2 * i + 1];
            let c = cos_row[i];
            let s = sin_row[i];
            row[2 * i] = c * x0 - s * x1;
            row[2 * i + 1] = s * x0 + c * x1;
        }
    }

    /// 逆方向の回転を in-place で適用する (backward, R(-θ))。
    /// 順回転に対する勾配の伝播に用いる: dL/dx = R(-θ) · dL/dy。
    pub fn apply_backward_in_place(&self, x: &mut Matrix) {
        let seq = x.rows();
        assert_eq!(x.cols(), self.d_head, "RoPE: x.cols mismatch");
        assert!(
            seq <= self.max_len,
            "RoPE: seq {} exceeds max_len {}",
            seq,
            self.max_len
        );
        let half = self.d_head / 2;
        for pos in 0..seq {
            let cos_row = &self.cos_table[pos];
            let sin_row = &self.sin_table[pos];
            let row = x.row_mut(pos);
            for i in 0..half {
                let x0 = row[2 * i];
                let x1 = row[2 * i + 1];
                let c = cos_row[i];
                let s = sin_row[i];
                row[2 * i] = c * x0 + s * x1;
                row[2 * i + 1] = -s * x0 + c * x1;
            }
        }
    }
}

impl Clone for Rope {
    fn clone(&self) -> Self {
        Self {
            cos_table: self.cos_table.clone(),
            sin_table: self.sin_table.clone(),
            max_len: self.max_len,
            d_head: self.d_head,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    #[test]
    fn position_zero_is_identity() {
        // 位置 0 では cos=1, sin=0 で回転なし → 入力そのまま
        let rope = Rope::new(8, 4, 10000.0);
        let x = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let mut m = Matrix::from_jagged(&x);
        rope.apply_in_place(&mut m);
        let y = m.to_jagged();
        for i in 0..4 {
            assert!(
                (y[0][i] - x[0][i]).abs() < 1e-6,
                "pos 0 should be identity at dim {i}: got {}, expected {}",
                y[0][i],
                x[0][i]
            );
        }
    }

    #[test]
    fn rotation_preserves_norm() {
        // 回転は等長変換: ||rotate(x)|| == ||x||
        let rope = Rope::new(8, 8, 10000.0);
        let mut rng = SmallRng::seed_from_u64(123);
        let x: Vec<Vec<f32>> = (0..8)
            .map(|_| (0..8).map(|_| rng.random_range(-2.0..2.0)).collect())
            .collect();
        let original_norms: Vec<f32> = (0..8)
            .map(|i| x[i].iter().map(|v| v * v).sum::<f32>().sqrt())
            .collect();
        let mut m = Matrix::from_jagged(&x);
        rope.apply_in_place(&mut m);
        for i in 0..8 {
            let n: f32 = m.row(i).iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (n - original_norms[i]).abs() < 1e-4,
                "norm should be preserved at row {i}: got {n}, expected {}",
                original_norms[i]
            );
        }
    }

    #[test]
    fn forward_then_backward_is_identity() {
        // R^T · R · x == x (回転 → 逆回転で元に戻る)
        let rope = Rope::new(8, 6, 10000.0);
        let original = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![-1.0, -2.0, 0.5, -0.5, 1.5, -1.5],
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ];
        let mut m = Matrix::from_jagged(&original);
        rope.apply_in_place(&mut m);
        rope.apply_backward_in_place(&mut m);
        let restored = m.to_jagged();
        for i in 0..3 {
            for j in 0..6 {
                assert!(
                    (restored[i][j] - original[i][j]).abs() < 1e-4,
                    "restore mismatch at ({i},{j}): got {}, expected {}",
                    restored[i][j],
                    original[i][j]
                );
            }
        }
    }

    #[test]
    fn relative_position_invariance() {
        // RoPE の中核性質: <Q'(m), K'(n)> は (m-n) のみに依存する。
        // 同じ Q, K でも位置の絶対値だけ変えると内積は同じ。
        let d_head = 8;
        let rope = Rope::new(20, d_head, 10000.0);
        let mut rng = SmallRng::seed_from_u64(42);
        let q: Vec<f32> = (0..d_head).map(|_| rng.random_range(-1.0..1.0)).collect();
        let k: Vec<f32> = (0..d_head).map(|_| rng.random_range(-1.0..1.0)).collect();

        // Q を位置 m、 K を位置 n に置いて回転後の内積を取る
        let dot_at = |m: usize, n: usize| -> f32 {
            let mut q_mat = Matrix::zeros(m + 1, d_head);
            for j in 0..d_head {
                q_mat.row_mut(m)[j] = q[j];
            }
            let mut k_mat = Matrix::zeros(n + 1, d_head);
            for j in 0..d_head {
                k_mat.row_mut(n)[j] = k[j];
            }
            rope.apply_in_place(&mut q_mat);
            rope.apply_in_place(&mut k_mat);
            q_mat
                .row(m)
                .iter()
                .zip(k_mat.row(n).iter())
                .map(|(a, b)| a * b)
                .sum()
        };

        // 全て差 = 2 になる組合せ
        let d_2_0 = dot_at(2, 0);
        let d_5_3 = dot_at(5, 3);
        let d_15_13 = dot_at(15, 13);
        assert!(
            (d_2_0 - d_5_3).abs() < 1e-4,
            "relative pos invariance failed: dot(2,0)={d_2_0} vs dot(5,3)={d_5_3}"
        );
        assert!(
            (d_2_0 - d_15_13).abs() < 1e-4,
            "relative pos invariance failed: dot(2,0)={d_2_0} vs dot(15,13)={d_15_13}"
        );
    }

    #[test]
    fn precomputed_tables_match_formula() {
        // cos/sin table が定義式 cos(m·θ_i), sin(m·θ_i) と一致するか
        // θ_i = base^(-2i/d_head)
        let d_head = 4;
        let base: f32 = 10000.0;
        let rope = Rope::new(3, d_head, base);
        for pos in 0..3 {
            for i in 0..2 {
                let theta = base.powf(-2.0 * i as f32 / d_head as f32);
                let angle = pos as f32 * theta;
                let expected_cos = angle.cos();
                let expected_sin = angle.sin();
                assert!(
                    (rope.cos_table[pos][i] - expected_cos).abs() < 1e-5,
                    "cos mismatch at pos={pos}, i={i}: got {}, expected {}",
                    rope.cos_table[pos][i],
                    expected_cos
                );
                assert!(
                    (rope.sin_table[pos][i] - expected_sin).abs() < 1e-5,
                    "sin mismatch at pos={pos}, i={i}: got {}, expected {}",
                    rope.sin_table[pos][i],
                    expected_sin
                );
            }
        }
    }

    #[test]
    fn apply_at_position_matches_full_sequence_rotation() {
        // apply_at_position(row, pos) と apply_in_place で pos 行目に置いたときの
        // 結果が一致する (= 単一行のショートカットが正しい) ことを確認。
        let d_head = 8;
        let rope = Rope::new(20, d_head, 10000.0);
        let mut rng = SmallRng::seed_from_u64(99);
        let single: Vec<f32> = (0..d_head).map(|_| rng.random_range(-1.0..1.0)).collect();

        for pos in [0usize, 1, 5, 19] {
            // (pos+1, d_head) の行列を作って pos 行目だけに値を入れて全体回転
            let mut full = Matrix::zeros(pos + 1, d_head);
            for j in 0..d_head {
                full.row_mut(pos)[j] = single[j];
            }
            rope.apply_in_place(&mut full);
            let expected = full.row(pos).to_vec();

            // 単一行を apply_at_position で回転
            let mut single_copy = single.clone();
            rope.apply_at_position(&mut single_copy, pos);

            for j in 0..d_head {
                assert!(
                    (single_copy[j] - expected[j]).abs() < 1e-6,
                    "mismatch at pos={pos}, j={j}: single={}, full={}",
                    single_copy[j],
                    expected[j]
                );
            }
        }
    }

    #[test]
    fn backward_matches_numerical_gradient() {
        // L = sum(rotate(x)) に対する dL/dx を解析的・数値的に比較
        // 解析: dL/dy = ones, dL/dx = R^T(ones)
        let d_head = 4;
        let seq = 3;
        let rope = Rope::new(seq, d_head, 10000.0);
        let mut rng = SmallRng::seed_from_u64(7);
        let x: Vec<Vec<f32>> = (0..seq)
            .map(|_| (0..d_head).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();

        let mut dl_dy = Matrix::from_jagged(&vec![vec![1.0; d_head]; seq]);
        rope.apply_backward_in_place(&mut dl_dy);
        let dl_dx_analytic = dl_dy.to_jagged();

        let h = 1e-3;
        for i in 0..seq {
            for j in 0..d_head {
                let mut x_p = x.clone();
                x_p[i][j] += h;
                let mut m_p = Matrix::from_jagged(&x_p);
                rope.apply_in_place(&mut m_p);
                let l_p: f32 = m_p.to_jagged().iter().flatten().sum();

                let mut x_m = x.clone();
                x_m[i][j] -= h;
                let mut m_m = Matrix::from_jagged(&x_m);
                rope.apply_in_place(&mut m_m);
                let l_m: f32 = m_m.to_jagged().iter().flatten().sum();

                let num = (l_p - l_m) / (2.0 * h);
                assert!(
                    (dl_dx_analytic[i][j] - num).abs() < 1e-3,
                    "mismatch at ({i},{j}): analytic={}, numerical={}",
                    dl_dx_analytic[i][j],
                    num
                );
            }
        }
    }
}
