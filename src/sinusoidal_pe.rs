use crate::matrix::Matrix;

pub struct SinusoidalPE {
    table: Vec<Vec<f32>>,
}

impl SinusoidalPE {
    pub fn new(max_len: usize, d_model: usize) -> Self {
        assert!(d_model % 2 == 0, "d_model must be even");
        let table: Vec<Vec<f32>> = (0..max_len)
            .map(|pos| {
                (0..d_model)
                    .map(|i| {
                        let div = 10000_f32.powf((i / 2 * 2) as f32 / d_model as f32);
                        let angle = pos as f32 / div;
                        if i % 2 == 0 { angle.sin() } else { angle.cos() }
                    })
                    .collect()
            })
            .collect();
        Self { table }
    }

    /// 推論専用: 単一 token に対して指定位置の位置エンコーディングを加算する。
    /// KV cache を使った逐次デコードで使用。
    pub fn add_at(&self, embedding: &[f32], pos: usize) -> Vec<f32> {
        assert!(pos < self.table.len(), "pos {} exceeds max_len", pos);
        embedding
            .iter()
            .zip(self.table[pos].iter())
            .map(|(e, pe)| e + pe)
            .collect()
    }

    /// Matrix 直叩き forward (Phase 7 高速化で導入)。
    /// 入力 (seq_len, d_model) に対し各行に PE を加算した新しい Matrix を返す。
    pub fn forward_matrix(&self, token_emb: &Matrix) -> Matrix {
        let (seq_len, d_model) = token_emb.shape();
        assert!(seq_len <= self.table.len(), "seq_len exceeded max_len");
        let mut data = token_emb.data().to_vec();
        for pos in 0..seq_len {
            let dst = &mut data[pos * d_model..(pos + 1) * d_model];
            let pe = &self.table[pos];
            for j in 0..d_model {
                dst[j] += pe[j];
            }
        }
        Matrix::from_flat(data, seq_len, d_model)
    }

    /// 旧 API: jagged。
    pub fn forward(&self, token_emb: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let m = Matrix::from_jagged(token_emb);
        self.forward_matrix(&m).to_jagged()
    }
}
