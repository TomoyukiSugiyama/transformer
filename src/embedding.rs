use std::io::Error;
use std::io::ErrorKind;
use std::io::Result;

use rand::RngExt;
use rand::rng;

use crate::adam_w::AdamW;
use crate::checkpoint::Checkpointable;
use crate::checkpoint::WeightMap;
use crate::matrix::Matrix;

pub struct Embedding {
    weight: Vec<Vec<f32>>,      // [vocab_size, d_model]
    grad_weight: Vec<Vec<f32>>, // [vocab_size, d_model]
    cache_ids: Vec<usize>,
    vocab_size: usize,
    d_model: usize,
    pad_id: Option<usize>,
}

impl Embedding {
    pub fn new(vocab_size: usize, d_model: usize, pad_id: Option<usize>) -> Self {
        let mut rng = rng();
        let scale = (1.0 / d_model as f32).sqrt();
        let mut weight: Vec<Vec<f32>> = (0..vocab_size)
            .map(|_| {
                (0..d_model)
                    .map(|_| rng.random_range(-scale..scale))
                    .collect()
            })
            .collect();

        if let Some(pad) = pad_id {
            weight[pad] = vec![0.0; d_model];
        }

        Self {
            weight,
            grad_weight: vec![vec![0.0; d_model]; vocab_size],
            cache_ids: Vec::new(),
            vocab_size,
            d_model,
            pad_id,
        }
    }

    fn lookup(&self, token_id: usize) -> &[f32] {
        assert!(token_id < self.vocab_size, "token_id out of vocab range");
        &self.weight[token_id]
    }

    /// Matrix 直叩き forward (Phase 7 高速化で導入)。
    /// 戻り値は (seq_len, d_model) の Matrix。
    pub fn forward_matrix(&mut self, token_ids: &[usize]) -> Matrix {
        self.cache_ids = token_ids.to_vec();
        let scale = (self.d_model as f32).sqrt();
        let n = token_ids.len();
        let d = self.d_model;
        let mut data = vec![0.0f32; n * d];
        for (i, &id) in token_ids.iter().enumerate() {
            let row = self.lookup(id);
            let dst = &mut data[i * d..(i + 1) * d];
            for j in 0..d {
                dst[j] = row[j] * scale;
            }
        }
        Matrix::from_flat(data, n, d)
    }

    /// 旧 API: jagged。 内部で Matrix 版を呼ぶ。
    pub fn forward(&mut self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        self.forward_matrix(token_ids).to_jagged()
    }

    /// 推論専用: 単一 token id を 1 ベクトルに埋め込む (内部 cache は触らない)。
    /// KV cache 利用時の 1 token 前進で使用。
    pub fn forward_one(&self, token_id: usize) -> Vec<f32> {
        let scale = (self.d_model as f32).sqrt();
        self.lookup(token_id).iter().map(|&v| v * scale).collect()
    }

    /// Matrix 直叩き backward (Phase 7 高速化で導入)。
    pub fn backward_matrix(&mut self, dl_dx: &Matrix) {
        let scale = (self.d_model as f32).sqrt();
        let d = self.d_model;
        for (i, &id) in self.cache_ids.iter().enumerate() {
            if let Some(pad) = self.pad_id {
                if id == pad {
                    continue;
                }
            }
            let dx_row = dl_dx.row(i);
            for j in 0..d {
                self.grad_weight[id][j] += dx_row[j] * scale;
            }
        }
    }

    /// 旧 API: jagged。
    pub fn backward(&mut self, dl_dx: &[Vec<f32>]) {
        let dy = Matrix::from_jagged(dl_dx);
        self.backward_matrix(&dy);
    }

    pub fn zero_grad(&mut self) {
        for row in &mut self.grad_weight {
            row.fill(0.0);
        }
    }

    pub fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        for id in 0..self.vocab_size {
            if let Some(pad) = self.pad_id {
                if id == pad {
                    continue;
                }
            }

            opt.step_vector(
                &format!("{prefix}.{id}"),
                &mut self.weight[id],
                &self.grad_weight[id],
            );
        }
    }
}

impl Checkpointable for Embedding {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_scalar("vocab_size", self.vocab_size as u64);
        map.insert_scalar("d_model", self.d_model as u64);
        map.insert_scalar("has_pad_id", self.pad_id.is_some() as u64);
        map.insert_scalar("pad_id", self.pad_id.unwrap_or(0) as u64);
        map.insert_matrix("weight", self.weight.clone());
        map
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> Result<()> {
        let vocab_size = map.get_scalar("vocab_size")? as usize;
        let d_model = map.get_scalar("d_model")? as usize;
        let has_pad_id = map.get_scalar("has_pad_id")? == 1;
        let pad_id = if has_pad_id {
            Some(map.get_scalar("pad_id")? as usize)
        } else {
            None
        };
        if vocab_size != self.vocab_size || d_model != self.d_model || pad_id != self.pad_id {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "embedding config mismatch",
            ));
        }
        self.weight = map.get_matrix("weight")?.clone();

        Ok(())
    }
}
