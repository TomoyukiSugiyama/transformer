use std::io::Error;
use std::io::ErrorKind;
use std::io::Result;

use crate::feed_forward::FeedForwardKind;
use crate::kv_cache::KvCache;
use crate::matrix::Matrix;
use crate::normalization::Normalization;
use crate::normalization::NormalizationKind;
use crate::normalization::load_normalization;
use crate::positional_encoding::PositionalEncodingKind;
use crate::rope::Rope;
use crate::{
    adam_w::AdamW,
    checkpoint::{Checkpointable, WeightMap},
    transformer_block::TransformerBlock,
};

pub struct Transformer {
    blocks: Vec<TransformerBlock>,
    final_norm: Box<dyn Normalization>,
}

impl Transformer {
    pub fn new(
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        max_len: usize,
        dropout_p: f32,
        normalization_kind: NormalizationKind,
        feed_forward_kind: FeedForwardKind,
        positional_encoding_kind: PositionalEncodingKind,
    ) -> Self {
        // RoPE 使用時は head ごとに同じ cos/sin テーブルを共有する。
        // 各 block / head に複製されるが、 サイズは max_len * d_head/2 * 2 で
        // d_head=64, max_len=256 の場合 64KB/layer 程度なので無視できる。
        let d_head = d_model / n_heads;
        let rope_template = match positional_encoding_kind {
            PositionalEncodingKind::Rope => Some(Rope::new(max_len, d_head, 10000.0)),
            PositionalEncodingKind::Sinusoidal => None,
        };
        Self {
            blocks: (0..n_layers)
                .map(|_| {
                    TransformerBlock::new(
                        d_model,
                        n_heads,
                        d_ff,
                        dropout_p,
                        normalization_kind,
                        feed_forward_kind,
                        rope_template.clone(),
                    )
                })
                .collect(),
            final_norm: load_normalization(normalization_kind, d_model),
        }
    }

    pub fn set_training(&mut self, training: bool) {
        for block in &mut self.blocks {
            block.set_training(training);
        }
    }

    /// Matrix 直叩き forward (Phase 7 高速化で導入)。
    pub fn forward_matrix(&mut self, x: &Matrix, mask: Option<&Vec<Vec<bool>>>) -> Matrix {
        let mut h = x.clone();
        for i in 0..self.blocks.len() {
            h = self.blocks[i].forward_matrix(&h, mask);
        }
        self.final_norm.forward_matrix(&h)
    }

    /// Matrix 直叩き backward (Phase 7 高速化で導入)。
    pub fn backward_matrix(&mut self, dl_dout: &Matrix) -> Matrix {
        let mut d1 = self.final_norm.backward_matrix(dl_dout);
        for i in (0..self.blocks.len()).rev() {
            d1 = self.blocks[i].backward_matrix(&d1);
        }
        d1
    }

    /// 旧 API: jagged → Matrix 経由。
    pub fn forward(&mut self, x: &[Vec<f32>], mask: Option<&Vec<Vec<bool>>>) -> Vec<Vec<f32>> {
        let xm = Matrix::from_jagged(x);
        self.forward_matrix(&xm, mask).to_jagged()
    }

    /// 旧 API: jagged → Matrix 経由。
    pub fn backward(&mut self, dl_dout: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let dy = Matrix::from_jagged(dl_dout);
        self.backward_matrix(&dy).to_jagged()
    }

    /// 推論専用: 各層の `KvCache` をプロンプト用に作成する。
    /// 容量は `max_len` (= 学習時の context window) に揃える。
    pub fn init_kv_caches(&self, max_len: usize, d_model: usize) -> Vec<KvCache> {
        (0..self.blocks.len())
            .map(|_| KvCache::new(max_len, d_model))
            .collect()
    }

    /// 推論専用: 1 token を全層通して前進。
    /// `caches` は `init_kv_caches` で作った layer 数ぶんの cache を渡す。
    pub fn forward_step(&mut self, x_new: &[f32], caches: &mut [KvCache]) -> Vec<f32> {
        assert_eq!(
            caches.len(),
            self.blocks.len(),
            "Transformer::forward_step: caches.len {} != n_layers {}",
            caches.len(),
            self.blocks.len()
        );
        let mut h = x_new.to_vec();
        for (block, cache) in self.blocks.iter_mut().zip(caches.iter_mut()) {
            h = block.forward_step(&h, cache);
        }
        // final_norm は (1, d_model) として扱う
        let single = vec![h];
        let normed = self.final_norm.forward(&single);
        normed.into_iter().next().unwrap()
    }

    pub fn zero_grad(&mut self) {
        for block in &mut self.blocks {
            block.zero_grad();
        }
        self.final_norm.zero_grad();
    }

    pub fn apply_gradients(&mut self, opt: &mut AdamW, prefix: &str) {
        self.final_norm
            .apply_gradients(opt, &format!("{prefix}.final_norm"));
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.apply_gradients(opt, &format!("{prefix}.block{i}"));
        }
    }
}

impl Checkpointable for Transformer {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_scalar("n_layers", self.blocks.len() as u64);
        for (i, block) in self.blocks.iter().enumerate() {
            map.merge(&format!("blocks.{i}"), block.to_weight_map());
        }
        map.merge("final_norm", self.final_norm.to_weight_map());
        map
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> Result<()> {
        let n_layers = map.get_scalar("n_layers")? as usize;
        if n_layers != self.blocks.len() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "transformer n_layer mismatch",
            ));
        }
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.from_weight_map(&map.scoped(&format!("blocks.{i}")))?;
        }
        self.final_norm.from_weight_map(&map.scoped("final_norm"))?;
        Ok(())
    }
}
