use std::collections::HashMap;

use crate::checkpoint::{Checkpointable, WeightMap};
use crate::matrix::Matrix;

pub struct AdamWParam {
    data: Vec<f32>, // パラメータ本体(W1, W2, b など)
    m: Vec<f32>,    // 一次モーメント
    v: Vec<f32>,    // 二次モーメント
}

impl AdamWParam {
    pub fn new(data: Vec<f32>) -> Self {
        let n = data.len();
        Self {
            data,
            m: vec![0.0f32; n],
            v: vec![0.0f32; n],
        }
    }

    pub fn step(
        &mut self,
        grad: &[f32],
        t: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
    ) {
        let t = t as f32;

        for i in 0..self.data.len() {
            let g = grad[i];
            // モーメント更新
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;

            let m_hat = self.m[i] / (1.0 - beta1.powf(t));
            let v_hat = self.v[i] / (1.0 - beta2.powf(t));

            self.data[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * self.data[i]);
        }
    }
}

pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    step_count: usize,
    moments: HashMap<String, AdamWParam>,
    grad_scale: f32,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            wd: 0.01,
            step_count: 0,
            moments: HashMap::new(),
            grad_scale: 1.0,
        }
    }

    pub fn set_grad_scale(&mut self, batch_size: usize) {
        self.grad_scale = 1.0 / batch_size as f32;
    }

    pub fn reset_grad_scale(&mut self) {
        self.grad_scale = 1.0;
    }

    pub fn increment_step(&mut self) {
        self.step_count += 1;
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    pub fn step_matrix(&mut self, param_id: &str, w: &mut Vec<Vec<f32>>, grad: &[Vec<f32>]) {
        let flat_w: Vec<f32> = w.iter().flat_map(|row| row.iter().cloned()).collect();
        let flat_g: Vec<f32> = grad
            .iter()
            .flat_map(|row| row.iter().cloned().map(|g| g * self.grad_scale))
            .collect();

        let param = self
            .moments
            .entry(param_id.to_string())
            .or_insert_with(|| AdamWParam::new(flat_w));
        param.step(
            &flat_g,
            self.step_count,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.wd,
        );

        let cols = w[0].len();
        for (row, chunk) in w.iter_mut().zip(param.data.chunks(cols)) {
            row.copy_from_slice(chunk);
        }
    }

    pub fn step_vector(&mut self, key: &str, param: &mut Vec<f32>, grad: &[f32]) {
        let mut param_mat = vec![param.clone()];
        self.step_matrix(key, &mut param_mat, &[grad.to_vec()]);
        *param = param_mat.remove(0);
    }

    /// flat な `Matrix` を直接受け取る版。`step_matrix` のような jagged ↔ flat の変換が
    /// 不要なため、`Matrix` を内部表現に持つレイヤー（移行後）から使う。
    pub fn step_matrix_flat(&mut self, param_id: &str, w: &mut Matrix, grad: &Matrix) {
        assert_eq!(
            w.shape(),
            grad.shape(),
            "step_matrix_flat shape mismatch: {:?} vs {:?}",
            w.shape(),
            grad.shape()
        );
        let scale = self.grad_scale;
        let scaled_grad: Vec<f32> = grad.data().iter().map(|&g| g * scale).collect();

        let param = self
            .moments
            .entry(param_id.to_string())
            .or_insert_with(|| AdamWParam::new(w.data().to_vec()));
        param.step(
            &scaled_grad,
            self.step_count,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.wd,
        );

        w.data_mut().copy_from_slice(&param.data);
    }
}

impl Checkpointable for AdamW {
    fn to_weight_map(&self) -> crate::checkpoint::WeightMap {
        let mut map = WeightMap::new();

        map.insert_scalar("lr", self.lr.to_bits() as u64);
        map.insert_scalar("beta1", self.beta1.to_bits() as u64);
        map.insert_scalar("beta2", self.beta2.to_bits() as u64);
        map.insert_scalar("eps", self.eps.to_bits() as u64);
        map.insert_scalar("wd", self.wd.to_bits() as u64);
        map.insert_scalar("step_count", self.step_count as u64);

        let mut keys: Vec<_> = self.moments.keys().collect();
        keys.sort();
        for key in keys {
            let p = &self.moments[key];
            map.insert_vector(&format!("moments.{key}.data"), p.data.clone());
            map.insert_vector(&format!("moments.{key}.m"), p.m.clone());
            map.insert_vector(&format!("moments.{key}.v"), p.v.clone());
        }

        map
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> std::io::Result<()> {
        self.lr = f32::from_bits(map.get_scalar("lr")? as u32);
        self.beta1 = f32::from_bits(map.get_scalar("beta1")? as u32);
        self.beta2 = f32::from_bits(map.get_scalar("beta2")? as u32);
        self.eps = f32::from_bits(map.get_scalar("eps")? as u32);
        self.wd = f32::from_bits(map.get_scalar("wd")? as u32);
        self.step_count = map.get_scalar("step_count")? as usize;

        self.moments.clear();
        let mut param_ids: Vec<String> = map
            .vector_keys()
            .filter(|k| k.starts_with("moments.") && k.ends_with(".data"))
            .map(|k| {
                k.strip_prefix("moments.")
                    .unwrap()
                    .strip_suffix(".data")
                    .unwrap()
                    .to_string()
            })
            .collect();
        param_ids.sort();

        for param_id in param_ids {
            let data = map.get_vector(&format!("moments.{param_id}.data"))?.clone();
            let m = map.get_vector(&format!("moments.{param_id}.m"))?.clone();
            let v = map.get_vector(&format!("moments.{param_id}.v"))?.clone();
            self.moments.insert(param_id, AdamWParam { data, m, v });
        }
        Ok(())
    }
}
