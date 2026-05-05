use std::collections::HashMap;

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
        }
    }

    pub fn step_matrix(&mut self, param_id: &str, w: &mut Vec<Vec<f32>>, grad: &[Vec<f32>]) {
        self.step_count += 1;

        let flat_w: Vec<f32> = w.iter().flat_map(|row| row.iter().cloned()).collect();
        let flat_g: Vec<f32> = grad.iter().flat_map(|row| row.iter().cloned()).collect();

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
}
