pub struct SinusoidalPE {
    table: Vec<Vec<f32>>,
    d_model: usize,
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
        Self { table, d_model }
    }

    pub fn forward(&self, token_enb: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = token_enb.len();
        assert!(seq_len <= self.table.len(), "seq_len exceeded max_len");
        token_enb
            .iter()
            .enumerate()
            .map(|(pos, enb)| {
                enb.iter()
                    .zip(self.table[pos].iter())
                    .map(|(e, pe)| e + pe)
                    .collect()
            })
            .collect()
    }
}
