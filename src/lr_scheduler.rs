use std::f32::consts::PI;

pub struct LrScheduler {
    lr_max: f32,
    lr_min: f32,
    warmup_steps: usize,
    total_steps: usize,
}

impl LrScheduler {
    pub fn new(lr_max: f32, lr_min: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            lr_max,
            lr_min,
            warmup_steps,
            total_steps,
        }
    }

    pub fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.lr_max * (step as f32 / self.warmup_steps as f32)
        } else {
            let progress =
                (step - self.warmup_steps) as f32 / (self.total_steps - self.warmup_steps) as f32;
            let progress = progress.min(1.0);
            self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + (PI * progress).cos())
        }
    }
}
