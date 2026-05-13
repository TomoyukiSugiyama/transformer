use std::f32::consts::PI;

/// 学習率スケジュールの種類。
///
/// - `WarmupCosine`: 古典的な warmup + cosine decay。 nanoGPT / GPT-3 慣例。
///   total_steps の最後に lr_min に到達。
/// - `WarmupStableDecay`: warmup → stable (lr_max 一定) → decay の 3 段。 MiniCPM や DeepSeek が採用。
///   ハイパラ調整なしに total_steps を伸ばしたり最終 checkpoint をそのまま追加学習に流用できる利点がある
///   (cosine は total_steps を変えると過去の lr 軌道が変わってしまう)。
///   decay 区間の関数形は `1 - sqrt(progress)` (MiniCPM 慣例) を採用。
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LrScheduleKind {
    WarmupCosine,
    /// stable_steps: warmup 完了後、 decay 開始までの「lr_max 一定」 区間長。
    /// stable_steps=0 にすると warmup → decay (cosine 不使用) という挙動になる。
    WarmupStableDecay { stable_steps: usize },
}

pub struct LrScheduler {
    lr_max: f32,
    lr_min: f32,
    warmup_steps: usize,
    total_steps: usize,
    kind: LrScheduleKind,
}

impl LrScheduler {
    /// 既存 API: warmup + cosine decay (デフォルト)。
    #[allow(dead_code)]
    pub fn new(lr_max: f32, lr_min: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self::with_kind(
            lr_max,
            lr_min,
            warmup_steps,
            total_steps,
            LrScheduleKind::WarmupCosine,
        )
    }

    /// schedule kind を明示的に指定するコンストラクタ。
    pub fn with_kind(
        lr_max: f32,
        lr_min: f32,
        warmup_steps: usize,
        total_steps: usize,
        kind: LrScheduleKind,
    ) -> Self {
        if let LrScheduleKind::WarmupStableDecay { stable_steps } = kind {
            assert!(
                warmup_steps + stable_steps <= total_steps,
                "WarmupStableDecay: warmup + stable ({}) must be <= total_steps ({})",
                warmup_steps + stable_steps,
                total_steps
            );
        }
        Self {
            lr_max,
            lr_min,
            warmup_steps,
            total_steps,
            kind,
        }
    }

    pub fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            return self.lr_max * (step as f32 / self.warmup_steps as f32);
        }
        match self.kind {
            LrScheduleKind::WarmupCosine => {
                let progress = (step - self.warmup_steps) as f32
                    / (self.total_steps - self.warmup_steps) as f32;
                let progress = progress.min(1.0);
                self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + (PI * progress).cos())
            }
            LrScheduleKind::WarmupStableDecay { stable_steps } => {
                let stable_end = self.warmup_steps + stable_steps;
                if step < stable_end {
                    return self.lr_max;
                }
                // decay 区間: 1 - sqrt(progress) で lr_max → lr_min
                // progress=0 → lr_max, progress=1 → lr_min
                let decay_total = self.total_steps - stable_end;
                if decay_total == 0 {
                    return self.lr_min;
                }
                let progress = (step - stable_end) as f32 / decay_total as f32;
                let progress = progress.min(1.0);
                self.lr_min + (self.lr_max - self.lr_min) * (1.0 - progress.sqrt())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn warmup_starts_at_zero_and_reaches_max_at_warmup_end() {
        let s = LrScheduler::new(1.0, 0.1, 100, 1000);
        assert!(approx_eq(s.get_lr(0), 0.0, 1e-6));
        assert!(approx_eq(s.get_lr(50), 0.5, 1e-6));
        // step=100 なら warmup を抜けて cosine の頭 (= lr_max)
        assert!(approx_eq(s.get_lr(100), 1.0, 1e-3));
    }

    #[test]
    fn warmup_cosine_reaches_lr_min_at_total_steps() {
        let s = LrScheduler::new(1.0, 0.1, 100, 1000);
        assert!(approx_eq(s.get_lr(1000), 0.1, 1e-3));
    }

    #[test]
    fn warmup_cosine_midpoint_is_at_half_amplitude() {
        // warmup=100, total=1100 → decay=1000、 中点 step=600 で progress=0.5
        // → lr = lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*0.5)) = lr_min + 0.5*(lr_max-lr_min)
        //      = 0.1 + 0.5 * 0.9 = 0.55
        let s = LrScheduler::new(1.0, 0.1, 100, 1100);
        assert!(approx_eq(s.get_lr(600), 0.55, 1e-3));
    }

    #[test]
    fn wsd_holds_lr_max_during_stable_phase() {
        let s = LrScheduler::with_kind(
            1.0,
            0.1,
            100,
            1000,
            LrScheduleKind::WarmupStableDecay { stable_steps: 500 },
        );
        // warmup
        assert!(approx_eq(s.get_lr(50), 0.5, 1e-6));
        // stable (warmup_end=100 〜 stable_end=600 の間)
        assert!(approx_eq(s.get_lr(100), 1.0, 1e-3));
        assert!(approx_eq(s.get_lr(300), 1.0, 1e-3));
        assert!(approx_eq(s.get_lr(599), 1.0, 1e-3));
    }

    #[test]
    fn wsd_decays_after_stable_phase() {
        let s = LrScheduler::with_kind(
            1.0,
            0.1,
            100,
            1000,
            LrScheduleKind::WarmupStableDecay { stable_steps: 500 },
        );
        // decay 開始: step=600 → progress=0 → lr=lr_max=1.0
        assert!(approx_eq(s.get_lr(600), 1.0, 1e-3));
        // 終端: step=1000 → progress=1 → lr=lr_min=0.1
        assert!(approx_eq(s.get_lr(1000), 0.1, 1e-3));
        // 中点 step=800 → progress=0.5 → 1 - sqrt(0.5) ≒ 0.293
        // → lr = 0.1 + 0.9 * 0.293 = 0.364
        assert!(approx_eq(s.get_lr(800), 0.364, 1e-2));
    }

    #[test]
    fn wsd_decays_monotonically() {
        let s = LrScheduler::with_kind(
            1.0,
            0.1,
            100,
            1000,
            LrScheduleKind::WarmupStableDecay { stable_steps: 500 },
        );
        let mut prev = f32::INFINITY;
        for step in (600..=1000).step_by(20) {
            let lr = s.get_lr(step);
            assert!(
                lr <= prev + 1e-5,
                "lr should decrease monotonically: step={step} prev={prev} cur={lr}"
            );
            prev = lr;
        }
    }

    #[test]
    #[should_panic(expected = "warmup + stable")]
    fn wsd_panics_when_warmup_and_stable_exceed_total() {
        let _ = LrScheduler::with_kind(
            1.0,
            0.1,
            500,
            1000,
            LrScheduleKind::WarmupStableDecay { stable_steps: 600 },
        );
    }
}
