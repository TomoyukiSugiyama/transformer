use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::language_model::LanguageModel;

/// 固定シードで val コーパスからランダム窓を `n_batches` 個取り、
/// それぞれの cross-entropy loss の平均を返す。
///
/// 同じ checkpoint・同じ val コーパスに対しては毎回同じ値を返すため、
/// 学習中の進捗比較や nanoGPT との直接比較に使える。
///
/// `chunk_len` は通常 `Config::max_len` と同じものを渡す。
/// val コーパスが `chunk_len + 1` token に満たない場合は `f32::NAN` を返す。
pub fn compute_val_loss(
    model: &mut LanguageModel,
    val_ids: &[usize],
    chunk_len: usize,
    n_batches: usize,
    seed: u64,
) -> f32 {
    if val_ids.len() <= chunk_len {
        return f32::NAN;
    }
    let max_offset = val_ids.len() - chunk_len;
    let pad_id = model.pad_id();

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut total = 0.0f32;
    let mut count = 0usize;

    for _ in 0..n_batches {
        let offset = rng.random_range(0..=max_offset);
        let chunk = &val_ids[offset..offset + chunk_len];
        let loss = model.forward_loss(chunk, pad_id);
        total += loss;
        count += 1;
    }

    if count == 0 {
        f32::NAN
    } else {
        total / count as f32
    }
}

/// loss → perplexity (e^loss) 変換。 inf/NaN は素通り。
pub fn perplexity(loss: f32) -> f32 {
    loss.exp()
}
