//! KV Cache for incremental autoregressive decoding.
//!
//! 推論時、 過去 token の K, V を保持しておくことで 1 token 生成あたりの計算量を
//! `O(n²·d) → O(n·d)` に削減する。
//!
//! # 格納するもの
//! - `k`: **RoPE 適用後** の K (shape `(cur_len, d_kv_model)`)
//!   - RoPE は位置に依存する回転なので、 K_pos が一度回転されたら以降そのまま再利用できる
//! - `v`: **未回転** の V (V には RoPE を掛けないため)
//! - `cur_len`: これまでに何 token 分キャッシュされたか
//!
//! # 使い方
//! 1. 推論開始時に `Transformer::init_kv_caches(max_len)` で各 layer 分の `KvCache` を作る
//! 2. プロンプトを 1 token ずつ `forward_step` に流して cache を構築 (= prefill)
//! 3. 各生成 step で `forward_step(new_token, &mut caches)` を呼んで新 token の logits を取得
//!
//! # 学習との関係
//! - 学習パスは一切触らない (`forward` / `backward` は従来通り)
//! - cache は推論専用で、 dropout は事前に `set_training(false)` で無効化される前提

use crate::matrix::Matrix;

/// 単一 attention 層のための K/V キャッシュ。
pub struct KvCache {
    /// RoPE 適用後の K。 shape `(cur_len, d_kv_model)`。
    /// Sinusoidal PE の場合は projection 後そのまま (回転は適用されない)。
    k: Matrix,
    /// 未回転の V。 shape `(cur_len, d_kv_model)`。
    v: Matrix,
    cur_len: usize,
    capacity: usize,
    d_kv_model: usize,
}

impl KvCache {
    /// `capacity`: 保持できる最大 token 数 (= max_len)
    /// `d_kv_model`: モデル次元 (全 head 連結後の次元)
    pub fn new(capacity: usize, d_kv_model: usize) -> Self {
        Self {
            k: Matrix::zeros(0, d_kv_model),
            v: Matrix::zeros(0, d_kv_model),
            cur_len: 0,
            capacity,
            d_kv_model,
        }
    }

    pub fn cur_len(&self) -> usize {
        self.cur_len
    }

    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[allow(dead_code)]
    pub fn d_kv_model(&self) -> usize {
        self.d_kv_model
    }

    /// 現在格納されている K (shape `(cur_len, d_kv_model)`) への参照。
    pub fn k(&self) -> &Matrix {
        &self.k
    }

    /// 現在格納されている V (shape `(cur_len, d_kv_model)`) への参照。
    pub fn v(&self) -> &Matrix {
        &self.v
    }

    /// 新しい K, V の 1 行を末尾に追加する。
    /// `k_row` は **既に RoPE 適用済** であること。
    /// `v_row` は未回転 (V には RoPE を掛けないため)。
    ///
    /// 容量を超えると panic。
    pub fn append(&mut self, k_row: &[f32], v_row: &[f32]) {
        assert!(
            self.cur_len < self.capacity,
            "KvCache overflow: cur_len {} >= capacity {}",
            self.cur_len,
            self.capacity
        );
        assert_eq!(
            k_row.len(),
            self.d_kv_model,
            "KvCache append: k_row.len {} != d_kv_model {}",
            k_row.len(),
            self.d_kv_model
        );
        assert_eq!(
            v_row.len(),
            self.d_kv_model,
            "KvCache append: v_row.len {} != d_kv_model {}",
            v_row.len(),
            self.d_kv_model
        );

        let new_len = self.cur_len + 1;
        // 旧データ + 新行を連結した新しい flat data を作って Matrix に詰める。
        // 1 step あたり O(cur_len * d_kv_model) の copy だが、 attention の O(cur_len * d_kv_model²)
        // に比べれば微小。 累計でも O(n² · d) で旧 forward と同じオーダー。
        let mut new_k = Vec::with_capacity(new_len * self.d_kv_model);
        new_k.extend_from_slice(self.k.data());
        new_k.extend_from_slice(k_row);
        self.k = Matrix::from_flat(new_k, new_len, self.d_kv_model);

        let mut new_v = Vec::with_capacity(new_len * self.d_kv_model);
        new_v.extend_from_slice(self.v.data());
        new_v.extend_from_slice(v_row);
        self.v = Matrix::from_flat(new_v, new_len, self.d_kv_model);

        self.cur_len = new_len;
    }

    /// 全ての履歴を破棄して空に戻す。 別プロンプトでの再利用に。
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.k = Matrix::zeros(0, self.d_kv_model);
        self.v = Matrix::zeros(0, self.d_kv_model);
        self.cur_len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newly_created_cache_is_empty() {
        let c = KvCache::new(16, 8);
        assert_eq!(c.cur_len(), 0);
        assert_eq!(c.capacity(), 16);
        assert_eq!(c.d_kv_model(), 8);
        assert_eq!(c.k().shape(), (0, 8));
        assert_eq!(c.v().shape(), (0, 8));
    }

    #[test]
    fn append_grows_by_one_row_each_call() {
        let mut c = KvCache::new(4, 3);
        c.append(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]);
        assert_eq!(c.cur_len(), 1);
        assert_eq!(c.k().shape(), (1, 3));
        assert_eq!(c.k().row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(c.v().row(0), &[10.0, 20.0, 30.0]);

        c.append(&[4.0, 5.0, 6.0], &[40.0, 50.0, 60.0]);
        assert_eq!(c.cur_len(), 2);
        assert_eq!(c.k().shape(), (2, 3));
        assert_eq!(c.k().row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(c.k().row(1), &[4.0, 5.0, 6.0]);
        assert_eq!(c.v().row(0), &[10.0, 20.0, 30.0]);
        assert_eq!(c.v().row(1), &[40.0, 50.0, 60.0]);
    }

    #[test]
    #[should_panic(expected = "KvCache overflow")]
    fn append_panics_when_exceeding_capacity() {
        let mut c = KvCache::new(2, 2);
        c.append(&[1.0, 2.0], &[3.0, 4.0]);
        c.append(&[5.0, 6.0], &[7.0, 8.0]);
        c.append(&[9.0, 10.0], &[11.0, 12.0]); // panic
    }

    #[test]
    fn reset_clears_history() {
        let mut c = KvCache::new(4, 2);
        c.append(&[1.0, 2.0], &[3.0, 4.0]);
        c.append(&[5.0, 6.0], &[7.0, 8.0]);
        assert_eq!(c.cur_len(), 2);
        c.reset();
        assert_eq!(c.cur_len(), 0);
        assert_eq!(c.k().shape(), (0, 2));
        // 再利用可能
        c.append(&[100.0, 200.0], &[300.0, 400.0]);
        assert_eq!(c.cur_len(), 1);
        assert_eq!(c.k().row(0), &[100.0, 200.0]);
    }
}
