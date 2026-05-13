use std::io::Result;

use crate::{
    adam_w::AdamW,
    checkpoint::{Checkpointable, WeightMap},
    cross_entropy_loss::CrossEntropyLoss,
    embedding::Embedding,
    feed_forward::FeedForwardKind,
    kv_cache::KvCache,
    matrix::Matrix,
    multi_head_attention::causal_mask,
    normalization::NormalizationKind,
    output_head::OutputHead,
    positional_encoding::PositionalEncodingKind,
    sinusoidal_pe::SinusoidalPE,
    tokenizer::{Tokenizer, TokenizerKind, load_tokenizer, train_tokenizer},
    transformer::Transformer,
};

pub struct LanguageModel {
    tokenizer: Box<dyn Tokenizer>,
    embedding: Embedding,
    /// Sinusoidal の場合のみ Some。 RoPE では位置情報は MHA 内で扱うため None。
    pe: Option<SinusoidalPE>,
    transformer: Transformer,
    output_head: OutputHead,
    // ハイパーパラメータ
    d_model: usize,
    n_heads: usize,
    d_ff: usize,
    n_layers: usize,
    max_len: usize,
    dropout_p: f32,
    normalization_kind: NormalizationKind,
    feed_forward_kind: FeedForwardKind,
    positional_encoding_kind: PositionalEncodingKind,
}

impl LanguageModel {
    #[allow(dead_code)]
    pub fn new(
        corpus_text: &str,
        tokenizer_kind: TokenizerKind,
        normalization_kind: NormalizationKind,
        feed_forward_kind: FeedForwardKind,
        positional_encoding_kind: PositionalEncodingKind,
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
        max_len: usize,
        dropout_p: f32,
    ) -> Self {
        let tokenizer = train_tokenizer(tokenizer_kind, corpus_text, vocab_size);
        Self::from_tokenizer(
            tokenizer,
            normalization_kind,
            feed_forward_kind,
            positional_encoding_kind,
            d_model,
            n_heads,
            d_ff,
            n_layers,
            max_len,
            dropout_p,
        )
    }

    /// 既に構築済みのトークナイザを使ってモデルを構築する。
    /// キャッシュからロードした BPE などを使う場合に呼ぶ。
    #[allow(clippy::too_many_arguments)]
    pub fn from_tokenizer(
        tokenizer: Box<dyn Tokenizer>,
        normalization_kind: NormalizationKind,
        feed_forward_kind: FeedForwardKind,
        positional_encoding_kind: PositionalEncodingKind,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
        max_len: usize,
        dropout_p: f32,
    ) -> Self {
        let vocab_size = tokenizer.vocab_size();
        let pad_id = tokenizer.pad_id();
        let pe = match positional_encoding_kind {
            PositionalEncodingKind::Sinusoidal => Some(SinusoidalPE::new(max_len, d_model)),
            PositionalEncodingKind::Rope => None,
        };
        Self {
            tokenizer,
            embedding: Embedding::new(vocab_size, d_model, Some(pad_id)),
            pe,
            transformer: Transformer::new(
                n_layers,
                d_model,
                n_heads,
                d_ff,
                max_len,
                dropout_p,
                normalization_kind,
                feed_forward_kind,
                positional_encoding_kind,
            ),
            output_head: OutputHead::new(d_model, vocab_size),
            d_model,
            n_heads,
            d_ff,
            n_layers,
            max_len,
            dropout_p,
            normalization_kind,
            feed_forward_kind,
            positional_encoding_kind,
        }
    }

    /// 学習ループから参照される pad id (cross-entropy のマスクに使用)。
    pub fn pad_id(&self) -> usize {
        self.tokenizer.pad_id()
    }

    /// このモデルが保持するトークナイザの種別 (ログ・診断用)。
    #[allow(dead_code)]
    pub fn tokenizer_kind(&self) -> TokenizerKind {
        self.tokenizer.kind()
    }

    /// dropout の有効/無効を切り替える。 推論・validation 時は false に設定する。
    /// LanguageModel コンストラクタ直後の状態は `training=true` (= dropout 有効)。
    pub fn set_training(&mut self, training: bool) {
        self.transformer.set_training(training);
    }

    /// 学習用に全コーパスを事前トークナイズ。
    /// BOS + content + EOS の token id 列を返す。
    pub fn tokenize_corpus(&self, text: &str) -> Vec<usize> {
        self.tokenizer.encode_long(text)
    }

    #[allow(dead_code)]
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    fn context_window<'a>(&self, ids: &'a [usize]) -> &'a [usize] {
        if ids.len() > self.max_len {
            &ids[ids.len() - self.max_len..]
        } else {
            ids
        }
    }

    /// Sinusoidal の場合は位置エンコーディングを加算、 RoPE の場合は埋め込みを素通し。
    /// (RoPE は MHA 内で Q, K に直接回転を掛けるためここで加算しない)
    fn apply_positional_encoding(&self, emb: Matrix) -> Matrix {
        match &self.pe {
            Some(pe) => pe.forward(&emb),
            None => emb,
        }
    }

    /// 生成用: 最後のトークン位置の logits だけ計算する。
    fn forward_ids_last(&mut self, token_ids: &[usize]) -> Vec<f32> {
        let seq = token_ids.len();
        let mask = causal_mask(seq);
        let emb = self.embedding.forward(token_ids);
        let x = self.apply_positional_encoding(emb);
        let h = self.transformer.forward(&x, Some(&mask));
        self.output_head.logits_last(h.row(seq - 1))
    }

    /// Forward のみで cross-entropy loss を計算する (backward は実行しない)。
    /// validation や評価用。 dropout は自動的に無効化されたあと、
    /// 関数終了時に元の training=true 状態へ戻る。
    /// 内部キャッシュは上書きされるため、 学習中に呼ぶ場合は
    /// **gradient apply / zero_grad の後**に挿入すること
    /// (次の forward_backward が改めてキャッシュを構築する)。
    pub fn forward_loss(&mut self, token_ids: &[usize], pad_id: usize) -> f32 {
        let seq = token_ids.len();
        assert!(seq >= 2, "seq_len must be >= 2");

        self.set_training(false);
        let mask = causal_mask(seq);
        let emb = self.embedding.forward(token_ids);
        let x = self.apply_positional_encoding(emb);
        let h = self.transformer.forward(&x, Some(&mask));

        // shifted: 先頭から seq-1 行
        let h_shifted = slice_rows(&h, 0, seq - 1);
        let logits = self.output_head.forward(&h_shifted);

        let targets = &token_ids[1..];
        let mask_ce: Vec<u8> = targets
            .iter()
            .map(|&t| if t == pad_id { 0 } else { 1 })
            .collect();
        let (loss, _) = CrossEntropyLoss::forward_sequence(&logits, targets, &mask_ce);
        self.set_training(true);
        loss
    }

    pub fn forward_backward(&mut self, token_ids: &[usize], pad_id: usize) -> f32 {
        let seq = token_ids.len();
        assert!(seq >= 2, "seq_len must be >= 2");

        let mask = causal_mask(seq);

        let emb = self.embedding.forward(token_ids);
        let x = self.apply_positional_encoding(emb);
        let h = self.transformer.forward(&x, Some(&mask));

        let h_shifted = slice_rows(&h, 0, seq - 1);
        let logits = self.output_head.forward(&h_shifted);

        let targets = &token_ids[1..];

        let mask_ce: Vec<u8> = targets
            .iter()
            .map(|&t| if t == pad_id { 0 } else { 1 })
            .collect();

        let (loss, dl_dlogits) =
            CrossEntropyLoss::forward_sequence(&logits, targets, &mask_ce);
        let dl_dh_shifted = self.output_head.backward(&dl_dlogits);
        let dl_dh_full = pad_grad_matrix(&dl_dh_shifted, seq);
        let dl_dh_full = clip_grad_norm_matrix(dl_dh_full, 1.0);
        let dl_dx = self.transformer.backward(&dl_dh_full);
        self.embedding.backward(&dl_dx);

        loss
    }

    pub fn apply_gradients(&mut self, opt: &mut AdamW) {
        self.output_head.apply_gradients(opt, "head");
        self.transformer.apply_gradients(opt, "transformer");
        self.embedding.apply_gradients(opt, "embedding");
    }

    pub fn zero_grad(&mut self) {
        self.embedding.zero_grad();
        self.transformer.zero_grad();
        self.output_head.zero_grad();
    }

    #[allow(dead_code)]
    pub fn generate(
        &mut self,
        prompt_text: &str,
        max_new_token: usize,
        repetition_penalty: f32,
    ) -> String {
        self.set_training(false);
        let mut ids = self.tokenizer.encode_prompt(prompt_text);
        let eos_id = self.tokenizer.eos_id();
        for _ in 0..max_new_token {
            let ctx = self.context_window(&ids);
            let mut logits = self.forward_ids_last(ctx);
            apply_repetition_penalty(&mut logits, &ids, repetition_penalty);
            let next_id = OutputHead::greedy(&logits);
            if next_id == eos_id {
                break;
            }
            ids.push(next_id);
        }
        let out = self.detokenize(&ids);
        self.set_training(true);
        out
    }

    #[allow(dead_code)]
    pub fn generate_top_k(
        &mut self,
        prompt_text: &str,
        max_new_token: usize,
        k: usize,
        temprature: f32,
        repetition_penalty: f32,
    ) -> String {
        self.set_training(false);
        let mut ids = self.tokenizer.encode_prompt(prompt_text);
        let eos_id = self.tokenizer.eos_id();
        for _ in 0..max_new_token {
            let ctx = self.context_window(&ids);
            let mut logits = self.forward_ids_last(ctx);
            apply_repetition_penalty(&mut logits, &ids, repetition_penalty);
            let next_id = OutputHead::top_k_sample(&logits, k, temprature);
            if next_id == eos_id {
                break;
            }
            ids.push(next_id);
        }
        let out = self.detokenize(&ids);
        self.set_training(true);
        out
    }

    #[allow(dead_code)]
    pub fn generate_top_p(
        &mut self,
        prompt_text: &str,
        max_new_token: usize,
        p: f32,
        temperature: f32,
        repetition_penalty: f32,
    ) -> String {
        self.set_training(false);
        let mut ids = self.tokenizer.encode_prompt(prompt_text);
        let eos_id = self.tokenizer.eos_id();
        for _ in 0..max_new_token {
            let ctx = self.context_window(&ids);
            let mut logits = self.forward_ids_last(ctx);
            apply_repetition_penalty(&mut logits, &ids, repetition_penalty);
            let next_id = OutputHead::top_p_sample(&logits, p, temperature);
            if next_id == eos_id {
                break;
            }
            ids.push(next_id);
        }
        let out = self.detokenize(&ids);
        self.set_training(true);
        out
    }

    /// 推論専用: 1 token を全モデル通して前進し、 その位置の logits を返す。
    /// 内部で Embedding → PE (Sinusoidal の場合のみ) → Transformer.forward_step
    /// → output_head.logits_last を呼び出す。
    /// `caches[0].cur_len()` が **これから推論する位置** に一致する前提
    /// (= cache が pos 個 token 分のデータを持っているとき、 pos 番目を計算する)。
    fn forward_step_last(&mut self, token_id: usize, caches: &mut [KvCache]) -> Vec<f32> {
        let pos = caches[0].cur_len();
        let emb = self.embedding.forward_one(token_id);
        let x = match &self.pe {
            Some(pe) => pe.add_at(&emb, pos),
            None => emb,
        };
        let h = self.transformer.forward_step(&x, caches);
        self.output_head.logits_last(&h)
    }

    /// `generate_top_k` と同じだが KV cache を使って高速化したバージョン。
    /// no-cache 版と **同じ乱数経路と repetition penalty** を踏むので、
    /// 同一サンプリング結果を期待できる (RNG seed が固定なら完全一致)。
    pub fn generate_top_k_with_cache(
        &mut self,
        prompt_text: &str,
        max_new_token: usize,
        k: usize,
        temperature: f32,
        repetition_penalty: f32,
    ) -> String {
        self.set_training(false);
        let mut ids = self.tokenizer.encode_prompt(prompt_text);
        let eos_id = self.tokenizer.eos_id();
        // 起点となる context (max_len を超えていたら末尾を切る)。
        let prompt_ctx: Vec<usize> = self.context_window(&ids).to_vec();

        let mut caches = self.transformer.init_kv_caches(self.max_len, self.d_model);

        // Prefill: 最後の token 以外を順に流して cache を埋める (logits は捨てる)。
        // 最後の token は logits を取って sampling に回すため、 別扱い。
        for &tid in &prompt_ctx[..prompt_ctx.len() - 1] {
            let _ = self.forward_step_last(tid, &mut caches);
        }
        let mut next_input = *prompt_ctx.last().unwrap();

        for _ in 0..max_new_token {
            // 容量超過のリスクチェック: cache.cur_len() == max_len なら sliding-window
            // 退避が必要だが、 ここでは単純に early-stop (将来 KV truncation を実装予定)。
            if caches[0].cur_len() >= self.max_len {
                break;
            }
            let mut logits = self.forward_step_last(next_input, &mut caches);
            apply_repetition_penalty(&mut logits, &ids, repetition_penalty);
            let next_id = OutputHead::top_k_sample(&logits, k, temperature);
            if next_id == eos_id {
                break;
            }
            ids.push(next_id);
            next_input = next_id;
        }
        let out = self.detokenize(&ids);
        self.set_training(true);
        out
    }

    /// `generate_top_p` の KV cache 版。
    pub fn generate_top_p_with_cache(
        &mut self,
        prompt_text: &str,
        max_new_token: usize,
        p: f32,
        temperature: f32,
        repetition_penalty: f32,
    ) -> String {
        self.set_training(false);
        let mut ids = self.tokenizer.encode_prompt(prompt_text);
        let eos_id = self.tokenizer.eos_id();
        let prompt_ctx: Vec<usize> = self.context_window(&ids).to_vec();
        let mut caches = self.transformer.init_kv_caches(self.max_len, self.d_model);

        for &tid in &prompt_ctx[..prompt_ctx.len() - 1] {
            let _ = self.forward_step_last(tid, &mut caches);
        }
        let mut next_input = *prompt_ctx.last().unwrap();

        for _ in 0..max_new_token {
            if caches[0].cur_len() >= self.max_len {
                break;
            }
            let mut logits = self.forward_step_last(next_input, &mut caches);
            apply_repetition_penalty(&mut logits, &ids, repetition_penalty);
            let next_id = OutputHead::top_p_sample(&logits, p, temperature);
            if next_id == eos_id {
                break;
            }
            ids.push(next_id);
            next_input = next_id;
        }
        let out = self.detokenize(&ids);
        self.set_training(true);
        out
    }

    /// 検証用: greedy decoding を KV cache で行う。 同じ入力に対し
    /// `forward_ids_last` を毎 step 回す方式と **完全一致** する logits 系列を出すことを
    /// 単体テストで保証する。
    #[allow(dead_code)]
    pub fn generate_greedy_with_cache(
        &mut self,
        prompt_text: &str,
        max_new_token: usize,
    ) -> String {
        self.set_training(false);
        let mut ids = self.tokenizer.encode_prompt(prompt_text);
        let eos_id = self.tokenizer.eos_id();
        let prompt_ctx: Vec<usize> = self.context_window(&ids).to_vec();
        let mut caches = self.transformer.init_kv_caches(self.max_len, self.d_model);

        for &tid in &prompt_ctx[..prompt_ctx.len() - 1] {
            let _ = self.forward_step_last(tid, &mut caches);
        }
        let mut next_input = *prompt_ctx.last().unwrap();

        for _ in 0..max_new_token {
            if caches[0].cur_len() >= self.max_len {
                break;
            }
            let logits = self.forward_step_last(next_input, &mut caches);
            let next_id = OutputHead::greedy(&logits);
            if next_id == eos_id {
                break;
            }
            ids.push(next_id);
            next_input = next_id;
        }
        let out = self.detokenize(&ids);
        self.set_training(true);
        out
    }

    fn detokenize(&self, ids: &[usize]) -> String {
        // BPE / Char いずれの decode 実装も特殊トークン (BOS/EOS/PAD/UNK) を
        // 自身でスキップするため、 ここでは単純に全 id を渡す。
        self.tokenizer.decode(ids)
    }

    fn build_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_scalar("meta.d_model", self.d_model as u64);
        map.insert_scalar("meta.n_heads", self.n_heads as u64);
        map.insert_scalar("meta.d_ff", self.d_ff as u64);
        map.insert_scalar("meta.n_layers", self.n_layers as u64);
        map.insert_scalar("meta.max_len", self.max_len as u64);
        map.insert_scalar("meta.dropout_p", self.dropout_p.to_bits() as u64);
        map.insert_scalar("meta.normalization_kind", self.normalization_kind.as_u64());
        map.insert_scalar("meta.feed_forward_kind", self.feed_forward_kind.as_u64());
        map.insert_scalar(
            "meta.positional_encoding_kind",
            self.positional_encoding_kind.as_u64(),
        );
        map.merge("tokenizer", self.tokenizer.to_weight_map());
        map.merge("embedding", self.embedding.to_weight_map());
        map.merge("transformer", self.transformer.to_weight_map());
        map.merge("output_head", self.output_head.to_weight_map());
        map
    }

    pub fn save_inference_checkpoint(&self, path: &str) -> Result<()> {
        self.build_weight_map().save(path)
    }

    pub fn save_training_checkpoint(&self, path: &str, opt: &AdamW, step: usize) -> Result<()> {
        let mut map = self.build_weight_map();
        map.insert_scalar("meta.step", step as u64);
        map.merge("optimizer", opt.to_weight_map());
        map.save(path)
    }

    fn restore_model(map: &WeightMap) -> Result<Self> {
        let d_model = map.get_scalar("meta.d_model")? as usize;
        let n_heads = map.get_scalar("meta.n_heads")? as usize;
        let d_ff = map.get_scalar("meta.d_ff")? as usize;
        let n_layers = map.get_scalar("meta.n_layers")? as usize;
        let max_len = map.get_scalar("meta.max_len")? as usize;
        // dropout_p は旧 checkpoint には存在しないので 0.0 を fallback に
        let dropout_p = map
            .get_scalar("meta.dropout_p")
            .map(|v| f32::from_bits(v as u32))
            .unwrap_or(0.0);
        let tokenizer = load_tokenizer(&map.scoped("tokenizer"))?;
        let normalization_kind = map
            .get_scalar("meta.normalization_kind")
            .and_then(NormalizationKind::from_u64)
            .unwrap_or(NormalizationKind::Layer);
        let feed_forward_kind = map
            .get_scalar("meta.feed_forward_kind")
            .and_then(FeedForwardKind::from_u64)
            .unwrap_or(FeedForwardKind::Gelu);
        // 旧 checkpoint には存在しないので Sinusoidal にフォールバック
        let positional_encoding_kind = map
            .get_scalar("meta.positional_encoding_kind")
            .and_then(PositionalEncodingKind::from_u64)
            .unwrap_or(PositionalEncodingKind::Sinusoidal);
        let vocab_size = tokenizer.vocab_size();
        let pad_id = tokenizer.pad_id();

        let pe = match positional_encoding_kind {
            PositionalEncodingKind::Sinusoidal => Some(SinusoidalPE::new(max_len, d_model)),
            PositionalEncodingKind::Rope => None,
        };
        let mut model = Self {
            tokenizer,
            embedding: Embedding::new(vocab_size, d_model, Some(pad_id)),
            pe,
            transformer: Transformer::new(
                n_layers,
                d_model,
                n_heads,
                d_ff,
                max_len,
                dropout_p,
                normalization_kind,
                feed_forward_kind,
                positional_encoding_kind,
            ),
            output_head: OutputHead::new(d_model, vocab_size),
            d_model,
            n_heads,
            d_ff,
            n_layers,
            max_len,
            dropout_p,
            normalization_kind,
            feed_forward_kind,
            positional_encoding_kind,
        };

        model.embedding.from_weight_map(&map.scoped("embedding"))?;
        model
            .transformer
            .from_weight_map(&map.scoped("transformer"))?;
        model
            .output_head
            .from_weight_map(&map.scoped("output_head"))?;

        Ok(model)
    }

    pub fn load_inference_checkpoint(path: &str) -> Result<Self> {
        let map = WeightMap::load(path)?;
        Self::restore_model(&map)
    }

    pub fn load_training_checkpoint(path: &str) -> Result<(Self, AdamW, usize)> {
        let map = WeightMap::load(path)?;
        let model = Self::restore_model(&map)?;
        let step = map.get_scalar("meta.step")? as usize;
        let mut opt = AdamW::new(0.0);
        opt.from_weight_map(&map.scoped("optimizer"))?;

        Ok((model, opt, step))
    }
}

/// 既に生成済み（プロンプトを含む）の token に対して logits を割引（penalty>1）する。
/// HuggingFace の repetition_penalty と同じ式（正は割り、負は掛ける）。
/// `penalty == 1.0` のとき何もしない。
fn apply_repetition_penalty(logits: &mut [f32], previous_ids: &[usize], penalty: f32) {
    if penalty == 1.0 || previous_ids.is_empty() {
        return;
    }
    use std::collections::HashSet;
    let unique: HashSet<usize> = previous_ids.iter().copied().collect();
    for id in unique {
        if id < logits.len() {
            let v = logits[id];
            logits[id] = if v > 0.0 { v / penalty } else { v * penalty };
        }
    }
}

/// Matrix の指定行範囲をコピーした新 Matrix を返す。
fn slice_rows(m: &Matrix, start: usize, end: usize) -> Matrix {
    assert!(end <= m.rows() && start <= end);
    let cols = m.cols();
    let n = end - start;
    let mut data = Vec::with_capacity(n * cols);
    for i in start..end {
        data.extend_from_slice(m.row(i));
    }
    Matrix::from_flat(data, n, cols)
}

/// `dl` (rows = seq-1) を末尾 0 行で padding して `seq` 行の Matrix にする。
fn pad_grad_matrix(dl: &Matrix, seq: usize) -> Matrix {
    let d_model = dl.cols();
    if dl.rows() == seq {
        return dl.clone();
    }
    let mut data = vec![0.0f32; seq * d_model];
    let n = dl.rows().min(seq);
    data[..n * d_model].copy_from_slice(&dl.data()[..n * d_model]);
    Matrix::from_flat(data, seq, d_model)
}

/// Matrix 全体ノルムでクリッピング。
fn clip_grad_norm_matrix(mut grads: Matrix, max_norm: f32) -> Matrix {
    let norm: f32 = grads.data().iter().map(|v| v.powi(2)).sum::<f32>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for v in grads.data_mut() {
            *v *= scale;
        }
    }
    grads
}

#[cfg(test)]
mod kv_cache_tests {
    use super::*;

    fn small_corpus() -> &'static str {
        "abcdefghijklmnopqrstuvwxyz0123456789 .,!?\n"
    }

    fn build_tiny_model(
        norm: NormalizationKind,
        ff: FeedForwardKind,
        pe: PositionalEncodingKind,
    ) -> LanguageModel {
        // 小さいモデルを作って no-cache vs with-cache を比較する。
        // 学習させていないランダム初期重みでも、 forward が決定的なら logits は一致するはず。
        LanguageModel::new(
            small_corpus(),
            TokenizerKind::Char,
            norm,
            ff,
            pe,
            0,    // vocab_size = 0 → tokenizer から自動算出 (Char)
            32,   // d_model
            4,    // n_heads (d_head=8 は偶数なので RoPE OK)
            64,   // d_ff
            2,    // n_layers
            32,   // max_len
            0.0,  // dropout (eval モードなら効かないが念のため 0)
        )
    }

    #[test]
    fn greedy_with_cache_matches_no_cache_rope_rms_swiglu() {
        let mut model = build_tiny_model(
            NormalizationKind::Rms,
            FeedForwardKind::SwiGlu,
            PositionalEncodingKind::Rope,
        );
        let prompt = "abc";
        let no_cache = model.generate(prompt, 8, 1.0);
        let with_cache = model.generate_greedy_with_cache(prompt, 8);
        assert_eq!(
            no_cache, with_cache,
            "RoPE+RMS+SwiGLU greedy 出力が一致しない\n  no_cache: {no_cache:?}\n  with_cache: {with_cache:?}",
        );
    }

    #[test]
    fn greedy_with_cache_matches_no_cache_sinusoidal_layernorm_gelu() {
        let mut model = build_tiny_model(
            NormalizationKind::Layer,
            FeedForwardKind::Gelu,
            PositionalEncodingKind::Sinusoidal,
        );
        let prompt = "xyz";
        let no_cache = model.generate(prompt, 8, 1.0);
        let with_cache = model.generate_greedy_with_cache(prompt, 8);
        assert_eq!(
            no_cache, with_cache,
            "Sinusoidal+LayerNorm+GELU greedy 出力が一致しない\n  no_cache: {no_cache:?}\n  with_cache: {with_cache:?}",
        );
    }

    #[test]
    fn greedy_with_cache_matches_no_cache_rope_layernorm_gelu() {
        // 組合せ違いも 1 つ確認 (RoPE × LN × GELU)
        let mut model = build_tiny_model(
            NormalizationKind::Layer,
            FeedForwardKind::Gelu,
            PositionalEncodingKind::Rope,
        );
        let prompt = "hello";
        let no_cache = model.generate(prompt, 12, 1.0);
        let with_cache = model.generate_greedy_with_cache(prompt, 12);
        assert_eq!(no_cache, with_cache);
    }

    #[test]
    fn forward_step_last_logits_match_forward_ids_last_per_position() {
        // 最も厳密な検証: 各位置で 「これまでの prefix を毎回 forward_ids_last に流す」
        // と 「逐次 forward_step_last」 の logits が (token-wise argmax で) 一致するか。
        let mut model = build_tiny_model(
            NormalizationKind::Rms,
            FeedForwardKind::SwiGlu,
            PositionalEncodingKind::Rope,
        );
        model.set_training(false);
        let ids = model.tokenizer.encode_prompt("hello world");
        let mut caches = model.transformer.init_kv_caches(model.max_len, model.d_model);

        // ids の各 prefix の最終位置 logits を 2 通りで計算
        for end in 1..=ids.len() {
            let prefix = &ids[..end];

            // no-cache: 毎回 prefix 全部を forward
            let logits_no_cache = model.forward_ids_last(prefix);
            let argmax_no_cache = OutputHead::greedy(&logits_no_cache);

            // with-cache: 直前の token を 1 つだけ feed
            let new_token = ids[end - 1];
            let logits_with_cache = model.forward_step_last(new_token, &mut caches);
            let argmax_with_cache = OutputHead::greedy(&logits_with_cache);

            assert_eq!(
                argmax_no_cache, argmax_with_cache,
                "argmax mismatch at position {} (1-indexed): no_cache={}, with_cache={}",
                end, argmax_no_cache, argmax_with_cache,
            );
            // 数値も近いことを確認 (FP 誤差の範囲)
            for (i, (a, b)) in logits_no_cache.iter().zip(&logits_with_cache).enumerate() {
                let diff = (a - b).abs();
                assert!(
                    diff < 1e-3,
                    "logit mismatch at position {} vocab {}: no_cache={}, with_cache={}, diff={}",
                    end, i, a, b, diff,
                );
            }
        }
    }
}

/// Phase 7 高速化の効果を見るベンチマーク。 Phase 6-c と同じモデル形状
/// (d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_len=1024) で
/// `forward_backward + apply_gradients` を数 step 実行し、 1 step あたりの
/// 平均所要時間を出力する。 比較対象は Phase 6-c の実測値 (~11,000 ms/step @ batch=16)。
///
/// `--nocapture` が無いと println! は出ないので、 走らせるときは:
///   cargo test --release bench_phase7_step_time -- --nocapture --ignored
#[cfg(test)]
mod bench_tests {
    use crate::adam_w::AdamW;
    use crate::feed_forward::FeedForwardKind;
    use crate::language_model::LanguageModel;
    use crate::normalization::NormalizationKind;
    use crate::positional_encoding::PositionalEncodingKind;
    use crate::tokenizer::TokenizerKind;
    use std::time::Instant;

    /// Phase 6-c 相当 (d_model=512, n_layers=6, max_len=1024) の 1 step あたり時間を計測。
    /// 重い (1 step で数秒〜十数秒) ので #[ignore] にしてある。
    #[test]
    #[ignore]
    fn bench_phase7_step_time() {
        // Phase 6-c と完全に同じ形状を作る (重みはランダム初期化だが計算量は同じ)。
        // tokenizer は Char で代用 (vocab=64 程度。 OutputHead/Embedding が小さくなるので
        // 実際の Phase 6-c (vocab=8000) より out-head の matmul が軽くなる点だけ留意)。
        let corpus = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?\nABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let mut model = LanguageModel::new(
            corpus,
            TokenizerKind::Char,
            NormalizationKind::Rms,
            FeedForwardKind::SwiGlu,
            PositionalEncodingKind::Rope,
            0,    // vocab_size 自動
            512,  // d_model
            8,    // n_heads
            2048, // d_ff
            6,    // n_layers
            1024, // max_len
            0.2,  // dropout
        );
        model.set_training(true);

        let pad_id = model.pad_id();
        let mut opt = AdamW::new_with_wd(7e-4, 0.1);

        // batch=2 にして 1 step ≒ 1.5-2 sec を狙う (Phase 6-c は batch=16 で ~11 sec)。
        // forward_backward / apply_gradients は 1 サンプルずつ呼ぶ実装が前提。
        let batch_size = 2;
        let seq_len = 1024;
        let warmup_steps = 1;
        // Phase 7-3 で 3 → 5 steps に増やしてばらつき低減 (1 step ≒ 1.3 sec)。
        let measure_steps = 5;

        // 適当な token id 列 (vocab を超えないようランダム)
        let vocab_size = 64; // Char tokenizer の概算 (実際の vocab はそれ未満)
        let make_batch = |seed: usize| -> Vec<Vec<usize>> {
            (0..batch_size)
                .map(|b| {
                    (0..seq_len)
                        .map(|i| (seed + b * 31 + i * 7) % vocab_size.min(20))
                        .collect()
                })
                .collect()
        };

        // warmup
        for s in 0..warmup_steps {
            let batch = make_batch(s + 100);
            for sample in &batch {
                let _ = model.forward_backward(sample, pad_id);
            }
            model.apply_gradients(&mut opt);
            model.zero_grad();
        }

        // 計測
        let start = Instant::now();
        for s in 0..measure_steps {
            let batch = make_batch(s);
            for sample in &batch {
                let _ = model.forward_backward(sample, pad_id);
            }
            model.apply_gradients(&mut opt);
            model.zero_grad();
        }
        let elapsed = start.elapsed();
        let per_step_ms = elapsed.as_secs_f64() * 1000.0 / measure_steps as f64;

        println!("=== Phase 7 step time benchmark ===");
        println!("  shape: d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_len=1024");
        println!("  batch_size={batch_size}, measure_steps={measure_steps}");
        println!("  per-step (Matrix-direct path): {per_step_ms:.1} ms");
        // Phase 6-c 実測 ~11000 ms/step @ batch=16
        // batch=2 換算で旧コード期待値 = 11000 * 2/16 = 1375 ms/step
        let phase6c_per_step_batch16_ms = 11000.0;
        let baseline_per_step_ms = phase6c_per_step_batch16_ms * (batch_size as f64) / 16.0;
        let speedup = baseline_per_step_ms / per_step_ms;
        println!("  Phase 6-c (old) per-step @ batch={batch_size}: ~{baseline_per_step_ms:.0} ms (推定)");
        println!("  speedup vs Phase 6-c (推定): {speedup:.2}x");
        // Phase 7-2 (transpose 残存) の bench で観測された値: ~1500 ms
        let phase72_per_step_ms = 1500.0;
        let phase73_speedup = phase72_per_step_ms / per_step_ms;
        println!("  Phase 7-2 (transpose 残存) per-step (実測): ~{phase72_per_step_ms:.0} ms");
        println!("  speedup vs Phase 7-2 (matmul_t1/t2 効果): {phase73_speedup:.2}x");
    }
}
