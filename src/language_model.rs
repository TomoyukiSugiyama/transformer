use std::io::Result;

use crate::{
    adam_w::AdamW,
    checkpoint::{Checkpointable, WeightMap},
    cross_entropy_loss::CrossEntropyLoss,
    embedding::Embedding,
    multi_head_attention::causal_mask,
    normalization::NormalizationKind,
    output_head::OutputHead,
    sinusoidal_pe::SinusoidalPE,
    tokenizer::{Tokenizer, TokenizerKind, load_tokenizer, train_tokenizer},
    transformer::Transformer,
};

pub struct LanguageModel {
    tokenizer: Box<dyn Tokenizer>,
    embedding: Embedding,
    pe: SinusoidalPE,
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
}

impl LanguageModel {
    pub fn new(
        corpus_text: &str,
        tokenizer_kind: TokenizerKind,
        normalization_kind: NormalizationKind,
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
        max_len: usize,
        dropout_p: f32,
    ) -> Self {
        let tokenizer = train_tokenizer(tokenizer_kind, corpus_text, vocab_size);
        let vocab_size = tokenizer.vocab_size();
        let pad_id = tokenizer.pad_id();
        Self {
            tokenizer,
            embedding: Embedding::new(vocab_size, d_model, Some(pad_id)),
            pe: SinusoidalPE::new(max_len, d_model),
            transformer: Transformer::new(
                n_layers,
                d_model,
                n_heads,
                d_ff,
                dropout_p,
                normalization_kind,
            ),
            output_head: OutputHead::new(d_model, vocab_size),
            d_model,
            n_heads,
            d_ff,
            n_layers,
            max_len,
            dropout_p,
            normalization_kind,
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

    #[allow(dead_code)]
    fn forward_ids(&mut self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        let seq = token_ids.len();
        let mask = causal_mask(seq);
        let emb = self.embedding.forward(token_ids);
        let x = self.pe.forward(&emb);
        let h = self.transformer.forward(&x, Some(&mask));
        self.output_head.forward(&h)
    }

    /// 生成用: 最後のトークン位置の logits だけ計算する。
    /// 全位置を計算する `forward_ids` よりも `O(seq)` 倍速い。
    fn forward_ids_last(&mut self, token_ids: &[usize]) -> Vec<f32> {
        let seq = token_ids.len();
        let mask = causal_mask(seq);
        let emb = self.embedding.forward(token_ids);
        let x = self.pe.forward(&emb);
        let h = self.transformer.forward(&x, Some(&mask));
        self.output_head.logits_last(&h[seq - 1])
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
        let x = self.pe.forward(&emb);
        let h = self.transformer.forward(&x, Some(&mask));

        let h_shifted = &h[..seq - 1];
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
        let x = self.pe.forward(&emb);
        let h = self.transformer.forward(&x, Some(&mask));

        let h_shifted = &h[..seq - 1];
        let logits = self.output_head.forward(&h_shifted);

        let targets = &token_ids[1..];

        let mask_ce: Vec<u8> = targets
            .iter()
            .map(|&t| if t == pad_id { 0 } else { 1 })
            .collect();

        let (loss, dl_dlogits) = CrossEntropyLoss::forward_sequence(&logits, targets, &mask_ce);
        let dl_dh_shifted = self.output_head.backward(&dl_dlogits);
        let dl_dh_full = pad_grad(dl_dh_shifted, seq);
        let dl_dh_full = clip_grad_norm(dl_dh_full, 1.0);
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
        let vocab_size = tokenizer.vocab_size();
        let pad_id = tokenizer.pad_id();

        let mut model = Self {
            tokenizer,
            embedding: Embedding::new(vocab_size, d_model, Some(pad_id)),
            pe: SinusoidalPE::new(max_len, d_model),
            transformer: Transformer::new(
                n_layers,
                d_model,
                n_heads,
                d_ff,
                dropout_p,
                normalization_kind,
            ),
            output_head: OutputHead::new(d_model, vocab_size),
            d_model,
            n_heads,
            d_ff,
            n_layers,
            max_len,
            dropout_p,
            normalization_kind,
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

fn pad_grad(mut dl: Vec<Vec<f32>>, seq: usize) -> Vec<Vec<f32>> {
    let d_model = dl[0].len();
    while dl.len() < seq {
        dl.push(vec![0.0; d_model]);
    }
    dl
}

fn clip_grad_norm(mut grads: Vec<Vec<f32>>, max_norm: f32) -> Vec<Vec<f32>> {
    let norm: f32 = grads
        .iter()
        .flatten()
        .map(|v| v.powi(2))
        .sum::<f32>()
        .sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        grads.iter_mut().flatten().for_each(|v| *v *= scale);
    }
    grads
}
