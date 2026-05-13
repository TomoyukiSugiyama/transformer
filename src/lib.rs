//! Transformer ライブラリ。 src/bin/*.rs から共通モジュールを参照するための公開エントリ。
//! バイナリ本体 (`src/main.rs`) もここから modules を再宣言する必要はなく、 そのまま動く。

pub mod adam_w;
pub mod bpe_tokenizer;
pub mod char_bpe_tokenizer;
pub mod char_tokenizer;
pub mod checkpoint;
pub mod cross_entropy_loss;
pub mod dropout;
pub mod embedding;
pub mod eval;
pub mod feed_forward;
pub mod feed_forward_network;
pub mod kv_cache;
pub mod language_model;
pub mod layer_normalization;
pub mod lr_scheduler;
pub mod matrix;
pub mod multi_head_attention;
pub mod normalization;
pub mod output_head;
pub mod positional_encoding;
pub mod root_mean_square_layer_normalization;
pub mod rope;
pub mod sinusoidal_pe;
pub mod swiglu_feed_forward_network;
pub mod tokenizer;
pub mod transformer;
pub mod transformer_block;
