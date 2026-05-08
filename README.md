# transformer

Rust で書かれた Transformer (decoder-only) 言語モデルの学習・推論実装。
外部 ML フレームワークに依存せず、行列演算から自前で実装している学習用プロジェクト。

## 依存

- Rust (edition 2024)
- `rand = "0.10.1"`

## 実行

### 学習

```bash
cargo run --release
```

`Config::tiny_shakespeare()` で定義された設定で学習が始まり、`checkpoints/<run_name>/` 配下に
チェックポイントが保存される。

### CSV ログとして保存

学習ログは CSV 形式 (`step,loss,ema,min,max,lr`) で標準出力に流れる。
コンパイラ出力やヘッダーコメントを除外して CSV を取り出すには:

```bash
cargo run --release -q 2>&1 | tee train.log
grep -E '^(step,|[0-9]+,)' train.log > train.csv
```

`-q` で cargo の `Compiling`/`Finished` メッセージを抑制し、`grep -E` で
ヘッダー (`step,...`) と数字始まりの行のみを抽出する。

## 設定

`src/main.rs` の `Config::tiny_shakespeare()` で全パラメータを指定する。

| 項目 | 役割 |
|------|------|
| `run_name` | チェックポイント保存先サブディレクトリ名 |
| `d_model` / `n_heads` / `d_ff` / `n_layers` | モデル構造 |
| `max_len` | 最大コンテキスト長（位置エンコーディング上限）|
| `vocab_size` | BPE トークナイザの語彙サイズ |
| `lr_max` / `lr_min` / `warmup_steps` | 学習率スケジュール（warmup + cosine decay） |
| `end_step` | 総学習ステップ数 |
| `batch_size` | ミニバッチサイズ |
| `save_every` / `log_every` | チェックポイント保存・ログ出力間隔 |

## ディレクトリ構成

```
src/
├── main.rs                    # エントリ・学習ループ
├── language_model.rs          # モデル全体（埋め込み→Transformer→出力）
├── transformer.rs             # Transformer (block の積み重ね)
├── transformer_block.rs       # 1 ブロック (MHA + FFN + LayerNorm)
├── multi_head_attention.rs    # マルチヘッドアテンション
├── feed_forward_network.rs    # 位置ごとの FFN
├── layer_normalization.rs
├── embedding.rs               # トークン埋め込み
├── sinusoidal_pe.rs           # 正弦波位置エンコーディング
├── output_head.rs             # 語彙への射影
├── adam_w.rs                  # AdamW オプティマイザ
├── lr_scheduler.rs            # warmup + cosine スケジューラ
├── cross_entropy_loss.rs
├── bpe_tokenizeer.rs          # BPE トークナイザ
├── checkpoint.rs              # 重み・状態の保存/読込
└── utility.rs                 # 行列演算ユーティリティ

corpus/
└── train.txt                  # 学習データ（1 行 1 サンプル、# でコメント）

checkpoints/<run_name>/
├── step_NNNNNN.bin            # 学習途中の checkpoint
├── latest.bin                 # 直近の checkpoint
└── inference.bin              # 学習完了後の推論用 checkpoint
```

## チェックポイントから再開・推論

`src/main.rs` の `main()` 内で対応する関数の呼び出しを切り替える:

```rust
fn main() {
    let corpus_strings = load_corpus("corpus/train.txt");
    let corpus: Vec<&str> = corpus_strings.iter().map(String::as_str).collect();
    let cfg = Config::tiny_shakespeare();

    // 新規学習
    training_and_inference(&corpus, &cfg);

    // チェックポイントから再開
    // training_from_checkpoint(&corpus, &cfg, "checkpoints/with_lr_sched/latest.bin");

    // チェックポイントを読み込んで推論のみ
    // inference_from_checkpoint("checkpoints/with_lr_sched/inference.bin");
}
```

## ログの読み方

```
# run_name=with_lr_sched
# d_model=128, n_heads=4, ...
# lr_max=0.0003, lr_min=0.000001, warmup_steps=200, ...
step,loss,ema,min,max,lr
20,6.7432,7.6912,6.7432,8.4294,3.000e-5
40,6.1839,6.7892,6.0621,6.7436,6.000e-5
...
```

| 列 | 意味 |
|----|------|
| `step` | 現在の学習ステップ |
| `loss` | 当該ステップでのバッチ平均 loss |
| `ema` | 指数移動平均 loss (α=0.05) |
| `min` / `max` | 直近 `log_every` ステップ内の最小/最大 loss |
| `lr` | 当該ステップでの学習率 |
