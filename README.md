# transformer

Rust で書かれた Transformer (decoder-only) 言語モデルの学習・推論実装。
外部 ML フレームワークに依存せず、行列演算から自前で実装している学習用プロジェクト。

## 依存

- Rust (edition 2024)
- `rand = "0.10.1"`
- `rayon = "1.10"` （行列演算・損失計算の並列化）

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
├── bpe_tokenizer.rs           # BPE トークナイザ
├── checkpoint.rs              # 重み・状態の保存/読込
└── utility.rs                 # 行列演算ユーティリティ（rayon で並列化）

corpus/
└── train.txt                  # 学習データ（全行を連結し、ランダム窓でサンプリング）

checkpoints/<run_name>/
├── step_NNNNNN.bin            # 学習途中の checkpoint
├── latest.bin                 # 直近の checkpoint
└── inference.bin              # 学習完了後の推論用 checkpoint
```

## チェックポイントから再開・推論

`src/main.rs` の `main()` 内で対応する関数の呼び出しを切り替える:

```rust
fn main() {
    let corpus_text = load_corpus("corpus/train.txt");
    let cfg = Config::tiny_shakespeare();

    // 新規学習
    training_and_inference(&corpus_text, &cfg);

    // チェックポイントから再開（同じモデル構造の checkpoint のみ）
    // training_from_checkpoint(&corpus_text, &cfg, "checkpoints/<run_name>/latest.bin");

    // チェックポイントを読み込んで推論のみ
    // inference_from_checkpoint(&cfg, "checkpoints/<run_name>/inference.bin");
}
```

`training_from_checkpoint` で再開する場合、checkpoint の
`d_model` / `n_heads` / `d_ff` / `n_layers` / `vocab_size` が `Config` と一致している必要がある。
モデル構造を変えた場合は `training_and_inference`（fresh start）を使う。

## 並列化と性能

CPU の全コアを活用するため、行列演算と損失計算を `rayon` で並列化している。

### 並列化の対象

| ファイル / 関数 | 並列化対象 | 並列粒度 |
|----------------|----------|---------|
| `utility::matmul` | 出力行 `i` ループ | 行ごと |
| `utility::softmax_rows` | 行ごとの softmax | 行ごと |
| `cross_entropy_loss::forward_sequence` | 各 token の softmax+loss+grad | token ごと |

### matmul / softmax を経由する主な処理

- `multi_head_attention`: Q/K/V/O 射影、scaled-dot-product attention
- `feed_forward_network`: forward / backward すべて
- `output_head`: forward / backward すべて

`Vec<Vec<f32>>` の jagged 表現のまま、`matmul` の内側ループ順序を `i-k-j` にすることで
キャッシュ効率と SIMD 自動ベクトル化を引き出している。

### 並列化されていない箇所

- `embedding`: token id ベースの lookup なので並列化のメリットが薄い
- `layer_normalization`: 行ごとの統計量計算で軽量
- `transpose`, `add_matrix_in_place`: 線形時間で軽量

### 効果（M1 Max / 10 コア環境での観察）

`d_model=128, max_len=64, batch_size=16` での 1 step あたりの所要時間がおよそ
**1 桁短く** なった（並列化前は MHA 以外がシリアル実行だったため）。
この高速化を前提に、`d_model=256, max_len=128` といった構成変更を
現実的な時間で試せる。

## チューニングのコツ

実装と実験を通じて得られた知見のまとめ。

### 学習データの整形
- **行ごと（短文）に切らず、コーパス全体を 1 つの token 列に連結**してランダム窓でサンプリングする方が、対話・改行・句読点などの構造を学習できる。
- 行ごとに `<EOS>` を付けると EOS バイアスが発生し、推論時に 1 トークンで打ち切られやすくなる。連結方式では EOS は不要（`max_new_tokens` で停止）。

### 学習率
- `lr_max ≒ 3e-4`、`lr_min ≒ 1e-5`、`warmup_steps ≒ 200` が小規模 Transformer の典型値。
- `lr_min` を `0` 付近にすると終盤が完全に止まるため、**`1e-5` 程度のフロアを残す**。
- checkpoint から end_step を伸ばして再開すると、scheduler の進捗がリセットされて lr が上振れする（**warm restart 効果**）。停滞局所解からの脱出に使える。

### バッチとミニバッチ勾配
- `batch_size` 個の `forward_backward` で勾配を累積し、まとめて 1 回 `apply_gradients` する。
- AdamW の `step_count` も 1 step に 1 回しか進めない（パラメータごとに進めない）。
- 累積勾配は `set_grad_scale(batch_size)` で平均化扱いにし、`apply_gradients` 後に `zero_grad` でリセットする。

### モデル構造の比率
- `d_ff = 4 × d_model` が標準（GPT-2、nanoGPT 等）。FFN は知識を蓄える主役なので `d_model` を増やすときは `d_ff` も比例して増やす。
- `d_head = d_model / n_heads = 32〜64` が扱いやすい。
- `max_len` を上げると attention は `O(n²)` で重くなる。学習対象（対話 1 ターン: 〜100 token、シーン: 〜500 token）に合わせて選ぶ。

### 推論時の繰り返し対策
- greedy はすぐに同じ語句に落ち込みやすいので、**top-k サンプリング + 適度な temperature**（例: `top_k=5, temperature=1.0`）の方が自然な文章になる。
- それでも `my lord, my lord, ...` のような繰り返しが出る場合は repetition penalty の導入を検討する。

### checkpoint 運用
- `run_name` を分けることで、設定違いの実験を上書きせずに並走できる（`checkpoints/<run_name>/`）。
- モデル構造（`d_model` 等）を変えると checkpoint 互換性が失われるため、構造変更時は fresh start（`training_and_inference`）を使う。

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
