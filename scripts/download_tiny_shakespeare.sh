#!/usr/bin/env bash
# Tiny Shakespeare corpus を Karpathy の char-rnn リポジトリから取得し、
# corpus/tiny_shakespeare.txt として保存する。
#
# 出典: https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare
#       (nanoGPT の data/shakespeare_char/prepare.py と同じ URL)
# 内容: シェイクスピア戯曲集の連結 (Coriolanus, Romeo and Juliet 等), ~1.1MB
#
# 必要なコマンド: curl

set -euo pipefail

URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT="corpus/tiny_shakespeare.txt"

mkdir -p corpus

echo "==> downloading $URL"
curl -L --silent --show-error "$URL" -o "$OUT"

WC=$(wc -c < "$OUT")
echo "==> done"
echo "wrote $OUT: $WC bytes"
ls -la "$OUT"
