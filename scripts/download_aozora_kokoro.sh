#!/usr/bin/env bash
# 青空文庫から夏目漱石「こころ」をダウンロードし、 ルビ・編集注記等を除去して
# UTF-8 のプレーンテキストとして corpus/aozora_kokoro.txt に保存する。
#
# 出典: 青空文庫 https://www.aozora.gr.jp/cards/000148/files/773_ruby_5968.zip
# 著者: 夏目漱石 (著作権切れ・パブリックドメイン)
#
# 必要なコマンド: curl, unzip, python3

set -euo pipefail

URL="https://www.aozora.gr.jp/cards/000148/files/773_ruby_5968.zip"
OUT="corpus/aozora_kokoro.txt"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

mkdir -p corpus

echo "==> downloading $URL"
curl -L --silent --show-error "$URL" -o "$TMPDIR/kokoro.zip"

echo "==> unzipping"
unzip -q -d "$TMPDIR" "$TMPDIR/kokoro.zip"

RAW="$(find "$TMPDIR" -maxdepth 1 -name '*.txt' | head -1)"
if [[ -z "$RAW" ]]; then
    echo "no .txt extracted from zip" >&2
    exit 1
fi
echo "==> extracted: $RAW"

echo "==> converting Shift-JIS -> UTF-8 and stripping annotations"
python3 - "$RAW" "$OUT" <<'PY'
import re
import sys

raw_path, out_path = sys.argv[1], sys.argv[2]

with open(raw_path, encoding="shift_jis", errors="replace") as f:
    text = f.read()

# CRLF -> LF
text = text.replace("\r\n", "\n").replace("\r", "\n")

# 青空文庫プレーンテキストの構造:
#   [タイトル / 著者]
#   -------------------------------------------------------
#   [凡例]
#   -------------------------------------------------------
#   [本文]
#   底本: ...
parts = re.split(r"^-{20,}\n", text, flags=re.M)
if len(parts) >= 3:
    body = parts[2]
elif len(parts) == 2:
    body = parts[1]
else:
    body = text

# 底本以降を削除
m = re.search(r"^底本[：:]", body, flags=re.M)
if m:
    body = body[: m.start()]

# ルビ: 親文字《よみがな》 → 親文字
body = re.sub(r"《[^》]*》", "", body)
# ルビ範囲指定マーカー (｜) を削除
body = body.replace("｜", "")
# 編集注記: ［＃...］ を削除
body = re.sub(r"［＃[^］]*］", "", body)

# 空白整理
body = re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"

with open(out_path, "w", encoding="utf-8") as f:
    f.write(body)

print(
    f"wrote {out_path}: "
    f"{len(body)} chars, {len(body.encode('utf-8'))} bytes"
)
PY

echo "==> done"
ls -la "$OUT"
