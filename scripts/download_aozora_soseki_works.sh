#!/usr/bin/env bash
# 青空文庫から夏目漱石の主要長編 7 作品 (全て新字新仮名) を取得し、
# ルビ・編集注記等を除去して 1 ファイルに連結 (約 2.36M char) する。
#
# 出力: corpus/aozora_soseki_works.txt
#       各作品の前に "===== <タイトル> =====" のヘッダ行を入れる
#       (BOS/EOS のような明示的境界はモデル側で BPE/Char tokenizer に任せる)
#
# 著者: 夏目漱石 (著作権切れ・パブリックドメイン)
# 仮名遣い: 全作品 「新字新仮名」 で統一 (それから/門 は仮名遣いが異なるため除外)
#
# 必要なコマンド: curl, unzip, python3

set -euo pipefail

URL_BASE="https://www.aozora.gr.jp/cards/000148/files"
OUT="corpus/aozora_soseki_works.txt"

# (slug | title | zip filename) - 出版年順
WORKS=(
    "wagahai_neko|吾輩は猫である|789_ruby_5639.zip"
    "botchan|坊っちゃん|752_ruby_2438.zip"
    "kusamakura|草枕|776_ruby_6020.zip"
    "sanshiro|三四郎|794_ruby_4237.zip"
    "kojin|行人|775_ruby_2064.zip"
    "kokoro|こころ|773_ruby_5968.zip"
    "michikusa|道草|783_ruby_1311.zip"
)

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

mkdir -p corpus

: > "$OUT"  # truncate

for entry in "${WORKS[@]}"; do
    slug="${entry%%|*}"
    rest="${entry#*|}"
    title="${rest%%|*}"
    zipname="${rest##*|}"

    echo "==> [$slug] downloading $title"
    curl -L --silent --show-error "$URL_BASE/$zipname" -o "$TMPDIR/$slug.zip"

    rm -f "$TMPDIR/$slug"/*
    mkdir -p "$TMPDIR/$slug"
    unzip -q -o -d "$TMPDIR/$slug" "$TMPDIR/$slug.zip"

    raw="$(find "$TMPDIR/$slug" -maxdepth 1 -name '*.txt' | head -1)"
    if [[ -z "$raw" ]]; then
        echo "  no .txt extracted from $zipname" >&2
        exit 1
    fi

    python3 - "$raw" "$slug" "$title" "$OUT" <<'PY'
import re
import sys

raw_path, slug, title, out_path = sys.argv[1:5]

with open(raw_path, encoding="shift_jis", errors="replace") as f:
    text = f.read()

text = text.replace("\r\n", "\n").replace("\r", "\n")

# 青空文庫プレーンテキスト構造: meta -- header notes -- body -- 底本
parts = re.split(r"^-{20,}\n", text, flags=re.M)
if len(parts) >= 3:
    body = parts[2]
elif len(parts) == 2:
    body = parts[1]
else:
    body = text

m = re.search(r"^底本[：:]", body, flags=re.M)
if m:
    body = body[: m.start()]

# ルビ・編集注記の除去
body = re.sub(r"《[^》]*》", "", body)         # ルビ
body = body.replace("｜", "")                    # ルビ範囲開始マーカー
body = re.sub(r"［＃[^］]*］", "", body)        # 編集注記

# 連続改行を整理
body = re.sub(r"\n{3,}", "\n\n", body).strip()

with open(out_path, "a", encoding="utf-8") as f:
    f.write(f"===== {title} =====\n\n")
    f.write(body)
    f.write("\n\n")

n_chars = len(body)
n_bytes = len(body.encode("utf-8"))
print(f"  [{slug}] {title}: {n_chars} chars ({n_bytes} bytes)")
PY
done

echo "==> done"
TOTAL_CHARS=$(python3 -c "print(len(open('$OUT', encoding='utf-8').read()))")
TOTAL_BYTES=$(wc -c < "$OUT")
echo "wrote $OUT: $TOTAL_CHARS chars ($TOTAL_BYTES bytes)"
ls -la "$OUT"
