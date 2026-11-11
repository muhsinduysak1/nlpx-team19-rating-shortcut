#!/usr/bin/env bash
# Downloads the IMDb data used for all reported numbers (HuggingFace mirror of Maas et al. 2011)
# into data/ and checks that the files are byte-identical to the ones we used.
set -euo pipefail
mkdir -p data
base=https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text
for f in train-00000-of-00001.parquet test-00000-of-00001.parquet; do
  [ -f "data/$f" ] || curl -L --fail -o "data/$f" "$base/$f"
done
cd data
cat > SHA256SUMS <<'SUMS'
db47d16b5c297cc0dd625e519c81319c24c9149e70e8496de5475f6fa928342c  train-00000-of-00001.parquet
b52e26e2f872d282ffac460bf9770b25ac6f102cda0e6ca7158df98c94e8b3da  test-00000-of-00001.parquet
SUMS
if command -v sha256sum >/dev/null; then sha256sum -c SHA256SUMS; else shasum -a 256 -c SHA256SUMS; fi
