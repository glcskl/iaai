#!/bin/bash
# Скачивание датасета EMNIST-letters для обучения распознавания букв A-Z.
set -e
cd "$(dirname "$0")/data"
BASE="https://huggingface.co/datasets/Heliosoph/EMNIST/resolve/main"
for f in emnist-letters-train-images-idx3-ubyte.gz \
         emnist-letters-train-labels-idx1-ubyte.gz \
         emnist-letters-test-images-idx3-ubyte.gz \
         emnist-letters-test-labels-idx1-ubyte.gz; do
  out="${f#emnist-letters-}"
  out="${out/-idx1-ubyte/}"    # оставляем .gz
  out="${out/-idx3-ubyte/}"
  [ -f "$out" ] && { echo "уже есть: $out"; continue; }
  echo "скачиваю: $f"
  curl -sL -o "$out" "$BASE/$f"
done
echo "Готово. Данные на месте."
ls -la