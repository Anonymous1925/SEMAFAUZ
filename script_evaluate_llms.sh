#!/bin/bash

LLAMA_TOKEN=$1
GPT_TOKEN=$2

MODELS=("llama" "gpt")
DATASETS=("imdb" "ecthr_a" "scotus" "ledgar" "eurlex")
SPLITS=("train" "validation" "test")


if [ -z "$LLAMA_TOKEN" ] || [ -z "$GPT_TOKEN" ]; then
  echo "Usage: ./script_test_llms.sh <LLAMA_TOKEN> <GPT_TOKEN>"
  exit 1
fi

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
      echo "Running: $MODEL $DATASET $SPLIT"
      python3 -u test_llms_originals.py "$MODEL" "$DATASET" "$SPLIT" "$LLAMA_TOKEN" "$GPT_TOKEN"
    done
  done
done
