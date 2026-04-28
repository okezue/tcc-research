#!/bin/bash
set -e
cd /workspace/tcc-research
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
mkdir -p artifacts/smt logs

train() {
  local arch=$1 r=$2 m=$3 lj=$4 tag=$5
  python -m two_channel.exp_smt_train \
    --arch $arch --corpus wikitext \
    --r $r --m $m --n_layers 12 --hr 4 --hm 4 --ff_r 512 --ff_m 1280 \
    --max_T 256 --seq_len 256 --batch_size 32 --lr 3e-4 \
    --steps 20000 --warmup 1000 \
    --lambda_jac $lj --probe_layers 4,6,8 \
    --n_train_samples 100000 \
    --seed 0 --dtype bfloat16 \
    --tag $tag --out_dir artifacts/smt \
    --log_every 500 --ckpt_every 10000 > logs/smt_${tag}.log 2>&1
}

measure() {
  local tag=$1
  local slug=$(ls artifacts/smt/*${tag}*.final.pt 2>/dev/null | head -1 | xargs -n1 basename | sed 's/.final.pt//')
  if [ -z "$slug" ]; then echo "no ckpt for $tag"; return; fi
  python -m two_channel.exp_smt_measure \
    --ckpt artifacts/smt/${slug}.final.pt \
    --info_json artifacts/smt/${slug}.json \
    --corpus wikitext --probe_layers 4,6,8 --n_cal 300 --n_pairs 500 --ctx 64 \
    --out artifacts/smt/${slug}.measure.json
}

echo "=== SMT main (r=128 m=640 lambda_jac=1e-3) ==="
train smt 128 640 1e-3 main

echo "=== SMT no-Jac (r=128 m=640 lambda_jac=0) ==="
train smt 128 640 0 nojac

echo "=== SMT r=64 (r=64 m=704 lambda_jac=1e-3) ==="
train smt 64 704 1e-3 r64

echo "=== Baseline GPT (matched param count) ==="
train baseline 128 640 0 baseline

echo "=== Measure all ==="
for tag in main nojac r64 baseline; do
  measure $tag
done

echo DONE_SMT
