#!/bin/bash
set -e
cd /workspace/tcc-research
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
mkdir -p artifacts/quotient_release logs

run_one() {
  local r=$1 b=$2 g=$3 s=$4 sd=$5
  local slug="r${r}_b${b}_g${g}_s${s}_seed${sd}"
  python -m two_channel.exp_quotient_release \
    --model openai-community/gpt2 --layer 6 \
    --r $r --beta $b --gamma $g --sigma_rel $s \
    --steps 50000 --batch_size 128 --lr 2e-4 --H_horizon 16 \
    --seq_len 64 --warmup 1000 --dtype bfloat16 \
    --seed $sd --out_dir artifacts/quotient_release \
    --log_every 1000 --ckpt_every 25000 > logs/${slug}.log 2>&1 &
}

echo "=== Missing batch 1: r=64 b=1e-1 (4 cells) ==="
run_one 64 1e-1 0.1 0.5 0
run_one 64 1e-1 0.3 0.1 0
run_one 64 1e-1 0.3 0.2 0
run_one 64 1e-1 0.3 0.5 0
wait

echo "=== Missing batch 2: Stage 2C (4 cells) ==="
run_one 64 1e-1 0.1 0.2 1
run_one 64 1e-1 0.1 0.2 2
run_one 32 1e-2 0.3 0.5 1
run_one 32 1e-2 0.3 0.5 2
wait

echo DONE_MISSING
