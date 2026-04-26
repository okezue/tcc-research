#!/bin/bash
set -e
cd /workspace/tcc-research
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
mkdir -p artifacts/quotient_release

run_one() {
  local r=$1 b=$2 g=$3 s=$4 sd=$5
  python -m two_channel.exp_quotient_release \
    --model openai-community/gpt2 --layer 6 \
    --r $r --beta $b --gamma $g --sigma_rel $s \
    --steps 100000 --batch_size 128 --lr 2e-4 --H_horizon 16 \
    --seq_len 64 --warmup 2000 --dtype float32 \
    --seed $sd --out_dir artifacts/quotient_release \
    --log_every 500 --ckpt_every 25000
}

# Stage 2A: utility-only floor (gamma=0, sigma=0); sweep r x beta
for r in 8 16 32 64 128; do
  for b in 1e-4 3e-4 1e-3 3e-3 1e-2; do
    run_one $r $b 0 0 0 2>&1 | tail -5
  done
done

# Stage 2B: gamma x sigma sweep at top 4 (r,b) from Stage 2A (will pick after, hardcode for now: r=64,b=1e-3)
for g in 0.03 0.1; do
  for s in 0.05 0.1 0.2 0.5; do
    run_one 64 1e-3 $g $s 0 2>&1 | tail -5
  done
done

echo DONE_STAGE2
