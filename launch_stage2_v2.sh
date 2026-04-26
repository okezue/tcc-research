#!/bin/bash
set -e
cd /workspace/tcc-research
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
mkdir -p artifacts/quotient_release logs

run_one() {
  local r=$1 b=$2 g=$3 s=$4 sd=$5
  local slug="r${r}_b${b}_g${g}_s${s}_seed${sd}"
  if [[ -f "artifacts/quotient_release/openai-community_gpt2_L6_${slug//e/e}.json" ]]; then
    echo "SKIP $slug (done)"
    return
  fi
  python -m two_channel.exp_quotient_release \
    --model openai-community/gpt2 --layer 6 \
    --r $r --beta $b --gamma $g --sigma_rel $s \
    --steps 100000 --batch_size 128 --lr 2e-4 --H_horizon 16 \
    --seq_len 64 --warmup 2000 --dtype float32 \
    --seed $sd --out_dir artifacts/quotient_release \
    --log_every 1000 --ckpt_every 50000 > logs/${slug}.log 2>&1
}

PARALLEL=4
sema=/tmp/pqr.sema
rm -f $sema
mkfifo $sema
exec 9<>$sema
for ((i=0;i<PARALLEL;i++)); do echo >&9; done

dispatch() {
  read -u 9
  ( "$@"; echo >&9 ) &
}

# Stage 2A: utility floor at gamma=0, sigma=0. r x beta grid.
echo "=== Stage 2A: utility floor (r x beta) ==="
for r in 8 16 32 64 128; do
  for b in 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1; do
    dispatch run_one $r $b 0 0 0
  done
done
wait

# Stage 2B: gamma x sigma at top (r,b). To stay simple we sweep over r in {32,64} and b in {3e-3,1e-2}
echo "=== Stage 2B: gamma x sigma sweep ==="
for r in 32 64; do
  for b in 3e-3 1e-2; do
    for g in 0.03 0.1 0.3; do
      for s in 0.05 0.1 0.2 0.5; do
        dispatch run_one $r $b $g $s 0
      done
    done
  done
done
wait

# Stage 2C: 3 seeds on the most promising config (defaults: r=64, b=1e-2, g=0.1, s=0.2)
echo "=== Stage 2C: 3 seeds ==="
for sd in 1 2; do
  dispatch run_one 64 1e-2 0.1 0.2 $sd
done
wait

echo DONE_STAGE2
