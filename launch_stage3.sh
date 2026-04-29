#!/bin/bash
set -e
cd /workspace/tcc-research
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
mkdir -p artifacts/inv_direct logs

train_one() {
  local def=$1 sigma=$2 fdiag=$3 tag=$4
  python -m two_channel.exp_inv_direct \
    --target_model openai-community/gpt2 --target_layer 6 \
    --defense $def --sigma $sigma --F_diag_path "$fdiag" \
    --seq_len 32 --max_T 64 --dm 512 --nhead 8 \
    --enc_layers 6 --dec_layers 6 --ff 2048 --drop 0.1 \
    --batch_size 64 --lr 3e-4 --steps 50000 --warmup 1000 \
    --seed 0 --target_dtype bfloat16 --out_dir artifacts/inv_direct \
    --log_every 1000 --ckpt_every 25000 > logs/inv_${tag}.log 2>&1
}

eval_one() {
  local def=$1 sigma=$2 fdiag=$3 tag=$4
  local slug=$(ls artifacts/inv_direct/openai-community_gpt2_L6_def-${def}*.final.pt 2>/dev/null | head -1 | xargs -n1 basename | sed s/.final.pt//)
  if [ -z "$slug" ]; then echo "no ckpt for $tag"; return; fi
  python -m two_channel.eval_inv_direct \
    --target_model openai-community/gpt2 --target_layer 6 \
    --inverter_ckpt artifacts/inv_direct/${slug}.final.pt \
    --defense $def --sigma $sigma --F_diag_path "$fdiag" \
    --n_test 500 --seq_len 32 --max_T 64 --beam 1 --target_dtype bfloat16 \
    --out artifacts/inv_direct/${slug}.eval.json
}

FDIAG=/workspace/tcc-research/artifacts/sigma_diag_validate/F_diag_openai-community_gpt2_L6.pt

echo "=== train clean ==="
train_one clean 0 "" clean

echo "=== train iso sigma=5 ==="
train_one isotropic 5.0 "" iso5

echo "=== train sigma_diag sigma=5 ==="
train_one sigma_diag 5.0 "$FDIAG" sd5

echo "=== eval all ==="
eval_one clean 0 "" clean
eval_one isotropic 5.0 "" iso5
eval_one sigma_diag 5.0 "$FDIAG" sd5

echo DONE_STAGE3
