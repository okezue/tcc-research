#!/bin/bash
set -e
cd /workspace/tcc-research
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
mkdir -p artifacts/sigma_diag_validate

# GPT-2 Small all 12 layers (cheap)
python -m two_channel.exp_sigma_diag_full \
  --model openai-community/gpt2 \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11 \
  --n_cal 300 --n_bank 50000 --n_query 2000 --n_each_adj 5000 --ctx 32 \
  --dtype float32 2>&1 | tee artifacts/sigma_diag_validate/log_gpt2.txt

# Mistral-7B at 8 layers
python -m two_channel.exp_sigma_diag_full \
  --model mistralai/Mistral-7B-v0.1 \
  --layers 4,8,12,16,20,24,28,31 \
  --n_cal 300 --n_bank 50000 --n_query 2000 --n_each_adj 5000 --ctx 32 \
  --dtype bfloat16 2>&1 | tee artifacts/sigma_diag_validate/log_mistral.txt

# Phi-2 at 4 layers
python -m two_channel.exp_sigma_diag_full \
  --model microsoft/phi-2 \
  --layers 4,12,20,28 \
  --n_cal 300 --n_bank 50000 --n_query 2000 --n_each_adj 5000 --ctx 32 \
  --dtype bfloat16 2>&1 | tee artifacts/sigma_diag_validate/log_phi2.txt

# Qwen3-14B at 4 layers
python -m two_channel.exp_sigma_diag_full \
  --model Qwen/Qwen3-14B \
  --layers 10,20,30,39 \
  --n_cal 300 --n_bank 50000 --n_query 2000 --n_each_adj 5000 --ctx 32 \
  --dtype bfloat16 2>&1 | tee artifacts/sigma_diag_validate/log_qwen3.txt

# DeepSeek-R1-Distill-Qwen-14B at 4 layers
python -m two_channel.exp_sigma_diag_full \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --layers 12,24,36,47 \
  --n_cal 300 --n_bank 50000 --n_query 2000 --n_each_adj 5000 --ctx 32 \
  --dtype bfloat16 2>&1 | tee artifacts/sigma_diag_validate/log_deepseek.txt

echo "DONE"
