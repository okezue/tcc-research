#!/bin/bash
# Measure G_Mah for all completed SMT/baseline checkpoints
set -e
cd /workspace/tcc-research
export HF_HOME=/workspace/.cache/huggingface
mkdir -p artifacts/smt_scaling
export CUDA_VISIBLE_DEVICES=${1:-0}

P30="2,4,5"
P90="4,6,8"
P300="6,10,13"

probes_for() {
  local nL=$1
  if [ "$nL" = "8" ]; then echo $P30
  elif [ "$nL" = "12" ]; then echo $P90
  elif [ "$nL" = "20" ]; then echo $P300
  fi
}

for ckpt in artifacts/smt_scaling/*.final.pt; do
  [ -f "$ckpt" ] || continue
  slug=$(basename "$ckpt" .final.pt)
  out="artifacts/smt_scaling/${slug}.measure.json"
  if [ -f "$out" ]; then echo "[SKIP $slug]"; continue; fi
  info_json="artifacts/smt_scaling/${slug}.json"
  nL=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_layers'])" "$info_json")
  probes=$(probes_for "$nL")
  echo "[MEASURE $slug nL=$nL probes=$probes]"
  python -m two_channel.exp_smt_measure \
    --ckpt "$ckpt" --info_json "$info_json" \
    --corpus wikitext --probe_layers "$probes" \
    --n_cal 300 --n_pairs 500 --ctx 64 \
    --out "$out"
done
echo "MEASURE DONE"
