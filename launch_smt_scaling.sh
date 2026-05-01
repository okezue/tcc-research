#!/bin/bash
# SMT scaling sweep: param × steps × seeds (4 tiers: 30M / 90M / 300M / 1B)
# Usage: launch_smt_scaling.sh <lane>   (lane in {0,1,2})
# Each lane runs sequentially on CUDA_VISIBLE_DEVICES = $LANE
set -e
cd /workspace/tcc-research
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
mkdir -p artifacts/smt_scaling logs_scaling

LANE=$1
if [ -z "$LANE" ]; then echo "usage: $0 <lane 0|1|2>"; exit 1; fi
export CUDA_VISIBLE_DEVICES=$LANE

train() {
  local arch=$1 r=$2 m=$3 nL=$4 ffr=$5 ffm=$6 hr=$7 hm=$8 lr=$9 steps=${10} seed=${11} probes=${12} tag=${13} lj=${14}
  local slug="${arch}_${tag}_st${steps}_s${seed}_r${r}_m${m}_nL${nL}_lj${lj}"
  local done_marker="artifacts/smt_scaling/${slug}.DONE"
  if [ -f "$done_marker" ]; then echo "[SKIP $slug]"; return; fi
  echo "[$(date -u +%H:%M:%S) GPU$LANE] START $slug"
  python -m two_channel.exp_smt_train \
    --arch $arch --corpus wikitext \
    --r $r --m $m --n_layers $nL --hr $hr --hm $hm --ff_r $ffr --ff_m $ffm \
    --max_T 256 --seq_len 256 --batch_size 32 --lr $lr \
    --steps $steps --warmup $((steps/20)) \
    --lambda_jac $lj --probe_layers $probes \
    --n_train_samples 200000 \
    --seed $seed --dtype bfloat16 \
    --tag "${tag}_st${steps}_s${seed}" --out_dir artifacts/smt_scaling \
    --log_every 500 --ckpt_every 5000 > logs_scaling/${slug}.log 2>&1
  touch "$done_marker"
  echo "[$(date -u +%H:%M:%S) GPU$LANE] DONE $slug"
}

# Tier configs:
# 30M:  r=64,  m=320,  nL=8,  ff_r=256,  ff_m=640,  hr=4, hm=4   d=384
# 90M:  r=128, m=640,  nL=12, ff_r=512,  ff_m=1280, hr=4, hm=4   d=768
# 300M: r=256, m=1024, nL=20, ff_r=1024, ff_m=2560, hr=4, hm=8   d=1280
# 1B:   r=512, m=1792, nL=24, ff_r=2048, ff_m=4480, hr=8, hm=8   d=2304   (SMT ~930M, base ~1.65B)
P30="2,4,5"
P90="4,6,8"
P300="6,10,13"
P1B="8,12,16"

# ---- Lane 0: 1B SMT (~21h) + 90M base s=0 + 90M base 5k + 30M SMT
if [ "$LANE" = "0" ]; then
  train smt      512 1792 24 2048 4480 8 8 1e-4 20000 0 $P1B  paramsweep 1e-3
  train baseline 128  640 12  512 1280 4 4 3e-4 20000 0 $P90  seedsweep  0
  train baseline 128  640 12  512 1280 4 4 3e-4  5000 0 $P90  stepsweep  0
  train smt       64  320  8  256  640 4 4 3e-4 20000 0 $P30  paramsweep 1e-3
fi

# ---- Lane 1: 1B baseline (~20h) + 90M SMT s=0 + 90M base s=1 + 90M SMT 5k + 30M base
if [ "$LANE" = "1" ]; then
  train baseline 512 1792 24 2048 4480 8 8 1e-4 20000 0 $P1B  paramsweep 0
  train smt      128  640 12  512 1280 4 4 3e-4 20000 0 $P90  seedsweep  1e-3
  train baseline 128  640 12  512 1280 4 4 3e-4 20000 1 $P90  seedsweep  0
  train smt      128  640 12  512 1280 4 4 3e-4  5000 0 $P90  stepsweep  1e-3
  train baseline  64  320  8  256  640 4 4 3e-4 20000 0 $P30  paramsweep 0
fi

# ---- Lane 2 (workhorse): 300M pair, 90M 80k pair, three 90M seeds
if [ "$LANE" = "2" ]; then
  train smt      256 1024 20 1024 2560 4 8 1.5e-4 20000 0 $P300 paramsweep 1e-3
  train baseline 256 1024 20 1024 2560 4 8 1.5e-4 20000 0 $P300 paramsweep 0
  train smt      128  640 12  512 1280 4 4 3e-4   80000 0 $P90  stepsweep  1e-3
  train baseline 128  640 12  512 1280 4 4 3e-4   80000 0 $P90  stepsweep  0
  train smt      128  640 12  512 1280 4 4 3e-4   20000 1 $P90  seedsweep  1e-3
  train smt      128  640 12  512 1280 4 4 3e-4   20000 2 $P90  seedsweep  1e-3
  train baseline 128  640 12  512 1280 4 4 3e-4   20000 2 $P90  seedsweep  0
fi

echo "[$(date -u +%H:%M:%S) GPU$LANE] LANE $LANE COMPLETE"
