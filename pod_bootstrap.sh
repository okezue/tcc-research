#!/bin/bash
# Run on the pod after SSH-in. Idempotent.
set -e
cd /workspace
if [ ! -d tcc-research ]; then
  git clone https://github.com/okezue/tcc-research.git
fi
cd tcc-research
git fetch origin && git checkout main && git pull origin main
pip install -q --upgrade pip
pip install -q torch transformers datasets accelerate sentencepiece tokenizers
mkdir -p artifacts/smt_scaling logs_scaling
mkdir -p /workspace/.cache/huggingface
chmod +x launch_smt_scaling.sh launch_smt_scaling_measure.sh
echo "[BOOTSTRAP DONE]"

# Start 3 tmux sessions, one per lane
tmux kill-server 2>/dev/null || true
sleep 1
for L in 0 1 2; do
  tmux new-session -d -s "lane$L" "cd /workspace/tcc-research && bash launch_smt_scaling.sh $L 2>&1 | tee logs_scaling/lane${L}_console.log; bash"
done
echo "[TMUX LANES STARTED: lane0, lane1, lane2]"
echo "Attach with: tmux attach -t lane0"
echo "Tail logs:   tail -f logs_scaling/*.log"
