#!/usr/bin/env bash
# Full pipeline: train 3 seeds -> evaluate (val-selected configuration) -> compare with MAG-Net.
#   bash run_all.sh                      # data folder auto-detected under /kaggle/input
#   bash run_all.sh /path/to/data/raw    # or give it explicitly
# Extra training options can be passed with TRAIN_ARGS, e.g. TRAIN_ARGS="--encoder efficientnet-b3" bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
DATA=${1:-}
DARG=""; [ -n "$DATA" ] && DARG="--data_dir $DATA"
TRAIN_ARGS=${TRAIN_ARGS:-}
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
mkdir -p runs logs
echo "GPUs: $NGPU"

if [ "$NGPU" -ge 2 ]; then
  # two seeds in parallel, one per GPU, then the third
  CUDA_VISIBLE_DEVICES=0 python train.py $DARG $TRAIN_ARGS --seed 0 --out runs/seed0 > logs/seed0.log 2>&1 &
  P0=$!
  CUDA_VISIBLE_DEVICES=1 python train.py $DARG $TRAIN_ARGS --seed 1 --out runs/seed1 > logs/seed1.log 2>&1 &
  P1=$!
  wait $P0; wait $P1
  python train.py $DARG $TRAIN_ARGS --seed 2 --out runs/seed2 2>&1 | tee logs/seed2.log
else
  for s in 0 1 2; do
    python train.py $DARG $TRAIN_ARGS --seed $s --out runs/seed$s 2>&1 | tee logs/seed$s.log
  done
fi
tail -n 2 logs/seed*.log

python evaluate.py $DARG --ckpts runs/seed0/best.pth runs/seed1/best.pth runs/seed2/best.pth --out results \
  2>&1 | tee logs/evaluate.log
python compare.py --new results/eval_per_slice.csv 2>&1 | tee logs/compare.log
python explain.py $DARG --ckpts runs/seed0/best.pth runs/seed1/best.pth runs/seed2/best.pth 2>&1 | tee logs/explain.log
python compare.py --new results_xai/eval_per_slice.csv --out results_xai/comparison --new_cam "val-selected CAM" \
  2>&1 | tee logs/compare_xai.log

zip -qr outputs_v6.zip results results_xai logs runs/*/history.csv runs/*/run_config.json runs/*/best_summary.json
echo "Done. Download outputs_v6.zip and runs/seed*/best.pth"
