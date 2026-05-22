#!/bin/bash
set -e

# Always run from Project folder where this script lives
cd "$(dirname "$0")" || exit 1

# Usage:
#   ./evaluate_frames_00001_00006.sh 4dhumans v1
#   ./evaluate_frames_00001_00006.sh tokenhmr v1
#   ./evaluate_frames_00001_00006.sh tokenhmr v2
#   ./evaluate_frames_00001_00006.sh 4dhumans v1 4D-Humans/demo_out/custom_dir
#
# Args:
#   $1 = MODEL: tokenhmr or 4dhumans
#   $2 = METHOD: v1 or v2
#   $3 = optional PRED_DIR override

MODEL=${1:-4dhumans}
METHOD=${2:-v1}
PRED_DIR_ARG=${3:-}
SMPL=./smpl

# ------------------------------------------------------------
# Choose default prediction directory from model + method
# ------------------------------------------------------------
if [ -n "$PRED_DIR_ARG" ]; then
    PRED_DIR="$PRED_DIR_ARG"
else
    case "$MODEL" in
      tokenhmr|TokenHMR|tkhmr)
        MODEL="tokenhmr"

        case "$METHOD" in
          v1)
            PRED_DIR="TokenHMR/demo_out/my_image_smplify_v1"
            ;;
          v2)
            PRED_DIR="TokenHMR/demo_out/my_image_smplify_v2"
            ;;
          *)
            echo "[ERROR] Unknown method for TokenHMR: $METHOD"
            echo "Valid methods: v1, v2"
            exit 1
            ;;
        esac
        ;;

      4dhumans|4DHumans|4dh|4D)
        MODEL="4dhumans"

        case "$METHOD" in
          v1)
            PRED_DIR="4D-Humans/demo_out/my_image_smplify_v1"
            ;;
          v2)
            PRED_DIR="4D-Humans/demo_out/my_image_smplify_v2"
            ;;
          *)
            echo "[ERROR] Unknown method for 4D-Humans: $METHOD"
            echo "Valid methods: v1, v2"
            exit 1
            ;;
        esac
        ;;

      *)
        echo "[ERROR] Unknown model: $MODEL"
        echo "Valid models: tokenhmr, 4dhumans"
        exit 1
        ;;
    esac
fi

echo "============================================================"
echo "Evaluating optimized meshes"
echo "MODEL: $MODEL"
echo "METHOD: $METHOD"
echo "PRED_DIR: $PRED_DIR"
echo "SMPL: $SMPL"
echo "Frames: 00001 to 00006"
echo "============================================================"

if [ ! -d "$PRED_DIR" ]; then
    echo "[ERROR] Prediction directory not found:"
    echo "$PRED_DIR"
    exit 1
fi

for FRAME in $(seq -f "%05g" 1 6); do
    echo
    echo "############################################################"
    echo "Evaluating frame $FRAME"
    echo "############################################################"

    python src/evaluate_optimized_vs_gt.py \
      --frame "$FRAME" \
      --model "$MODEL" \
      --method "$METHOD" \
      --pred_dir "$PRED_DIR" \
      --smpl_model_dir "$SMPL" \
      --out_csv "metrics_${FRAME}_${MODEL}_${METHOD}_optimized_vs_gt.csv" \
      --summary_csv "metrics_${FRAME}_${MODEL}_${METHOD}_optimized_vs_gt_summary.csv"
done

echo
echo "Done. Generated metrics files:"
ls -lh metrics_*_${MODEL}_${METHOD}_optimized_vs_gt*.csv