#!/bin/bash
set -e

# Always run relative to the Project folder where this script lives
cd "$(dirname "$0")" || exit 1

MODEL=${1:-4dhumans}

SMPL=./smpl

case "$MODEL" in
  tokenhmr|TokenHMR|tkhmr)
    MODEL_NAME="TokenHMR"
    PRED_DIR=../TokenHMR/demo_out/my_image
    OUT=joint_baseline_TokenHMR_auto_matched.txt
    ;;

  4dhumans|4DHumans|4dh|4D)
    MODEL_NAME="4DHumans"
    PRED_DIR=../4D-Humans/demo_out/my_image
    OUT=joint_baseline_4DHumans_auto_matched.txt
    ;;

  *)
    echo "[ERROR] Unknown model: $MODEL"
    echo
    echo "Usage:"
    echo "  $0 tokenhmr"
    echo "  $0 4dhumans"
    exit 1
    ;;
esac

GT_ROOT=./data/mesh_cam_unscaled/cam01/rgb

if [ ! -d "$PRED_DIR" ]; then
  echo "[ERROR] Prediction directory not found:"
  echo "$PRED_DIR"
  exit 1
fi

if [ ! -d "$GT_ROOT" ]; then
  echo "[ERROR] GT root not found:"
  echo "$GT_ROOT"
  exit 1
fi

{
  echo "================ AUTO-MATCHED JOINT-RELATED BASELINE RESULTS ================"
  echo "Model: $MODEL_NAME"
  echo "Prediction dir: $PRED_DIR"
  echo "GT root: $GT_ROOT"
  echo "SMPL model dir: $SMPL"
  echo "Output file: $OUT"
  echo "Generated at: $(date)"
  echo
  echo "Identity matching:"
  echo "  Per-frame auto matching by minimum root-aligned SMPL-joint MPJPE"
  echo "  This fixes unstable detection IDs across frames."
  echo

  frames=$(find "$PRED_DIR" -maxdepth 1 -name "*.obj" \
    ! -name "*_all.obj" \
    -printf "%f\n" | awk -F'_' '{print $1}' | sort -u)

  if [ -z "$frames" ]; then
    echo "[ERROR] No prediction OBJ files found in:"
    echo "$PRED_DIR"
    exit 1
  fi

  for frame in $frames; do
    GT_DIR="$GT_ROOT/$frame"

    echo
    echo "################################################################"
    echo "Model: $MODEL_NAME"
    echo "Frame: $frame"
    echo "Prediction frame files: $PRED_DIR/${frame}_*.obj"
    echo "GT frame dir: $GT_DIR"
    echo "################################################################"

    if [ ! -d "$GT_DIR" ]; then
      echo "[Warning] GT directory not found: $GT_DIR"
      continue
    fi

    python ./src/baseline_mpjpe_same_regressor.py \
      --auto_match_frame \
      --pred_dir "$PRED_DIR" \
      --gt_dir "$GT_DIR" \
      --frame "$frame" \
      --smpl_model_dir "$SMPL"
  done

} 2>&1 | tee "$OUT"

echo
echo "Saved auto-matched joint baseline results to: $OUT"