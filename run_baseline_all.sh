#!/bin/bash
set -euo pipefail

# Always run relative to the Project folder where this script lives
cd "$(dirname "$0")" || exit 1

MODEL=${1:-4dhumans}

SMPL=./smpl
GT_ROOT=./data/mesh_cam_unscaled/cam01/rgb
FRAMES=("00001" "00002" "00003" "00004" "00005" "00006")

case "$MODEL" in
  tokenhmr|TokenHMR|tkhmr)
    MODEL_NAME="TokenHMR"
    PRED_DIR=./TokenHMR/demo_out/my_image_with_smpl_params
    OUT=joint_baseline_TokenHMR_00001_00006_auto_matched.txt
    ;;

  4dhumans|4DHumans|4dh|4D)
    MODEL_NAME="4DHumans"
    PRED_DIR=./4D-Humans/demo_out/my_image_with_smpl_params
    OUT=joint_baseline_4DHumans_00001_00006_auto_matched.txt
    ;;

  *)
    echo "[ERROR] Unknown model: $MODEL"
    echo "Usage:"
    echo "  $0 tokenhmr"
    echo "  $0 4dhumans"
    exit 1
    ;;
esac

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

if [ ! -f "$SMPL/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl" ]; then
  echo "[ERROR] SMPL model file not found:"
  echo "$SMPL/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
  echo
  echo "Either copy the SMPL file into ./smpl/ or change SMPL to the correct path."
  exit 1
fi

{
  echo "================ AUTO-MATCHED JOINT-RELATED BASELINE RESULTS ================"
  echo "Model: $MODEL_NAME"
  echo "Frames: ${FRAMES[*]}"
  echo "Prediction dir: $PRED_DIR"
  echo "GT root: $GT_ROOT"
  echo "SMPL model dir: $SMPL"
  echo "Output file: $OUT"
  echo "Generated at: $(date)"
  echo
  echo "Identity matching:"
  echo "  Per-frame auto matching by minimum root-aligned SMPL-joint MPJPE"
  echo

  for FRAME in "${FRAMES[@]}"; do
    GT_DIR="$GT_ROOT/$FRAME"

    echo
    echo "################################################################"
    echo "Model: $MODEL_NAME"
    echo "Frame: $FRAME"
    echo "Prediction frame files: $PRED_DIR/${FRAME}_*.obj"
    echo "GT frame dir: $GT_DIR"
    echo "################################################################"

    if [ ! -d "$GT_DIR" ]; then
      echo "[Warning] GT directory not found: $GT_DIR"
      continue
    fi

    if ! ls "$PRED_DIR"/${FRAME}_*.obj >/dev/null 2>&1; then
      echo "[Warning] No prediction OBJ files found for frame $FRAME:"
      echo "$PRED_DIR/${FRAME}_*.obj"
      continue
    fi

    python ./src/baseline_mpjpe_same_regressor.py \
      --auto_match_frame \
      --pred_dir "$PRED_DIR" \
      --gt_dir "$GT_DIR" \
      --frame "$FRAME" \
      --smpl_model_dir "$SMPL"
  done

} 2>&1 | tee "$OUT"

echo
echo "Saved auto-matched joint baseline results to: $OUT"