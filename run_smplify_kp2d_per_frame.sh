#!/bin/bash
set -e

# Always run relative to the Project folder where this script lives
cd "$(dirname "$0")" || exit 1

# Usage:
#   ./run_smplify_kp2d_per_frame.sh 00002 tokenhmr adam "0:aria02,1:aria03,2:aria04,3:aria01"
#
# Defaults:
#   FRAME=00001
#   MODEL=tokenhmr
#   OPTIMIZER=lbfgs
#   MAPPING=0:aria03,1:aria02,2:aria01,3:aria04

FRAME=${1:-00001}
MODEL=${2:-tokenhmr}
OPTIMIZER=${3:-lbfgs}
MAPPING=${4:-"0:aria03,1:aria02,2:aria01,3:aria04"}

TARGET_DIR=/work/courses/digital_human/team6/Project/head_targets_unscaled
SMPL=/work/courses/digital_human/team6/
EXO_CAM=cam01

# ------------------------------------------------------------
# Validate optimizer
# ------------------------------------------------------------
case "$OPTIMIZER" in
  adam|Adam|ADAM)
    OPTIMIZER_NAME="adam"
    LR=0.01
    OPT_ARGS="--optimizer adam --iters 300 --lr $LR"
    ;;

  lbfgs|LBFGS|L-BFGS|l-bfgs)
    OPTIMIZER_NAME="lbfgs"
    LR=1.0
    OPT_ARGS="--optimizer lbfgs --lr $LR --lbfgs_max_iter 50"
    ;;

  *)
    echo "[ERROR] Unknown optimizer: $OPTIMIZER"
    echo
    echo "Usage:"
    echo "  $0 00001 tokenhmr lbfgs"
    echo "  $0 00001 tokenhmr adam"
    echo "  $0 00002 tokenhmr adam \"0:aria02,1:aria03,2:aria04,3:aria01\""
    echo "  $0 00001 4dhumans lbfgs"
    exit 1
    ;;
esac

# ------------------------------------------------------------
# Choose model-specific prediction/output directories
# ------------------------------------------------------------
case "$MODEL" in
  tokenhmr|TokenHMR|tkhmr)
    MODEL_NAME="TokenHMR"
    PRED_DIR=/work/courses/digital_human/team6/Project/TokenHMR/demo_out/my_image_smplify_v1/
    OUT_DIR=/work/courses/digital_human/team6/Project/TokenHMR/demo_out/my_image_smplify_${OPTIMIZER_NAME}_kp2d
    NPZ_SUFFIX=".npz"
    ;;

  4dhumans|4DHumans|4dh|4D)
    MODEL_NAME="4DHumans"
    PRED_DIR=/work/courses/digital_human/team6/Project/4D-Humans/demo_out/my_image
    OUT_DIR=/work/courses/digital_human/team6/Project/4D-Humans/demo_out/my_image_smplify_${OPTIMIZER_NAME}_kp2d
    NPZ_SUFFIX="_4dhumans.npz"
    ;;

  *)
    echo "[ERROR] Unknown model: $MODEL"
    echo
    echo "Usage:"
    echo "  $0 00001 tokenhmr lbfgs"
    echo "  $0 00001 4dhumans lbfgs"
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"

# ------------------------------------------------------------
# Parse detection-id to EgoHumans identity mapping
# Example:
#   MAPPING="0:aria02,1:aria03,2:aria04,3:aria01"
# ------------------------------------------------------------
declare -A MAP

IFS=',' read -ra PAIRS <<< "$MAPPING"
for PAIR in "${PAIRS[@]}"; do
    DET_ID="${PAIR%%:*}"
    ARIA="${PAIR#*:}"

    DET_ID="$(echo "$DET_ID" | xargs)"
    ARIA="$(echo "$ARIA" | xargs)"

    if [[ "$ARIA" != aria* ]]; then
        ARIA="aria${ARIA}"
    fi

    MAP[$DET_ID]="$ARIA"
done

echo "============================================================"
echo "Optimizing all people"
echo "Model: $MODEL_NAME"
echo "Frame: $FRAME"
echo "Optimizer: $OPTIMIZER_NAME"
echo "Learning rate: $LR"
echo "Prediction dir: $PRED_DIR"
echo "Output dir: $OUT_DIR"
echo "Target dir: $TARGET_DIR"
echo "Exo cam: $EXO_CAM"
echo "SMPL dir: $SMPL"
echo "Mapping string: $MAPPING"
echo "Parsed mapping:"
for DET_ID in 0 1 2 3; do
    echo "  det $DET_ID -> ${MAP[$DET_ID]}"
done
echo "============================================================"

if [ ! -d "$PRED_DIR" ]; then
    echo "[ERROR] Prediction directory not found:"
    echo "$PRED_DIR"
    exit 1
fi

for DET_ID in 0 1 2 3; do
    ARIA=${MAP[$DET_ID]}

    if [ -z "$ARIA" ]; then
        echo "[ERROR] No Aria identity specified for detection id $DET_ID"
        echo "Mapping was: $MAPPING"
        exit 1
    fi

    PRED_NPZ="$PRED_DIR/${FRAME}_${DET_ID}${NPZ_SUFFIX}"
    PRED_OBJ="$PRED_DIR/${FRAME}_${DET_ID}.obj"
    TARGET="$TARGET_DIR/${FRAME}_${ARIA}_${EXO_CAM}_egohumans_style.txt"
    OUT_OBJ="$OUT_DIR/${FRAME}_${DET_ID}.obj"

    echo
    echo "============================================================"
    echo "Model: $MODEL_NAME"
    echo "Frame: $FRAME"
    echo "Optimizer: $OPTIMIZER_NAME"
    echo "Detection id: $DET_ID"
    echo "Matched Aria: $ARIA"
    echo "Pred NPZ: $PRED_NPZ"
    echo "Pred OBJ: $PRED_OBJ"
    echo "Head target: $TARGET"
    echo "Out OBJ: $OUT_OBJ"
    echo "============================================================"

    if [ ! -f "$PRED_NPZ" ]; then
        echo "[ERROR] Missing prediction NPZ: $PRED_NPZ"
        echo
        echo "For TokenHMR, expected:"
        echo "  ${FRAME}_${DET_ID}_tkhmr.npz"
        echo
        echo "For 4DHumans, this script expects:"
        echo "  ${FRAME}_${DET_ID}_4dhumans.npz"
        echo
        echo "If your 4DHumans NPZ has a different suffix, modify NPZ_SUFFIX in this script."
        exit 1
    fi

    if [ ! -f "$PRED_OBJ" ]; then
        echo "[ERROR] Missing prediction OBJ: $PRED_OBJ"
        exit 1
    fi

    if [ ! -f "$TARGET" ]; then
        echo "[ERROR] Missing head target file: $TARGET"
        echo "Generate it first, for example:"
        echo "  ./extract_all_aria_targets.sh $FRAME $EXO_CAM"
        exit 1
    fi

    python src/smplify_2d_keypoints.py \
      --pred_npz "$PRED_NPZ" \
      --pred_obj_template "$PRED_OBJ" \
      --smpl_model_dir "$SMPL" \
      --out_obj "$OUT_OBJ" \
      --keypoints_2d_json /work/courses/digital_human/team6/Project/keypoints2d/${FRAME}_${MAP[$DET_ID]}_cam01.json
      
done

echo
echo "Done. Optimized meshes saved to:"
echo "$OUT_DIR"