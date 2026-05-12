#!/bin/bash
set -e

# Always run relative to the Project folder where this script lives
cd "$(dirname "$0")" || exit 1

# Usage:
#   ./run_smplify_all_people.sh 00001 tokenhmr lbfgs
#   ./run_smplify_all_people.sh 00001 tokenhmr adam
#   ./run_smplify_all_people.sh 00001 4dhumans lbfgs
#
# Defaults:
#   FRAME=00001
#   MODEL=tokenhmr
#   OPTIMIZER=lbfgs

FRAME=${1:-00001}
MODEL=${2:-tokenhmr}
OPTIMIZER=${3:-lbfgs}

TARGET_DIR=./head_targets
SMPL=./smpl
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
    PRED_DIR=./TokenHMR/demo_out/my_image
    OUT_DIR=./TokenHMR/demo_out/my_image_smplify_${OPTIMIZER_NAME}
    NPZ_SUFFIX="_tkhmr.npz"
    ;;

  4dhumans|4DHumans|4dh|4D)
    MODEL_NAME="4DHumans"
    PRED_DIR=./4D-Humans/demo_out/my_image
    OUT_DIR=./4D-Humans/demo_out/my_image_smplify_${OPTIMIZER_NAME}
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
echo "============================================================"

if [ ! -d "$PRED_DIR" ]; then
    echo "[ERROR] Prediction directory not found:"
    echo "$PRED_DIR"
    exit 1
fi

# detection id -> EgoHumans identity
declare -A MAP
MAP[0]=aria03
MAP[1]=aria02
MAP[2]=aria01
MAP[3]=aria04

for DET_ID in 0 1 2 3; do
    ARIA=${MAP[$DET_ID]}

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

    python src/smplify_v1_translation.py \
      --pred_npz "$PRED_NPZ" \
      --pred_obj_template "$PRED_OBJ" \
      --head_target_file "$TARGET" \
      --head_target_coord camera \
      --smpl_model_dir "$SMPL" \
      --out_obj "$OUT_OBJ" \
      $OPT_ARGS \
      --w_head 5.0 \
      --w_trans_prior 0.1
done

echo
echo "Done. Optimized meshes saved to:"
echo "$OUT_DIR"