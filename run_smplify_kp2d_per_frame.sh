#!/bin/bash
set -e

# Always run relative to the Project folder where this script lives
cd "$(dirname "$0")" || exit 1

# Usage:
#   ./run_smplify_kp2d_per_frame.sh 00001 tokenhmr lbfgs
#   ./run_smplify_kp2d_per_frame.sh 00001 4dhumans lbfgs
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

PROJECT_ROOT="$(pwd)"

TARGET_DIR="$PROJECT_ROOT/head_targets_unscaled"
KEYPOINTS_DIR="$PROJECT_ROOT/keypoints2d"
SMPL="$PROJECT_ROOT/smpl"
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

    # Input is Stage-1 output.
    # Expected files:
    #   TokenHMR/demo_out/my_image_smplify_v1/00001_0.obj
    #   TokenHMR/demo_out/my_image_smplify_v1/00001_0.npz
    PRED_DIR="$PROJECT_ROOT/TokenHMR/demo_out/my_image_smplify_v1"
    OUT_DIR="$PROJECT_ROOT/TokenHMR/demo_out/my_image_smplify_${OPTIMIZER_NAME}_kp2d"
    NPZ_SUFFIX=".npz"
    ;;

  4dhumans|4DHumans|4dh|4D)
    MODEL_NAME="4DHumans"

    # Input is Stage-1 output.
    # Expected files:
    #   4D-Humans/demo_out/my_image_smplify_v1/00001_0.obj
    #   4D-Humans/demo_out/my_image_smplify_v1/00001_0.npz
    PRED_DIR="$PROJECT_ROOT/4D-Humans/demo_out/my_image_smplify_v1"
    OUT_DIR="$PROJECT_ROOT/4D-Humans/demo_out/my_image_smplify_${OPTIMIZER_NAME}_kp2d"
    NPZ_SUFFIX=".npz"
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
echo "Optimizing with 2D keypoints"
echo "Project root: $PROJECT_ROOT"
echo "Model: $MODEL_NAME"
echo "Frame: $FRAME"
echo "Optimizer: $OPTIMIZER_NAME"
echo "Learning rate: $LR"
echo "Prediction dir: $PRED_DIR"
echo "Output dir: $OUT_DIR"
echo "Target dir: $TARGET_DIR"
echo "Keypoints dir: $KEYPOINTS_DIR"
echo "Exo cam: $EXO_CAM"
echo "SMPL dir: $SMPL"
echo "Mapping string: $MAPPING"
echo "Parsed mapping:"
for DET_ID in 0 1 2 3; do
    echo "  det $DET_ID -> ${MAP[$DET_ID]}"
done
echo "============================================================"

# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------
if [ ! -d "$PRED_DIR" ]; then
    echo "[ERROR] Prediction directory not found:"
    echo "$PRED_DIR"
    echo
    echo "Run Stage-1 first, for example:"
    echo "  ./run_smplify_v1_frame.sh"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "[ERROR] Head target directory not found:"
    echo "$TARGET_DIR"
    echo
    echo "Generate targets first, for example:"
    echo "  ./extract_all_aria_targets.sh $FRAME $EXO_CAM"
    exit 1
fi

if [ ! -d "$KEYPOINTS_DIR" ]; then
    echo "[ERROR] Keypoints directory not found:"
    echo "$KEYPOINTS_DIR"
    echo
    echo "Generate keypoints first, for example:"
    echo "  ./make_all_keypoints2d_json_frame.sh $FRAME $EXO_CAM"
    exit 1
fi

if [ ! -d "$SMPL" ]; then
    echo "[ERROR] SMPL directory not found:"
    echo "$SMPL"
    exit 1
fi

# ------------------------------------------------------------
# Run optimization for each detection/person
# ------------------------------------------------------------
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
    KEYPOINTS_JSON="$KEYPOINTS_DIR/${FRAME}_${ARIA}_${EXO_CAM}.json"
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
    echo "2D keypoints: $KEYPOINTS_JSON"
    echo "Out OBJ: $OUT_OBJ"
    echo "============================================================"

    if [ ! -f "$PRED_NPZ" ]; then
        echo "[ERROR] Missing prediction NPZ: $PRED_NPZ"
        echo
        echo "This script expects Stage-1 NPZ:"
        echo "  ${FRAME}_${DET_ID}.npz"
        echo
        echo "Run Stage-1 first."
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

    if [ ! -f "$KEYPOINTS_JSON" ]; then
        echo "[ERROR] Missing 2D keypoints JSON: $KEYPOINTS_JSON"
        echo "Generate it first, for example:"
        echo "  ./make_all_keypoints2d_json_frame.sh $FRAME $EXO_CAM"
        exit 1
    fi

    python src/smplify_2d_keypoints.py \
      --pred_npz "$PRED_NPZ" \
      --pred_obj_template "$PRED_OBJ" \
      --smpl_model_dir "$SMPL" \
      --out_obj "$OUT_OBJ" \
      --keypoints_2d_json "$KEYPOINTS_JSON" \
      $OPT_ARGS
done

echo
echo "Done. Optimized meshes saved to:"
echo "$OUT_DIR"