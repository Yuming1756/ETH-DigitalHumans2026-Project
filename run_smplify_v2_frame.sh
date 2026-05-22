#!/bin/bash
set -e

# Always run relative to the Project folder where this script lives
cd "$(dirname "$0")" || exit 1

# Usage:
#   ./run_smplify_v2_frame.sh 00001 tokenhmr
#   ./run_smplify_v2_frame.sh 00001 4dhumans
#   ./run_smplify_v2_frame.sh 00001 tokenhmr "0:aria03,1:aria02,2:aria01,3:aria04"
#
# Defaults:
#   FRAME=00001
#   MODEL=tokenhmr
#   MAPPING=0:aria03,1:aria02,2:aria01,3:aria04

FRAME=${1:-00001}
MODEL=${2:-tokenhmr}
MAPPING=${3:-"0:aria03,1:aria02,2:aria01,3:aria04"}

EXO_CAM=cam01
SMPL=./smpl
TARGET_DIR=./head_targets_unscaled
KEYPOINTS_DIR=./keypoints2d

# ------------------------------------------------------------
# Choose model-specific directories
# ------------------------------------------------------------
case "$MODEL" in
  tokenhmr|TokenHMR|tkhmr)
    MODEL_NAME="TokenHMR"
    STAGE1_DIR=./TokenHMR/demo_out/my_image_smplify_v1
    OUT_DIR=./TokenHMR/demo_out/my_image_smplify_v2
    ;;

  4dhumans|4DHumans|4dh|4D)
    MODEL_NAME="4DHumans"
    STAGE1_DIR=./4D-Humans/demo_out/my_image_smplify_v1
    OUT_DIR=./4D-Humans/demo_out/my_image_smplify_v2
    ;;

  *)
    echo "[ERROR] Unknown model: $MODEL"
    echo "Usage:"
    echo "  $0 00001 tokenhmr"
    echo "  $0 00001 4dhumans"
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
echo "Running SMPLify Stage 2: optimize global_orient/body_pose/betas"
echo "Model: $MODEL_NAME"
echo "Frame: $FRAME"
echo "Stage-1 dir: $STAGE1_DIR"
echo "Output dir: $OUT_DIR"
echo "Target dir: $TARGET_DIR"
echo "Keypoints dir: $KEYPOINTS_DIR"
echo "SMPL dir: $SMPL"
echo "Exo cam: $EXO_CAM"
echo "Mapping string: $MAPPING"
echo "Parsed mapping:"
for DET_ID in 0 1 2 3; do
    echo "  det $DET_ID -> ${MAP[$DET_ID]}"
done
echo "============================================================"

if [ ! -d "$STAGE1_DIR" ]; then
    echo "[ERROR] Stage-1 directory not found:"
    echo "$STAGE1_DIR"
    echo
    echo "Run Stage 1 first, for example:"
    echo "  ./run_smplify_v1_frame.sh $FRAME $MODEL"
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

if [ ! -d "$TARGET_DIR" ]; then
    echo "[ERROR] Head target directory not found:"
    echo "$TARGET_DIR"
    echo
    echo "Generate unscaled head targets first, for example:"
    echo "  ./extract_all_aria_targets.sh $FRAME $EXO_CAM"
    exit 1
fi

for DET_ID in 0 1 2 3; do
    ARIA=${MAP[$DET_ID]}

    if [ -z "$ARIA" ]; then
        echo "[ERROR] No Aria identity specified for detection id $DET_ID"
        echo "Mapping was: $MAPPING"
        exit 1
    fi

    STAGE1_NPZ="$STAGE1_DIR/${FRAME}_${DET_ID}.npz"
    KEYPOINTS_JSON="$KEYPOINTS_DIR/${FRAME}_${ARIA}_${EXO_CAM}.json"
    TARGET="$TARGET_DIR/${FRAME}_${ARIA}_${EXO_CAM}_egohumans_style.txt"
    OUT_OBJ="$OUT_DIR/${FRAME}_${DET_ID}.obj"

    echo
    echo "============================================================"
    echo "Stage 2 full-parameter optimization"
    echo "Model: $MODEL_NAME"
    echo "Frame: $FRAME"
    echo "Detection id: $DET_ID"
    echo "Matched Aria: $ARIA"
    echo "Stage-1 NPZ: $STAGE1_NPZ"
    echo "2D keypoints: $KEYPOINTS_JSON"
    echo "Head target: $TARGET"
    echo "Out OBJ: $OUT_OBJ"
    echo "============================================================"

    if [ ! -f "$STAGE1_NPZ" ]; then
        echo "[ERROR] Missing Stage-1 NPZ: $STAGE1_NPZ"
        echo
        echo "Run Stage 1 first, for example:"
        echo "  ./run_smplify_v1_frame.sh $FRAME $MODEL \"$MAPPING\""
        exit 1
    fi

    if [ ! -f "$KEYPOINTS_JSON" ]; then
        echo "[ERROR] Missing 2D keypoints JSON: $KEYPOINTS_JSON"
        echo
        echo "Generate it first, for example:"
        echo "  ./make_all_keypoints2d_json_frame.sh $FRAME $EXO_CAM"
        exit 1
    fi

    if [ ! -f "$TARGET" ]; then
        echo "[ERROR] Missing head target file: $TARGET"
        echo
        echo "Generate it first, for example:"
        echo "  ./extract_all_aria_targets.sh $FRAME $EXO_CAM"
        exit 1
    fi

    python src/smplify_v2_all_param.py \
      --stage1_npz "$STAGE1_NPZ" \
      --keypoints_2d_json "$KEYPOINTS_JSON" \
      --head_target_file "$TARGET" \
      --head_target_coord camera \
      --smpl_model_dir "$SMPL" \
      --out_obj "$OUT_OBJ" \
      --optimizer lbfgs \
      --lr 0.05 \
      --lbfgs_max_iter 10000 \
      --w_2d 10.0 \
      --w_head 0.0 \
      --proxy_radius 0.0 \
      --w_orient_prior 0.5 \
      --w_pose_prior 0.5 \
      --w_betas_prior 1.0 \
      --w_betas_l2 1.0 \
      --w_z_positive 1.0 \
      --freeze_beta
done

echo
echo "Done. Stage-2 optimized meshes saved to:"
echo "$OUT_DIR"