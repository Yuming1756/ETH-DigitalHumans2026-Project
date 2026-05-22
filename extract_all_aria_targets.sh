#!/bin/bash
set -e

# Always run relative to the Project folder where this script lives
cd "$(dirname "$0")" || exit 1

DATA_ROOT=./data
SMPL=./smpl

# Use unscaled GT and save unscaled head targets.
GT_TYPE=mesh_cam_unscaled
TARGET_DIR=./head_targets_unscaled

# Usage:
#   ./extract_all_aria_targets.sh 00001 cam01
#
# Defaults:
#   FRAME=00001
#   EXO_CAM=cam01
FRAME=${1:-00001}
EXO_CAM=${2:-cam01}
ANCHOR_ARIA=aria01

GT_ROOT="$DATA_ROOT/$GT_TYPE/$EXO_CAM/rgb"

mkdir -p "$TARGET_DIR"

echo "============================================================"
echo "Extracting UNSCALED EgoHumans-style Aria camera/head targets"
echo "DATA_ROOT: $DATA_ROOT"
echo "FRAME: $FRAME"
echo "EXO_CAM: $EXO_CAM"
echo "ANCHOR_ARIA: $ANCHOR_ARIA"
echo "GT_TYPE: $GT_TYPE"
echo "TARGET_DIR: $TARGET_DIR"
echo "GT_ROOT: $GT_ROOT"
echo "============================================================"

if [ ! -d "$GT_ROOT/$FRAME" ]; then
    echo "[ERROR] GT frame directory not found:"
    echo "$GT_ROOT/$FRAME"
    echo
    echo "You probably need to generate/copy mesh_cam_unscaled first."
    exit 1
fi

for ARIA in aria01 aria02 aria03 aria04; do
    GT_OBJ="$GT_ROOT/$FRAME/mesh_${ARIA}.obj"
    SAVE_TARGET="$TARGET_DIR/${FRAME}_${ARIA}_${EXO_CAM}_egohumans_style.txt"

    echo
    echo "------------------------------------------------------------"
    echo "ARIA: $ARIA"
    echo "GT_OBJ: $GT_OBJ"
    echo "SAVE_TARGET: $SAVE_TARGET"
    echo "------------------------------------------------------------"

    if [ ! -f "$GT_OBJ" ]; then
        echo "[Warning] GT mesh not found: $GT_OBJ"
        echo "Skipping $ARIA"
        continue
    fi

    python src/extract_egohumans_aria_location.py \
      --data_root "$DATA_ROOT" \
      --aria "$ARIA" \
      --frame "$FRAME" \
      --exo_cam "$EXO_CAM" \
      --anchor_aria "$ANCHOR_ARIA" \
      --gt_obj "$GT_OBJ" \
      --smpl_model_dir "$SMPL" \
      --save_target "$SAVE_TARGET" \
      --remove_colmap_scale
done

echo
echo "Done. Saved UNSCALED targets to:"
echo "$TARGET_DIR"

echo
echo "Generated files:"
ls -lh "$TARGET_DIR"/${FRAME}_aria*_${EXO_CAM}_egohumans_style.txt 2>/dev/null || true