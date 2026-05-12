#!/bin/bash
set -e

cd ~/DigitalHumans || exit 1

# Usage:
#   ./visualize_smplify_frame.sh 00001 tokenhmr lbfgs
#   ./visualize_smplify_frame.sh 00001 tokenhmr adam
#   ./visualize_smplify_frame.sh 00001 4dhumans lbfgs
#
# Defaults:
#   FRAME=00001
#   MODEL=tokenhmr
#   OPTIMIZER=lbfgs

FRAME=${1:-00001}
MODEL=${2:-tokenhmr}
OPTIMIZER=${3:-lbfgs}

GT_ROOT=~/DigitalHumans/data/01_tagging/001_tagging/processed_data/mesh_cam/cam01/rgb
TARGET_DIR=~/DigitalHumans/head_targets
VIS_SCRIPT=~/DigitalHumans/visualize_frame_before_after_all_people.py

case "$OPTIMIZER" in
  adam|Adam|ADAM)
    OPTIMIZER_NAME="adam"
    ;;
  lbfgs|LBFGS|L-BFGS|l-bfgs)
    OPTIMIZER_NAME="lbfgs"
    ;;
  *)
    echo "[ERROR] Unknown optimizer: $OPTIMIZER"
    echo "Usage: $0 00001 tokenhmr lbfgs"
    echo "       $0 00001 tokenhmr adam"
    echo "       $0 00001 4dhumans lbfgs"
    exit 1
    ;;
esac

case "$MODEL" in
  tokenhmr|TokenHMR|tkhmr)
    MODEL_NAME="TokenHMR"
    BEFORE_DIR=~/DigitalHumans/TokenHMR/demo_out/my_image
    AFTER_DIR=~/DigitalHumans/TokenHMR/demo_out/my_image_smplify_${OPTIMIZER_NAME}
    OUT_HTML=~/DigitalHumans/visualizations/${FRAME}_TokenHMR_before_after_${OPTIMIZER_NAME}.html
    ;;

  4dhumans|4DHumans|4dh|4D)
    MODEL_NAME="4DHumans"
    BEFORE_DIR=~/DigitalHumans/4D-Humans/demo_out/my_image_real_intrinsics
    AFTER_DIR=~/DigitalHumans/4D-Humans/demo_out/my_image_real_intrinsics_smplify_${OPTIMIZER_NAME}
    OUT_HTML=~/DigitalHumans/visualizations/${FRAME}_4DHumans_before_after_${OPTIMIZER_NAME}.html
    ;;

  *)
    echo "[ERROR] Unknown model: $MODEL"
    echo "Usage: $0 00001 tokenhmr lbfgs"
    echo "       $0 00001 4dhumans lbfgs"
    exit 1
    ;;
esac

mkdir -p ~/DigitalHumans/visualizations

echo "============================================================"
echo "Visualizing before/after optimization"
echo "Model: $MODEL_NAME"
echo "Frame: $FRAME"
echo "Optimizer: $OPTIMIZER_NAME"
echo "Before dir: $BEFORE_DIR"
echo "After dir: $AFTER_DIR"
echo "GT root: $GT_ROOT"
echo "Target dir: $TARGET_DIR"
echo "Output HTML: $OUT_HTML"
echo "============================================================"

if [ ! -f "$VIS_SCRIPT" ]; then
    echo "[ERROR] Visualization Python script not found:"
    echo "$VIS_SCRIPT"
    echo
    echo "You need visualize_frame_before_after_all_people.py first."
    exit 1
fi

if [ ! -d "$BEFORE_DIR" ]; then
    echo "[ERROR] Before directory not found:"
    echo "$BEFORE_DIR"
    exit 1
fi

if [ ! -d "$AFTER_DIR" ]; then
    echo "[ERROR] After directory not found:"
    echo "$AFTER_DIR"
    echo
    echo "Run optimization first, for example:"
    echo "  ./run_smplify_all_people.sh $FRAME $MODEL_NAME $OPTIMIZER_NAME"
    exit 1
fi

if [ ! -d "$GT_ROOT/$FRAME" ]; then
    echo "[ERROR] GT frame directory not found:"
    echo "$GT_ROOT/$FRAME"
    exit 1
fi

python "$VIS_SCRIPT" \
  --frame "$FRAME" \
  --before_dir "$BEFORE_DIR" \
  --after_dir "$AFTER_DIR" \
  --gt_root "$GT_ROOT" \
  --target_dir "$TARGET_DIR" \
  --draw_lines \
  --save_html "$OUT_HTML"

echo
echo "Done. Open visualization:"
echo "$OUT_HTML"