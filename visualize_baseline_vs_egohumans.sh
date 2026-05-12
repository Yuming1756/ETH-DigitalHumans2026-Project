#!/bin/bash
set -e

# Always run relative to the Project folder where this script lives
cd "$(dirname "$0")" || exit 1

# Usage:
#   ./visualize_model_vs_egohumans.sh 00001 tokenhmr
#   ./visualize_model_vs_egohumans.sh 00001 4dhumans
#
# Defaults:
#   FRAME=00001
#   MODEL=tokenhmr

FRAME=${1:-00001}
MODEL=${2:-tokenhmr}

SCRIPT=./src/visualize_baseline_vs_egohumans_mesh3d.py
GT_ROOT=./data/mesh_cam/cam01/rgb
MAPPING="0:aria03,1:aria02,2:aria01,3:aria04"

case "$MODEL" in
  tokenhmr|TokenHMR|tkhmr)
    MODEL_NAME="TokenHMR"
    PRED_DIR=./TokenHMR/demo_out/my_image
    OUT_HTML=./visualizations/${FRAME}_TokenHMR_vs_EgoHumans_mesh3d.html
    ;;

  4dhumans|4DHumans|4dh|4D)
    MODEL_NAME="4DHumans"
    PRED_DIR=./4D-Humans/demo_out/my_image
    OUT_HTML=./visualizations/${FRAME}_4DHumans_vs_EgoHumans_mesh3d.html
    ;;

  *)
    echo "[ERROR] Unknown model: $MODEL"
    echo "Usage:"
    echo "  $0 00001 tokenhmr"
    echo "  $0 00001 4dhumans"
    exit 1
    ;;
esac

mkdir -p ./visualizations

echo "============================================================"
echo "Visualizing model vs EgoHumans"
echo "Frame: $FRAME"
echo "Model: $MODEL_NAME"
echo "Pred dir: $PRED_DIR"
echo "GT root: $GT_ROOT"
echo "Output: $OUT_HTML"
echo "============================================================"

if [ ! -f "$SCRIPT" ]; then
    echo "[ERROR] Python visualization script not found:"
    echo "$SCRIPT"
    exit 1
fi

if [ ! -d "$PRED_DIR" ]; then
    echo "[ERROR] Prediction directory not found:"
    echo "$PRED_DIR"
    exit 1
fi

if [ ! -d "$GT_ROOT/$FRAME" ]; then
    echo "[ERROR] GT frame directory not found:"
    echo "$GT_ROOT/$FRAME"
    exit 1
fi

python "$SCRIPT" \
  --frame "$FRAME" \
  --model "$MODEL" \
  --pred_dir "$PRED_DIR" \
  --gt_root "$GT_ROOT" \
  --mapping "$MAPPING" \
  --draw_centroid_lines \
  --axis \
  --save_html "$OUT_HTML"

echo
echo "Done. Open:"
echo "$OUT_HTML"