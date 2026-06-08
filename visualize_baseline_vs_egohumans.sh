#!/bin/bash
set -e

# Always run relative to the Project folder where this script lives
cd "$(dirname "$0")" || exit 1

# Usage:
#   ./visualize_baseline_vs_egohumans.sh
#
# Visualizes baseline model prediction vs EgoHumans GT
# for frames 00001–00006 and both TokenHMR / 4D-Humans.

PROJECT_ROOT=~/DigitalHumans/Project

VIS_SCRIPT="$PROJECT_ROOT/src/visualize_baseline_vs_egohumans_mesh3d.py"
VIS_DIR="$PROJECT_ROOT/visualizations"
GT_ROOT="$PROJECT_ROOT/data/mesh_cam_unscaled/cam01/rgb"

mkdir -p "$VIS_DIR"

get_mapping_for_frame() {
    local FRAME="$1"

    case "$FRAME" in
      00001)
        echo "0:aria03,1:aria02,2:aria01,3:aria04"
        ;;
      00002)
        echo "0:aria02,1:aria03,2:aria04,3:aria01"
        ;;
      00003)
        echo "0:aria03,1:aria02,2:aria04,3:aria01"
        ;;
      00004)
        echo "0:aria03,1:aria04,2:aria02,3:aria01"
        ;;
      00005)
        echo "0:aria03,1:aria04,2:aria02,3:aria01"
        ;;
      00006)
        echo "0:aria04,1:aria03,2:aria02,3:aria01"
        ;;
      *)
        echo "[ERROR] No mapping defined for frame $FRAME" >&2
        exit 1
        ;;
    esac
}

get_model_config() {
    local MODEL="$1"
    local FRAME="$2"

    case "$MODEL" in
      tokenhmr)
        MODEL_NAME="TokenHMR"
        MODEL_ARG="tokenhmr"
        PRED_DIR="$PROJECT_ROOT/TokenHMR/demo_out/my_image_with_smpl_params"
        OUT_HTML="$VIS_DIR/${FRAME}_TokenHMR_baseline_vs_GT_clean.html"
        ;;

      4dhumans)
        MODEL_NAME="4DHumans"
        MODEL_ARG="4dhumans"
        PRED_DIR="$PROJECT_ROOT/4D-Humans/demo_out/my_image_with_smpl_params"
        OUT_HTML="$VIS_DIR/${FRAME}_4DHumans_baseline_vs_GT_clean.html"
        ;;

      *)
        echo "[ERROR] Unknown model: $MODEL"
        exit 1
        ;;
    esac
}

if [ ! -f "$VIS_SCRIPT" ]; then
    echo "[ERROR] Visualization Python script not found:"
    echo "$VIS_SCRIPT"
    exit 1
fi

if [ ! -d "$GT_ROOT" ]; then
    echo "[ERROR] GT root not found:"
    echo "$GT_ROOT"
    exit 1
fi

for MODEL in tokenhmr 4dhumans; do
    for FRAME in 00001 00002 00003 00004 00005 00006; do
        get_model_config "$MODEL" "$FRAME"
        MAPPING=$(get_mapping_for_frame "$FRAME")

        echo
        echo "============================================================"
        echo "Visualizing baseline vs EgoHumans GT"
        echo "Model: $MODEL_NAME"
        echo "Frame: $FRAME"
        echo "Mapping: $MAPPING"
        echo "Prediction dir: $PRED_DIR"
        echo "GT root: $GT_ROOT"
        echo "Output HTML: $OUT_HTML"
        echo "============================================================"

        if [ ! -d "$PRED_DIR" ]; then
            echo "[Warning] Prediction directory not found:"
            echo "$PRED_DIR"
            echo "Skipping $MODEL_NAME frame $FRAME."
            continue
        fi

        if [ ! -d "$GT_ROOT/$FRAME" ]; then
            echo "[Warning] GT frame directory not found:"
            echo "$GT_ROOT/$FRAME"
            echo "Skipping $MODEL_NAME frame $FRAME."
            continue
        fi

        python "$VIS_SCRIPT" \
          --model "$MODEL_ARG" \
          --frame "$FRAME" \
          --pred_dir "$PRED_DIR" \
          --gt_root "$GT_ROOT" \
          --mapping "$MAPPING" \
          --save_html "$OUT_HTML" \
          --show_camera \
          --camera_scale 0.25 \
          --scene_padding 1.30 \
          --camera_eye_scale 1.0 \
          --auto_rotate \
          --rotation_degrees 360 \
          --hide_axis \
          --hide_legend
        echo
        echo "Saved:"
        echo "$OUT_HTML"
    done
done

echo
echo "============================================================"
echo "Done. Generated visualizations:"
echo "============================================================"
ls -lh "$VIS_DIR"/*_baseline_vs_GT_clean.html 2>/dev/null || true