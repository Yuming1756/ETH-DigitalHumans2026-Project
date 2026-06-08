#!/bin/bash
set -e

# Always run relative to the Project folder where this script lives
cd "$(dirname "$0")" || exit 1

# Usage:
#   ./visualize_smplify_v2_frame.sh
#
# This visualizes:
#   baseline model prediction
#   SMPLify-v1 prediction
#   SMPLify-v2 prediction
#   EgoHumans GT
#
# For:
#   frames 00001 to 00006
#   models tokenhmr and 4dhumans
#
# Outputs:
#   visualizations/00001_TokenHMR_baseline_vs_v1_vs_v2_vs_GT.html
#   visualizations/00001_4DHumans_baseline_vs_v1_vs_v2_vs_GT.html
#   ...

PROJECT_ROOT=~/DigitalHumans/Project

GT_ROOT="$PROJECT_ROOT/data/mesh_cam_unscaled/cam01/rgb"
TARGET_DIR="$PROJECT_ROOT/head_targets_unscaled"

VIS_SCRIPT="$PROJECT_ROOT/src/visualize_frame_baseline_v1_v2_all_people.py"
VIS_DIR="$PROJECT_ROOT/visualizations"

mkdir -p "$VIS_DIR"

# ------------------------------------------------------------
# Frame-specific detection-id to EgoHumans identity mapping
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Model-specific directories
# ------------------------------------------------------------
get_model_config() {
    local MODEL="$1"
    local FRAME="$2"

    case "$MODEL" in
      tokenhmr)
        MODEL_NAME="TokenHMR"
        BASELINE_DIR="$PROJECT_ROOT/TokenHMR/demo_out/my_image_with_smpl_params"
        V1_DIR="$PROJECT_ROOT/TokenHMR/demo_out/my_image_smplify_v1"
        V2_DIR="$PROJECT_ROOT/TokenHMR/demo_out/my_image_smplify_v2"
        OUT_HTML="$VIS_DIR/${FRAME}_TokenHMR_baseline_vs_v1_vs_v2_vs_GT.html"
        ;;

      4dhumans)
        MODEL_NAME="4DHumans"
        BASELINE_DIR="$PROJECT_ROOT/4D-Humans/demo_out/my_image_with_smpl_params"
        V1_DIR="$PROJECT_ROOT/4D-Humans/demo_out/my_image_smplify_v1"
        V2_DIR="$PROJECT_ROOT/4D-Humans/demo_out/my_image_smplify_v2"
        OUT_HTML="$VIS_DIR/${FRAME}_4DHumans_baseline_vs_v1_vs_v2_vs_GT.html"
        ;;

      *)
        echo "[ERROR] Unknown model: $MODEL"
        exit 1
        ;;
    esac
}

# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------
if [ ! -f "$VIS_SCRIPT" ]; then
    echo "[ERROR] Visualization Python script not found:"
    echo "$VIS_SCRIPT"
    echo
    echo "Please create:"
    echo "  src/visualize_frame_baseline_v1_v2_all_people.py"
    echo
    echo "Expected arguments:"
    echo "  --frame"
    echo "  --baseline_dir"
    echo "  --v1_dir"
    echo "  --v2_dir"
    echo "  --gt_root"
    echo "  --target_dir"
    echo "  --mapping"
    echo "  --draw_lines"
    echo "  --save_html"
    exit 1
fi

if [ ! -d "$GT_ROOT" ]; then
    echo "[ERROR] GT root not found:"
    echo "$GT_ROOT"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "[WARNING] Target directory not found:"
    echo "$TARGET_DIR"
    echo "Continuing anyway, but target markers may be missing."
fi

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
for MODEL in tokenhmr 4dhumans; do
    for FRAME in 00001 00002 00003 00004 00005 00006; do
        get_model_config "$MODEL" "$FRAME"
        MAPPING=$(get_mapping_for_frame "$FRAME")

        echo
        echo "============================================================"
        echo "Visualizing baseline vs SMPLify-v1 vs SMPLify-v2 vs GT"
        echo "Model: $MODEL_NAME"
        echo "Frame: $FRAME"
        echo "Mapping: $MAPPING"
        echo "Baseline dir: $BASELINE_DIR"
        echo "V1 dir: $V1_DIR"
        echo "V2 dir: $V2_DIR"
        echo "GT root: $GT_ROOT"
        echo "Target dir: $TARGET_DIR"
        echo "Output HTML: $OUT_HTML"
        echo "============================================================"

        if [ ! -d "$BASELINE_DIR" ]; then
            echo "[WARNING] Baseline directory not found:"
            echo "$BASELINE_DIR"
            echo "Skipping $MODEL_NAME frame $FRAME."
            continue
        fi

        if [ ! -d "$V1_DIR" ]; then
            echo "[WARNING] V1 directory not found:"
            echo "$V1_DIR"
            echo "Skipping $MODEL_NAME frame $FRAME."
            continue
        fi

        if [ ! -d "$V2_DIR" ]; then
            echo "[WARNING] V2 directory not found:"
            echo "$V2_DIR"
            echo "Skipping $MODEL_NAME frame $FRAME."
            continue
        fi

        if [ ! -d "$GT_ROOT/$FRAME" ]; then
            echo "[WARNING] GT frame directory not found:"
            echo "$GT_ROOT/$FRAME"
            echo "Skipping $MODEL_NAME frame $FRAME."
            continue
        fi

        python "$VIS_SCRIPT" \
          --frame "$FRAME" \
          --baseline_dir "$BASELINE_DIR" \
          --v1_dir "$V1_DIR" \
          --v2_dir "$V2_DIR" \
          --gt_root "$GT_ROOT" \
          --target_dir "$TARGET_DIR" \
          --mapping "$MAPPING" \
          --draw_lines \
          --save_html "$OUT_HTML" \
          --show_camera \
          --camera_scale 0.25 \
          --scene_padding 1.30 \
          --camera_eye_scale 1.0 \
          --auto_rotate \
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
ls -lh "$VIS_DIR"/*_baseline_vs_v1_vs_v2_vs_GT.html 2>/dev/null || true