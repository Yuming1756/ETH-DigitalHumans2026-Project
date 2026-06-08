#!/bin/bash
set -euo pipefail

# Always run relative to the Project folder where this script lives
cd "$(dirname "$0")" || exit 1

# Usage:
#   ./visualize_smplify_frame.sh
#
# This visualizes:
#   baseline model prediction
#   SMPLify-v1 optimized prediction
#   EgoHumans GT
#
# For:
#   frames 00001 to 00006
#   models TokenHMR and 4D-Humans
#
# Outputs:
#   visualizations/00001_TokenHMR_baseline_vs_v1_vs_GT.html
#   visualizations/00001_4DHumans_baseline_vs_v1_vs_GT.html
#   ...

PROJECT_ROOT="$HOME/DigitalHumans/Project"

GT_ROOT="$PROJECT_ROOT/data/mesh_cam_unscaled/cam01/rgb"
TARGET_DIR="$PROJECT_ROOT/head_targets_unscaled"

VIS_SCRIPT="$PROJECT_ROOT/src/visualize_frame_before_after_all_people.py"
VIS_DIR="$PROJECT_ROOT/visualizations"

FRAMES=("00001" "00002" "00003" "00004" "00005" "00006")
MODELS=("tokenhmr" "4dhumans")

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
      tokenhmr|TokenHMR|tkhmr)
        MODEL_NAME="TokenHMR"

        # Baseline prediction with saved SMPL parameters
        BEFORE_DIR="$PROJECT_ROOT/TokenHMR/demo_out/my_image_with_smpl_params"

        # SMPLify-v1 optimized prediction
        AFTER_DIR="$PROJECT_ROOT/TokenHMR/demo_out/my_image_smplify_v1"

        OUT_HTML="$VIS_DIR/${FRAME}_TokenHMR_baseline_vs_v1_vs_GT.html"
        ;;

      4dhumans|4DHumans|4dh|4D)
        MODEL_NAME="4DHumans"

        # Baseline prediction with saved SMPL parameters
        BEFORE_DIR="$PROJECT_ROOT/4D-Humans/demo_out/my_image_with_smpl_params"

        # SMPLify-v1 optimized prediction
        AFTER_DIR="$PROJECT_ROOT/4D-Humans/demo_out/my_image_smplify_v1"

        OUT_HTML="$VIS_DIR/${FRAME}_4DHumans_baseline_vs_v1_vs_GT.html"
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
    echo "You need visualize_frame_before_after_all_people.py first."
    exit 1
fi

if [ ! -d "$GT_ROOT" ]; then
    echo "[ERROR] GT root not found:"
    echo "$GT_ROOT"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "[Warning] Target directory not found:"
    echo "$TARGET_DIR"
    echo "Continuing anyway, but target markers may be missing."
fi

# ------------------------------------------------------------
# Main loop: both models, frames 00001–00006
# ------------------------------------------------------------
for MODEL in "${MODELS[@]}"; do
    for FRAME in "${FRAMES[@]}"; do
        get_model_config "$MODEL" "$FRAME"
        MAPPING=$(get_mapping_for_frame "$FRAME")

        echo
        echo "============================================================"
        echo "Visualizing baseline vs SMPLify-v1 vs EgoHumans GT"
        echo "Model: $MODEL_NAME"
        echo "Frame: $FRAME"
        echo "Mapping: $MAPPING"
        echo "Baseline dir: $BEFORE_DIR"
        echo "SMPLify-v1 dir: $AFTER_DIR"
        echo "GT root: $GT_ROOT"
        echo "Target dir: $TARGET_DIR"
        echo "Output HTML: $OUT_HTML"
        echo "============================================================"

        if [ ! -d "$BEFORE_DIR" ]; then
            echo "[Warning] Baseline directory not found:"
            echo "$BEFORE_DIR"
            echo "Skipping $MODEL_NAME frame $FRAME."
            continue
        fi

        if [ ! -d "$AFTER_DIR" ]; then
            echo "[Warning] SMPLify-v1 directory not found:"
            echo "$AFTER_DIR"
            echo "Skipping $MODEL_NAME frame $FRAME."
            continue
        fi

        if [ ! -d "$GT_ROOT/$FRAME" ]; then
            echo "[Warning] GT frame directory not found:"
            echo "$GT_ROOT/$FRAME"
            echo "Skipping $MODEL_NAME frame $FRAME."
            continue
        fi

        # Check at least one baseline OBJ and one v1 OBJ exist for this frame.
        if ! ls "$BEFORE_DIR"/${FRAME}_*.obj >/dev/null 2>&1; then
            echo "[Warning] No baseline OBJ files found for frame $FRAME:"
            echo "$BEFORE_DIR/${FRAME}_*.obj"
            echo "Skipping $MODEL_NAME frame $FRAME."
            continue
        fi

        if ! ls "$AFTER_DIR"/${FRAME}_*.obj >/dev/null 2>&1; then
            echo "[Warning] No SMPLify-v1 OBJ files found for frame $FRAME:"
            echo "$AFTER_DIR/${FRAME}_*.obj"
            echo "Skipping $MODEL_NAME frame $FRAME."
            continue
        fi

        python "$VIS_SCRIPT" \
          --frame "$FRAME" \
          --before_dir "$BEFORE_DIR" \
          --after_dir "$AFTER_DIR" \
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
ls -lh "$VIS_DIR"/*_baseline_vs_v1_vs_GT.html 2>/dev/null || true