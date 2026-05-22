#!/bin/bash
set -e

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
#   models tokenhmr and 4dhumans
#
# Outputs:
#   visualizations/00001_TokenHMR_baseline_vs_v1_vs_GT.html
#   visualizations/00001_4DHumans_baseline_vs_v1_vs_GT.html
#   ...

PROJECT_ROOT=~/DigitalHumans/Project

GT_ROOT="$PROJECT_ROOT/data/mesh_cam_unscaled/cam01/rgb"
TARGET_DIR="$PROJECT_ROOT/head_targets_unscaled"

VIS_SCRIPT="$PROJECT_ROOT/src/visualize_frame_before_after_all_people.py"
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

        # Baseline prediction
        # Use my_image if your baseline OBJ files are there.
        # If you want the NPZ/SMPL-param version, change this to my_image_with_smpl_params.
        BEFORE_DIR="$PROJECT_ROOT/TokenHMR/demo_out/my_image"

        # SMPLify-v1 optimized prediction
        AFTER_DIR="$PROJECT_ROOT/TokenHMR/demo_out/my_image_smplify_v1"

        OUT_HTML="$VIS_DIR/${FRAME}_TokenHMR_baseline_vs_v1_vs_GT.html"
        ;;

      4dhumans)
        MODEL_NAME="4DHumans"

        # Baseline prediction
        BEFORE_DIR="$PROJECT_ROOT/4D-Humans/demo_out/my_image"

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
for MODEL in tokenhmr 4dhumans; do
    for FRAME in 00001 00002 00003 00004 00005 00006; do
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

        python "$VIS_SCRIPT" \
          --frame "$FRAME" \
          --before_dir "$BEFORE_DIR" \
          --after_dir "$AFTER_DIR" \
          --gt_root "$GT_ROOT" \
          --target_dir "$TARGET_DIR" \
          --mapping "$MAPPING" \
          --draw_lines \
          --save_html "$OUT_HTML"

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