#!/bin/bash
set -e

# Always run relative to the Project folder where this script lives
cd "$(dirname "$0")" || exit 1

# Usage:
#   ./run_smplify_v1_frame.sh
#
# This runs Stage-1 translation-only optimization for:
#   - frames 00001 to 00006
#   - both tokenhmr and 4dhumans
#
# Output:
#   TokenHMR/demo_out/my_image_smplify_v1/
#   4D-Humans/demo_out/my_image_smplify_v1/

TARGET_DIR=./head_targets_unscaled
SMPL=./smpl
EXO_CAM=cam01

# ------------------------------------------------------------
# Fixed optimizer: Adam only
# ------------------------------------------------------------
OPTIMIZER_NAME="adam"
OPT_ARGS="--optimizer adam --iters 300 --lr 0.01"

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

    case "$MODEL" in
      tokenhmr)
        MODEL_NAME="TokenHMR"
        PRED_DIR=./TokenHMR/demo_out/my_image_with_smpl_params
        OUT_DIR=./TokenHMR/demo_out/my_image_smplify_v1
        NPZ_SUFFIX="_tkhmr.npz"
        ;;

      4dhumans)
        MODEL_NAME="4DHumans"
        PRED_DIR=./4D-Humans/demo_out/my_image_with_smpl_params
        OUT_DIR=./4D-Humans/demo_out/my_image_smplify_v1
        NPZ_SUFFIX="_4dhumans.npz"
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
if [ ! -d "$TARGET_DIR" ]; then
    echo "[ERROR] Target directory not found:"
    echo "$TARGET_DIR"
    echo
    echo "Generate unscaled head targets first, for example:"
    echo "  ./extract_all_aria_targets.sh 00001 $EXO_CAM"
    exit 1
fi

if [ ! -d "$SMPL" ]; then
    echo "[ERROR] SMPL directory not found:"
    echo "$SMPL"
    exit 1
fi

# ------------------------------------------------------------
# Main loop: both models, frames 00001–00006
# ------------------------------------------------------------
for MODEL in tokenhmr 4dhumans; do
    get_model_config "$MODEL"

    mkdir -p "$OUT_DIR"

    if [ ! -d "$PRED_DIR" ]; then
        echo "[ERROR] Prediction directory not found for $MODEL_NAME:"
        echo "$PRED_DIR"
        echo
        echo "Make sure you generated my_image_with_smpl_params first."
        exit 1
    fi

    for FRAME in 00001 00002 00003 00004 00005 00006; do
        MAPPING=$(get_mapping_for_frame "$FRAME")

        # Parse mapping into associative array
        declare -A MAP
        MAP=()

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

        echo
        echo "============================================================"
        echo "Running SMPLify Stage 1: translation-only"
        echo "Model: $MODEL_NAME"
        echo "Frame: $FRAME"
        echo "Optimizer: Adam"
        echo "Prediction dir: $PRED_DIR"
        echo "Output dir: $OUT_DIR"
        echo "Target dir: $TARGET_DIR"
        echo "Exo cam: $EXO_CAM"
        echo "SMPL dir: $SMPL"
        echo "Mapping string: $MAPPING"
        echo "Parsed mapping:"
        for DET_ID in 0 1 2 3; do
            echo "  det $DET_ID -> ${MAP[$DET_ID]}"
        done
        echo "============================================================"

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
            OUT_OBJ="$OUT_DIR/${FRAME}_${DET_ID}.obj"

            echo
            echo "------------------------------------------------------------"
            echo "Stage 1 translation-only optimization"
            echo "Model: $MODEL_NAME"
            echo "Frame: $FRAME"
            echo "Detection id: $DET_ID"
            echo "Matched Aria: $ARIA"
            echo "Pred NPZ: $PRED_NPZ"
            echo "Pred OBJ: $PRED_OBJ"
            echo "Head target: $TARGET"
            echo "Out OBJ: $OUT_OBJ"
            echo "------------------------------------------------------------"

            if [ ! -f "$PRED_NPZ" ]; then
                echo "[Warning] Missing prediction NPZ: $PRED_NPZ"
                echo "Skipping this detection."
                continue
            fi

            if [ ! -f "$PRED_OBJ" ]; then
                echo "[Warning] Missing prediction OBJ: $PRED_OBJ"
                echo "Skipping this detection."
                continue
            fi

            if [ ! -f "$TARGET" ]; then
                echo "[Warning] Missing head target file: $TARGET"
                echo "Skipping this detection."
                continue
            fi

            python src/smplify_v1_translation.py \
              --pred_npz "$PRED_NPZ" \
              --pred_obj_template "$PRED_OBJ" \
              --head_target_file "$TARGET" \
              --head_target_coord camera \
              --target_vertex_ids 2787 3639 6248 2817 \
              --smpl_model_dir "$SMPL" \
              --out_obj "$OUT_OBJ" \
              $OPT_ARGS \
              --w_head 5.0 \
              --w_2d 0.0 \
              --w_trans_prior 0.0 \
              --proxy_radius 0.0 \
              --max_shift 0.0 \
              --w_trust 0.0 \
              --w_z_prior 0.0 \
              --w_z_positive 5.0
        done
    done
done

echo
echo "============================================================"
echo "Done. Stage-1 optimized meshes saved to:"
echo "  ./TokenHMR/demo_out/my_image_smplify_v1"
echo "  ./4D-Humans/demo_out/my_image_smplify_v1"
echo "============================================================"