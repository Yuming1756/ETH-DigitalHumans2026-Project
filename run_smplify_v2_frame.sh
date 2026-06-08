#!/bin/bash
set -euo pipefail

# Always run relative to the Project folder where this script lives
cd "$(dirname "$0")" || exit 1

# Usage:
#   ./run_smplify_v2_all_frames.sh
#
# This runs SMPLify-v2 for:
#   - TokenHMR and 4D-Humans
#   - frames 00001 to 00006
#   - fixed frame-specific detection-to-Aria mappings
#
# Required before running:
#   1. Stage-1 results must already exist:
#        TokenHMR/demo_out/my_image_smplify_v1/
#        4D-Humans/demo_out/my_image_smplify_v1/
#   2. 2D keypoint JSONs must exist:
#        keypoints2d/${FRAME}_${ARIA}_cam01.json
#   3. Head target files must exist:
#        head_targets_unscaled/${FRAME}_${ARIA}_cam01_egohumans_style.txt

EXO_CAM=cam01
SMPL=./smpl
TARGET_DIR=./head_targets_unscaled
KEYPOINTS_DIR=./keypoints2d

FRAMES=("00001" "00002" "00003" "00004" "00005" "00006")
MODELS=("tokenhmr" "4dhumans")

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
        exit 1
        ;;
    esac
}

# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------
if [ ! -d "$SMPL" ]; then
    echo "[ERROR] SMPL directory not found:"
    echo "$SMPL"
    exit 1
fi

if [ ! -d "$KEYPOINTS_DIR" ]; then
    echo "[ERROR] Keypoints directory not found:"
    echo "$KEYPOINTS_DIR"
    echo
    echo "Generate keypoints first, for example:"
    echo "  ./make_all_keypoints2d_json_frame.sh 00001 $EXO_CAM"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "[ERROR] Head target directory not found:"
    echo "$TARGET_DIR"
    echo
    echo "Generate unscaled head targets first, for example:"
    echo "  ./extract_all_aria_targets.sh 00001 $EXO_CAM"
    exit 1
fi

# ------------------------------------------------------------
# Main loop: both models, frames 00001–00006
# ------------------------------------------------------------
for MODEL in "${MODELS[@]}"; do
    get_model_config "$MODEL"

    mkdir -p "$OUT_DIR"

    if [ ! -d "$STAGE1_DIR" ]; then
        echo "[ERROR] Stage-1 directory not found for $MODEL_NAME:"
        echo "$STAGE1_DIR"
        echo
        echo "Run Stage 1 first."
        exit 1
    fi

    for FRAME in "${FRAMES[@]}"; do
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
        echo "Running SMPLify Stage 2"
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
            echo "------------------------------------------------------------"
            echo "Stage 2 full-parameter optimization"
            echo "Model: $MODEL_NAME"
            echo "Frame: $FRAME"
            echo "Detection id: $DET_ID"
            echo "Matched Aria: $ARIA"
            echo "Stage-1 NPZ: $STAGE1_NPZ"
            echo "2D keypoints: $KEYPOINTS_JSON"
            echo "Head target: $TARGET"
            echo "Out OBJ: $OUT_OBJ"
            echo "------------------------------------------------------------"

            if [ ! -f "$STAGE1_NPZ" ]; then
                echo "[Warning] Missing Stage-1 NPZ:"
                echo "$STAGE1_NPZ"
                echo "Skipping this detection."
                continue
            fi

            if [ ! -f "$KEYPOINTS_JSON" ]; then
                echo "[Warning] Missing 2D keypoints JSON:"
                echo "$KEYPOINTS_JSON"
                echo "Skipping this detection."
                continue
            fi

            if [ ! -f "$TARGET" ]; then
                echo "[Warning] Missing head target file:"
                echo "$TARGET"
                echo "Skipping this detection."
                continue
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
              --freeze_betas
        done
    done
done

echo
echo "============================================================"
echo "Done. Stage-2 optimized meshes saved to:"
echo "  ./TokenHMR/demo_out/my_image_smplify_v2"
echo "  ./4D-Humans/demo_out/my_image_smplify_v2"
echo "============================================================"