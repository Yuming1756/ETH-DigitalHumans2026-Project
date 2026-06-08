#!/bin/bash
set -euo pipefail

# Always run from Project folder where this script lives
cd "$(dirname "$0")" || exit 1

# Usage:
#   ./evaluate_v1_v2_frames_00001_00006_fixed_mapping.sh
#
# This automatically evaluates:
#   - TokenHMR v1
#   - TokenHMR v2
#   - 4D-Humans v1
#   - 4D-Humans v2
#
# Frames:
#   00001 to 00006
#
# Evaluation:
#   Uses fixed frame-specific detection-to-Aria mapping.
#   Does NOT use auto-matching.

SMPL=./smpl
GT_ROOT=./data/mesh_cam_unscaled/cam01/rgb
OUT_DIR=metrics_00001_00006

FRAMES=("00001" "00002" "00003" "00004" "00005" "00006")
MODELS=("tokenhmr" "4dhumans")
METHODS=("v1" "v2")

mkdir -p "$OUT_DIR"

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
# Model/method-specific prediction directory
# ------------------------------------------------------------
get_pred_dir() {
    local MODEL="$1"
    local METHOD="$2"

    case "$MODEL" in
      tokenhmr)
        case "$METHOD" in
          v1)
            echo "TokenHMR/demo_out/my_image_smplify_v1"
            ;;
          v2)
            echo "TokenHMR/demo_out/my_image_smplify_v2"
            ;;
          *)
            echo "[ERROR] Unknown method for TokenHMR: $METHOD" >&2
            exit 1
            ;;
        esac
        ;;

      4dhumans)
        case "$METHOD" in
          v1)
            echo "4D-Humans/demo_out/my_image_smplify_v1"
            ;;
          v2)
            echo "4D-Humans/demo_out/my_image_smplify_v2"
            ;;
          *)
            echo "[ERROR] Unknown method for 4D-Humans: $METHOD" >&2
            exit 1
            ;;
        esac
        ;;

      *)
        echo "[ERROR] Unknown model: $MODEL" >&2
        exit 1
        ;;
    esac
}

get_model_label() {
    local MODEL="$1"

    case "$MODEL" in
      tokenhmr)
        echo "TokenHMR"
        ;;
      4dhumans)
        echo "4D-Humans"
        ;;
      *)
        echo "$MODEL"
        ;;
    esac
}

# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------
if [ ! -d "$GT_ROOT" ]; then
    echo "[ERROR] GT root not found:"
    echo "$GT_ROOT"
    exit 1
fi

if [ ! -d "$SMPL" ]; then
    echo "[ERROR] SMPL directory not found:"
    echo "$SMPL"
    exit 1
fi

if [ ! -f "$SMPL/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl" ]; then
    echo "[ERROR] SMPL model file not found:"
    echo "$SMPL/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    echo
    echo "Place the SMPL neutral model under ./smpl/ or modify SMPL in this script."
    exit 1
fi

if ! python src/evaluate_optimized_vs_gt.py --help | grep -q -- "--mapping"; then
    echo "[ERROR] src/evaluate_optimized_vs_gt.py does not support --mapping."
    echo "Please use the modified evaluator with fixed-mapping support."
    exit 1
fi

echo "============================================================"
echo "Evaluating v1/v2 with fixed mapping"
echo "Models: ${MODELS[*]}"
echo "Methods: ${METHODS[*]}"
echo "Frames: ${FRAMES[*]}"
echo "GT root: $GT_ROOT"
echo "SMPL: $SMPL"
echo "Output dir: $OUT_DIR"
echo "============================================================"

# ------------------------------------------------------------
# Evaluate each model + method + frame
# ------------------------------------------------------------
for MODEL in "${MODELS[@]}"; do
    MODEL_LABEL=$(get_model_label "$MODEL")

    for METHOD in "${METHODS[@]}"; do
        PRED_DIR=$(get_pred_dir "$MODEL" "$METHOD")

        echo
        echo "============================================================"
        echo "Model: $MODEL_LABEL"
        echo "Method: $METHOD"
        echo "Prediction dir: $PRED_DIR"
        echo "============================================================"

        if [ ! -d "$PRED_DIR" ]; then
            echo "[Warning] Prediction directory not found, skipping:"
            echo "$PRED_DIR"
            continue
        fi

        FRAME_CSV_LIST=""

        for FRAME in "${FRAMES[@]}"; do
            MAPPING=$(get_mapping_for_frame "$FRAME")
            GT_FRAME_DIR="$GT_ROOT/$FRAME"

            OUT_CSV_FRAME="$OUT_DIR/${MODEL}_${METHOD}_${FRAME}_fixed_mapping.csv"
            OUT_SUMMARY_FRAME="$OUT_DIR/${MODEL}_${METHOD}_${FRAME}_fixed_mapping_summary.csv"

            echo
            echo "------------------------------------------------------------"
            echo "Evaluating"
            echo "Model: $MODEL_LABEL"
            echo "Method: $METHOD"
            echo "Frame: $FRAME"
            echo "Mapping: $MAPPING"
            echo "Pred dir: $PRED_DIR"
            echo "GT frame dir: $GT_FRAME_DIR"
            echo "Out CSV: $OUT_CSV_FRAME"
            echo "------------------------------------------------------------"

            if [ ! -d "$GT_FRAME_DIR" ]; then
                echo "[Warning] Missing GT frame directory, skipping:"
                echo "$GT_FRAME_DIR"
                continue
            fi

            if ! ls "$PRED_DIR"/${FRAME}_*.obj >/dev/null 2>&1; then
                echo "[Warning] Missing prediction OBJ files for frame $FRAME, skipping:"
                echo "$PRED_DIR/${FRAME}_*.obj"
                continue
            fi

            python src/evaluate_optimized_vs_gt.py \
              --frame "$FRAME" \
              --model "$MODEL" \
              --method "$METHOD" \
              --pred_dir "$PRED_DIR" \
              --gt_root "$GT_ROOT" \
              --smpl_model_dir "$SMPL" \
              --mapping "$MAPPING" \
              --out_csv "$OUT_CSV_FRAME" \
              --summary_csv "$OUT_SUMMARY_FRAME"

            FRAME_CSV_LIST="${FRAME_CSV_LIST} ${OUT_CSV_FRAME}"
        done

        # ------------------------------------------------------------
        # Merge per-frame CSVs for this model/method
        # ------------------------------------------------------------
        MERGED_CSV="$OUT_DIR/${MODEL}_${METHOD}_00001_00006_fixed_mapping.csv"
        SUMMARY_CSV="$OUT_DIR/${MODEL}_${METHOD}_00001_00006_fixed_mapping_summary.csv"

        python - <<PY
import pandas as pd
from pathlib import Path
import numpy as np
import sys

csvs = """$FRAME_CSV_LIST""".strip().split()

if len(csvs) == 0:
    print("[Warning] No frame CSVs to merge for ${MODEL} ${METHOD}")
    sys.exit(0)

dfs = []
for p in csvs:
    path = Path(p)
    if path.exists():
        dfs.append(pd.read_csv(path))
    else:
        print("[Warning] missing CSV:", path)

if len(dfs) == 0:
    print("[Warning] No existing CSVs to merge for ${MODEL} ${METHOD}")
    sys.exit(0)

df = pd.concat(dfs, ignore_index=True)
out = Path("$MERGED_CSV")
df.to_csv(out, index=False)
print("[Saved merged CSV]", out)

metric_keys = [
    "raw_mpjpe_mm",
    "root_aligned_mpjpe_mm",
    "pa_mpjpe_mm",
    "mean_vertex_error_mm",
    "global_root_drift_m",
]

rows = []
for key in metric_keys:
    if key not in df.columns:
        continue
    values = df[key].dropna().to_numpy(dtype=float)
    rows.append({
        "metric": key,
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    })

summary = pd.DataFrame(rows)
summary_out = Path("$SUMMARY_CSV")
summary.to_csv(summary_out, index=False)
print("[Saved merged summary]", summary_out)
print(summary)
PY

    done
done

echo
echo "============================================================"
echo "Done. Generated files:"
echo "============================================================"
ls -lh "$OUT_DIR"/*.csv 2>/dev/null || true