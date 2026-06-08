#!/bin/bash
set -e

cd "$(dirname "$0")" || exit 1

# Usage:
#   ./run_ablation_three_pipelines_fixed_mapping.sh 4dhumans
#   ./run_ablation_three_pipelines_fixed_mapping.sh tokenhmr
#
# Pipelines:
#   (1) translation_only:
#       Stage-1 translation-only optimization.
#
#   (2) two_stage:
#       Stage-1 translation-only optimization,
#       then Stage-2 full-parameter refinement with fixed T_opt.
#
#   (3) one_stage_all:
#       optimize T + global_orient + body_pose + betas together.
#
# Frames:
#   00001 and 00025
#
# Frame-specific mapping:
#   00001: 0:aria03,1:aria02,2:aria01,3:aria04
#   00025: 0:aria03,1:aria02,2:aria04,3:aria01
#
# Occlusion comparison:
#   aria01: non-occluded in 00001, occluded in 00025
#   aria04: occluded in 00001, non-occluded in 00025

MODEL=${1:-4dhumans}

TARGET_DIR=./head_targets_unscaled
KEYPOINTS_DIR=./keypoints2d
GT_ROOT=./data/mesh_cam_unscaled/cam01/rgb
SMPL=./smpl
EXO_CAM=cam01

# Compatible with your previous run_smplify_v1_frame.sh.
# Your previous Stage-1 used 2800 6260.
HEAD_PROXY_VERTICES="2800 6260"

FRAMES=("00019" "00020" "00021" "00022" "00023" "00024" "00025")

# Set to 0 if you only want to re-run evaluation using existing OBJ files.
RUN_OPTIMIZATION=${RUN_OPTIMIZATION:-1}

mkdir -p metrics_ablation

# ------------------------------------------------------------
# Frame-specific mapping
# ------------------------------------------------------------
get_mapping_for_frame() {
    local FRAME="$1"

    case "$FRAME" in
      00019)
        echo "0:aria03,1:aria04,2:aria02,3:aria01"
        ;;
      00020)
        echo "0:aria03,1:aria02,2:aria04,3:aria01"
        ;;
      00021)
        echo "0:aria03,1:aria02,2:aria04,3:aria01"
        ;;
      00022)
        echo "0:aria02,1:aria03,2:aria04,3:aria01"
        ;;
      00023)
        echo "0:aria02,1:aria04,2:aria03,3:aria01"
        ;;
      00024)
        echo "0:aria04,1:aria02,2:aria03,3:aria01"
        ;;
      00025)
        echo "0:aria03,1:aria02,2:aria04,3:aria01"
        ;;
      *)
        echo "[ERROR] No mapping defined for frame $FRAME" >&2
        exit 1
        ;;
    esac
}

# ------------------------------------------------------------
# Model-specific paths
# ------------------------------------------------------------
case "$MODEL" in
  tokenhmr|TokenHMR|tkhmr)
    MODEL_NAME="tokenhmr"
    MODEL_LABEL="TokenHMR"

    PRED_DIR=./TokenHMR/demo_out/my_image_with_smpl_params
    NPZ_SUFFIX="_tkhmr.npz"

    TRANSLATION_DIR=./TokenHMR/demo_out/ablation_00019_00025_translation_only_fixed
    TWO_STAGE_DIR=./TokenHMR/demo_out/ablation_00019_00025_two_stage_fixed
    ONE_STAGE_DIR=./TokenHMR/demo_out/ablation_00019_00025_one_stage_all_fixed
    ;;

  4dhumans|4DHumans|4dh|4D)
    MODEL_NAME="4dhumans"
    MODEL_LABEL="4DHumans"

    PRED_DIR=./4D-Humans/demo_out/my_image_with_smpl_params
    NPZ_SUFFIX="_4dhumans.npz"

    TRANSLATION_DIR=./4D-Humans/demo_out/ablation_00019_00025_translation_only_fixed
    TWO_STAGE_DIR=./4D-Humans/demo_out/ablation_00019_00025_two_stage_fixed
    ONE_STAGE_DIR=./4D-Humans/demo_out/ablation_00019_00025_one_stage_all_fixed
    ;;

  *)
    echo "[ERROR] Unknown model: $MODEL"
    exit 1
    ;;
esac

mkdir -p "$TRANSLATION_DIR" "$TWO_STAGE_DIR" "$ONE_STAGE_DIR"

echo "============================================================"
echo "Three-pipeline ablation with frame-specific fixed evaluation"
echo "Model: $MODEL_LABEL"
echo "Frames: ${FRAMES[*]}"
echo "Run optimization: $RUN_OPTIMIZATION"
echo "Prediction dir: $PRED_DIR"
echo "Translation-only dir: $TRANSLATION_DIR"
echo "Two-stage dir: $TWO_STAGE_DIR"
echo "One-stage-all dir: $ONE_STAGE_DIR"
echo "Head proxy vertices: $HEAD_PROXY_VERTICES"
echo "============================================================"

# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------
for DIR in "$PRED_DIR" "$TARGET_DIR" "$KEYPOINTS_DIR" "$GT_ROOT" "$SMPL"; do
    if [ ! -d "$DIR" ]; then
        echo "[ERROR] Missing directory: $DIR"
        exit 1
    fi
done

if ! python src/evaluate_optimized_vs_gt.py --help | grep -q -- "--mapping"; then
    echo "[ERROR] src/evaluate_optimized_vs_gt.py does not support --mapping yet."
    echo "Please modify evaluate_optimized_vs_gt.py first."
    exit 1
fi

# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------
if [ "$RUN_OPTIMIZATION" = "1" ]; then
    for FRAME in "${FRAMES[@]}"; do
        MAPPING=$(get_mapping_for_frame "$FRAME")

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
        echo "Frame: $FRAME"
        echo "Optimization mapping: $MAPPING"
        echo "============================================================"

        for DET_ID in 0 1 2 3; do
            ARIA=${MAP[$DET_ID]}

            PRED_NPZ="$PRED_DIR/${FRAME}_${DET_ID}${NPZ_SUFFIX}"
            PRED_OBJ="$PRED_DIR/${FRAME}_${DET_ID}.obj"
            TARGET="$TARGET_DIR/${FRAME}_${ARIA}_${EXO_CAM}_egohumans_style.txt"
            KEYPOINTS_JSON="$KEYPOINTS_DIR/${FRAME}_${ARIA}_${EXO_CAM}.json"

            TRANSLATION_OBJ="$TRANSLATION_DIR/${FRAME}_${DET_ID}.obj"
            TRANSLATION_NPZ="$TRANSLATION_DIR/${FRAME}_${DET_ID}.npz"

            TWO_STAGE_OBJ="$TWO_STAGE_DIR/${FRAME}_${DET_ID}.obj"
            ONE_STAGE_OBJ="$ONE_STAGE_DIR/${FRAME}_${DET_ID}.obj"

            echo
            echo "------------------------------------------------------------"
            echo "Frame: $FRAME | det $DET_ID -> $ARIA"
            echo "Pred NPZ:       $PRED_NPZ"
            echo "Pred OBJ:       $PRED_OBJ"
            echo "Target:         $TARGET"
            echo "2D keypoints:   $KEYPOINTS_JSON"
            echo "Translation:    $TRANSLATION_OBJ"
            echo "Two-stage:      $TWO_STAGE_OBJ"
            echo "One-stage-all:  $ONE_STAGE_OBJ"
            echo "------------------------------------------------------------"

            if [ ! -f "$PRED_NPZ" ]; then
                echo "[Warning] Missing prediction NPZ: $PRED_NPZ"
                echo "Skipping."
                continue
            fi

            if [ ! -f "$PRED_OBJ" ]; then
                echo "[Warning] Missing prediction OBJ: $PRED_OBJ"
                echo "Skipping."
                continue
            fi

            if [ ! -f "$TARGET" ]; then
                echo "[Warning] Missing head target file: $TARGET"
                echo "Skipping."
                continue
            fi

            if [ ! -f "$KEYPOINTS_JSON" ]; then
                echo "[Warning] Missing 2D keypoints JSON: $KEYPOINTS_JSON"
                echo "Skipping."
                continue
            fi

            # ========================================================
            # (1) Translation-only
            # Exactly matches your old run_smplify_v1_frame.sh settings.
            # ========================================================
            echo
            echo "============================================================"
            echo "(1) TRANSLATION-ONLY"
            echo "============================================================"

            python src/smplify_v1_translation.py \
              --pred_npz "$PRED_NPZ" \
              --pred_obj_template "$PRED_OBJ" \
              --head_target_file "$TARGET" \
              --head_target_coord camera \
              --target_vertex_ids $HEAD_PROXY_VERTICES \
              --smpl_model_dir "$SMPL" \
              --out_obj "$TRANSLATION_OBJ" \
              --optimizer adam \
              --iters 300 \
              --lr 0.01 \
              --w_head 5.0 \
              --w_2d 0.0 \
              --w_trans_prior 0.0 \
              --proxy_radius 0.0 \
              --max_shift 0.0 \
              --w_trust 0.0 \
              --w_z_prior 0.0 \
              --w_z_positive 5.0

            if [ ! -f "$TRANSLATION_NPZ" ]; then
                echo "[ERROR] Translation-only NPZ not produced:"
                echo "$TRANSLATION_NPZ"
                exit 1
            fi

            # ========================================================
            # (2) Two-stage: fixed-T Stage 2
            # Matches your old run_smplify_v2_frame.sh settings,
            # but uses the corrected --freeze_betas flag.
            # ========================================================
            echo
            echo "============================================================"
            echo "(2) TWO-STAGE: FIXED-T STAGE-2 REFINEMENT"
            echo "============================================================"

            python src/smplify_v2_all_param.py \
              --stage1_npz "$TRANSLATION_NPZ" \
              --keypoints_2d_json "$KEYPOINTS_JSON" \
              --head_target_file "$TARGET" \
              --head_target_coord camera \
              --smpl_model_dir "$SMPL" \
              --out_obj "$TWO_STAGE_OBJ" \
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

            # ========================================================
            # (3) One-stage all-together
            # ========================================================
            echo
            echo "============================================================"
            echo "(3) ONE-STAGE ALL-TOGETHER"
            echo "============================================================"

            python src/smplify_one_stage_all_param.py \
              --pred_npz "$PRED_NPZ" \
              --keypoints_2d_json "$KEYPOINTS_JSON" \
              --head_target_file "$TARGET" \
              --head_target_coord camera \
              --head_proxy_vertex_ids $HEAD_PROXY_VERTICES \
              --smpl_model_dir "$SMPL" \
              --out_obj "$ONE_STAGE_OBJ" \
              --optimizer lbfgs \
              --lr 0.05 \
              --lbfgs_max_iter 10000 \
              --w_2d 10.0 \
              --w_face_2d 10.0 \
              --w_head 5.0 \
              --proxy_radius 0.0 \
              --w_trans_prior 0.0 \
              --w_trust 0.0 \
              --max_shift 0.0 \
              --w_z_prior 0.0 \
              --w_orient_prior 0.5 \
              --w_pose_prior 0.5 \
              --w_betas_prior 1.0 \
              --w_betas_l2 1.0 \
              --w_z_positive 1.0 \
              --freeze_betas
        done
    done
else
    echo
    echo "[Info] RUN_OPTIMIZATION=0, skipping optimization and running evaluation only."
fi

# ------------------------------------------------------------
# Evaluation helper with frame-specific fixed mapping
# ------------------------------------------------------------
evaluate_pipeline() {
    local PIPELINE_NAME="$1"
    local PRED_DIR_EVAL="$2"

    echo
    echo "============================================================"
    echo "Evaluating pipeline: $PIPELINE_NAME"
    echo "Prediction dir: $PRED_DIR_EVAL"
    echo "============================================================"

    FRAME_CSV_LIST=""

    for FRAME in "${FRAMES[@]}"; do
        MAPPING=$(get_mapping_for_frame "$FRAME")

        OUT_CSV_FRAME="metrics_ablation/${MODEL_NAME}_${FRAME}_${PIPELINE_NAME}_vs_gt.csv"
        OUT_SUMMARY_FRAME="metrics_ablation/${MODEL_NAME}_${FRAME}_${PIPELINE_NAME}_vs_gt_summary.csv"

        echo
        echo "------------------------------------------------------------"
        echo "Evaluating frame: $FRAME"
        echo "Fixed evaluation mapping: $MAPPING"
        echo "Output: $OUT_CSV_FRAME"
        echo "------------------------------------------------------------"

        python src/evaluate_optimized_vs_gt.py \
          --frame "$FRAME" \
          --model "${MODEL_NAME}_${PIPELINE_NAME}" \
          --method v2 \
          --pred_dir "$PRED_DIR_EVAL" \
          --gt_root "$GT_ROOT" \
          --smpl_model_dir "$SMPL" \
          --mapping "$MAPPING" \
          --out_csv "$OUT_CSV_FRAME" \
          --summary_csv "$OUT_SUMMARY_FRAME"

        FRAME_CSV_LIST="${FRAME_CSV_LIST} ${OUT_CSV_FRAME}"
    done

    MERGED_CSV="metrics_ablation/${MODEL_NAME}_00019_00025_${PIPELINE_NAME}_vs_gt.csv"

    python - <<PY
import pandas as pd
from pathlib import Path

csvs = """$FRAME_CSV_LIST""".strip().split()
dfs = [pd.read_csv(p) for p in csvs]
df = pd.concat(dfs, ignore_index=True)

out = Path("$MERGED_CSV")
df.to_csv(out, index=False)
print("[Saved merged pipeline CSV]", out)
PY
}

evaluate_pipeline "translation_only" "$TRANSLATION_DIR"
evaluate_pipeline "two_stage" "$TWO_STAGE_DIR"
evaluate_pipeline "one_stage_all" "$ONE_STAGE_DIR"

# ------------------------------------------------------------
# Merge all pipelines and add visibility labels
# ------------------------------------------------------------
python - <<PY
import pandas as pd
from pathlib import Path

model = "${MODEL_NAME}"

paths = {
    "translation_only": Path(f"metrics_ablation/{model}_00019_00025_translation_only_vs_gt.csv"),
    "two_stage": Path(f"metrics_ablation/{model}_00019_00025_two_stage_vs_gt.csv"),
    "one_stage_all": Path(f"metrics_ablation/{model}_00019_00025_one_stage_all_vs_gt.csv"),
}

dfs = []

for pipeline, path in paths.items():
    if not path.exists():
        print("[Warning] missing:", path)
        continue

    df = pd.read_csv(path)
    df["pipeline"] = pipeline
    dfs.append(df)

if len(dfs) == 0:
    raise RuntimeError("No CSV files found to merge.")

df = pd.concat(dfs, ignore_index=True)

df["frame_str"] = df["frame"].astype(str).str.zfill(5)

def visibility_label(row):
    frame = row["frame_str"]
    aria = row["aria"]

    if frame == "00001" and aria == "aria01":
        return "non_occluded"
    if frame == "00025" and aria == "aria01":
        return "occluded"

    if frame == "00001" and aria == "aria04":
        return "occluded"
    if frame == "00025" and aria == "aria04":
        return "non_occluded"

    return "other"

df["visibility_case"] = df.apply(visibility_label, axis=1)

main_out = Path(f"metrics_ablation/{model}_00019_00025_three_pipeline_ablation_fixed.csv")
df.to_csv(main_out, index=False)

focus = df[df["visibility_case"].isin(["occluded", "non_occluded"])].copy()
focus_out = Path(f"metrics_ablation/{model}_00019_00025_three_pipeline_ablation_focus_aria01_aria04.csv")
focus.to_csv(focus_out, index=False)

metrics = [
    "raw_mpjpe_mm",
    "root_aligned_mpjpe_mm",
    "pa_mpjpe_mm",
    "mean_vertex_error_mm",
    "global_root_drift_m",
]

summary_all = (
    df.groupby(["visibility_case", "pipeline"])[metrics]
      .agg(["mean", "std", "median", "min", "max", "count"])
)

summary_focus = (
    focus.groupby(["visibility_case", "pipeline"])[metrics]
         .agg(["mean", "std", "median", "min", "max", "count"])
)

summary_all_out = Path(f"metrics_ablation/{model}_00019_00025_three_pipeline_summary_all.csv")
summary_focus_out = Path(f"metrics_ablation/{model}_00019_00025_three_pipeline_summary_focus.csv")

summary_all.to_csv(summary_all_out)
summary_focus.to_csv(summary_focus_out)

print()
print("Saved main CSV:")
print(main_out)

print()
print("Saved focus CSV:")
print(focus_out)

print()
print("Saved all-case summary:")
print(summary_all_out)
print(summary_all)

print()
print("Saved focus summary:")
print(summary_focus_out)
print(summary_focus)
PY

echo
echo "============================================================"
echo "Done."
echo "Main CSV:"
echo "  metrics_ablation/${MODEL_NAME}_00019_00025_three_pipeline_ablation_fixed.csv"
echo "Focus CSV:"
echo "  metrics_ablation/${MODEL_NAME}_00019_00025_three_pipeline_ablation_focus_aria01_aria04.csv"
echo "Focus summary:"
echo "  metrics_ablation/${MODEL_NAME}_00019_00025_three_pipeline_summary_focus.csv"
echo "============================================================"