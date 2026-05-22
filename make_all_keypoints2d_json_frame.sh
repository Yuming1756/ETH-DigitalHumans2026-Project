#!/bin/bash
set -e

cd "$(dirname "$0")" || exit 1

FRAME=${1:-00001}
EXO_CAM=${2:-cam01}

POSES2D="data/poses2d/$EXO_CAM/rgb/$FRAME.npy"
OUT_DIR="keypoints2d"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Generating 2D keypoint JSONs from EgoHumans"
echo "FRAME: $FRAME"
echo "EXO_CAM: $EXO_CAM"
echo "POSES2D: $POSES2D"
echo "OUT_DIR: $OUT_DIR"
echo "============================================================"

if [ ! -f "$POSES2D" ]; then
    echo "[ERROR] poses2d file not found:"
    echo "$POSES2D"
    exit 1
fi

for ARIA in aria01 aria02 aria03 aria04; do
    OUT_JSON="$OUT_DIR/${FRAME}_${ARIA}_${EXO_CAM}.json"

    echo
    echo "------------------------------------------------------------"
    echo "ARIA: $ARIA"
    echo "OUT_JSON: $OUT_JSON"
    echo "------------------------------------------------------------"

    python src/make_keypoints2d_json_from_egohumans.py \
      --poses2d "$POSES2D" \
      --aria "$ARIA" \
      --out_json "$OUT_JSON" \
      --min_conf 0.2
done

echo
echo "Done."
ls -lh "$OUT_DIR"/${FRAME}_aria*_${EXO_CAM}.json