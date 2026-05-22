#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SMPL_JOINT_NAMES = {
    1: "L_hip",
    2: "R_hip",
    4: "L_knee",
    5: "R_knee",
    7: "L_ankle",
    8: "R_ankle",
    16: "L_shoulder",
    17: "R_shoulder",
    18: "L_elbow",
    19: "R_elbow",
    20: "L_wrist",
    21: "R_wrist",
}

# Skeleton edges in SMPL joint IDs.
EDGES = [
    (16, 18),  # left shoulder -> left elbow
    (18, 20),  # left elbow -> left wrist
    (17, 19),  # right shoulder -> right elbow
    (19, 21),  # right elbow -> right wrist
    (16, 17),  # shoulders
    (1, 2),    # hips
    (16, 1),   # left torso
    (17, 2),   # right torso
    (1, 4),    # left hip -> left knee
    (4, 7),    # left knee -> left ankle
    (2, 5),    # right hip -> right knee
    (5, 8),    # right knee -> right ankle
]


def load_keypoints_json(path):
    path = Path(path).expanduser()

    with open(path, "r") as f:
        data = json.load(f)

    joint_ids = np.asarray(data["joint_ids"], dtype=np.int64)
    keypoints = np.asarray(data["keypoints"], dtype=np.float32)

    if keypoints.ndim != 2 or keypoints.shape[1] != 3:
        raise ValueError(f"Expected keypoints shape (N,3), got {keypoints.shape}")

    if len(joint_ids) != len(keypoints):
        raise ValueError("joint_ids length does not match keypoints length")

    return joint_ids, keypoints


def conf_to_color(conf):
    """
    Green for high confidence, yellow/orange for lower confidence.
    OpenCV uses BGR.
    """
    conf = float(np.clip(conf, 0.0, 1.0))

    if conf >= 0.8:
        return (0, 255, 0)       # green
    elif conf >= 0.5:
        return (0, 220, 255)     # yellow
    else:
        return (0, 128, 255)     # orange


def draw_keypoints(img, joint_ids, keypoints, min_conf=0.0, draw_names=True):
    out = img.copy()

    # Build lookup: SMPL joint id -> (u, v, conf)
    kpt_by_joint = {}
    for jid, kpt in zip(joint_ids, keypoints):
        u, v, conf = kpt
        if conf >= min_conf:
            kpt_by_joint[int(jid)] = (float(u), float(v), float(conf))

    # Draw skeleton lines first.
    for a, b in EDGES:
        if a not in kpt_by_joint or b not in kpt_by_joint:
            continue

        ua, va, ca = kpt_by_joint[a]
        ub, vb, cb = kpt_by_joint[b]

        color = (255, 255, 255)
        thickness = 2

        cv2.line(
            out,
            (int(round(ua)), int(round(va))),
            (int(round(ub)), int(round(vb))),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    # Draw points and labels.
    for jid, (u, v, conf) in kpt_by_joint.items():
        color = conf_to_color(conf)
        center = (int(round(u)), int(round(v)))

        cv2.circle(out, center, 8, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, center, 10, (0, 0, 0), 2, lineType=cv2.LINE_AA)

        label = f"{jid}"
        if draw_names:
            name = SMPL_JOINT_NAMES.get(jid, "joint")
            label = f"{jid}:{name}"

        label += f" {conf:.2f}"

        cv2.putText(
            out,
            label,
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            lineType=cv2.LINE_AA,
        )

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Full image path, e.g. data/images/cam01/rgb/00001.jpg")
    parser.add_argument("--keypoints_json", required=True, help="keypoints2d JSON file")
    parser.add_argument("--out", required=True, help="Output visualization image")
    parser.add_argument("--min_conf", type=float, default=0.0)
    parser.add_argument("--no_names", action="store_true")
    args = parser.parse_args()
    print("meow")
    image_path = Path(args.image).expanduser()
    json_path = Path(args.keypoints_json).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    joint_ids, keypoints = load_keypoints_json(json_path)

    print("Image:", image_path)
    print("Image shape:", img.shape)
    print("Keypoints JSON:", json_path)
    print("joint_ids:", joint_ids.tolist())
    print("keypoints shape:", keypoints.shape)

    for jid, (u, v, conf) in zip(joint_ids, keypoints):
        name = SMPL_JOINT_NAMES.get(int(jid), "joint")
        print(f"  joint {jid:2d} {name:12s}: u={u:8.2f}, v={v:8.2f}, conf={conf:.3f}")

    vis = draw_keypoints(
        img,
        joint_ids,
        keypoints,
        min_conf=args.min_conf,
        draw_names=not args.no_names,
    )

    cv2.imwrite(str(out_path), vis)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()