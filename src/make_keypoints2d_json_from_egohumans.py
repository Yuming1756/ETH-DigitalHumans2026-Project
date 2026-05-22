from pathlib import Path
import argparse
import json
import numpy as np


# COCO-17 body keypoint index -> SMPL-24 joint index
COCO_TO_SMPL24 = {
    5: 16,   # left_shoulder
    6: 17,   # right_shoulder
    7: 18,   # left_elbow
    8: 19,   # right_elbow
    9: 20,   # left_wrist
    10: 21,  # right_wrist
    11: 1,   # left_hip
    12: 2,   # right_hip
    13: 4,   # left_knee
    14: 5,   # right_knee
    15: 7,   # left_ankle
    16: 8,   # right_ankle
}


# COCO-17 face/head keypoint indices
COCO_FACE_NAMES = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
}


def load_poses2d(path):
    path = Path(path).expanduser()
    data = np.load(path, allow_pickle=True)

    # EgoHumans format: ndarray shape=(num_people,), dtype=object
    if isinstance(data, np.ndarray) and data.dtype == object:
        return list(data)

    return data


def extract_keypoints_for_aria(data, aria):
    """
    EgoHumans format:
        list/array of dicts:
        {
            'bbox': ...,
            'human_name': 'aria03',
            'human_id': 2,
            'keypoints': (133, 3)
        }

    Returns:
        keypoints: (133, 3), COCO-WholeBody format.
        First 17 are COCO body keypoints.
    """
    for item in data:
        if not isinstance(item, dict):
            continue

        if item.get("human_name") == aria:
            if "keypoints" not in item:
                raise KeyError(f"Found {aria}, but no 'keypoints' field.")

            return np.asarray(item["keypoints"], dtype=np.float32)

    available = [
        item.get("human_name")
        for item in data
        if isinstance(item, dict) and "human_name" in item
    ]
    raise KeyError(f"Could not find {aria}. Available identities: {available}")


def is_valid_keypoint(kpt, min_conf):
    u, v, conf = kpt[:3]

    if not np.isfinite(u) or not np.isfinite(v) or not np.isfinite(conf):
        return False

    if conf < min_conf:
        return False

    return True


def convert_to_optimizer_json(
    kpts,
    min_conf_body,
    min_conf_face,
    face_vertex_map,
):
    kpts = np.asarray(kpts, dtype=np.float32)

    if kpts.ndim != 2 or kpts.shape[1] < 3:
        raise ValueError(f"Expected keypoints shape (N,3), got {kpts.shape}")

    if kpts.shape[0] < 17:
        raise ValueError(f"Expected at least 17 COCO keypoints, got {kpts.shape}")

    # COCO-WholeBody 133 format: first 17 are COCO body keypoints.
    coco17 = kpts[:17]

    # ------------------------------------------------------------
    # 1. Body keypoints: COCO body -> SMPL-24 joints
    # ------------------------------------------------------------
    joint_ids = []
    keypoints = []

    for coco_idx, smpl_idx in COCO_TO_SMPL24.items():
        kpt = coco17[coco_idx, :3]

        if not is_valid_keypoint(kpt, min_conf_body):
            continue

        u, v, conf = kpt
        joint_ids.append(int(smpl_idx))
        keypoints.append([float(u), float(v), float(conf)])

    if len(joint_ids) == 0:
        raise RuntimeError("No valid body keypoints after confidence filtering.")

    # ------------------------------------------------------------
    # 2. Face/head keypoints: COCO face/head -> SMPL vertices
    # ------------------------------------------------------------
    vertex_ids = []
    vertex_keypoints = []
    vertex_names = []

    for coco_idx, name in COCO_FACE_NAMES.items():
        if name not in face_vertex_map:
            continue

        vertex_id = face_vertex_map[name]

        # If user passes -1, skip this keypoint.
        if vertex_id is None or vertex_id < 0:
            continue

        kpt = coco17[coco_idx, :3]

        if not is_valid_keypoint(kpt, min_conf_face):
            continue

        u, v, conf = kpt
        vertex_ids.append(int(vertex_id))
        vertex_keypoints.append([float(u), float(v), float(conf)])
        vertex_names.append(name)

    out = {
        "joint_ids": joint_ids,
        "keypoints": keypoints,
        "joint_note": "SMPL-24 joint IDs. keypoints are [u, v, confidence].",

        "vertex_ids": vertex_ids,
        "vertex_keypoints": vertex_keypoints,
        "vertex_names": vertex_names,
        "vertex_note": (
            "SMPL vertex IDs for face/head 2D reprojection. "
            "vertex_keypoints are [u, v, confidence]. "
            "The third value is confidence, not depth."
        ),
    }

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--poses2d", required=True)
    parser.add_argument("--aria", required=True)
    parser.add_argument("--out_json", required=True)

    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.2,
        help="Minimum confidence for body keypoints.",
    )

    parser.add_argument(
        "--min_conf_face",
        type=float,
        default=0.2,
        help="Minimum confidence for face/head keypoints.",
    )

    # ------------------------------------------------------------
    # SMPL vertex IDs for COCO face/head keypoints.
    #
    # You must choose these from MeshLab / nearest-vertex analysis.
    # Use -1 to skip a keypoint.
    # ------------------------------------------------------------
    parser.add_argument("--nose_vertex", type=int, default=-1)
    parser.add_argument("--left_eye_vertex", type=int, default=-1)
    parser.add_argument("--right_eye_vertex", type=int, default=-1)
    parser.add_argument("--left_ear_vertex", type=int, default=-1)
    parser.add_argument("--right_ear_vertex", type=int, default=-1)

    args = parser.parse_args()

    face_vertex_map = {
        "nose": args.nose_vertex,
        "left_eye": args.left_eye_vertex,
        "right_eye": args.right_eye_vertex,
        "left_ear": args.left_ear_vertex,
        "right_ear": args.right_ear_vertex,
    }

    data = load_poses2d(args.poses2d)
    kpts = extract_keypoints_for_aria(data, args.aria)

    out = convert_to_optimizer_json(
        kpts,
        min_conf_body=args.min_conf,
        min_conf_face=args.min_conf_face,
        face_vertex_map=face_vertex_map,
    )

    out_path = Path(args.out_json).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("Saved:", out_path)
    print("aria:", args.aria)
    print("input keypoints shape:", kpts.shape)

    print("\nBody keypoints:")
    print("used joints:", len(out["joint_ids"]))
    print("SMPL joint_ids:", out["joint_ids"])

    print("\nFace/head vertex keypoints:")
    print("used vertices:", len(out["vertex_ids"]))
    print("vertex names:", out["vertex_names"])
    print("vertex ids:", out["vertex_ids"])


if __name__ == "__main__":
    main()