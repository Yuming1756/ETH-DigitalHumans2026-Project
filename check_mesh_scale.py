from pathlib import Path
import argparse
import pickle
import numpy as np


def load_obj_vertices(path):
    path = Path(path).expanduser()
    verts = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                p = line.strip().split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])

    verts = np.asarray(verts, dtype=np.float32)

    if verts.shape != (6890, 3):
        raise ValueError(f"Expected (6890, 3), got {verts.shape} from {path}")

    return verts


def patch_numpy_for_old_smpl():
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }

    for name, typ in aliases.items():
        try:
            getattr(np, name)
        except AttributeError:
            setattr(np, name, typ)


def load_smpl_joint_regressor(smpl_model_dir):
    patch_numpy_for_old_smpl()

    smpl_model_dir = Path(smpl_model_dir).expanduser()
    smpl_path = smpl_model_dir / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"

    with open(smpl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    J = data["J_regressor"]

    if hasattr(J, "toarray"):
        J = J.toarray()

    return np.asarray(J, dtype=np.float32)


def summarize(name, verts, J):
    joints = J @ verts

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    extent = bbox_max - bbox_min

    root = joints[0]
    head = joints[15]

    # Common SMPL limb-ish distances
    root_to_head = np.linalg.norm(head - root)

    left_hip = joints[1]
    right_hip = joints[2]
    left_knee = joints[4]
    right_knee = joints[5]
    left_ankle = joints[7]
    right_ankle = joints[8]

    left_leg = np.linalg.norm(left_hip - left_knee) + np.linalg.norm(left_knee - left_ankle)
    right_leg = np.linalg.norm(right_hip - right_knee) + np.linalg.norm(right_knee - right_ankle)

    print(f"\n========== {name} ==========")
    print("bbox min:", bbox_min)
    print("bbox max:", bbox_max)
    print("bbox extent x/y/z:", extent)
    print("mesh center:", verts.mean(axis=0))
    print("SMPL root:", root)
    print("SMPL head:", head)
    print(f"root-to-head: {root_to_head:.4f} m")
    print(f"left leg length approx:  {left_leg:.4f} m")
    print(f"right leg length approx: {right_leg:.4f} m")

    return {
        "extent": extent,
        "root_to_head": root_to_head,
        "left_leg": left_leg,
        "right_leg": right_leg,
    }


def ratio(name, pred_val, gt_val):
    print(f"{name}: pred={pred_val:.4f}, gt={gt_val:.4f}, pred/gt={pred_val / gt_val:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--smpl_model_dir", required=True)
    args = parser.parse_args()

    pred = load_obj_vertices(args.pred)
    gt = load_obj_vertices(args.gt)
    J = load_smpl_joint_regressor(args.smpl_model_dir)

    pred_stats = summarize("Prediction", pred, J)
    gt_stats = summarize("GT", gt, J)

    print("\n========== SCALE RATIOS ==========")
    for i, ax in enumerate(["x_extent", "y_extent", "z_extent"]):
        ratio(ax, pred_stats["extent"][i], gt_stats["extent"][i])

    ratio("root_to_head", pred_stats["root_to_head"], gt_stats["root_to_head"])
    ratio("left_leg", pred_stats["left_leg"], gt_stats["left_leg"])
    ratio("right_leg", pred_stats["right_leg"], gt_stats["right_leg"])

    print("\nInterpretation:")
    print("- If pred/gt is close to 1.0, scale is probably fine.")
    print("- If pred/gt is consistently around 0.5, 0.7, 1.3, etc., there may be a scale bug.")
    print("- Translation affects center/root position, but should not affect body extents or limb lengths.")


if __name__ == "__main__":
    main()