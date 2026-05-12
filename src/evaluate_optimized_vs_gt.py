#!/usr/bin/env python3

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from itertools import permutations


def load_obj_vertices(obj_path):
    obj_path = Path(obj_path)
    vertices = []

    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    vertices = np.asarray(vertices, dtype=np.float32)

    if vertices.shape != (6890, 3):
        raise ValueError(f"Expected vertices shape (6890, 3), got {vertices.shape} from {obj_path}")

    return vertices


def load_smpl_joint_regressor(smpl_model_dir):
    smpl_model_dir = Path(smpl_model_dir)

    if smpl_model_dir.is_dir():
        smpl_path = smpl_model_dir / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    else:
        smpl_path = smpl_model_dir

    if not smpl_path.exists():
        raise FileNotFoundError(f"SMPL model file not found: {smpl_path}")

    # Compatibility for old SMPL pickle files
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)

    with open(smpl_path, "rb") as f:
        smpl_data = pickle.load(f, encoding="latin1")

    J_regressor = smpl_data["J_regressor"]

    if hasattr(J_regressor, "toarray"):
        J_regressor = J_regressor.toarray()

    J_regressor = np.asarray(J_regressor, dtype=np.float32)

    if J_regressor.shape[1] != 6890:
        raise ValueError(f"Expected J_regressor shape (?, 6890), got {J_regressor.shape}")

    print(f"[SMPL] Loaded: {smpl_path}")
    print(f"[SMPL] J_regressor shape: {J_regressor.shape}")

    return J_regressor


def vertices_to_joints(vertices, J_regressor):
    return J_regressor @ vertices


def mean_l2_error(pred, gt):
    return float(np.linalg.norm(pred - gt, axis=1).mean())


def procrustes_align(pred, gt):
    X = pred.astype(np.float64)
    Y = gt.astype(np.float64)

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)

    X0 = X - mu_x
    Y0 = Y - mu_y

    var_x = np.mean(np.sum(X0 ** 2, axis=1))
    K = (Y0.T @ X0) / X.shape[0]

    U, S, Vt = np.linalg.svd(K)

    Z = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        Z[-1, -1] = -1

    R = U @ Z @ Vt
    scale = np.trace(np.diag(S) @ Z) / var_x
    t = mu_y - scale * (R @ mu_x)

    X_aligned = scale * (R @ X.T).T + t
    return X_aligned.astype(np.float32)


def compute_metrics(pred_vertices, gt_vertices, J_regressor, root_index=0, unit_scale_to_mm=1000.0):
    pred_joints = vertices_to_joints(pred_vertices, J_regressor)
    gt_joints = vertices_to_joints(gt_vertices, J_regressor)

    raw_mpjpe = mean_l2_error(pred_joints, gt_joints)

    pred_ra = pred_joints - pred_joints[root_index : root_index + 1]
    gt_ra = gt_joints - gt_joints[root_index : root_index + 1]
    root_aligned_mpjpe = mean_l2_error(pred_ra, gt_ra)

    pred_pa = procrustes_align(pred_joints, gt_joints)
    pa_mpjpe = mean_l2_error(pred_pa, gt_joints)

    root_drift = float(np.linalg.norm(pred_joints[root_index] - gt_joints[root_index]))

    vertex_error = mean_l2_error(pred_vertices, gt_vertices)

    return {
        "raw_mpjpe_mm": raw_mpjpe * unit_scale_to_mm,
        "root_aligned_mpjpe_mm": root_aligned_mpjpe * unit_scale_to_mm,
        "pa_mpjpe_mm": pa_mpjpe * unit_scale_to_mm,
        "global_root_drift_m": root_drift,
        "mean_vertex_error_mm": vertex_error * unit_scale_to_mm,
    }


def default_prediction_dir(project_root, model, optimizer):
    if model.lower() in ["tokenhmr", "tkhmr"]:
        return project_root / "TokenHMR" / "demo_out" / f"my_image_smplify_{optimizer}"
    elif model.lower() in ["4dhumans", "4dh", "4d"]:
        return project_root / "4D-Humans" / "demo_out" / f"my_image_smplify_{optimizer}"
    else:
        raise ValueError(f"Unknown model: {model}")

def root_aligned_mpjpe_mm(pred_joints, gt_joints, root_index=0):
    pred_ra = pred_joints - pred_joints[root_index : root_index + 1]
    gt_ra = gt_joints - gt_joints[root_index : root_index + 1]
    return mean_l2_error(pred_ra, gt_ra) * 1000.0


def find_best_identity_mapping(pred_items, gt_items, root_index=0):
    """
    Auto-match detection IDs to EgoHumans identities for one frame.

    Matching criterion:
        minimum total root-aligned SMPL-joint MPJPE

    pred_items:
        list of {
            "det_id": str,
            "path": Path,
            "vertices": np.ndarray,
            "joints": np.ndarray,
        }

    gt_items:
        list of {
            "aria": str,
            "path": Path,
            "vertices": np.ndarray,
            "joints": np.ndarray,
        }
    """
    n_pred = len(pred_items)
    n_gt = len(gt_items)

    if n_pred == 0:
        raise ValueError("No prediction items for matching.")
    if n_gt == 0:
        raise ValueError("No GT items for matching.")

    score_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)

    for i, p in enumerate(pred_items):
        for j, g in enumerate(gt_items):
            score_matrix[i, j] = root_aligned_mpjpe_mm(
                p["joints"],
                g["joints"],
                root_index=root_index,
            )

    best_score = float("inf")
    best_perm = None

    # Match each prediction to one GT identity.
    # If there are fewer predictions than GT, only a subset of GT is used.
    for perm in permutations(range(n_gt), min(n_pred, n_gt)):
        total = 0.0
        for i, j in enumerate(perm):
            total += score_matrix[i, j]

        if total < best_score:
            best_score = total
            best_perm = perm

    mapping = {}

    for i, j in enumerate(best_perm):
        mapping[pred_items[i]["det_id"]] = gt_items[j]["aria"]

    return mapping, score_matrix, best_score


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--frame", default=None, help="Evaluate one frame, e.g. 00001. If omitted, evaluate all frames found.")
    parser.add_argument("--model", default="4dhumans", help="tokenhmr or 4dhumans")
    parser.add_argument("--optimizer", default="lbfgs", help="lbfgs or adam")

    parser.add_argument("--pred_dir", default=None, help="Directory containing optimized OBJ files")
    parser.add_argument("--gt_root", default="./data/mesh_cam/cam01/rgb", help="GT root: data/mesh_cam/cam01/rgb")
    parser.add_argument("--smpl_model_dir", default="./smpl", help="SMPL folder containing basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")

    parser.add_argument("--out_csv", default=None, help="Output per-pair CSV")
    parser.add_argument("--summary_csv", default=None, help="Output summary CSV")

    parser.add_argument("--root_index", type=int, default=0)
    parser.add_argument("--unit_scale_to_mm", type=float, default=1000.0)

    args = parser.parse_args()

    project_root = Path(".").resolve()

    if args.pred_dir is None:
        pred_dir = default_prediction_dir(project_root, args.model, args.optimizer)
    else:
        pred_dir = Path(args.pred_dir)

    gt_root = Path(args.gt_root)
    smpl_model_dir = Path(args.smpl_model_dir)

    if args.out_csv is None:
        args.out_csv = f"metrics_{args.model}_{args.optimizer}_optimized_vs_gt.csv"

    if args.summary_csv is None:
        args.summary_csv = f"metrics_{args.model}_{args.optimizer}_optimized_vs_gt_summary.csv"

    out_csv = Path(args.out_csv)
    summary_csv = Path(args.summary_csv)

    print("============================================================")
    print("Evaluate optimized meshes against EgoHumans GT")
    print(f"Model: {args.model}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Frame: {args.frame if args.frame else 'ALL'}")
    print(f"Prediction dir: {pred_dir}")
    print(f"GT root: {gt_root}")
    print(f"SMPL model dir: {smpl_model_dir}")
    print(f"Output CSV: {out_csv}")
    print(f"Summary CSV: {summary_csv}")
    print("============================================================")

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    if not gt_root.exists():
        raise FileNotFoundError(f"GT root not found: {gt_root}")

    J_regressor = load_smpl_joint_regressor(smpl_model_dir)

    # detection id -> EgoHumans identity
    pred_files_all = sorted(pred_dir.glob("*.obj"))
    pred_files_all = [p for p in pred_files_all if "_all" not in p.stem]

    if args.frame is not None:
        pred_files_all = [
            p for p in pred_files_all
            if p.stem.startswith(f"{args.frame}_")
        ]

    if len(pred_files_all) == 0:
        raise FileNotFoundError(f"No optimized OBJ files found in {pred_dir}")

    # Group prediction files by frame
    frame_to_pred_files = {}

    for pred_obj in pred_files_all:
        parts = pred_obj.stem.split("_")

        if len(parts) < 2:
            print(f"[SKIP] Cannot parse frame/detection id from: {pred_obj.name}")
            continue

        frame = parts[0]
        frame_to_pred_files.setdefault(frame, []).append(pred_obj)

    rows = []

    for frame in sorted(frame_to_pred_files.keys()):
        print()
        print("################################################################")
        print(f"Frame: {frame}")
        print("Auto-matching optimized predictions to EgoHumans GT")
        print("################################################################")

        pred_files = sorted(
            frame_to_pred_files[frame],
            key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else p.stem.split("_")[1],
        )

        gt_dir = gt_root / frame

        if not gt_dir.exists():
            print(f"[SKIP] Missing GT frame directory: {gt_dir}")
            continue

        gt_files = sorted(gt_dir.glob("mesh_aria*.obj"))

        if len(gt_files) == 0:
            print(f"[SKIP] No GT files found in: {gt_dir}")
            continue

        pred_items = []
        gt_items = []

        print("\nLoading optimized prediction meshes:")
        for pred_obj in pred_files:
            parts = pred_obj.stem.split("_")
            det_id = parts[1]

            pred_vertices = load_obj_vertices(pred_obj)
            pred_joints = vertices_to_joints(pred_vertices, J_regressor)

            pred_items.append(
                {
                    "det_id": det_id,
                    "path": pred_obj,
                    "vertices": pred_vertices,
                    "joints": pred_joints,
                }
            )

            print(f"  det {det_id}: {pred_obj}")

        print("\nLoading EgoHumans GT meshes:")
        for gt_obj in gt_files:
            aria = gt_obj.stem.replace("mesh_", "")

            gt_vertices = load_obj_vertices(gt_obj)
            gt_joints = vertices_to_joints(gt_vertices, J_regressor)

            gt_items.append(
                {
                    "aria": aria,
                    "path": gt_obj,
                    "vertices": gt_vertices,
                    "joints": gt_joints,
                }
            )

            print(f"  {aria}: {gt_obj}")

        mapping, score_matrix, best_score = find_best_identity_mapping(
            pred_items,
            gt_items,
            root_index=args.root_index,
        )

        pred_ids = [p["det_id"] for p in pred_items]
        gt_ids = [g["aria"] for g in gt_items]

        print("\n---------- AUTO IDENTITY MATCHING ----------")
        print("Criterion: minimum total root-aligned SMPL-joint MPJPE")
        print("Prediction det ids:", pred_ids)
        print("GT person ids:", gt_ids)
        print("\nPairwise root-aligned MPJPE matrix, mm")
        print("Rows = prediction det_id, columns = GT aria")
        print(score_matrix)
        print(f"\nBest total matching score: {best_score:.2f} mm")

        print("\nAuto detection-to-GT identity mapping:")
        for det_id in sorted(mapping.keys(), key=lambda x: int(x) if x.isdigit() else x):
            print(f"  det {det_id} -> {mapping[det_id]}")
        print("--------------------------------------------")

        pred_by_id = {p["det_id"]: p for p in pred_items}
        gt_by_aria = {g["aria"]: g for g in gt_items}

        for det_id in sorted(mapping.keys(), key=lambda x: int(x) if x.isdigit() else x):
            aria = mapping[det_id]

            p = pred_by_id[det_id]
            g = gt_by_aria[aria]

            pred_obj = p["path"]
            gt_obj = g["path"]

            print()
            print("------------------------------------------------------------")
            print(f"Frame: {frame}")
            print(f"Detection id: {det_id}")
            print(f"Auto-matched GT: {aria}")
            print(f"Pred: {pred_obj}")
            print(f"GT:   {gt_obj}")

            metrics = compute_metrics(
                p["vertices"],
                g["vertices"],
                J_regressor,
                root_index=args.root_index,
                unit_scale_to_mm=args.unit_scale_to_mm,
            )

            row = {
                "model": args.model,
                "optimizer": args.optimizer,
                "frame": frame,
                "det_id": det_id,
                "aria": aria,
                "pred_obj": str(pred_obj),
                "gt_obj": str(gt_obj),
                "matching_score_total_root_aligned_mpjpe_mm": best_score,
                **metrics,
            }

            rows.append(row)

            print(f"Raw MPJPE:              {metrics['raw_mpjpe_mm']:.2f} mm")
            print(f"Root-aligned MPJPE:     {metrics['root_aligned_mpjpe_mm']:.2f} mm")
            print(f"PA-MPJPE:               {metrics['pa_mpjpe_mm']:.2f} mm")
            print(f"Mean vertex error:      {metrics['mean_vertex_error_mm']:.2f} mm")
            print(f"Global root drift:      {metrics['global_root_drift_m']:.4f} m")

    if len(rows) == 0:
        raise RuntimeError("No valid prediction/GT pairs were evaluated.")

    fieldnames = list(rows[0].keys())

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_keys = [
        "raw_mpjpe_mm",
        "root_aligned_mpjpe_mm",
        "pa_mpjpe_mm",
        "mean_vertex_error_mm",
        "global_root_drift_m",
    ]

    summary_rows = []

    for key in metric_keys:
        values = np.asarray([r[key] for r in rows], dtype=np.float64)
        summary_rows.append(
            {
                "metric": key,
                "n": len(values),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "median": float(np.median(values)),
            }
        )

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "n", "mean", "std", "min", "max", "median"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("============================================================")
    print("Summary")
    print("============================================================")

    for row in summary_rows:
        print(
            f"{row['metric']}: "
            f"mean={row['mean']:.2f}, "
            f"std={row['std']:.2f}, "
            f"median={row['median']:.2f}, "
            f"min={row['min']:.2f}, "
            f"max={row['max']:.2f}, "
            f"n={row['n']}"
        )

    print()
    print(f"Saved per-pair metrics to: {out_csv}")
    print(f"Saved summary metrics to:  {summary_csv}")


if __name__ == "__main__":
    main()