import argparse
from pathlib import Path
from itertools import permutations

import numpy as np


def load_obj_vertices(obj_path):
    """
    Load vertices from an OBJ file.

    Supports:
        v x y z
        v x y z r g b
    """
    obj_path = Path(obj_path).expanduser()
    vertices = []

    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    vertices = np.asarray(vertices, dtype=np.float32)

    if vertices.shape != (6890, 3):
        raise ValueError(f"Expected OBJ vertices shape (6890, 3), got {vertices.shape} from {obj_path}")

    return vertices


def load_prediction_vertices(pred_path):
    """
    Load TokenHMR / 4DHumans prediction.

    Supports:
        .obj
        .npz with pred_vertices_full
        .npz with pred_vertices_crop + pred_cam_t_full
    """
    pred_path = Path(pred_path).expanduser()

    if pred_path.suffix == ".obj":
        print(f"[Prediction] Loading OBJ: {pred_path}")
        return load_obj_vertices(pred_path)

    if pred_path.suffix == ".npz":
        print(f"[Prediction] Loading NPZ: {pred_path}")
        data = np.load(pred_path)

        print("[Prediction] keys:", list(data.files))

        if "pred_vertices_full" in data.files:
            vertices = data["pred_vertices_full"]

        elif "pred_vertices_hmr" in data.files and "pred_cam_t_full" in data.files:
            vertices = data["pred_vertices_hmr"] + data["pred_cam_t_full"][None, :]

        elif "pred_vertices_crop" in data.files and "pred_cam_t_full" in data.files:
            vertices = data["pred_vertices_crop"] + data["pred_cam_t_full"][None, :]

        elif "pred_vertices" in data.files and "pred_cam_t_full" in data.files:
            vertices = data["pred_vertices"] + data["pred_cam_t_full"][None, :]

        elif "pred_vertices_crop" in data.files:
            print("[Warning] Using pred_vertices_crop directly.")
            vertices = data["pred_vertices_crop"]

        elif "pred_vertices" in data.files:
            print("[Warning] Using pred_vertices directly.")
            vertices = data["pred_vertices"]

        else:
            raise KeyError("Could not find predicted vertices in NPZ.")

        vertices = vertices.astype(np.float32)

        if vertices.shape != (6890, 3):
            raise ValueError(f"Expected prediction vertices shape (6890, 3), got {vertices.shape}")

        return vertices

    raise ValueError(f"Unsupported prediction file type: {pred_path.suffix}")


def load_smpl_joint_regressor(smpl_model_path):
    """
    Load SMPL J_regressor directly from basicModel_neutral_lbs_10_207_0_v1.0.0.pkl.
    """
    import pickle

    # Compatibility patch for old SMPL pickle files
    if not hasattr(np, "bool"):
        np.bool = bool
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "complex"):
        np.complex = complex
    if not hasattr(np, "object"):
        np.object = object
    if not hasattr(np, "str"):
        np.str = str
    if not hasattr(np, "unicode"):
        np.unicode = str

    smpl_model_path = Path(smpl_model_path).expanduser()

    if smpl_model_path.is_dir():
        smpl_model_path = smpl_model_path / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"

    if not smpl_model_path.exists():
        raise FileNotFoundError(f"SMPL model file not found: {smpl_model_path}")

    with open(smpl_model_path, "rb") as f:
        smpl_data = pickle.load(f, encoding="latin1")

    J_regressor = smpl_data["J_regressor"]

    if hasattr(J_regressor, "toarray"):
        J_regressor = J_regressor.toarray()

    J_regressor = np.asarray(J_regressor, dtype=np.float32)

    print("[SMPL] Loaded J_regressor from:", smpl_model_path)
    print("[SMPL] J_regressor shape:", J_regressor.shape)

    if J_regressor.shape[1] != 6890:
        raise ValueError(f"Expected J_regressor shape (?, 6890), got {J_regressor.shape}")

    return J_regressor


def vertices_to_joints(vertices, J_regressor):
    return J_regressor @ vertices


def mean_l2_error(pred, gt, valid=None):
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, gt {gt.shape}")

    errors = np.linalg.norm(pred - gt, axis=1)

    if valid is not None:
        errors = errors[valid]

    return float(errors.mean())


def procrustes_align(pred, gt):
    """
    Similarity-align pred to gt.
    """
    X = pred.astype(np.float64)
    Y = gt.astype(np.float64)

    if X.shape != Y.shape:
        raise ValueError(f"Shape mismatch: pred {X.shape}, gt {Y.shape}")

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


def compute_mpjpe_metrics(pred_joints, gt_joints, root_index=0, unit_scale_to_mm=1000.0):
    raw_mpjpe = mean_l2_error(pred_joints, gt_joints)

    pred_root_aligned = pred_joints - pred_joints[root_index : root_index + 1]
    gt_root_aligned = gt_joints - gt_joints[root_index : root_index + 1]
    root_aligned_mpjpe = mean_l2_error(pred_root_aligned, gt_root_aligned)

    pred_pa = procrustes_align(pred_joints, gt_joints)
    pa_mpjpe = mean_l2_error(pred_pa, gt_joints)

    global_root_drift = float(
        np.linalg.norm(pred_joints[root_index] - gt_joints[root_index])
    )

    return {
        "raw_mpjpe_mm": raw_mpjpe * unit_scale_to_mm,
        "root_aligned_mpjpe_mm": root_aligned_mpjpe * unit_scale_to_mm,
        "pa_mpjpe_mm": pa_mpjpe * unit_scale_to_mm,
        "global_root_drift_m": global_root_drift,
    }


def root_aligned_mpjpe_mm(pred_joints, gt_joints, root_index=0):
    pred_ra = pred_joints - pred_joints[root_index : root_index + 1]
    gt_ra = gt_joints - gt_joints[root_index : root_index + 1]
    return mean_l2_error(pred_ra, gt_ra) * 1000.0


def find_best_identity_mapping(pred_items, gt_items, root_index=0):
    """
    pred_items:
        list of dicts:
            {
                "det_id": "0",
                "path": Path(...),
                "joints": np.ndarray [J, 3],
            }

    gt_items:
        list of dicts:
            {
                "person": "aria03",
                "path": Path(...),
                "joints": np.ndarray [J, 3],
            }

    Returns best mapping by minimum total root-aligned MPJPE.
    """
    n_pred = len(pred_items)
    n_gt = len(gt_items)

    if n_pred == 0:
        raise ValueError("No prediction items.")
    if n_gt == 0:
        raise ValueError("No GT items.")
    if n_pred > n_gt:
        print(f"[Warning] More predictions than GT: {n_pred} predictions, {n_gt} GT. Only matching subset.")

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

    for perm in permutations(range(n_gt), min(n_pred, n_gt)):
        total = 0.0
        for i, j in enumerate(perm):
            total += score_matrix[i, j]

        if total < best_score:
            best_score = total
            best_perm = perm

    mapping = {}
    for i, j in enumerate(best_perm):
        mapping[pred_items[i]["det_id"]] = gt_items[j]["person"]

    return mapping, score_matrix, best_score


def print_pair_metrics(pred_path, gt_path, pred_joints, gt_joints, args):
    metrics = compute_mpjpe_metrics(
        pred_joints,
        gt_joints,
        root_index=args.root_index,
        unit_scale_to_mm=args.unit_scale_to_mm,
    )

    print("\n================ SMPL-Regressed Joint Metrics ================")
    print(f"Prediction: {pred_path}")
    print(f"Groundtruth: {gt_path}")
    print("--------------------------------------------------------------")
    print(f"Raw MPJPE:              {metrics['raw_mpjpe_mm']:.2f} mm")
    print(f"Root-aligned MPJPE:     {metrics['root_aligned_mpjpe_mm']:.2f} mm")
    print(f"PA-MPJPE:               {metrics['pa_mpjpe_mm']:.2f} mm")
    print(f"Global root drift:      {metrics['global_root_drift_m']:.4f} m")
    print("==============================================================")


def run_single_pair(args):
    pred_vertices = load_prediction_vertices(args.pred)
    gt_vertices = load_obj_vertices(args.gt)

    print("[Prediction vertices]", pred_vertices.shape)
    print("[GT vertices]", gt_vertices.shape)

    J_regressor = load_smpl_joint_regressor(args.smpl_model_dir)

    pred_joints = vertices_to_joints(pred_vertices, J_regressor)
    gt_joints = vertices_to_joints(gt_vertices, J_regressor)

    print("[Pred joints]", pred_joints.shape)
    print("[GT joints]", gt_joints.shape)

    print_pair_metrics(args.pred, args.gt, pred_joints, gt_joints, args)


def run_auto_match_frame(args):
    pred_dir = Path(args.pred_dir).expanduser()
    gt_dir = Path(args.gt_dir).expanduser()
    frame = args.frame

    J_regressor = load_smpl_joint_regressor(args.smpl_model_dir)

    pred_files = sorted(pred_dir.glob(f"{frame}_*.obj"))
    pred_files = [p for p in pred_files if "_all" not in p.stem]

    gt_files = sorted(gt_dir.glob("mesh_aria*.obj"))

    if len(pred_files) == 0:
        print(f"[Warning] No prediction files found: {pred_dir}/{frame}_*.obj")
        return

    if len(gt_files) == 0:
        print(f"[Warning] No GT files found: {gt_dir}/mesh_aria*.obj")
        return

    pred_items = []
    gt_items = []

    print("\nLoading prediction meshes for auto matching:")
    for pred_path in pred_files:
        det_id = pred_path.stem.split("_")[-1]

        vertices = load_prediction_vertices(pred_path)
        joints = vertices_to_joints(vertices, J_regressor)

        pred_items.append(
            {
                "det_id": det_id,
                "path": pred_path,
                "vertices": vertices,
                "joints": joints,
            }
        )

    print("\nLoading GT meshes for auto matching:")
    for gt_path in gt_files:
        person = gt_path.stem.replace("mesh_", "")

        vertices = load_obj_vertices(gt_path)
        joints = vertices_to_joints(vertices, J_regressor)

        gt_items.append(
            {
                "person": person,
                "path": gt_path,
                "vertices": vertices,
                "joints": joints,
            }
        )

    mapping, score_matrix, best_score = find_best_identity_mapping(
        pred_items,
        gt_items,
        root_index=args.root_index,
    )

    pred_ids = [p["det_id"] for p in pred_items]
    gt_ids = [g["person"] for g in gt_items]

    print("\n---------- AUTO IDENTITY MATCHING ----------")
    print("Matching criterion: minimum total root-aligned SMPL-joint MPJPE")
    print("Prediction det ids:", pred_ids)
    print("GT person ids:", gt_ids)
    print("\nPairwise root-aligned MPJPE matrix, mm")
    print("Rows = prediction det_id, columns = GT person")
    print(score_matrix)
    print(f"\nBest total matching score: {best_score:.2f} mm")

    print("\nAuto detection-to-GT identity mapping:")
    for det_id in sorted(mapping.keys(), key=lambda x: int(x) if x.isdigit() else x):
        print(f"  det {det_id} -> {mapping[det_id]}")
    print("--------------------------------------------")

    pred_by_id = {p["det_id"]: p for p in pred_items}
    gt_by_person = {g["person"]: g for g in gt_items}

    for det_id in sorted(mapping.keys(), key=lambda x: int(x) if x.isdigit() else x):
        person = mapping[det_id]
        p = pred_by_id[det_id]
        g = gt_by_person[person]

        print()
        print("############################################################")
        print(f"Frame: {frame}")
        print(f"TokenHMR detection id: {det_id}")
        print(f"Auto-matched GT person: {person}")
        print(f"Prediction: {p['path']}")
        print(f"Ground truth: {g['path']}")
        print("############################################################")

        print_pair_metrics(
            p["path"],
            g["path"],
            p["joints"],
            g["joints"],
            args,
        )

    print("\nImportant:")
    print("- Detection ids are not assumed to be stable across frames.")
    print("- Matching is recomputed independently for each frame.")
    print("- Matching uses root-aligned joints, so it is less affected by global depth drift.")
    print("- Raw MPJPE/root drift are still meaningful only if both meshes are in the same camera coordinate frame.")
    print("- PA-MPJPE is safer when coordinate alignment is uncertain.")


def main():
    parser = argparse.ArgumentParser()

    # Old single-pair mode
    parser.add_argument("--pred", default=None, help="Prediction .obj or .npz")
    parser.add_argument("--gt", default=None, help="EgoHumans GT mesh_cam OBJ")

    # New auto-frame mode
    parser.add_argument("--auto_match_frame", action="store_true")
    parser.add_argument("--pred_dir", default=None, help="Prediction OBJ directory")
    parser.add_argument("--gt_dir", default=None, help="GT frame directory, e.g. .../rgb/00001")
    parser.add_argument("--frame", default=None, help="Frame id, e.g. 00001")

    parser.add_argument(
        "--smpl_model_dir",
        required=True,
        help="Folder containing SMPL model files, e.g. ~/DigitalHumans/4D-Humans/data",
    )

    parser.add_argument(
        "--root_index",
        type=int,
        default=0,
        help="Root joint index. For standard SMPL regressor, 0 is usually pelvis.",
    )

    parser.add_argument(
        "--unit_scale_to_mm",
        type=float,
        default=1000.0,
        help="Use 1000 if coordinates are meters; use 1 if already millimeters.",
    )

    args = parser.parse_args()

    if args.auto_match_frame:
        if args.pred_dir is None or args.gt_dir is None or args.frame is None:
            raise ValueError("--auto_match_frame requires --pred_dir, --gt_dir, and --frame")
        run_auto_match_frame(args)
    else:
        if args.pred is None or args.gt is None:
            raise ValueError("Single-pair mode requires --pred and --gt")
        run_single_pair(args)


if __name__ == "__main__":
    main()