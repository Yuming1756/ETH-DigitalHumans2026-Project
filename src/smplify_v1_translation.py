import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from compute_total_loss import compute_total_loss


# ---------------------------------------------------------------------
# Coordinate convention
# ---------------------------------------------------------------------
# Current TokenHMR OBJ export does this:
#
#   mesh = vertices + camera_translation
#   then rotate 180 degrees around X
#
# So:
#
#   TokenHMR camera coordinate:
#       X =  x
#       Y =  y
#       Z =  positive depth
#
#   Exported OBJ / EgoHumans mesh_cam-like coordinate:
#       X =  x
#       Y = -y
#       Z = -z
#
# Therefore conversion between them is:
#
#   [x, y, z] camera  <->  [x, -y, -z] mesh_cam
# ---------------------------------------------------------------------


def mesh_cam_to_camera(points):
    points = np.asarray(points, dtype=np.float32).copy()
    points[..., 1] *= -1.0
    points[..., 2] *= -1.0
    return points


def camera_to_mesh_cam(points):
    points = np.asarray(points, dtype=np.float32).copy()
    points[..., 1] *= -1.0
    points[..., 2] *= -1.0
    return points


def load_obj_vertices(obj_path, expected_vertices=6890):
    obj_path = Path(obj_path).expanduser()
    vertices = []

    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                p = line.strip().split()
                vertices.append([float(p[1]), float(p[2]), float(p[3])])

    vertices = np.asarray(vertices, dtype=np.float32)

    if expected_vertices is not None and vertices.shape != (expected_vertices, 3):
        raise ValueError(
            f"Expected OBJ vertices shape ({expected_vertices}, 3), "
            f"got {vertices.shape} from {obj_path}"
        )

    return vertices


def save_obj_like_template(out_path, vertices_mesh_cam, template_obj, color=(0.65098039, 0.74117647, 0.85882353)):
    """
    Save optimized vertices as OBJ, copying faces from an existing TokenHMR OBJ.

    vertices_mesh_cam should already be in exported OBJ / EgoHumans mesh_cam convention.
    """
    out_path = Path(out_path).expanduser()
    template_obj = Path(template_obj).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    face_lines = []
    with open(template_obj, "r") as f:
        for line in f:
            if line.startswith("f "):
                face_lines.append(line)

    with open(out_path, "w") as f:
        for v in vertices_mesh_cam:
            f.write(
                f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} "
                f"{color[0]:.8f} {color[1]:.8f} {color[2]:.8f}\n"
            )

        for line in face_lines:
            f.write(line)

    print(f"[Saved] optimized OBJ: {out_path}")


def patch_numpy_for_old_smpl():
    # Old SMPL pickle/chumpy compatibility.
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
    smpl_model_dir = Path(smpl_model_dir).expanduser()

    if smpl_model_dir.is_dir():
        smpl_path = smpl_model_dir / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    else:
        smpl_path = smpl_model_dir

    if not smpl_path.exists():
        raise FileNotFoundError(f"SMPL model file not found: {smpl_path}")

    patch_numpy_for_old_smpl()

    with open(smpl_path, "rb") as f:
        smpl_data = pickle.load(f, encoding="latin1")

    J_regressor = smpl_data["J_regressor"]

    if hasattr(J_regressor, "toarray"):
        J_regressor = J_regressor.toarray()

    J_regressor = np.asarray(J_regressor, dtype=np.float32)

    if J_regressor.shape[1] != 6890:
        raise ValueError(f"Expected J_regressor shape (?, 6890), got {J_regressor.shape}")

    print("[SMPL] J_regressor:", J_regressor.shape)
    return J_regressor


def load_head_target_file(path):
    """
    Supports a simple text file containing:

        x y z

    Lines starting with # are ignored.
    """
    path = Path(path).expanduser()

    vals = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.replace(",", " ").split()
            vals.extend([float(x) for x in parts])

    if len(vals) < 3:
        raise ValueError(f"Could not read x y z from head target file: {path}")

    return np.asarray(vals[:3], dtype=np.float32)


def load_keypoints_2d_json(path):
    """
    Optional JSON format:

    {
      "joint_ids": [0, 1, 2, 3, ...],
      "keypoints": [
        [u, v, confidence],
        [u, v, confidence],
        ...
      ]
    }

    The joint_ids should refer to SMPL joint indices from the loaded SMPL regressor.
    For minimal v1, you can skip 2D keypoints entirely.
    """
    path = Path(path).expanduser()

    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        keypoints = np.asarray(data["keypoints"], dtype=np.float32)
        joint_ids = np.asarray(data.get("joint_ids", list(range(len(keypoints)))), dtype=np.int64)
    else:
        keypoints = np.asarray(data, dtype=np.float32)
        joint_ids = np.arange(len(keypoints), dtype=np.int64)

    if keypoints.ndim != 2 or keypoints.shape[1] not in [2, 3]:
        raise ValueError(f"Expected keypoints shape (J,2) or (J,3), got {keypoints.shape}")

    if keypoints.shape[1] == 2:
        conf = np.ones((keypoints.shape[0], 1), dtype=np.float32)
        keypoints = np.concatenate([keypoints, conf], axis=1)

    if len(joint_ids) != len(keypoints):
        raise ValueError("joint_ids length must match keypoints length.")

    return joint_ids, keypoints


def project_points_camera(points_camera, fx, fy, cx, cy, eps=1e-6):
    """
    Perspective projection.

    points_camera: torch tensor, shape (N, 3), positive Z.
    """
    x = points_camera[:, 0]
    y = points_camera[:, 1]
    z = torch.clamp(points_camera[:, 2], min=eps)

    u = fx * x / z + cx
    v = fy * y / z + cy

    return torch.stack([u, v], dim=-1)


def make_translation(opt_param, init_T, opt_mode):
    """
    opt_mode:
      xyz: optimize Tx, Ty, Tz
      z:   optimize only Tz, keep Tx/Ty from init
    """
    if opt_mode == "xyz":
        return opt_param

    if opt_mode == "z":
        return torch.stack([init_T[0], init_T[1], opt_param[0]])

    raise ValueError(f"Unknown opt_mode: {opt_mode}")


def compute_metrics(pred_joints_camera, T, head_index, head_target_camera=None):
    out = {}

    pred_joints_full = pred_joints_camera + T[None, :]
    pred_head = pred_joints_full[head_index]

    out["pred_head_camera"] = pred_head.detach().cpu().numpy()

    if head_target_camera is not None:
        head_err = torch.linalg.norm(pred_head - head_target_camera)
        out["head_error_m"] = float(head_err.detach().cpu().item())

    return out



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_npz", required=True, help="TokenHMR NPZ saved by demo.py, e.g. 00001_0_tkhmr.npz")
    parser.add_argument("--pred_obj_template", required=True, help="Existing TokenHMR OBJ used only for copying faces.")
    parser.add_argument("--smpl_model_dir", required=True, help="Folder containing basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")
    parser.add_argument("--out_obj", required=True, help="Output optimized OBJ path.")

    # Head target from EgoHumans egocentric camera pose.
    parser.add_argument("--head_target_file", default=None, help="Text file containing x y z head target.")
    parser.add_argument(
        "--head_target_coord",
        choices=["camera", "mesh_cam"],
        default="mesh_cam",
        help=(
            "Coordinate convention of head target. "
            "Use mesh_cam if target is in EgoHumans mesh_cam / GT OBJ convention. "
            "Use camera if target is already TokenHMR positive-Z camera convention."
        ),
    )

    # Debug-only option. Do not use for final method.
    parser.add_argument(
        "--gt_head_debug_obj",
        default=None,
        help="DEBUG ONLY: use GT mesh head joint as head target to verify optimizer.",
    )

    parser.add_argument("--head_index", type=int, default=15, help="SMPL head joint index. Usually 15.")
    parser.add_argument("--root_index", type=int, default=0, help="SMPL root/pelvis joint index. Usually 0.")

    # Optional offset if your target is Aria camera center rather than SMPL head joint.
    parser.add_argument(
        "--head_target_offset_camera",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help=(
            "Optional offset added to the head target in TokenHMR camera coords. "
            "Use this if the target is Aria camera center and you know the camera-to-head offset."
        ),
    )

    # Optional 2D reprojection term.
    parser.add_argument("--keypoints_2d_json", default=None, help="Optional 2D keypoints JSON.")
    parser.add_argument("--fx", type=float, default=1915.346061677637)
    parser.add_argument("--fy", type=float, default=1915.5916704165572)
    parser.add_argument("--cx", type=float, default=1920.0)
    parser.add_argument("--cy", type=float, default=1080.0)

    # Optimization settings.
    parser.add_argument("--opt_mode", choices=["xyz", "z"], default="xyz")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.03)

    parser.add_argument(
        "--optimizer",
        choices=["adam", "lbfgs"],
        default="adam",
        help="Optimizer to use: adam or lbfgs.",
    )

    parser.add_argument(
        "--lbfgs_max_iter",
        type=int,
        default=50,
        help="Internal max_iter for torch.optim.LBFGS.",
    )

    # Loss weights.
    parser.add_argument("--w_head", type=float, default=50.0)
    parser.add_argument("--w_2d", type=float, default=1.0)
    parser.add_argument("--w_trans_prior", type=float, default=0.05)
    parser.add_argument("--w_z_positive", type=float, default=10.0)

    args = parser.parse_args()

    pred_npz = Path(args.pred_npz).expanduser()
    pred_obj_template = Path(args.pred_obj_template).expanduser()
    out_obj = Path(args.out_obj).expanduser()

    data = np.load(pred_npz)

    if "pred_vertices_local" not in data.files:
        raise KeyError(
            f"{pred_npz} does not contain pred_vertices_local. "
            "Add the NPZ-saving block to demo.py and rerun TokenHMR."
        )

    pred_vertices_local = data["pred_vertices_local"].astype(np.float32)

    if pred_vertices_local.shape != (6890, 3):
        raise ValueError(f"Expected pred_vertices_local shape (6890,3), got {pred_vertices_local.shape}")

    if "pred_cam_t_full" in data.files:
        init_T = data["pred_cam_t_full"].astype(np.float32)
    else:
        raise KeyError(
            f"{pred_npz} does not contain pred_cam_t_full. "
            "Add the NPZ-saving block to demo.py and rerun TokenHMR."
        )

    if init_T.shape != (3,):
        init_T = init_T.reshape(3).astype(np.float32)

    J_regressor = load_smpl_joint_regressor(args.smpl_model_dir)
    pred_joints_local = J_regressor @ pred_vertices_local

    if pred_joints_local.shape[0] <= args.head_index:
        raise ValueError(
            f"head_index={args.head_index} is invalid for joints shape {pred_joints_local.shape}"
        )

    # ------------------------------------------------------------
    # Load head target
    # ------------------------------------------------------------
    head_target_camera_np = None

    if args.head_target_file is not None:
        head_target_np = load_head_target_file(args.head_target_file)

        if args.head_target_coord == "mesh_cam":
            head_target_camera_np = mesh_cam_to_camera(head_target_np)
        else:
            head_target_camera_np = head_target_np.astype(np.float32)

        print("[Head target] loaded from:", args.head_target_file)

    elif args.gt_head_debug_obj is not None:
        # DEBUG ONLY. This uses GT mesh, so do not use it for final method.
        gt_vertices_mesh_cam = load_obj_vertices(args.gt_head_debug_obj)
        gt_joints_mesh_cam = J_regressor @ gt_vertices_mesh_cam
        head_target_mesh_cam = gt_joints_mesh_cam[args.head_index]
        head_target_camera_np = mesh_cam_to_camera(head_target_mesh_cam)

        print("[DEBUG] using GT mesh head joint as target.")
        print("[DEBUG] do NOT use --gt_head_debug_obj for final method.")

    # Optional offset.
    if head_target_camera_np is not None:
        head_target_camera_np = head_target_camera_np + np.asarray(args.head_target_offset_camera, dtype=np.float32)

    # ------------------------------------------------------------
    # Optional 2D keypoints
    # ------------------------------------------------------------
    kpt_joint_ids_np = None
    keypoints_2d_np = None

    if args.keypoints_2d_json is not None:
        kpt_joint_ids_np, keypoints_2d_np = load_keypoints_2d_json(args.keypoints_2d_json)
        print("[2D] loaded keypoints:", keypoints_2d_np.shape)

    if head_target_camera_np is None and keypoints_2d_np is None:
        raise ValueError(
            "No optimization target was provided. "
            "Provide --head_target_file, --gt_head_debug_obj, or --keypoints_2d_json."
        )

    # ------------------------------------------------------------
    # Torch tensors
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pred_vertices_local_t = torch.tensor(pred_vertices_local, dtype=torch.float32, device=device)
    pred_joints_local_t = torch.tensor(pred_joints_local, dtype=torch.float32, device=device)
    init_T_t = torch.tensor(init_T, dtype=torch.float32, device=device)

    if head_target_camera_np is not None:
        head_target_camera_t = torch.tensor(head_target_camera_np, dtype=torch.float32, device=device)
    else:
        head_target_camera_t = None

    if keypoints_2d_np is not None:
        kpt_joint_ids_t = torch.tensor(kpt_joint_ids_np, dtype=torch.long, device=device)
        keypoints_2d_t = torch.tensor(keypoints_2d_np, dtype=torch.float32, device=device)
    else:
        kpt_joint_ids_t = None
        keypoints_2d_t = None

    fx = torch.tensor(float(args.fx), dtype=torch.float32, device=device)
    fy = torch.tensor(float(args.fy), dtype=torch.float32, device=device)
    cx = torch.tensor(float(args.cx), dtype=torch.float32, device=device)
    cy = torch.tensor(float(args.cy), dtype=torch.float32, device=device)

    # ------------------------------------------------------------
    # Initialize optimization variable
    # ------------------------------------------------------------
    if args.opt_mode == "xyz":
        opt_param = torch.tensor(init_T, dtype=torch.float32, device=device, requires_grad=True)
    elif args.opt_mode == "z":
        opt_param = torch.tensor([init_T[2]], dtype=torch.float32, device=device, requires_grad=True)
    else:
        raise ValueError(args.opt_mode)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam([opt_param], lr=args.lr)

    elif args.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(
            [opt_param],
            lr=args.lr,
            max_iter=args.lbfgs_max_iter,
            line_search_fn="strong_wolfe",
        )
    else:
        raise ValueError(args.optimizer)

    print("\n========== SMPLify-v1 translation-only ==========")
    print("pred_npz:", pred_npz)
    print("template OBJ:", pred_obj_template)
    print("out_obj:", out_obj)
    print("initial T camera:", init_T)
    print("opt_mode:", args.opt_mode)
    print("head_index:", args.head_index)

    if head_target_camera_np is not None:
        print("head target camera:", head_target_camera_np)
        print("head target mesh_cam/export:", camera_to_mesh_cam(head_target_camera_np))

    with torch.no_grad():
        T0 = make_translation(opt_param, init_T_t, args.opt_mode)
        m0 = compute_metrics(
            pred_joints_local_t,
            T0,
            args.head_index,
            head_target_camera_t,
        )
        print("initial predicted head camera:", m0["pred_head_camera"])
        if "head_error_m" in m0:
            print(f"initial head error: {m0['head_error_m'] * 1000.0:.2f} mm")

    # ------------------------------------------------------------
    # Optimization loop
    # ------------------------------------------------------------
    print("optimizer:", args.optimizer)

    if args.optimizer == "adam":
        for it in range(args.iters):
            optimizer.zero_grad()

            T = make_translation(opt_param, init_T_t, args.opt_mode)
            loss, loss_terms = compute_total_loss(
                T,
                args=args,
                device=device,
                pred_joints_local_t=pred_joints_local_t,
                pred_vertices_local_t=pred_vertices_local_t,
                init_T_t=init_T_t,
                head_target_camera_t=head_target_camera_t,
                keypoints_2d_t=keypoints_2d_t,
                kpt_joint_ids_t=kpt_joint_ids_t,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                project_points_camera=project_points_camera,
            )

            loss.backward()
            optimizer.step()

            if it % 50 == 0 or it == args.iters - 1:
                with torch.no_grad():
                    T_now = make_translation(opt_param, init_T_t, args.opt_mode)
                    metrics_now = compute_metrics(
                        pred_joints_local_t,
                        T_now,
                        args.head_index,
                        head_target_camera_t,
                    )

                    msg = (
                        f"iter {it:04d} | "
                        f"loss={loss.item():.6f} | "
                        f"T={T_now.detach().cpu().numpy()}"
                    )

                    if "head_error_m" in metrics_now:
                        msg += f" | head_err={metrics_now['head_error_m'] * 1000.0:.2f}mm"

                    if keypoints_2d_t is not None:
                        joints_sel = pred_joints_local_t[kpt_joint_ids_t] + T_now[None, :]
                        proj = project_points_camera(joints_sel, fx, fy, cx, cy)
                        diff_px = proj - keypoints_2d_t[:, :2]
                        conf = keypoints_2d_t[:, 2].clamp(min=0.0)
                        px_err = torch.sum(conf * torch.linalg.norm(diff_px, dim=1)) / torch.clamp(conf.sum(), min=1.0)
                        msg += f" | reproj={px_err.item():.2f}px"

                    print(msg)

    elif args.optimizer == "lbfgs":
        lbfgs_iter = {"count": 0}

        def closure():
            optimizer.zero_grad()

            T = make_translation(opt_param, init_T_t, args.opt_mode)
            loss, loss_terms = compute_total_loss(
                T,
                args=args,
                device=device,
                pred_joints_local_t=pred_joints_local_t,
                pred_vertices_local_t=pred_vertices_local_t,
                init_T_t=init_T_t,
                head_target_camera_t=head_target_camera_t,
                keypoints_2d_t=keypoints_2d_t,
                kpt_joint_ids_t=kpt_joint_ids_t,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                project_points_camera=project_points_camera,
            )

            loss.backward()

            lbfgs_iter["count"] += 1

            if lbfgs_iter["count"] % 10 == 0:
                with torch.no_grad():
                    T_now = make_translation(opt_param, init_T_t, args.opt_mode)
                    metrics_now = compute_metrics(
                        pred_joints_local_t,
                        T_now,
                        args.head_index,
                        head_target_camera_t,
                    )

                    msg = (
                        f"lbfgs_eval {lbfgs_iter['count']:04d} | "
                        f"loss={loss.item():.6f} | "
                        f"T={T_now.detach().cpu().numpy()}"
                    )

                    if "head_error_m" in metrics_now:
                        msg += f" | head_err={metrics_now['head_error_m'] * 1000.0:.2f}mm"

                    print(msg)

            return loss

        optimizer.step(closure)

        with torch.no_grad():
            T_now = make_translation(opt_param, init_T_t, args.opt_mode)
            loss, _ = compute_total_loss(
                T_now,
                args=args,
                device=device,
                pred_joints_local_t=pred_joints_local_t,
                pred_vertices_local_t=pred_vertices_local_t,
                init_T_t=init_T_t,
                head_target_camera_t=head_target_camera_t,
                keypoints_2d_t=keypoints_2d_t,
                kpt_joint_ids_t=kpt_joint_ids_t,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                project_points_camera=project_points_camera,
            )
            msg = (
                f"lbfgs_final | "
                f"loss={loss.item():.6f} | "
                f"T={T_now.detach().cpu().numpy()}"
            )
            metrics_now = compute_metrics(
                pred_joints_local_t,
                T_now,
                args.head_index,
                head_target_camera_t,
            )

            if "head_error_m" in metrics_now:
                msg += f" | head_err={metrics_now['head_error_m'] * 1000.0:.2f}mm"

            print(msg)

    else:
        raise ValueError(args.optimizer)

    # ------------------------------------------------------------
    # Save optimized result
    # ------------------------------------------------------------
    with torch.no_grad():
        T_final = make_translation(opt_param, init_T_t, args.opt_mode)
        pred_vertices_full_camera = pred_vertices_local_t + T_final[None, :]
        pred_vertices_full_camera_np = pred_vertices_full_camera.detach().cpu().numpy()

    pred_vertices_export_np = camera_to_mesh_cam(pred_vertices_full_camera_np)

    save_obj_like_template(
        out_obj,
        pred_vertices_export_np,
        pred_obj_template,
    )

    out_npz = out_obj.with_suffix(".npz")
    np.savez(
        out_npz,
        pred_vertices_local=pred_vertices_local.astype(np.float32),
        pred_vertices_full_camera=pred_vertices_full_camera_np.astype(np.float32),
        pred_vertices_export=pred_vertices_export_np.astype(np.float32),
        init_T=init_T.astype(np.float32),
        T_opt=T_final.detach().cpu().numpy().astype(np.float32),
        head_target_camera=None if head_target_camera_np is None else head_target_camera_np.astype(np.float32),
        head_target_export=None if head_target_camera_np is None else camera_to_mesh_cam(head_target_camera_np).astype(np.float32),
    )

    print(f"[Saved] optimized NPZ: {out_npz}")

    print("\n========== Final summary ==========")
    print("Initial T camera:", init_T)
    print("Optimized T camera:", T_final.detach().cpu().numpy())
    print("Optimized T mesh_cam/export:", camera_to_mesh_cam(T_final.detach().cpu().numpy()))

    if head_target_camera_np is not None:
        final_metrics = compute_metrics(
            pred_joints_local_t,
            T_final,
            args.head_index,
            head_target_camera_t,
        )
        print(f"Final head error: {final_metrics['head_error_m'] * 1000.0:.2f} mm")

    print("===================================")


if __name__ == "__main__":
    main()