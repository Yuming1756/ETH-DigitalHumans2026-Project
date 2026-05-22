#!/usr/bin/env python3

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import smplx


# ---------------------------------------------------------------------
# Coordinate convention
# ---------------------------------------------------------------------
# Positive-Z camera coordinate:
#     [x, y, z]
#
# Exported OBJ / EgoHumans mesh_cam-like coordinate:
#     [x, -y, -z]
# ---------------------------------------------------------------------


def camera_to_mesh_cam(points):
    points = np.asarray(points, dtype=np.float32).copy()
    points[..., 1] *= -1.0
    points[..., 2] *= -1.0
    return points


def mesh_cam_to_camera(points):
    points = np.asarray(points, dtype=np.float32).copy()
    points[..., 1] *= -1.0
    points[..., 2] *= -1.0
    return points


def patch_numpy_for_old_smpl():
    for name, typ in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }.items():
        try:
            getattr(np, name)
        except AttributeError:
            setattr(np, name, typ)


def prepare_smplx_model_path(smpl_model_dir):
    """
    smplx.create(model_type='smpl') expects:

        model_path/
            smpl/
                SMPL_NEUTRAL.pkl

    Your project often has:

        smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
    """
    smpl_model_dir = Path(smpl_model_dir).expanduser().resolve()

    expected = smpl_model_dir / "smpl" / "SMPL_NEUTRAL.pkl"
    basic = smpl_model_dir / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"

    if expected.exists():
        return smpl_model_dir

    if not basic.exists():
        raise FileNotFoundError(
            f"Could not find SMPL model. Expected either:\n"
            f"  {expected}\n"
            f"or:\n"
            f"  {basic}"
        )

    expected.parent.mkdir(parents=True, exist_ok=True)

    if expected.is_symlink() and not expected.exists():
        expected.unlink()

    if not expected.exists():
        os.symlink(str(basic.resolve()), str(expected))
        print(f"[SMPL] Created symlink:\n  {expected} -> {basic.resolve()}")

    return smpl_model_dir


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

    J = smpl_data["J_regressor"]

    if hasattr(J, "toarray"):
        J = J.toarray()

    J = np.asarray(J, dtype=np.float32)

    if J.shape[1] != 6890:
        raise ValueError(f"Expected J_regressor shape (?, 6890), got {J.shape}")

    print("[SMPL] J_regressor:", J.shape)
    return J


def save_obj(out_path, vertices, faces, color=(0.65098039, 0.74117647, 0.85882353)):
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for v in vertices:
            f.write(
                f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} "
                f"{color[0]:.8f} {color[1]:.8f} {color[2]:.8f}\n"
            )

        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    print(f"[Saved OBJ] {out_path}")


def load_head_target_file(path):
    path = Path(path).expanduser()

    vals = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            vals.extend([float(x) for x in line.replace(",", " ").split()])

    if len(vals) < 3:
        raise ValueError(f"Could not read xyz from {path}")

    return np.asarray(vals[:3], dtype=np.float32)

def load_vector_file(path):
    """
    Load a 3D direction vector from a text file.

    Expected format:
        x y z

    This should be the Aria forward direction in cam01 positive-Z camera coordinates.
    The vector is normalized before returning.
    """
    path = Path(path).expanduser()

    vals = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            vals.extend([float(x) for x in line.replace(",", " ").split()])

    if len(vals) < 3:
        raise ValueError(f"Could not read xyz vector from {path}")

    v = np.asarray(vals[:3], dtype=np.float32)
    norm = np.linalg.norm(v)

    if norm < 1e-8:
        raise ValueError(f"Cannot normalize near-zero vector from {path}: {v}")

    return v / norm


def load_keypoints_2d_json(path):
    path = Path(path).expanduser()

    with open(path, "r") as f:
        data = json.load(f)

    # ------------------------------------------------------------
    # Body joint keypoints
    # ------------------------------------------------------------
    keypoints = np.asarray(data["keypoints"], dtype=np.float32)
    joint_ids = np.asarray(data["joint_ids"], dtype=np.int64)

    if keypoints.ndim != 2 or keypoints.shape[1] != 3:
        raise ValueError(f"Expected keypoints shape (N,3), got {keypoints.shape}")

    if len(joint_ids) != len(keypoints):
        raise ValueError("joint_ids length must match keypoints length.")

    # ------------------------------------------------------------
    # Optional face/head vertex keypoints
    # ------------------------------------------------------------
    if "vertex_ids" in data and "vertex_keypoints" in data:
        vertex_ids = np.asarray(data["vertex_ids"], dtype=np.int64)
        vertex_keypoints = np.asarray(data["vertex_keypoints"], dtype=np.float32)

        if len(vertex_ids) == 0:
            vertex_ids = None
            vertex_keypoints = None
        else:
            if vertex_keypoints.ndim != 2 or vertex_keypoints.shape[1] != 3:
                raise ValueError(
                    f"Expected vertex_keypoints shape (M,3), got {vertex_keypoints.shape}"
                )

            if len(vertex_ids) != len(vertex_keypoints):
                raise ValueError("vertex_ids length must match vertex_keypoints length.")
    else:
        vertex_ids = None
        vertex_keypoints = None

    return joint_ids, keypoints, vertex_ids, vertex_keypoints


def project_points_camera(points_camera, fx, fy, cx, cy, eps=1e-6):
    x = points_camera[:, 0]
    y = points_camera[:, 1]
    z = torch.clamp(points_camera[:, 2], min=eps)

    u = fx * x / z + cx
    v = fy * y / z + cy

    return torch.stack([u, v], dim=-1)


def huber_loss(residual, delta):
    return torch.where(
        residual <= delta,
        0.5 * residual ** 2,
        delta * (residual - 0.5 * delta),
    )


# ---------------------------------------------------------------------
# Rotation representation
# ---------------------------------------------------------------------
# We optimize 6D rotations, not raw 3x3 matrices.
#
# Why:
#   Saved TokenHMR / 4D-Humans poses are rotation matrices.
#   If we directly optimize 3x3 matrices, they may stop being valid rotations.
#   6D rotation representation is easier and stable.
# ---------------------------------------------------------------------


def rotmat_to_6d_np(R):
    """
    R shape:
        (..., 3, 3)

    Returns:
        (..., 6), first two rotation columns concatenated.
    """
    R = np.asarray(R, dtype=np.float32)

    col1 = R[..., :, 0]
    col2 = R[..., :, 1]

    return np.concatenate([col1, col2], axis=-1).astype(np.float32)


def rot6d_to_rotmat(x):
    """
    x shape:
        (..., 6)

    Returns:
        (..., 3, 3), valid rotation matrices.
    """
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]

    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack as columns of rotation matrix.
    return torch.stack([b1, b2, b3], dim=-1)


def get_param(data, key, fallback=None):
    if key in data.files:
        return data[key]
    if fallback is not None and fallback in data.files:
        return data[fallback]
    raise KeyError(f"Missing {key}" + (f" / {fallback}" if fallback else ""))


def compute_reprojection_loss(
    joints24_camera,
    kpt_joint_ids_t,
    keypoints_2d_t,
    fx,
    fy,
    cx,
    cy,
    huber_delta_2d,
):
    joints_sel = joints24_camera[kpt_joint_ids_t]
    proj = project_points_camera(joints_sel, fx, fy, cx, cy)

    diff_px = proj - keypoints_2d_t[:, :2]
    conf = keypoints_2d_t[:, 2].clamp(min=0.0)

    diff_norm = torch.stack(
        [diff_px[:, 0] / fx, diff_px[:, 1] / fy],
        dim=-1,
    )

    residual = torch.linalg.norm(diff_norm, dim=1)
    loss_per_joint = huber_loss(residual, huber_delta_2d)

    loss_2d = torch.sum(conf * loss_per_joint) / torch.clamp(conf.sum(), min=1.0)

    reproj_px = torch.sum(
        conf * torch.linalg.norm(diff_px, dim=1)
    ) / torch.clamp(conf.sum(), min=1.0)

    return loss_2d, reproj_px

def compute_vertex_reprojection_loss(
    vertices_camera,
    vertex_ids_t,
    vertex_keypoints_2d_t,
    fx,
    fy,
    cx,
    cy,
    huber_delta_2d,
):
    """
    Reprojection loss for SMPL surface vertices.

    vertices_camera:
        (6890, 3), positive-Z camera coordinates

    vertex_ids_t:
        (M,), SMPL vertex indices

    vertex_keypoints_2d_t:
        (M, 3), [u, v, confidence]
    """
    vertices_sel = vertices_camera[vertex_ids_t]
    proj = project_points_camera(vertices_sel, fx, fy, cx, cy)

    diff_px = proj - vertex_keypoints_2d_t[:, :2]
    conf = vertex_keypoints_2d_t[:, 2].clamp(min=0.0)

    diff_norm = torch.stack(
        [diff_px[:, 0] / fx, diff_px[:, 1] / fy],
        dim=-1,
    )

    residual = torch.linalg.norm(diff_norm, dim=1)
    loss_per_vertex = huber_loss(residual, huber_delta_2d)

    loss_vertex_2d = torch.sum(conf * loss_per_vertex) / torch.clamp(conf.sum(), min=1.0)

    reproj_px = torch.sum(
        conf * torch.linalg.norm(diff_px, dim=1)
    ) / torch.clamp(conf.sum(), min=1.0)

    return loss_vertex_2d, reproj_px

def compute_aria_orientation_loss(
    vertices_camera,
    nose_vertex,
    left_eye_vertex,
    right_eye_vertex,
    aria_forward_t,
):
    """
    Align SMPL face-forward direction with Aria camera forward direction.

    vertices_camera:
        (6890, 3), SMPL vertices in positive-Z camera coordinates.

    nose_vertex / left_eye_vertex / right_eye_vertex:
        SMPL vertex indices defining the face-forward direction.

    aria_forward_t:
        (3,), Aria +Z forward direction in cam01 positive-Z camera coordinates.

    SMPL face-forward direction:
        nose - eye_center

    Loss:
        1 - cosine_similarity(SMPL_forward, Aria_forward)
    """
    nose = vertices_camera[nose_vertex]
    left_eye = vertices_camera[left_eye_vertex]
    right_eye = vertices_camera[right_eye_vertex]

    eye_center = 0.5 * (left_eye + right_eye)

    smpl_forward = nose - eye_center
    smpl_forward = smpl_forward / torch.clamp(
        torch.linalg.norm(smpl_forward),
        min=1e-8,
    )

    aria_forward = aria_forward_t / torch.clamp(
        torch.linalg.norm(aria_forward_t),
        min=1e-8,
    )

    cosine = torch.sum(smpl_forward * aria_forward).clamp(-1.0, 1.0)

    # 0 if perfectly aligned, 2 if exactly opposite.
    loss_orient = 1.0 - cosine

    return loss_orient, cosine.detach(), smpl_forward.detach()

def compute_aria_up_loss(
    joints24_camera,
    neck_index,
    head_index,
    aria_up_t,
):
    """
    Align SMPL head-neck upward direction with Aria camera upward direction.

    joints24_camera:
        (24, 3), SMPL joints in positive-Z camera coordinates.

    SMPL up direction:
        head_joint - neck_joint

    Loss:
        1 - cosine_similarity(SMPL_up, Aria_up)
    """
    neck = joints24_camera[neck_index]
    head = joints24_camera[head_index]

    smpl_up = head - neck
    smpl_up = smpl_up / torch.clamp(
        torch.linalg.norm(smpl_up),
        min=1e-8,
    )

    aria_up = aria_up_t / torch.clamp(
        torch.linalg.norm(aria_up_t),
        min=1e-8,
    )

    cosine = torch.sum(smpl_up * aria_up).clamp(-1.0, 1.0)

    # 0 if perfectly aligned, 2 if exactly opposite.
    loss_up = 1.0 - cosine

    return loss_up, cosine.detach(), smpl_up.detach()

def compute_head_up_forward_perp_loss(
    joints24_camera,
    neck_index,
    head_index,
    aria_forward_t,
):
    """
    Encourage SMPL head-neck direction to be perpendicular to Aria forward.

    SMPL_up = normalize(head_joint - neck_joint)
    Loss = dot(SMPL_up, Aria_forward)^2
    """
    neck = joints24_camera[neck_index]
    head = joints24_camera[head_index]

    smpl_up = head - neck
    smpl_up = smpl_up / torch.clamp(
        torch.linalg.norm(smpl_up),
        min=1e-8,
    )

    aria_forward = aria_forward_t / torch.clamp(
        torch.linalg.norm(aria_forward_t),
        min=1e-8,
    )

    dot_val = torch.sum(smpl_up * aria_forward).clamp(-1.0, 1.0)

    loss_perp = dot_val ** 2

    return loss_perp, dot_val.detach(), smpl_up.detach()


def summarize_vertices(name, vertices):
    print(f"\n========== {name} ==========")
    print("shape:", vertices.shape)
    print("center:", vertices.mean(axis=0))
    print("min:   ", vertices.min(axis=0))
    print("max:   ", vertices.max(axis=0))
    print("extent:", vertices.max(axis=0) - vertices.min(axis=0))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--stage1_npz", required=True, help="Stage-1 NPZ containing T_opt and SMPL params")
    parser.add_argument("--keypoints_2d_json", required=True)
    parser.add_argument("--head_target_file", default=None)
    parser.add_argument("--head_target_coord", choices=["camera", "mesh_cam"], default="camera")
    parser.add_argument("--smpl_model_dir", required=True)
    parser.add_argument("--out_obj", required=True)
    parser.add_argument("--out_npz", default=None)

    parser.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    parser.add_argument("--fx", type=float, default=1915.346061677637)
    parser.add_argument("--fy", type=float, default=1915.5916704165572)
    parser.add_argument("--cx", type=float, default=1920.0)
    parser.add_argument("--cy", type=float, default=1080.0)

    parser.add_argument("--optimizer", choices=["adam", "lbfgs"], default="lbfgs")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--lbfgs_max_iter", type=int, default=50)

    # What to optimize
    parser.add_argument("--freeze_betas", action="store_true", help="Freeze betas during Stage 2")
    parser.add_argument("--freeze_body_pose", action="store_true", help="Freeze body pose during Stage 2")
    parser.add_argument("--freeze_global_orient", action="store_true", help="Freeze global orient during Stage 2")

    # Loss weights
    parser.add_argument("--w_2d", type=float, default=10.0)
    parser.add_argument(
        "--w_face_2d",
        type=float,
        default=10.0,
        help="Weight for 2D reprojection loss on SMPL face/head vertices.",
    )
    parser.add_argument("--huber_delta_2d", type=float, default=0.05)
    # Aria orientation loss
    parser.add_argument(
        "--aria_forward_file",
        default=None,
        help=(
            "Text file containing Aria +Z forward direction in cam01 positive-Z "
            "camera coordinates."
        ),
    )

    parser.add_argument(
        "--w_aria_orient",
        type=float,
        default=0.0,
        help="Weight for Aria orientation loss.",
    )

    parser.add_argument(
        "--aria_up_file",
        default=None,
        help=(
            "Text file containing Aria up direction in cam01 positive-Z "
            "camera coordinates. Usually extracted from Aria local -Y or +Y."
        ),
    )

    parser.add_argument(
        "--w_aria_up",
        type=float,
        default=0.0,
        help="Weight for Aria up-vector orientation loss.",
    )

    parser.add_argument(
        "--neck_index",
        type=int,
        default=12,
        help="SMPL neck joint index used for up-vector loss. Usually 12.",
    )
    
    parser.add_argument(
        "--nose_vertex",
        type=int,
        default=-1,
        help="SMPL vertex index for nose / nose bridge.",
    )

    parser.add_argument(
        "--left_eye_vertex",
        type=int,
        default=-1,
        help="SMPL vertex index for anatomical left eye / eye corner.",
    )

    parser.add_argument(
        "--right_eye_vertex",
        type=int,
        default=-1,
        help="SMPL vertex index for anatomical right eye / eye corner.",
    )

    parser.add_argument(
        "--w_head_up_perp",
        type=float,
        default=0.0,
        help="Weight for perpendicularity loss between SMPL head-neck vector and Aria forward vector.",
    )

    parser.add_argument(
        "--train_body_pose_indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional body_pose indices to train. "
            "body_pose[i] corresponds to SMPL joint i+1. "
            "Example: --train_body_pose_indices 11 14 trains neck/head only."
        ),
    )
    

    # Mild Stage-1 style head loss
    parser.add_argument("--w_head", type=float, default=1.0)
    parser.add_argument("--proxy_radius", type=float, default=0.18)
    parser.add_argument("--head_index", type=int, default=15)
    parser.add_argument(
        "--target_vertex_index",
        type=int,
        default=None,
        help="Optional SMPL vertex index near glasses/forehead. If set, use this instead of head joint.",
    )

    # Priors
    parser.add_argument("--w_orient_prior", type=float, default=5.0)
    parser.add_argument("--w_pose_prior", type=float, default=1.0)
    parser.add_argument("--w_betas_prior", type=float, default=10.0)
    parser.add_argument("--w_betas_l2", type=float, default=0.1)
    parser.add_argument("--w_z_positive", type=float, default=5.0)

    args = parser.parse_args()

    stage1_npz = Path(args.stage1_npz).expanduser()
    out_obj = Path(args.out_obj).expanduser()
    out_npz = Path(args.out_npz).expanduser() if args.out_npz else out_obj.with_suffix(".npz")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    print("============================================================")
    print("SMPLify Stage 2: optimize orient/body pose/betas with fixed T_opt")
    print("stage1_npz:", stage1_npz)
    print("out_obj:", out_obj)
    print("out_npz:", out_npz)
    print("device:", device)
    print("============================================================")

    data = np.load(stage1_npz)

    # ------------------------------------------------------------
    # Required SMPL params
    # ------------------------------------------------------------
    global_orient_np = get_param(data, "global_orient", "pred_smpl_global_orient").astype(np.float32)
    body_pose_np = get_param(data, "body_pose", "pred_smpl_body_pose").astype(np.float32)
    betas_np = get_param(data, "betas", "pred_smpl_betas").astype(np.float32).reshape(10)

    if "T_opt" not in data.files:
        raise KeyError("Stage-2 requires T_opt in the stage1_npz.")

    T_fixed_np = data["T_opt"].astype(np.float32).reshape(3)

    print("\nLoaded params:")
    print("global_orient:", global_orient_np.shape)
    print("body_pose:", body_pose_np.shape)
    print("betas:", betas_np.shape)
    print("T_fixed from T_opt:", T_fixed_np)

    # Expected saved format: rotation matrices.
    if global_orient_np.shape != (1, 3, 3):
        raise ValueError(f"Expected global_orient shape (1,3,3), got {global_orient_np.shape}")
    if body_pose_np.shape != (23, 3, 3):
        raise ValueError(f"Expected body_pose shape (23,3,3), got {body_pose_np.shape}")

    # Convert initial rotation matrices to 6D.
    global_orient_6d_np = rotmat_to_6d_np(global_orient_np)       # (1, 6)
    body_pose_6d_np = rotmat_to_6d_np(body_pose_np)               # (23, 6)

    global_orient_6d_init = torch.tensor(global_orient_6d_np, dtype=torch.float32, device=device)
    body_pose_6d_init = torch.tensor(body_pose_6d_np, dtype=torch.float32, device=device)
    betas_init = torch.tensor(betas_np.reshape(1, 10), dtype=torch.float32, device=device)
    T_fixed_t = torch.tensor(T_fixed_np.reshape(1, 3), dtype=torch.float32, device=device)

    # Parameters to optimize.
    global_orient_6d = global_orient_6d_init.clone().detach().requires_grad_(not args.freeze_global_orient)
    body_pose_6d = body_pose_6d_init.clone().detach().requires_grad_(not args.freeze_body_pose)
    betas = betas_init.clone().detach().requires_grad_(not args.freeze_betas)

    if args.train_body_pose_indices is not None and body_pose_6d.requires_grad:
        body_pose_grad_mask = torch.zeros_like(body_pose_6d)

        for idx in args.train_body_pose_indices:
            if idx < 0 or idx >= body_pose_grad_mask.shape[0]:
                raise ValueError(
                    f"Invalid body_pose index {idx}. "
                    f"Expected 0 to {body_pose_grad_mask.shape[0] - 1}."
                )
            body_pose_grad_mask[idx, :] = 1.0

        def mask_body_pose_grad(grad):
            return grad * body_pose_grad_mask

        body_pose_6d.register_hook(mask_body_pose_grad)

        print("[Body pose training] only optimizing body_pose indices:", args.train_body_pose_indices)

    opt_params = []
    if global_orient_6d.requires_grad:
        opt_params.append(global_orient_6d)
    if body_pose_6d.requires_grad:
        opt_params.append(body_pose_6d)
    if betas.requires_grad:
        opt_params.append(betas)

    if len(opt_params) == 0:
        raise ValueError("No parameters to optimize. All of global_orient/body_pose/betas are frozen.")

    # ------------------------------------------------------------
    # Load targets
    # ------------------------------------------------------------
    (
        kpt_joint_ids_np,
        keypoints_2d_np,
        vertex_ids_np,
        vertex_keypoints_2d_np,
    ) = load_keypoints_2d_json(args.keypoints_2d_json)

    kpt_joint_ids_t = torch.tensor(kpt_joint_ids_np, dtype=torch.long, device=device)
    keypoints_2d_t = torch.tensor(keypoints_2d_np, dtype=torch.float32, device=device)

    if vertex_ids_np is not None:
        vertex_ids_t = torch.tensor(vertex_ids_np, dtype=torch.long, device=device)
        vertex_keypoints_2d_t = torch.tensor(vertex_keypoints_2d_np, dtype=torch.float32, device=device)

        print("[Face/head 2D] loaded vertex keypoints:", vertex_keypoints_2d_np.shape)
        print("[Face/head 2D] vertex_ids:", vertex_ids_np.tolist())
    else:
        vertex_ids_t = None
        vertex_keypoints_2d_t = None

    print("[Body 2D] loaded joint keypoints:", keypoints_2d_np.shape)
    print("[Body 2D] joint_ids:", kpt_joint_ids_np.tolist())

    if args.head_target_file is not None:
        head_target_np = load_head_target_file(args.head_target_file)

        if args.head_target_coord == "mesh_cam":
            head_target_np = mesh_cam_to_camera(head_target_np)
        else:
            head_target_np = head_target_np.astype(np.float32)

        head_target_t = torch.tensor(head_target_np, dtype=torch.float32, device=device)
        print("head_target_camera:", head_target_np)
    else:
        head_target_t = None
    
     # ------------------------------------------------------------
    # Optional Aria orientation target
    # ------------------------------------------------------------
    if args.aria_forward_file is not None:
        aria_forward_np = load_vector_file(args.aria_forward_file)
        aria_forward_t = torch.tensor(
            aria_forward_np,
            dtype=torch.float32,
            device=device,
        )

        print("aria_forward_camera:", aria_forward_np)
    else:
        aria_forward_t = None
    
    if args.aria_up_file is not None:
        aria_up_np = load_vector_file(args.aria_up_file)
        aria_up_t = torch.tensor(
            aria_up_np,
            dtype=torch.float32,
            device=device,
        )

        print("aria_up_camera:", aria_up_np)
    else:
        aria_up_t = None

    fx = torch.tensor(float(args.fx), dtype=torch.float32, device=device)
    fy = torch.tensor(float(args.fy), dtype=torch.float32, device=device)
    cx = torch.tensor(float(args.cx), dtype=torch.float32, device=device)
    cy = torch.tensor(float(args.cy), dtype=torch.float32, device=device)

    huber_delta_t = torch.tensor(float(args.huber_delta_2d), dtype=torch.float32, device=device)
    proxy_radius_t = torch.tensor(float(args.proxy_radius), dtype=torch.float32, device=device)

    # ------------------------------------------------------------
    # SMPL model and joint regressor
    # ------------------------------------------------------------
    patch_numpy_for_old_smpl()
    smplx_model_path = prepare_smplx_model_path(args.smpl_model_dir)

    smpl_model = smplx.create(
        model_path=str(smplx_model_path),
        model_type="smpl",
        gender=args.gender,
        num_betas=10,
        batch_size=1,
    ).to(device)
    smpl_model.eval()

    J_np = load_smpl_joint_regressor(args.smpl_model_dir)
    J_t = torch.tensor(J_np, dtype=torch.float32, device=device)

    # ------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------
    def forward_smpl():
        global_rotmat = rot6d_to_rotmat(global_orient_6d).reshape(1, 1, 3, 3)
        body_rotmat = rot6d_to_rotmat(body_pose_6d).reshape(1, 23, 3, 3)

        smpl_out = smpl_model(
            global_orient=global_rotmat,
            body_pose=body_rotmat,
            betas=betas,
            transl=T_fixed_t,
            pose2rot=False,
            return_verts=True,
        )

        vertices_camera = smpl_out.vertices[0]          # (6890, 3), already with fixed T
        joints24_camera = J_t @ vertices_camera         # (24, 3), same evaluation regressor

        return vertices_camera, joints24_camera, global_rotmat, body_rotmat

    def compute_loss():
        vertices_camera, joints24_camera, global_rotmat, body_rotmat = forward_smpl()

        loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        terms = {}

        # 2D reprojection
        loss_2d, reproj_px = compute_reprojection_loss(
            joints24_camera,
            kpt_joint_ids_t,
            keypoints_2d_t,
            fx,
            fy,
            cx,
            cy,
            huber_delta_t,
        )
        loss = loss + args.w_2d * loss_2d
        terms["2d"] = loss_2d
        terms["reproj_px"] = reproj_px.detach()

        # Face/head vertex 2D reprojection
        if vertex_ids_t is not None and args.w_face_2d > 0:
            loss_face_2d, face_reproj_px = compute_vertex_reprojection_loss(
                vertices_camera,
                vertex_ids_t,
                vertex_keypoints_2d_t,
                fx,
                fy,
                cx,
                cy,
                huber_delta_t,
            )
            loss = loss + args.w_face_2d * loss_face_2d
            terms["face_2d"] = loss_face_2d
            terms["face_reproj_px"] = face_reproj_px.detach()
        
        # Aria orientation loss
        if aria_forward_t is not None and args.w_aria_orient > 0:
            if args.nose_vertex < 0 or args.left_eye_vertex < 0 or args.right_eye_vertex < 0:
                raise ValueError(
                    "Aria orientation loss requires "
                    "--nose_vertex, --left_eye_vertex, and --right_eye_vertex."
                )

            loss_aria_orient, aria_cosine, smpl_forward = compute_aria_orientation_loss(
                vertices_camera,
                args.nose_vertex,
                args.left_eye_vertex,
                args.right_eye_vertex,
                aria_forward_t,
            )

            loss = loss + args.w_aria_orient * loss_aria_orient
            terms["aria_orient"] = loss_aria_orient
            terms["aria_cosine"] = aria_cosine
        
        if aria_forward_t is not None and args.w_head_up_perp > 0:
            loss_perp, head_up_dot_forward, smpl_up = compute_head_up_forward_perp_loss(
                joints24_camera,
                args.neck_index,
                args.head_index,
                aria_forward_t,
            )

            loss = loss + args.w_head_up_perp * loss_perp
            terms["head_up_perp"] = loss_perp
            terms["head_up_dot_forward"] = head_up_dot_forward
        
        # Aria up-vector loss
        if aria_up_t is not None and args.w_aria_up > 0:
            loss_aria_up, aria_up_cosine, smpl_up = compute_aria_up_loss(
                joints24_camera,
                args.neck_index,
                args.head_index,
                aria_up_t,
            )

            loss = loss + args.w_aria_up * loss_aria_up
            terms["aria_up"] = loss_aria_up
            terms["aria_up_cosine"] = aria_up_cosine

        # Mild first-stage head/proxy loss
        if head_target_t is not None and args.w_head > 0:
            if args.target_vertex_index is not None:
                target_point = vertices_camera[args.target_vertex_index]
            else:
                target_point = joints24_camera[args.head_index]

            head_dist = torch.linalg.norm(target_point - head_target_t)
            head_res = torch.relu(head_dist - proxy_radius_t)
            loss_head = head_res ** 2

            loss = loss + args.w_head * loss_head
            terms["head"] = loss_head
            terms["head_dist"] = head_dist.detach()

        # Priors against Stage-1 / original HMR params
        if args.w_orient_prior > 0 and not args.freeze_global_orient:
            loss_orient_prior = torch.mean((global_orient_6d - global_orient_6d_init) ** 2)
            loss = loss + args.w_orient_prior * loss_orient_prior
            terms["orient_prior"] = loss_orient_prior

        if args.w_pose_prior > 0 and not args.freeze_body_pose:
            loss_pose_prior = torch.mean((body_pose_6d - body_pose_6d_init) ** 2)
            loss = loss + args.w_pose_prior * loss_pose_prior
            terms["pose_prior"] = loss_pose_prior

        if args.w_betas_prior > 0 and not args.freeze_betas:
            loss_betas_prior = torch.mean((betas - betas_init) ** 2)
            loss = loss + args.w_betas_prior * loss_betas_prior
            terms["betas_prior"] = loss_betas_prior

        if args.w_betas_l2 > 0 and not args.freeze_betas:
            loss_betas_l2 = torch.mean(betas ** 2)
            loss = loss + args.w_betas_l2 * loss_betas_l2
            terms["betas_l2"] = loss_betas_l2

        # Positive-depth safety
        if args.w_z_positive > 0:
            min_z = torch.min(vertices_camera[:, 2])
            loss_z_pos = torch.relu(0.10 - min_z) ** 2

            loss = loss + args.w_z_positive * loss_z_pos
            terms["z_positive"] = loss_z_pos

        return loss, terms

    # ------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(opt_params, lr=args.lr)

        for it in range(args.iters):
            optimizer.zero_grad()
            loss, terms = compute_loss()
            loss.backward()
            optimizer.step()

            if it % 50 == 0 or it == args.iters - 1:
                msg = f"iter {it:04d} | loss={loss.item():.6f}"
                msg += f" | reproj={terms['reproj_px'].item():.2f}px"
                if "face_reproj_px" in terms:
                    msg += f" | face_reproj={terms['face_reproj_px'].item():.2f}px"
                if "aria_cosine" in terms:
                    msg += f" | aria_cos={terms['aria_cosine'].item():.3f}"
                if "aria_up_cosine" in terms:
                    msg += f" | aria_up_cos={terms['aria_up_cosine'].item():.3f}"
                if "head_up_dot_forward" in terms:
                    msg += f" | up_dot_fwd={terms['head_up_dot_forward'].item():.3f}"
                if "head_dist" in terms:
                    msg += f" | head_dist={terms['head_dist'].item() * 1000.0:.1f}mm"

                if "pose_prior" in terms:
                    msg += f" | pose_prior={terms['pose_prior'].item():.6f}"

                if "betas_prior" in terms:
                    msg += f" | betas_prior={terms['betas_prior'].item():.6f}"

                print(msg)

    elif args.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(
            opt_params,
            lr=args.lr,
            max_iter=args.lbfgs_max_iter,
            line_search_fn="strong_wolfe",
        )

        eval_count = {"n": 0}

        def closure():
            optimizer.zero_grad()
            loss, terms = compute_loss()
            loss.backward()

            eval_count["n"] += 1

            if eval_count["n"] % 10 == 0:
                msg = f"lbfgs_eval {eval_count['n']:04d} | loss={loss.item():.6f}"
                msg += f" | reproj={terms['reproj_px'].item():.2f}px"
                if "face_reproj_px" in terms:
                    msg += f" | face_reproj={terms['face_reproj_px'].item():.2f}px"

                if "aria_cosine" in terms:
                    msg += f" | aria_cos={terms['aria_cosine'].item():.3f}"
                if "aria_up_cosine" in terms:
                    msg += f" | aria_up_cos={terms['aria_up_cosine'].item():.3f}"
                if "head_up_dot_forward" in terms:
                    msg += f" | up_dot_fwd={terms['head_up_dot_forward'].item():.3f}"

                if "head_dist" in terms:
                    msg += f" | head_dist={terms['head_dist'].item() * 1000.0:.1f}mm"

                if "pose_prior" in terms:
                    msg += f" | pose_prior={terms['pose_prior'].item():.6f}"

                if "betas_prior" in terms:
                    msg += f" | betas_prior={terms['betas_prior'].item():.6f}"

                print(msg)

            return loss

        optimizer.step(closure)

    else:
        raise ValueError(args.optimizer)

    # ------------------------------------------------------------
    # Save final mesh and NPZ
    # ------------------------------------------------------------
    with torch.no_grad():
        vertices_camera_t, joints24_camera_t, global_rotmat_t, body_rotmat_t = forward_smpl()

        vertices_camera = vertices_camera_t.detach().cpu().numpy().astype(np.float32)
        vertices_mesh_cam = camera_to_mesh_cam(vertices_camera)

        global_rotmat_np = global_rotmat_t.detach().cpu().numpy().astype(np.float32).reshape(1, 3, 3)
        body_rotmat_np = body_rotmat_t.detach().cpu().numpy().astype(np.float32).reshape(23, 3, 3)
        betas_np_opt = betas.detach().cpu().numpy().astype(np.float32).reshape(10)

    summarize_vertices("Final camera vertices", vertices_camera)
    summarize_vertices("Final mesh_cam vertices", vertices_mesh_cam)

    save_obj(out_obj, vertices_mesh_cam, smpl_model.faces)

    # Preserve original Stage-1 keys, then overwrite optimized params/vertices.
    save_dict = {}
    for key in data.files:
        save_dict[key] = data[key]

    save_dict.update(
        {
            "T_opt": T_fixed_np.astype(np.float32),
            "T_fixed_stage2": T_fixed_np.astype(np.float32),

            "global_orient": global_rotmat_np.astype(np.float32),
            "body_pose": body_rotmat_np.astype(np.float32),
            "betas": betas_np_opt.astype(np.float32),

            "pred_vertices_full_camera": vertices_camera.astype(np.float32),
            "pred_vertices_export": vertices_mesh_cam.astype(np.float32),

            # Because pose/shape changed, local vertices also changed.
            "pred_vertices_local": (vertices_camera - T_fixed_np[None, :]).astype(np.float32),

            "stage2_optimized_global_orient": np.array(not args.freeze_global_orient, dtype=bool),
            "stage2_optimized_body_pose": np.array(not args.freeze_body_pose, dtype=bool),
            "stage2_optimized_betas": np.array(not args.freeze_betas, dtype=bool),

            "stage2_used_face_2d": np.array(vertex_ids_np is not None, dtype=bool),
            "stage2_w_face_2d": np.array(args.w_face_2d, dtype=np.float32),
            "stage2_used_aria_orientation": np.array(
                aria_forward_t is not None and args.w_aria_orient > 0,
                dtype=bool,
            ),
            "stage2_w_aria_orient": np.array(args.w_aria_orient, dtype=np.float32),
            "stage2_nose_vertex": np.array(args.nose_vertex, dtype=np.int32),
            "stage2_left_eye_vertex": np.array(args.left_eye_vertex, dtype=np.int32),
            "stage2_right_eye_vertex": np.array(args.right_eye_vertex, dtype=np.int32),
                        "stage2_used_aria_up": np.array(
                aria_up_t is not None and args.w_aria_up > 0,
                dtype=bool,
            ),
            "stage2_w_aria_up": np.array(args.w_aria_up, dtype=np.float32),
            "stage2_neck_index": np.array(args.neck_index, dtype=np.int32),
        }
    )
    if vertex_ids_np is not None:
        save_dict["stage2_face_vertex_ids"] = vertex_ids_np.astype(np.int64)
        save_dict["stage2_face_vertex_keypoints"] = vertex_keypoints_2d_np.astype(np.float32)
    if aria_forward_t is not None:
        save_dict["stage2_aria_forward_camera"] = (
            aria_forward_t.detach().cpu().numpy().astype(np.float32)
        )
    if aria_up_t is not None:
        save_dict["stage2_aria_up_camera"] = (
            aria_up_t.detach().cpu().numpy().astype(np.float32)
        )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **save_dict)

    print(f"[Saved NPZ] {out_npz}")
    print("Done.")


if __name__ == "__main__":
    main()