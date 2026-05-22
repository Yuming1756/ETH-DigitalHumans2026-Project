#!/usr/bin/env python3

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import smplx


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

    Your repo often has:

        smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl

    This function creates the expected symlink if needed.
    """
    smpl_model_dir = Path(smpl_model_dir).expanduser()

    expected = smpl_model_dir / "smpl" / "SMPL_NEUTRAL.pkl"
    if expected.exists():
        return smpl_model_dir

    basic = smpl_model_dir / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    if not basic.exists():
        raise FileNotFoundError(
            f"Could not find SMPL model. Expected either:\n"
            f"  {expected}\n"
            f"or:\n"
            f"  {basic}"
        )

    expected.parent.mkdir(parents=True, exist_ok=True)

    if not expected.exists():
        os.symlink(basic, expected)
        print(f"[SMPL] Created symlink:\n  {expected} -> {basic}")

    return smpl_model_dir


def camera_to_mesh_cam(points):
    """
    Positive-Z camera coordinate:
        [x, y, z]

    Exported OBJ / EgoHumans mesh_cam-like coordinate:
        [x, -y, -z]
    """
    points = np.asarray(points, dtype=np.float32).copy()
    points[..., 1] *= -1.0
    points[..., 2] *= -1.0
    return points


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


def as_tensor_rotmat_global_orient(x, device):
    """
    Supports:
        (1, 3, 3)
        (3, 3)
        (3,) axis-angle
    """
    x = np.asarray(x, dtype=np.float32)

    if x.shape == (1, 3, 3):
        return torch.tensor(x[None, ...], dtype=torch.float32, device=device), False
        # shape becomes (1, 1, 3, 3), pose2rot=False

    if x.shape == (3, 3):
        return torch.tensor(x[None, None, ...], dtype=torch.float32, device=device), False

    if x.shape == (3,):
        return torch.tensor(x[None, :], dtype=torch.float32, device=device), True
        # shape becomes (1, 3), pose2rot=True

    raise ValueError(f"Unsupported global_orient shape: {x.shape}")


def as_tensor_body_pose(x, device):
    """
    Supports:
        (23, 3, 3)
        (1, 23, 3, 3)
        (69,) axis-angle
        (1, 69) axis-angle
    """
    x = np.asarray(x, dtype=np.float32)

    if x.shape == (23, 3, 3):
        return torch.tensor(x[None, ...], dtype=torch.float32, device=device), False
        # shape becomes (1, 23, 3, 3), pose2rot=False

    if x.shape == (1, 23, 3, 3):
        return torch.tensor(x, dtype=torch.float32, device=device), False

    if x.shape == (69,):
        return torch.tensor(x[None, :], dtype=torch.float32, device=device), True

    if x.shape == (1, 69):
        return torch.tensor(x, dtype=torch.float32, device=device), True

    raise ValueError(f"Unsupported body_pose shape: {x.shape}")


def load_required_smpl_params(npz):
    """
    Supports shared names:
        global_orient, body_pose, betas

    Also supports 4D-Humans aliases:
        pred_smpl_global_orient, pred_smpl_body_pose, pred_smpl_betas
    """
    if "global_orient" in npz.files:
        global_orient = npz["global_orient"]
    elif "pred_smpl_global_orient" in npz.files:
        global_orient = npz["pred_smpl_global_orient"]
    else:
        raise KeyError("Missing global_orient / pred_smpl_global_orient")

    if "body_pose" in npz.files:
        body_pose = npz["body_pose"]
    elif "pred_smpl_body_pose" in npz.files:
        body_pose = npz["pred_smpl_body_pose"]
    else:
        raise KeyError("Missing body_pose / pred_smpl_body_pose")

    if "betas" in npz.files:
        betas = npz["betas"]
    elif "pred_smpl_betas" in npz.files:
        betas = npz["pred_smpl_betas"]
    else:
        raise KeyError("Missing betas / pred_smpl_betas")

    return global_orient, body_pose, betas


def summarize_vertices(name, vertices):
    print(f"\n========== {name} ==========")
    print("shape:", vertices.shape)
    print("center:", vertices.mean(axis=0))
    print("min:   ", vertices.min(axis=0))
    print("max:   ", vertices.max(axis=0))
    print("extent:", vertices.max(axis=0) - vertices.min(axis=0))


def mean_vertex_error(a, b):
    return float(np.linalg.norm(a - b, axis=1).mean())


def compare_if_available(name, recovered, npz, key):
    if key not in npz.files:
        return

    target = npz[key].astype(np.float32)

    if target.shape != recovered.shape:
        print(f"[Compare] {name}: shape mismatch recovered={recovered.shape}, saved={target.shape}")
        return

    err = mean_vertex_error(recovered, target)
    max_err = float(np.linalg.norm(recovered - target, axis=1).max())

    print(f"[Compare] {name}: mean vertex error = {err * 1000:.6f} mm, max = {max_err * 1000:.6f} mm")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--npz", required=True, help="Input HMR NPZ with SMPL params")
    parser.add_argument("--smpl_model_dir", default="./smpl")
    parser.add_argument("--out_obj", required=True)
    parser.add_argument("--out_npz", default=None)

    parser.add_argument(
        "--output_coord",
        choices=["local", "camera", "mesh_cam"],
        default="mesh_cam",
        help=(
            "Which coordinate to export as OBJ. "
            "local = SMPL local before global translation; "
            "camera = positive-Z camera coordinates; "
            "mesh_cam = exported OBJ convention [x,-y,-z]."
        ),
    )

    parser.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    npz_path = Path(args.npz).expanduser()
    out_obj = Path(args.out_obj).expanduser()

    if args.out_npz is None:
        out_npz = out_obj.with_suffix(".npz")
    else:
        out_npz = Path(args.out_npz).expanduser()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    print("============================================================")
    print("Recover SMPL mesh from saved HMR SMPL parameters")
    print("NPZ:", npz_path)
    print("SMPL model dir:", args.smpl_model_dir)
    print("Output OBJ:", out_obj)
    print("Output coord:", args.output_coord)
    print("Device:", device)
    print("============================================================")

    data = np.load(npz_path)

    print("\nAvailable NPZ keys:")
    for k in data.files:
        print(" ", k, data[k].shape, data[k].dtype)

    global_orient_np, body_pose_np, betas_np = load_required_smpl_params(data)

    print("\nLoaded SMPL params:")
    print("global_orient:", global_orient_np.shape, global_orient_np.dtype)
    print("body_pose:    ", body_pose_np.shape, body_pose_np.dtype)
    print("betas:        ", betas_np.shape, betas_np.dtype)

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

    global_orient_t, global_orient_is_aa = as_tensor_rotmat_global_orient(global_orient_np, device)
    body_pose_t, body_pose_is_aa = as_tensor_body_pose(body_pose_np, device)

    if global_orient_is_aa != body_pose_is_aa:
        raise ValueError(
            "global_orient and body_pose are mixed axis-angle/rotmat formats. "
            "This script expects both to be axis-angle or both to be rotation matrices."
        )

    pose2rot = global_orient_is_aa

    betas_np = np.asarray(betas_np, dtype=np.float32).reshape(1, -1)
    betas_t = torch.tensor(betas_np, dtype=torch.float32, device=device)

    if betas_t.shape[1] != 10:
        raise ValueError(f"Expected betas shape (1,10), got {betas_t.shape}")

    with torch.no_grad():
        smpl_out = smpl_model(
            global_orient=global_orient_t,
            body_pose=body_pose_t,
            betas=betas_t,
            transl=torch.zeros((1, 3), dtype=torch.float32, device=device),
            pose2rot=pose2rot,
            return_verts=True,
        )

    vertices_local = smpl_out.vertices[0].detach().cpu().numpy().astype(np.float32)

    if "T_opt" in data.files:
        T = data["T_opt"].astype(np.float32).reshape(3)
        print("[Translation] using T_opt")
    elif "pred_cam_t_full" in data.files:
        T = data["pred_cam_t_full"].astype(np.float32).reshape(3)
        print("[Translation] using pred_cam_t_full")
    else:
        T = np.zeros(3, dtype=np.float32)
        print("[Translation] no T_opt or pred_cam_t_full found; using zero translation")

    vertices_camera = vertices_local + T[None, :]
    vertices_mesh_cam = camera_to_mesh_cam(vertices_camera)

    summarize_vertices("Recovered local vertices", vertices_local)
    summarize_vertices("Recovered camera vertices", vertices_camera)
    summarize_vertices("Recovered mesh_cam/export vertices", vertices_mesh_cam)

    compare_if_available("local vs pred_vertices_local", vertices_local, data, "pred_vertices_local")
    compare_if_available("camera vs pred_vertices_full_camera", vertices_camera, data, "pred_vertices_full_camera")
    compare_if_available("mesh_cam vs pred_vertices_export", vertices_mesh_cam, data, "pred_vertices_export")

    if args.output_coord == "local":
        vertices_to_save = vertices_local
    elif args.output_coord == "camera":
        vertices_to_save = vertices_camera
    elif args.output_coord == "mesh_cam":
        vertices_to_save = vertices_mesh_cam
    else:
        raise ValueError(args.output_coord)

    save_obj(out_obj, vertices_to_save, smpl_model.faces)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        vertices_local=vertices_local.astype(np.float32),
        vertices_camera=vertices_camera.astype(np.float32),
        vertices_mesh_cam=vertices_mesh_cam.astype(np.float32),
        T=T.astype(np.float32),
        global_orient=global_orient_np.astype(np.float32),
        body_pose=body_pose_np.astype(np.float32),
        betas=betas_np.astype(np.float32),
        pose2rot=np.array(pose2rot, dtype=bool),
    )
    print(f"[Saved NPZ] {out_npz}")


if __name__ == "__main__":
    main()