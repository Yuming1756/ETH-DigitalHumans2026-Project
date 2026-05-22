#!/usr/bin/env python3

import argparse
import os
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

    Your project often has:

        smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl

    This function creates the expected absolute symlink if needed.
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

    # Remove broken symlink if present.
    if expected.is_symlink() and not expected.exists():
        expected.unlink()

    if not expected.exists():
        os.symlink(str(basic.resolve()), str(expected))
        print(f"[SMPL] Created symlink:\n  {expected} -> {basic.resolve()}")

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


def load_smpl_params_from_npz(npz_path):
    """
    Supports shared names:
        global_orient, body_pose, betas

    Also supports 4D-Humans aliases:
        pred_smpl_global_orient, pred_smpl_body_pose, pred_smpl_betas
    """
    data = np.load(Path(npz_path).expanduser())

    if "global_orient" in data.files:
        global_orient = data["global_orient"]
    elif "pred_smpl_global_orient" in data.files:
        global_orient = data["pred_smpl_global_orient"]
    else:
        raise KeyError("Missing global_orient / pred_smpl_global_orient")

    if "body_pose" in data.files:
        body_pose = data["body_pose"]
    elif "pred_smpl_body_pose" in data.files:
        body_pose = data["pred_smpl_body_pose"]
    else:
        raise KeyError("Missing body_pose / pred_smpl_body_pose")

    if "betas" in data.files:
        betas = data["betas"]
    elif "pred_smpl_betas" in data.files:
        betas = data["pred_smpl_betas"]
    else:
        raise KeyError("Missing betas / pred_smpl_betas")

    # Translation is optional.
    # Priority:
    #   T_opt from optimization
    #   pred_cam_t_full from original HMR
    #   zero translation
    if "T_opt" in data.files:
        transl = data["T_opt"].astype(np.float32).reshape(3)
        transl_source = "T_opt"
    elif "pred_cam_t_full" in data.files:
        transl = data["pred_cam_t_full"].astype(np.float32).reshape(3)
        transl_source = "pred_cam_t_full"
    else:
        transl = np.zeros(3, dtype=np.float32)
        transl_source = "zero"

    return {
        "global_orient": global_orient.astype(np.float32),
        "body_pose": body_pose.astype(np.float32),
        "betas": betas.astype(np.float32),
        "transl": transl.astype(np.float32),
        "transl_source": transl_source,
    }


def as_tensor_global_orient(x, device):
    """
    Supports:
        rotation matrix: (1, 3, 3) or (3, 3)
        axis-angle:      (3,)
    """
    x = np.asarray(x, dtype=np.float32)

    if x.shape == (1, 3, 3):
        return torch.tensor(x[None, ...], dtype=torch.float32, device=device), False
        # output shape: (1, 1, 3, 3), pose2rot=False

    if x.shape == (3, 3):
        return torch.tensor(x[None, None, ...], dtype=torch.float32, device=device), False
        # output shape: (1, 1, 3, 3), pose2rot=False

    if x.shape == (3,):
        return torch.tensor(x[None, :], dtype=torch.float32, device=device), True
        # output shape: (1, 3), pose2rot=True

    raise ValueError(f"Unsupported global_orient shape: {x.shape}")


def as_tensor_body_pose(x, device):
    """
    Supports:
        rotation matrix: (23, 3, 3) or (1, 23, 3, 3)
        axis-angle:      (69,) or (1, 69)
    """
    x = np.asarray(x, dtype=np.float32)

    if x.shape == (23, 3, 3):
        return torch.tensor(x[None, ...], dtype=torch.float32, device=device), False
        # output shape: (1, 23, 3, 3), pose2rot=False

    if x.shape == (1, 23, 3, 3):
        return torch.tensor(x, dtype=torch.float32, device=device), False

    if x.shape == (69,):
        return torch.tensor(x[None, :], dtype=torch.float32, device=device), True
        # output shape: (1, 69), pose2rot=True

    if x.shape == (1, 69):
        return torch.tensor(x, dtype=torch.float32, device=device), True

    raise ValueError(f"Unsupported body_pose shape: {x.shape}")


def generate_smpl_mesh(
    *,
    smpl_model,
    global_orient_np,
    body_pose_np,
    betas_np,
    transl_np=None,
    device,
):
    """
    Generate SMPL vertices from SMPL parameters.

    Returns:
        vertices_local:  (6890, 3), no global translation
        vertices_camera: (6890, 3), with translation
    """
    global_orient_t, global_is_aa = as_tensor_global_orient(global_orient_np, device)
    body_pose_t, body_is_aa = as_tensor_body_pose(body_pose_np, device)

    if global_is_aa != body_is_aa:
        raise ValueError(
            "global_orient and body_pose are mixed formats. "
            "Both should be rotation matrices or both should be axis-angle."
        )

    pose2rot = global_is_aa

    betas_np = np.asarray(betas_np, dtype=np.float32).reshape(1, -1)
    if betas_np.shape[1] != 10:
        raise ValueError(f"Expected betas shape (10,), got {betas_np.shape}")

    betas_t = torch.tensor(betas_np, dtype=torch.float32, device=device)

    zero_transl_t = torch.zeros((1, 3), dtype=torch.float32, device=device)

    with torch.no_grad():
        smpl_out = smpl_model(
            global_orient=global_orient_t,
            body_pose=body_pose_t,
            betas=betas_t,
            transl=zero_transl_t,
            pose2rot=pose2rot,
            return_verts=True,
        )

    vertices_local = smpl_out.vertices[0].detach().cpu().numpy().astype(np.float32)

    if transl_np is None:
        transl_np = np.zeros(3, dtype=np.float32)

    transl_np = np.asarray(transl_np, dtype=np.float32).reshape(3)
    vertices_camera = vertices_local + transl_np[None, :]

    return vertices_local, vertices_camera, pose2rot


def summarize_vertices(name, vertices):
    print(f"\n========== {name} ==========")
    print("shape:", vertices.shape)
    print("center:", vertices.mean(axis=0))
    print("min:   ", vertices.min(axis=0))
    print("max:   ", vertices.max(axis=0))
    print("extent:", vertices.max(axis=0) - vertices.min(axis=0))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--npz", required=True, help="Input NPZ containing SMPL parameters")
    parser.add_argument("--smpl_model_dir", default="./smpl")
    parser.add_argument("--out_obj", required=True)
    parser.add_argument("--out_npz", default=None)

    parser.add_argument(
        "--output_coord",
        choices=["local", "camera", "mesh_cam"],
        default="mesh_cam",
        help=(
            "Coordinate for saved OBJ. "
            "local = SMPL local without translation; "
            "camera = positive-Z camera coordinate with translation; "
            "mesh_cam = exported OBJ convention [x, -y, -z]."
        ),
    )

    parser.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    npz_path = Path(args.npz).expanduser()
    out_obj = Path(args.out_obj).expanduser()
    out_npz = Path(args.out_npz).expanduser() if args.out_npz else out_obj.with_suffix(".npz")

    print("============================================================")
    print("Generate SMPL mesh from SMPL parameters")
    print("NPZ:", npz_path)
    print("SMPL model dir:", args.smpl_model_dir)
    print("Output OBJ:", out_obj)
    print("Output coord:", args.output_coord)
    print("Device:", device)
    print("============================================================")

    params = load_smpl_params_from_npz(npz_path)

    print("\nLoaded parameters:")
    print("global_orient:", params["global_orient"].shape, params["global_orient"].dtype)
    print("body_pose:    ", params["body_pose"].shape, params["body_pose"].dtype)
    print("betas:        ", params["betas"].shape, params["betas"].dtype)
    print("transl:       ", params["transl"].shape, params["transl"].dtype)
    print("transl source:", params["transl_source"])

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

    vertices_local, vertices_camera, pose2rot = generate_smpl_mesh(
        smpl_model=smpl_model,
        global_orient_np=params["global_orient"],
        body_pose_np=params["body_pose"],
        betas_np=params["betas"],
        transl_np=params["transl"],
        device=device,
    )

    vertices_mesh_cam = camera_to_mesh_cam(vertices_camera)

    summarize_vertices("SMPL local vertices", vertices_local)
    summarize_vertices("SMPL camera vertices", vertices_camera)
    summarize_vertices("SMPL mesh_cam/export vertices", vertices_mesh_cam)

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
        global_orient=params["global_orient"].astype(np.float32),
        body_pose=params["body_pose"].astype(np.float32),
        betas=params["betas"].astype(np.float32),
        transl=params["transl"].astype(np.float32),
        pose2rot=np.array(pose2rot, dtype=bool),
    )

    print(f"[Saved NPZ] {out_npz}")


if __name__ == "__main__":
    main()