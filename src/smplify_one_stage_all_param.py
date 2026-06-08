#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch

# Reuse your existing Stage-2 utilities.
from smplify_v2_all_param import (
    camera_to_mesh_cam,
    mesh_cam_to_camera,
    patch_numpy_for_old_smpl,
    prepare_smplx_model_path,
    load_smpl_joint_regressor,
    save_obj,
    load_head_target_file,
    load_vector_file,
    load_keypoints_2d_json,
    project_points_camera,
    rotmat_to_6d_np,
    rot6d_to_rotmat,
    get_param,
    compute_reprojection_loss,
    compute_vertex_reprojection_loss,
    compute_aria_orientation_loss,
    compute_aria_up_loss,
    compute_head_up_forward_perp_loss,
    summarize_vertices,
)
import smplx


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_npz", required=True)
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

    # What to optimize.
    parser.add_argument("--freeze_trans", action="store_true")
    parser.add_argument("--freeze_betas", action="store_true")
    parser.add_argument("--freeze_body_pose", action="store_true")
    parser.add_argument("--freeze_global_orient", action="store_true")

    # Optional gradient mask for body pose.
    parser.add_argument(
        "--train_body_pose_indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional body_pose indices to train. "
            "body_pose[i] corresponds to SMPL joint i+1. "
            "Example: --train_body_pose_indices 2 5 8 11 14."
        ),
    )

    # Reprojection losses.
    parser.add_argument("--w_2d", type=float, default=5.0)
    parser.add_argument("--w_face_2d", type=float, default=10.0)
    parser.add_argument("--huber_delta_2d", type=float, default=0.05)

    # Head / glasses proxy.
    parser.add_argument("--w_head", type=float, default=5.0)
    parser.add_argument("--proxy_radius", type=float, default=0.03)
    parser.add_argument("--head_index", type=int, default=15)
    parser.add_argument("--neck_index", type=int, default=12)
    parser.add_argument(
        "--head_proxy_vertex_ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "SMPL vertex IDs used as mesh-side egocentric camera proxy. "
            "For your final proxy use: --head_proxy_vertex_ids 2800 6280"
        ),
    )
    parser.add_argument(
        "--target_vertex_index",
        type=int,
        default=None,
        help="Fallback single vertex proxy. Ignored if --head_proxy_vertex_ids is given.",
    )

    # Aria orientation.
    parser.add_argument("--aria_forward_file", default=None)
    parser.add_argument("--aria_up_file", default=None)
    parser.add_argument("--w_aria_orient", type=float, default=0.0)
    parser.add_argument("--w_aria_up", type=float, default=0.0)
    parser.add_argument("--w_head_up_perp", type=float, default=0.0)
    parser.add_argument("--nose_vertex", type=int, default=-1)
    parser.add_argument("--left_eye_vertex", type=int, default=-1)
    parser.add_argument("--right_eye_vertex", type=int, default=-1)

    # Priors.
    parser.add_argument("--w_trans_prior", type=float, default=0.1)
    parser.add_argument("--w_trust", type=float, default=2.0)
    parser.add_argument("--max_shift", type=float, default=0.60)
    parser.add_argument("--w_z_prior", type=float, default=0.0)

    parser.add_argument("--w_orient_prior", type=float, default=5.0)
    parser.add_argument("--w_pose_prior", type=float, default=3.0)
    parser.add_argument("--w_betas_prior", type=float, default=10.0)
    parser.add_argument("--w_betas_l2", type=float, default=0.1)
    parser.add_argument("--w_z_positive", type=float, default=5.0)

    args = parser.parse_args()

    pred_npz = Path(args.pred_npz).expanduser()
    out_obj = Path(args.out_obj).expanduser()
    out_npz = Path(args.out_npz).expanduser() if args.out_npz else out_obj.with_suffix(".npz")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    print("============================================================")
    print("SMPLify One-Stage: optimize T + orient/body pose/betas jointly")
    print("pred_npz:", pred_npz)
    print("out_obj:", out_obj)
    print("out_npz:", out_npz)
    print("device:", device)
    print("============================================================")

    data = np.load(pred_npz)

    # ------------------------------------------------------------
    # Load initial SMPL params
    # ------------------------------------------------------------
    global_orient_np = get_param(data, "global_orient", "pred_smpl_global_orient").astype(np.float32)
    body_pose_np = get_param(data, "body_pose", "pred_smpl_body_pose").astype(np.float32)
    betas_np = get_param(data, "betas", "pred_smpl_betas").astype(np.float32).reshape(10)

    if "T_opt" in data.files:
        init_T_np = data["T_opt"].astype(np.float32).reshape(3)
        print("[Init] using T_opt")
    elif "pred_cam_t_full" in data.files:
        init_T_np = data["pred_cam_t_full"].astype(np.float32).reshape(3)
        print("[Init] using pred_cam_t_full")
    else:
        raise KeyError("Input NPZ must contain T_opt or pred_cam_t_full.")

    if global_orient_np.shape != (1, 3, 3):
        raise ValueError(f"Expected global_orient shape (1,3,3), got {global_orient_np.shape}")
    if body_pose_np.shape != (23, 3, 3):
        raise ValueError(f"Expected body_pose shape (23,3,3), got {body_pose_np.shape}")

    global_orient_6d_np = rotmat_to_6d_np(global_orient_np)  # (1, 6)
    body_pose_6d_np = rotmat_to_6d_np(body_pose_np)          # (23, 6)

    global_orient_6d_init = torch.tensor(global_orient_6d_np, dtype=torch.float32, device=device)
    body_pose_6d_init = torch.tensor(body_pose_6d_np, dtype=torch.float32, device=device)
    betas_init = torch.tensor(betas_np.reshape(1, 10), dtype=torch.float32, device=device)
    T_init = torch.tensor(init_T_np.reshape(1, 3), dtype=torch.float32, device=device)

    global_orient_6d = global_orient_6d_init.clone().detach().requires_grad_(not args.freeze_global_orient)
    body_pose_6d = body_pose_6d_init.clone().detach().requires_grad_(not args.freeze_body_pose)
    betas = betas_init.clone().detach().requires_grad_(not args.freeze_betas)
    T = T_init.clone().detach().requires_grad_(not args.freeze_trans)

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
    for p in [global_orient_6d, body_pose_6d, betas, T]:
        if p.requires_grad:
            opt_params.append(p)

    if len(opt_params) == 0:
        raise ValueError("No parameters to optimize.")

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
        print("[Face/head 2D] vertex_ids:", vertex_ids_np.tolist())
    else:
        vertex_ids_t = None
        vertex_keypoints_2d_t = None

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

    if args.aria_forward_file is not None:
        aria_forward_np = load_vector_file(args.aria_forward_file)
        aria_forward_t = torch.tensor(aria_forward_np, dtype=torch.float32, device=device)
        print("aria_forward_camera:", aria_forward_np)
    else:
        aria_forward_t = None

    if args.aria_up_file is not None:
        aria_up_np = load_vector_file(args.aria_up_file)
        aria_up_t = torch.tensor(aria_up_np, dtype=torch.float32, device=device)
        print("aria_up_camera:", aria_up_np)
    else:
        aria_up_t = None

    fx = torch.tensor(float(args.fx), dtype=torch.float32, device=device)
    fy = torch.tensor(float(args.fy), dtype=torch.float32, device=device)
    cx = torch.tensor(float(args.cx), dtype=torch.float32, device=device)
    cy = torch.tensor(float(args.cy), dtype=torch.float32, device=device)
    huber_delta_t = torch.tensor(float(args.huber_delta_2d), dtype=torch.float32, device=device)
    proxy_radius_t = torch.tensor(float(args.proxy_radius), dtype=torch.float32, device=device)

    if args.head_proxy_vertex_ids is not None:
        head_proxy_vertex_ids_t = torch.tensor(args.head_proxy_vertex_ids, dtype=torch.long, device=device)
        print("[Head proxy] using vertex IDs:", args.head_proxy_vertex_ids)
    else:
        head_proxy_vertex_ids_t = None

    # ------------------------------------------------------------
    # SMPL model
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
    # Forward / loss
    # ------------------------------------------------------------
    def forward_smpl():
        global_rotmat = rot6d_to_rotmat(global_orient_6d).reshape(1, 1, 3, 3)
        body_rotmat = rot6d_to_rotmat(body_pose_6d).reshape(1, 23, 3, 3)

        smpl_out = smpl_model(
            global_orient=global_rotmat,
            body_pose=body_rotmat,
            betas=betas,
            transl=T,
            pose2rot=False,
            return_verts=True,
        )

        vertices_camera = smpl_out.vertices[0]
        joints24_camera = J_t @ vertices_camera

        return vertices_camera, joints24_camera, global_rotmat, body_rotmat

    def compute_loss():
        vertices_camera, joints24_camera, global_rotmat, body_rotmat = forward_smpl()

        loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        terms = {}

        # Body 2D reprojection
        if args.w_2d > 0:
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

        # Face/head vertex reprojection
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

        # Head / glasses target
        if head_target_t is not None and args.w_head > 0:
            if head_proxy_vertex_ids_t is not None:
                target_point = vertices_camera[head_proxy_vertex_ids_t].mean(dim=0)
            elif args.target_vertex_index is not None:
                target_point = vertices_camera[args.target_vertex_index]
            else:
                target_point = joints24_camera[args.head_index]

            head_dist = torch.linalg.norm(target_point - head_target_t)
            head_res = torch.relu(head_dist - proxy_radius_t)
            loss_head = head_res ** 2

            loss = loss + args.w_head * loss_head
            terms["head"] = loss_head
            terms["head_dist"] = head_dist.detach()

        # Aria forward orientation
        if aria_forward_t is not None and args.w_aria_orient > 0:
            if args.nose_vertex < 0 or args.left_eye_vertex < 0 or args.right_eye_vertex < 0:
                raise ValueError(
                    "Aria orientation loss requires --nose_vertex, "
                    "--left_eye_vertex, and --right_eye_vertex."
                )

            loss_aria_orient, aria_cosine, _ = compute_aria_orientation_loss(
                vertices_camera,
                args.nose_vertex,
                args.left_eye_vertex,
                args.right_eye_vertex,
                aria_forward_t,
            )
            loss = loss + args.w_aria_orient * loss_aria_orient
            terms["aria_orient"] = loss_aria_orient
            terms["aria_cosine"] = aria_cosine

        # SMPL up perpendicular to Aria forward
        if aria_forward_t is not None and args.w_head_up_perp > 0:
            loss_perp, head_up_dot_forward, _ = compute_head_up_forward_perp_loss(
                joints24_camera,
                args.neck_index,
                args.head_index,
                aria_forward_t,
            )
            loss = loss + args.w_head_up_perp * loss_perp
            terms["head_up_perp"] = loss_perp
            terms["head_up_dot_forward"] = head_up_dot_forward

        # Aria up loss
        if aria_up_t is not None and args.w_aria_up > 0:
            loss_aria_up, aria_up_cosine, _ = compute_aria_up_loss(
                joints24_camera,
                args.neck_index,
                args.head_index,
                aria_up_t,
            )
            loss = loss + args.w_aria_up * loss_aria_up
            terms["aria_up"] = loss_aria_up
            terms["aria_up_cosine"] = aria_up_cosine

        # Translation priors
        if args.w_trans_prior > 0 and not args.freeze_trans:
            loss_trans_prior = torch.mean((T - T_init) ** 2)
            loss = loss + args.w_trans_prior * loss_trans_prior
            terms["trans_prior"] = loss_trans_prior

        if args.w_trust > 0 and not args.freeze_trans:
            shift = torch.linalg.norm(T.reshape(3) - T_init.reshape(3))
            loss_trust = torch.relu(shift - args.max_shift) ** 2
            loss = loss + args.w_trust * loss_trust
            terms["shift"] = shift.detach()
            terms["trust"] = loss_trust

        if args.w_z_prior > 0 and not args.freeze_trans:
            loss_z_prior = (T.reshape(3)[2] - T_init.reshape(3)[2]) ** 2
            loss = loss + args.w_z_prior * loss_z_prior
            terms["z_prior"] = loss_z_prior

        # Pose/shape priors
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

                if "reproj_px" in terms:
                    msg += f" | reproj={terms['reproj_px'].item():.2f}px"
                if "face_reproj_px" in terms:
                    msg += f" | face_reproj={terms['face_reproj_px'].item():.2f}px"
                if "head_dist" in terms:
                    msg += f" | head_dist={terms['head_dist'].item() * 1000.0:.1f}mm"
                if "shift" in terms:
                    msg += f" | shift={terms['shift'].item():.3f}m"
                if "aria_cosine" in terms:
                    msg += f" | aria_cos={terms['aria_cosine'].item():.3f}"
                if "head_up_dot_forward" in terms:
                    msg += f" | up_dot_fwd={terms['head_up_dot_forward'].item():.3f}"

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

                if "reproj_px" in terms:
                    msg += f" | reproj={terms['reproj_px'].item():.2f}px"
                if "face_reproj_px" in terms:
                    msg += f" | face_reproj={terms['face_reproj_px'].item():.2f}px"
                if "head_dist" in terms:
                    msg += f" | head_dist={terms['head_dist'].item() * 1000.0:.1f}mm"
                if "shift" in terms:
                    msg += f" | shift={terms['shift'].item():.3f}m"
                if "aria_cosine" in terms:
                    msg += f" | aria_cos={terms['aria_cosine'].item():.3f}"
                if "head_up_dot_forward" in terms:
                    msg += f" | up_dot_fwd={terms['head_up_dot_forward'].item():.3f}"

                print(msg)

            return loss

        optimizer.step(closure)

    else:
        raise ValueError(args.optimizer)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    with torch.no_grad():
        vertices_camera_t, joints24_camera_t, global_rotmat_t, body_rotmat_t = forward_smpl()

        vertices_camera = vertices_camera_t.detach().cpu().numpy().astype(np.float32)
        vertices_mesh_cam = camera_to_mesh_cam(vertices_camera)

        global_rotmat_np = global_rotmat_t.detach().cpu().numpy().astype(np.float32).reshape(1, 3, 3)
        body_rotmat_np = body_rotmat_t.detach().cpu().numpy().astype(np.float32).reshape(23, 3, 3)
        betas_np_opt = betas.detach().cpu().numpy().astype(np.float32).reshape(10)
        T_opt_np = T.detach().cpu().numpy().astype(np.float32).reshape(3)

    summarize_vertices("Final camera vertices", vertices_camera)
    summarize_vertices("Final mesh_cam vertices", vertices_mesh_cam)

    save_obj(out_obj, vertices_mesh_cam, smpl_model.faces)

    save_dict = {}
    for key in data.files:
        save_dict[key] = data[key]

    save_dict.update(
        {
            "T_opt": T_opt_np,
            "T_one_stage": T_opt_np,
            "T_init_one_stage": init_T_np.astype(np.float32),
            "T_delta_from_init": (T_opt_np - init_T_np.astype(np.float32)).astype(np.float32),

            "global_orient": global_rotmat_np.astype(np.float32),
            "body_pose": body_rotmat_np.astype(np.float32),
            "betas": betas_np_opt.astype(np.float32),

            "pred_vertices_full_camera": vertices_camera.astype(np.float32),
            "pred_vertices_export": vertices_mesh_cam.astype(np.float32),
            "pred_vertices_local": (vertices_camera - T_opt_np[None, :]).astype(np.float32),

            "one_stage_optimized_translation": np.array(not args.freeze_trans, dtype=bool),
            "one_stage_optimized_global_orient": np.array(not args.freeze_global_orient, dtype=bool),
            "one_stage_optimized_body_pose": np.array(not args.freeze_body_pose, dtype=bool),
            "one_stage_optimized_betas": np.array(not args.freeze_betas, dtype=bool),
        }
    )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **save_dict)

    print(f"[Saved OBJ] {out_obj}")
    print(f"[Saved NPZ] {out_npz}")
    print("Done.")


if __name__ == "__main__":
    main()