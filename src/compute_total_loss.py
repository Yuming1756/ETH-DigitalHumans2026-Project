import torch


def huber_loss(residual, delta):
    """
    residual: non-negative tensor
    delta: float
    """
    return torch.where(
        residual <= delta,
        0.5 * residual ** 2,
        delta * (residual - 0.5 * delta),
    )


def compute_total_loss(
    T,
    *,
    args,
    device,
    pred_joints_local_t,
    pred_vertices_local_t,
    init_T_t,
    head_target_camera_t=None,
    target_vertex_ids_t=None,
    keypoints_2d_t=None,
    kpt_joint_ids_t=None,
    fx=None,
    fy=None,
    cx=None,
    cy=None,
    project_points_camera=None,
):
    loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    loss_terms = {}

    # ------------------------------------------------------------
    # 1. Head / Aria proxy loss with dead zone
    # ------------------------------------------------------------
    if head_target_camera_t is not None and args.w_head > 0:
    # ------------------------------------------------------------
    # Choose what point should match the Aria/glasses target.
    #
    # Option 1:
    #   Use the mean of selected SMPL surface vertices.
    #   This is better for glasses / between-eye / forehead targets.
    #
    # Option 2:
    #   Fall back to SMPL head joint.
    # ------------------------------------------------------------
        if target_vertex_ids_t is not None:
            pred_target_point = pred_vertices_local_t[target_vertex_ids_t].mean(dim=0) + T
            loss_terms["target_type"] = "vertex_mean"
        else:
            pred_target_point = pred_joints_local_t[args.head_index] + T
            loss_terms["target_type"] = "head_joint"

        target_dist = torch.linalg.norm(pred_target_point - head_target_camera_t)
    
        proxy_radius = torch.tensor(
            args.proxy_radius,
            dtype=torch.float32,
            device=device,
        )

        target_residual = torch.relu(target_dist - proxy_radius)
        
        loss_head = target_dist ** 2

        loss = loss + args.w_head * loss_head
        loss_terms["head_deadzone"] = loss_head
        loss_terms["head_dist"] = target_dist.detach()

    # ------------------------------------------------------------
    # 2. Optional 2D reprojection loss with Huber
    # ------------------------------------------------------------
    if keypoints_2d_t is not None and args.w_2d > 0:
        if project_points_camera is None:
            raise ValueError("project_points_camera must be provided when using 2D keypoints.")

        if fx is None or fy is None or cx is None or cy is None:
            raise ValueError("fx, fy, cx, cy must be provided when using 2D keypoints.")

        if kpt_joint_ids_t is None:
            raise ValueError("kpt_joint_ids_t must be provided when using 2D keypoints.")

        joints_sel = pred_joints_local_t[kpt_joint_ids_t] + T[None, :]
        proj = project_points_camera(joints_sel, fx, fy, cx, cy)

        diff_px = proj - keypoints_2d_t[:, :2]
        conf = keypoints_2d_t[:, 2].clamp(min=0.0)

        # Optional safety clipping in pixels.
        # If you use Huber, you can set --reproj_clip_px 0.
        if args.reproj_clip_px > 0:
            diff_norm_px = torch.linalg.norm(diff_px, dim=1, keepdim=True).clamp(min=1e-6)
            scale = torch.clamp(args.reproj_clip_px / diff_norm_px, max=1.0)
            diff_px = diff_px * scale

        # Normalize by focal length.
        diff_norm = torch.stack(
            [diff_px[:, 0] / fx, diff_px[:, 1] / fy],
            dim=-1,
        )

        # Per-joint normalized reprojection residual.
        residual = torch.linalg.norm(diff_norm, dim=1)

        # Huber loss. Example:
        # huber_delta_2d = 0.05 means about 0.05 * 1915 ≈ 96 px.
        delta = torch.tensor(
            args.huber_delta_2d,
            dtype=torch.float32,
            device=device,
        )

        loss_2d_per_joint = huber_loss(residual, delta)

        loss_2d = torch.sum(conf * loss_2d_per_joint) / torch.clamp(conf.sum(), min=1.0)

        loss = loss + args.w_2d * loss_2d
        loss_terms["2d"] = loss_2d

        # For logging only: true weighted mean reprojection error in pixels.
        reproj_err_px = torch.sum(
            conf * torch.linalg.norm(proj - keypoints_2d_t[:, :2], dim=1)
        ) / torch.clamp(conf.sum(), min=1.0)

        loss_terms["reproj_px"] = reproj_err_px.detach()

    # ------------------------------------------------------------
    # 3. Translation prior
    # ------------------------------------------------------------
    if args.w_trans_prior > 0:
        loss_prior = torch.sum((T - init_T_t) ** 2)
        loss = loss + args.w_trans_prior * loss_prior
        loss_terms["prior"] = loss_prior

    # ------------------------------------------------------------
    # 4. Trust-region translation prior
    # ------------------------------------------------------------
    if args.w_trust > 0:
        shift = torch.linalg.norm(T - init_T_t)

        max_shift = torch.tensor(
            args.max_shift,
            dtype=torch.float32,
            device=device,
        )

        loss_trust = torch.relu(shift - max_shift) ** 2

        loss = loss + args.w_trust * loss_trust
        loss_terms["trust"] = loss_trust
        loss_terms["shift"] = shift.detach()

    # ------------------------------------------------------------
    # 5. Extra depth prior
    # ------------------------------------------------------------
    if args.w_z_prior > 0:
        loss_z_prior = (T[2] - init_T_t[2]) ** 2
        loss = loss + args.w_z_prior * loss_z_prior
        loss_terms["z_prior"] = loss_z_prior

    # ------------------------------------------------------------
    # 6. Positive-depth safety term
    # ------------------------------------------------------------
    if args.w_z_positive > 0:
        vertices_z = pred_vertices_local_t[:, 2] + T[2]
        min_z = torch.min(vertices_z)

        loss_z_pos = torch.relu(0.10 - min_z) ** 2

        loss = loss + args.w_z_positive * loss_z_pos
        loss_terms["z_positive"] = loss_z_pos

    return loss, loss_terms