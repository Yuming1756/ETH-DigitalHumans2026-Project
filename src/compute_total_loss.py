import torch


def compute_total_loss(
    T,
    *,
    args,
    device,
    pred_joints_local_t,
    pred_vertices_local_t,
    init_T_t,
    head_target_camera_t=None,
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

    # Head-position term
    if head_target_camera_t is not None and args.w_head > 0:
        pred_head = pred_joints_local_t[args.head_index] + T
        loss_head = torch.sum((pred_head - head_target_camera_t) ** 2)
        loss = loss + args.w_head * loss_head
        loss_terms["head"] = loss_head

    # Optional 2D reprojection term
    if keypoints_2d_t is not None and args.w_2d > 0:
        if project_points_camera is None:
            raise ValueError("project_points_camera must be provided when using 2D keypoints.")

        if fx is None or fy is None or cx is None or cy is None:
            raise ValueError("fx, fy, cx, cy must be provided when using 2D keypoints.")

        joints_sel = pred_joints_local_t[kpt_joint_ids_t] + T[None, :]
        proj = project_points_camera(joints_sel, fx, fy, cx, cy)

        diff_px = proj - keypoints_2d_t[:, :2]
        conf = keypoints_2d_t[:, 2].clamp(min=0.0)

        diff_norm = torch.stack(
            [diff_px[:, 0] / fx, diff_px[:, 1] / fy],
            dim=-1,
        )

        loss_2d_per_joint = torch.sum(diff_norm ** 2, dim=1)
        loss_2d = torch.sum(conf * loss_2d_per_joint) / torch.clamp(conf.sum(), min=1.0)

        loss = loss + args.w_2d * loss_2d
        loss_terms["2d"] = loss_2d

    # Translation prior
    if args.w_trans_prior > 0:
        loss_prior = torch.sum((T - init_T_t) ** 2)
        loss = loss + args.w_trans_prior * loss_prior
        loss_terms["prior"] = loss_prior

    # Positive-depth safety term
    if args.w_z_positive > 0:
        vertices_z = pred_vertices_local_t[:, 2] + T[2]
        min_z = torch.min(vertices_z)
        loss_z_pos = torch.relu(0.10 - min_z) ** 2
        loss = loss + args.w_z_positive * loss_z_pos
        loss_terms["z_positive"] = loss_z_pos

    return loss, loss_terms