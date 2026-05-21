import torch


def _robust_loss(x, alpha=1.0, scale=1.0):
    """
    Barron's general robust loss (https://arxiv.org/abs/1701.03077).
    - alpha=2 -> L2, alpha=1 -> pseudo-Huber/Charbonnier, alpha=0 -> Cauchy
    - scale: transition point from quadratic to robust regime (in units of x)
    """
    x_scaled = x / scale
    if alpha == 2.0:
        return 0.5 * x_scaled ** 2
    elif alpha == 0.0:
        return torch.log(0.5 * x_scaled ** 2 + 1.0)
    else:
        return (abs(alpha - 2.0) / alpha) * (
            (x_scaled ** 2 / abs(alpha - 2.0) + 1.0) ** (alpha / 2.0) - 1.0
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
    keypoints_2d_t=None,
    kpt_joint_ids_t=None,
    fx=None,
    fy=None,
    cx=None,
    cy=None,
    project_points_camera=None,
    head_robust_alpha: float = 1.0,
    head_robust_scale_m: float = 0.05,
    kpt_conf_thresh: float = 0.2,
    kpt_robust_alpha: float = 1.0,
    kpt_robust_scale_px: float = 8.0,
    prior_xy_scale: float = 1.0,
    prior_z_scale: float = 0.3,
    z_min_margin_m: float = 0.10,
    z_depth_reg_weight: float = 0.0,
    z_depth_reg_target_m: float = 5.0,
):
    loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    loss_terms = {}

    # ------------------------------------------------------------------
    # 1. Head-position term (robust, scale-normalised)
    # ------------------------------------------------------------------
    if head_target_camera_t is not None and args.w_head > 0:
        pred_head = pred_joints_local_t[args.head_index] + T
        diff_head = pred_head - head_target_camera_t          # (3,) metres

        robust_per_axis = _robust_loss(
            diff_head,
            alpha=head_robust_alpha,
            scale=head_robust_scale_m,
        )
        loss_head = robust_per_axis.sum()

        loss = loss + args.w_head * loss_head
        loss_terms["head"] = loss_head.detach()

    # ------------------------------------------------------------------
    # 2. 2D reprojection term (confidence threshold + robust kernel)
    # ------------------------------------------------------------------
    if keypoints_2d_t is not None and args.w_2d > 0:
        if project_points_camera is None:
            raise ValueError(
                "project_points_camera must be provided when using 2D keypoints."
            )
        if fx is None or fy is None or cx is None or cy is None:
            raise ValueError(
                "fx, fy, cx, cy must be provided when using 2D keypoints."
            )

        joints_sel = pred_joints_local_t[kpt_joint_ids_t] + T[None, :]    # (J,3)
        proj       = project_points_camera(joints_sel, fx, fy, cx, cy)     # (J,2) pixels

        diff_px = proj - keypoints_2d_t[:, :2]                              # (J,2) pixels
        err_px  = torch.linalg.norm(diff_px, dim=1)                         # (J,)  pixels

        robust_err = _robust_loss(
            err_px,
            alpha=kpt_robust_alpha,
            scale=kpt_robust_scale_px,
        )                                                                    # (J,)

        conf     = keypoints_2d_t[:, 2].clamp(min=0.0)                      # (J,)
        mask     = (conf >= kpt_conf_thresh).float()
        eff_conf = conf * mask                                               # (J,)

        loss_2d = (
            torch.sum(eff_conf * robust_err)
            / torch.clamp(eff_conf.sum(), min=1.0)
        )

        loss = loss + args.w_2d * loss_2d
        loss_terms["2d"] = loss_2d.detach()

        with torch.no_grad():
            loss_terms["2d_px"] = (
                torch.sum(eff_conf * err_px) / torch.clamp(eff_conf.sum(), min=1.0)
            ).detach()

    # ------------------------------------------------------------------
    # 3. Anisotropic translation prior
    # ------------------------------------------------------------------
    if args.w_trans_prior > 0:
        dT = T - init_T_t                                                   # (3,)

        loss_prior_xy = torch.sum(dT[:2] ** 2) / (prior_xy_scale ** 2)
        loss_prior_z  = (dT[2] ** 2) / (prior_z_scale ** 2)
        loss_prior    = loss_prior_xy + loss_prior_z

        loss = loss + args.w_trans_prior * loss_prior
        loss_terms["prior"]    = loss_prior.detach()
        loss_terms["prior_xy"] = loss_prior_xy.detach()
        loss_terms["prior_z"]  = loss_prior_z.detach()

    # ------------------------------------------------------------------
    # 4. Optional depth regularisation (pulls Z toward plausible depth)
    # ------------------------------------------------------------------
    if z_depth_reg_weight > 0.0:
        T_z = T[2]
        loss_depth_reg = (T_z - z_depth_reg_target_m) ** 2
        loss = loss + z_depth_reg_weight * loss_depth_reg
        loss_terms["depth_reg"] = loss_depth_reg.detach()

    # ------------------------------------------------------------------
    # 5. Positive-depth safety term
    # ------------------------------------------------------------------
    if args.w_z_positive > 0:
        vertices_z = pred_vertices_local_t[:, 2] + T[2]
        min_z      = torch.min(vertices_z)
        loss_z_pos = torch.relu(z_min_margin_m - min_z) ** 2

        loss = loss + args.w_z_positive * loss_z_pos
        loss_terms["z_positive"] = loss_z_pos.detach()

    return loss, loss_terms