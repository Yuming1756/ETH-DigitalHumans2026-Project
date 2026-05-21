import argparse
from ast import arg
import json
import pickle
from pathlib import Path
from turtle import shape
import numpy as np
import torch
import torch.optim as optim
from smplx import SMPL
import scipy

from smplify_v1_translation import load_smpl_joint_regressor, load_head_target_file, mesh_cam_to_camera, load_obj_vertices, load_keypoints_2d_json

def extract_egohumans_input_data(args):
    """
    Parses, transforms, and validates all input data required for EgoHumans 
    translation optimization.
    
    Coordinates are unified into TokenHMR standard positive-Z camera space:
        X: Right, Y: Down, Z: Forward (Positive Depth)

    Parameters:
        args: Namespace object containing all validated parsed command-line configurations.

    Returns:
        dict: A dictionary containing the following extracted arrays and metadata:
            - "pred_vertices_local": np.ndarray (6890, 3), localized SMPL vertices.
            - "pred_joints_local":   np.ndarray (J, 3), regressed joints in local space.
            - "init_T":              np.ndarray (3,), starting [Tx, Ty, Tz] camera translation.
            - "head_target_camera":  np.ndarray (3,) or None, unified head target coords [X, Y, Z].
            - "kpt_joint_ids":       np.ndarray (K,) or None, joint indices for 2D tracking.
            - "keypoints_2d":        np.ndarray (K, 3) or None, pixels + confidence [u, v, conf].
    """
    extracted_data = {}

    # -------------------------------------------------------------------------
    # 1. EXTRACT LOCAL PREDICTIONS & TRANSLATION INITIALIZATION (.npz)
    # -------------------------------------------------------------------------
    pred_npz_path = Path(args.pred_npz).expanduser()
    loaded_npz = np.load(pred_npz_path)

    # Validate and read localized vertices
    if "pred_vertices_local" not in loaded_npz.files:
        raise KeyError(f"Missing 'pred_vertices_local' key in target NPZ: {pred_npz_path}")
    
    # Format: Matrix size (6890, 3) representing [x, y, z] values for a standard neutral SMPL body mesh
    pred_vertices_local = loaded_npz["pred_vertices_local"].astype(np.float32)
    if pred_vertices_local.shape != (6890, 3):
        raise ValueError(f"Expected vertices shape (6890, 3), got {pred_vertices_local.shape}")
    extracted_data["pred_vertices_local"] = pred_vertices_local

    # Validate and read the default estimated system camera translation matrix
    if "pred_cam_t_full" not in loaded_npz.files:
        raise KeyError(f"Missing camera translation 'pred_cam_t_full' in target NPZ: {pred_npz_path}")
    
    # Format: Flat vector array size (3,) representing translation coordinates [Tx, Ty, Tz]
    init_T = loaded_npz["T_opt"].astype(np.float32).reshape(3)
    extracted_data["init_T"] = init_T

    extracted_data["init_smpl_global_orient"] = loaded_npz["global_orient"]
    extracted_data["init_smpl_body_pose"] = loaded_npz["body_pose"]
    extracted_data["init_smpl_betas"] = loaded_npz["betas"]

    # -------------------------------------------------------------------------
    # 2. EXTRACT & APPLY SMPL SKELETAL REGRESSOR MAPPING (.pkl)
    # -------------------------------------------------------------------------
    # Format: Matrix size (J, 6890) where row rows correspond to structural joints
    J_regressor = load_smpl_joint_regressor(args.smpl_model_dir)
    
    # Linear projection matrix operation to construct explicit skeletal coordinate values
    # Matrix shape outcome: (J, 6890) @ (6890, 3) -> (J, 3) representing [x, y, z] coordinates
    pred_joints_local = J_regressor @ pred_vertices_local
    
    if pred_joints_local.shape[0] <= args.head_index:
        raise ValueError(f"Configured head_index ({args.head_index}) bounds past joint size ({pred_joints_local.shape[0]})")
    extracted_data["pred_joints_local"] = pred_joints_local

    # -------------------------------------------------------------------------
    # 3. EXTRACT REFERENCE TARGET COORDINATES & UNIFY SPATIAL CONVENTIONS
    # -------------------------------------------------------------------------
    head_target_camera = None

    if args.head_target_file is not None:
        # Load localized array profile. Expected data layout: flat array [x, y, z]
        head_target_raw = load_head_target_file(args.head_target_file)

        # Coordinate Conversion Logic:
        # If the input target follows the EgoHumans dataset or exported OBJ convention ('mesh_cam'),
        # invert the Y and Z coordinates to match standard positive-Z TokenHMR camera space.
        if args.head_target_coord == "mesh_cam":
            head_target_camera = mesh_cam_to_camera(head_target_raw)
        else:
            head_target_camera = head_target_raw.astype(np.float32)
            
        print(f"[Extraction] Head tracking target loaded from text file: {args.head_target_file}")

    elif args.gt_head_debug_obj is not None:
        # DEBUG TRACKER ROUTINE: Extracts target head joint location dynamically from ground-truth mesh
        gt_vertices_mesh_cam = load_obj_vertices(args.gt_head_debug_obj)
        gt_joints_mesh_cam = J_regressor @ gt_vertices_mesh_cam
        head_target_mesh_cam = gt_joints_mesh_cam[args.head_index]
        
        # Converted target matrix location from inverted mesh_cam orientation to standard camera coordinates
        head_target_camera = mesh_cam_to_camera(head_target_mesh_cam)
        print(f"[Extraction] [DEBUG ONLY] Ground-truth vertex source active: {args.gt_head_debug_obj}")

    # Apply rigid sensor/structural translation tracking offsets if declared
    if head_target_camera is not None:
        # Format: Flat vector array offset translation addition [x + dx, y + dy, z + dz]
        offset_vector = np.asarray(args.head_target_offset_camera, dtype=np.float32)
        head_target_camera = head_target_camera + offset_vector

    extracted_data["head_target_camera"] = head_target_camera

    # -------------------------------------------------------------------------
    # 4. EXTRACT OPTIONAL 2D REPROJECTION CONSTRAINTS (.json)
    # -------------------------------------------------------------------------
    kpt_joint_ids = None
    keypoints_2d = None

    if args.keypoints_2d_json is not None:
        # Extracted Array Output Formats:
        # kpt_joint_ids: Shape (K,) containing tracking matrix index mappings matching the model joints.
        # keypoints_2d: Shape (K, 3) containing structured 2D positions [u, v, confidence]
        kpt_joint_ids, keypoints_2d = load_keypoints_2d_json(args.keypoints_2d_json)
        print(f"[Extraction] 2D Projection tracking active. Features found: {keypoints_2d.shape[0]}")

    extracted_data["kpt_joint_ids"] = kpt_joint_ids
    extracted_data["keypoints_2d"] = keypoints_2d
    
    

    # -------------------------------------------------------------------------
    # 5. DATASET INTEGRITY VALIDATION STEP
    # -------------------------------------------------------------------------
    if extracted_data["head_target_camera"] is None and extracted_data["keypoints_2d"] is None:
        raise ValueError(
            "Extraction sequence halted: Optimization targets missing. "
            "Provide at least one tracking constraint: --head_target_file, "
            "--gt_head_debug_obj, or --keypoints_2d_json."
        )
    

    return extracted_data



# ---------- projection & losses ----------
def perspective_project(joints3d, t, focal, princpt):
    # joints3d: (K,3), t: (3,)
    X = joints3d + t.unsqueeze(0)            # camera-space translation (assumes SMPL is in camera-centered local coords)
    x = X[:, 0] / (X[:, 2] + 1e-9)
    y = X[:, 1] / (X[:, 2] + 1e-9)
    if isinstance(focal, (list,tuple,np.ndarray)):
        fx, fy = float(focal[0]), float(focal[1])
    else:
        fx = fy = float(focal)
    cx, cy = float(princpt[0]), float(princpt[1])
    u = fx * x + cx
    v = fy * y + cy
    return torch.stack([u, v], dim=-1)

def weighted_kp_loss(pred, gt_uv_conf, mask=None):
    # gt_uv_conf: (K,3) -> u,v,conf
    gt = gt_uv_conf[:, :2]
    conf = gt_uv_conf[:, 2]
    if mask is not None:
        conf = conf * mask.float()
    conf = conf.unsqueeze(-1)               # (K,1)
    diff = (pred - gt)**2
    loss = (diff * conf).sum() / (conf.sum()+1e-6)
    return loss

def l2(x, w=1.0): return w * (x**2).mean()

# ---------- main optimizer that consumes your extraction output ----------
def optimize_from_extracted(extracted, smpl_model,
                            focal, princpt,
                            iters=300, lr=5e-3,
                            temp_w_pose=1e-2, temp_w_trans=1e-2,
                            device='cuda'):
    """
    extracted: dict from extract_egohumans_input_data()
    smpl_model: SMPL instance
    focal: scalar or (fx,fy)
    princpt: (cx,cy)
    """
    device = torch.device(device)
    smpl_model.to(device)

    verts_local = torch.tensor(extracted["pred_vertices_local"], dtype=torch.float32, device=device)  # (6890,3)
    joints_local = torch.tensor(extracted["pred_joints_local"], dtype=torch.float32, device=device)    # (J,3)
    init_T = torch.tensor(extracted["init_T"], dtype=torch.float32, device=device)                    # (3,)
    poses_init_mats = extracted["init_smpl_body_pose"]
    poses_init_rotvecs = np.zeros(shape=(23,3))
    for i in range(poses_init_mats.shape[0]):
        poses_init_rotvecs[i] = scipy.spatial.transform.Rotation.from_matrix(poses_init_mats[i]).as_rotvec()
        print(f"pose rotvec {i}: {poses_init_rotvecs[i]}")
    poses_init = poses_init_rotvecs # (23,3)
    betas_init = extracted["init_smpl_betas"] # (10,)
    global_orient_init = scipy.spatial.transform.Rotation.from_matrix(extracted["init_smpl_global_orient"]).as_rotvec().flatten()[np.newaxis,:] # (1,3)
    head_target = extracted["head_target_camera"]                                                     # (3,) or None

    # print(f"joints_local.shape = {joints_local.shape}")
    # print(f"init_T.shape = {init_T.shape}")
    # print(f"poses_init.shape = {poses_init.shape}")
    # print(f"betas_init.shape = {betas_init.shape}")
    # print(f"global_orient_init.shape = {global_orient_init.shape}")
    # print(f"head_target.shape = {head_target.shape}")

    # Keypoints: (K,3) or None. kpt_joint_ids: (K,) mapping into SMPL joint index space
    keypoints_2d = extracted["keypoints_2d"]
    kpt_joint_ids = extracted["kpt_joint_ids"]

    # Build sequence length T. EgoHumans loader may be per-frame or sequence; we assume a single-frame or single-seq:
    # If keypoints_2d is per-frame sequence, it should be (T,K,3). If it's single-frame (K,3), make T=1.
    if keypoints_2d is None:
        kp_seq = None
        kp_mask_seq = None
    else:
        kp_arr = np.array(keypoints_2d)
        if kp_arr.ndim == 2 and kp_arr.shape[1] == 3:
            # single frame
            kp_seq = kp_arr[None, ...]        # (1,K,3)
        elif kp_arr.ndim == 3:
            kp_seq = kp_arr                    # (T,K,3)
        else:
            raise ValueError("Unexpected keypoints_2d shape")
        # confidence mask: conf > 0.05
        kp_mask_seq = (kp_seq[..., 2] > 0.05).astype(np.float32)  # (T,K)

    # If no keypoints but head target present, we can optimize single-frame pose+trans toward head target by using a synthetic 2D target (skip for brevity)
    # Build T
    T = kp_seq.shape[0] if kp_seq is not None else 1

    # Initialize params:
    # per-frame poses (T,69)
    poses = torch.tensor(poses_init, dtype=torch.float32, device=device, requires_grad=True)
    print(f"Tensor pose shape: {poses.shape}")
    # shared betas (10,)
    betas = torch.tensor(betas_init, dtype=torch.float32, device=device, requires_grad=True)
    # per-frame full translation (T,3) initialize from init_T
    trans_init = init_T.cpu().numpy().repeat(T, axis=0)
    trans = torch.tensor(trans_init, dtype=torch.float32, device=device, requires_grad=False)  # we can optionally optimize translation, but start with fixed for stability

    # Prepare mapping tensor if available
    if kpt_joint_ids is not None:
        kp_to_smpl = torch.tensor(np.asarray(kpt_joint_ids, dtype=np.int64), dtype=torch.long, device=device)  # (K,)
    else:
        kp_to_smpl = None

    # Convert kp_seq to torch
    if kp_seq is not None:
        kp_seq_t = torch.tensor(kp_seq, dtype=torch.float32, device=device)  # (T,K,3)
        kp_mask_t = torch.tensor(kp_mask_seq, dtype=torch.float32, device=device)  # (T,K)
    else:
        kp_seq_t = None
        kp_mask_t = None

    global_orient = torch.tensor(global_orient_init, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = optim.Adam([poses, betas, trans], lr=lr)

    focal_in = focal
    princpt_in = princpt

    for it in range(iters):
        optimizer.zero_grad()
        loss = 0.0

        for t in range(T):
            pose_t = poses # (T,69) -> (69,) for frame t, we optimize a single shared pose across all frames for simplicity, but you can also make it per-frame by using poses[t]
            print("pose_t shape:", pose_t.shape)
            global_orient = global_orient  # (1,3) # TODO: should optimize global_orient per-frame 
            body_pose = poses
            print(f"Iteration {it}, Frame {t}, body_pose shape: {body_pose.shape}, Global Orient: {global_orient.shape}, Betas: {betas.shape},")
            smpl_out = smpl_model( transl=trans.unsqueeze(0), body_pose=body_pose.unsqueeze(0),
                                  betas=betas.unsqueeze(0), return_verts=True, return_joints=True, global_orient=global_orient.unsqueeze(0))
            joints3d = smpl_out.joints[0]   # (J,3)

            if kp_seq_t is not None and kp_to_smpl is not None:
                # select mapped joints
                sel = kp_to_smpl.clamp(0, joints3d.shape[0]-1)
                joints_sel = joints3d[sel]                       # (K,3)
                kp_pred = perspective_project(joints_sel, trans[t], focal_in, princpt_in)  # (K,2)
                loss_kp = weighted_kp_loss(kp_pred, kp_seq_t[t], mask=kp_mask_t[t] if kp_mask_t is not None else None)
                loss = loss + loss_kp

            # Optionally add head-target 3D loss (if provided) - match head joint to target in camera coords
            if head_target is not None:
                head_idx = int(extracted.get("head_index", 15))  # try to read or default to 15
                if head_idx < joints3d.shape[0]:
                    head_j3d = joints3d[head_idx] + trans[t]  # (3,)
                    head_target_t = torch.tensor(head_target, dtype=torch.float32, device=device)
                    loss = loss + l2(head_j3d - head_target_t, w=1e-2)

            # small parameter priors per frame
            loss = loss + l2(pose_t, w=1e-5) + l2(trans[t], w=1e-5)

        # shared betas prior
        loss = loss + l2(betas, w=1e-4)

        # temporal smoothness on poses and translations
        if T > 1:
            dp = poses[1:] - poses[:-1]
            dt = trans[1:] - trans[:-1]
            loss = loss + l2(dp, w=temp_w_pose) + l2(dt, w=temp_w_trans)

        loss.backward()
        optimizer.step()

        # simple LR drop mid-run
        if it == (iters//2):
            for g in optimizer.param_groups:
                g['lr'] = lr * 0.2

    return {
        "poses": poses.detach().cpu().numpy(),    # (T,72)
        "betas": betas.detach().cpu().numpy(),    # (10,)
        "trans": trans.detach().cpu().numpy(),     # (T,3)
        "vertices": smpl_out.vertices.detach().cpu().numpy(), # (T,6890,3) 
    }

# ---------- usage example ----------
# extracted = extract_egohumans_input_data(args)
# smpl = SMPL(model_path='models/smpl', gender='NEUTRAL')
# out = optimize_from_extracted(extracted, smpl, focal=(fx,fy), princpt=(cx,cy), iters=250, device='cuda')

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
    parser.add_argument("--keypoints_2d_json", default=None, help="Optional 2D keypoints JSON.")
    parser.add_argument("--fx", type=float, default=1915.346061677637)
    parser.add_argument("--fy", type=float, default=1915.5916704165572)
    parser.add_argument("--cx", type=float, default=1920.0)
    parser.add_argument("--cy", type=float, default=1080.0)
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
    
    args = parser.parse_args()
    extracted = extract_egohumans_input_data(args)
    smpl = SMPL(model_path=args.smpl_model_dir + "/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")
    out = optimize_from_extracted(extracted, smpl, focal=(args.fx,args.fy), princpt=(args.cx,args.cy), iters=250, device='cpu')
    
if __name__ == "__main__":
    main()


