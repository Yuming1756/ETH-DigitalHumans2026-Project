from pathlib import Path
import argparse
import pickle
import numpy as np


def qvec_to_rotmat(qvec):
    """
    COLMAP quaternion format: qw, qx, qy, qz.
    Returns world-to-camera rotation matrix.
    """
    qw, qx, qy, qz = qvec

    return np.array([
        [
            1 - 2 * qy * qy - 2 * qz * qz,
            2 * qx * qy - 2 * qw * qz,
            2 * qx * qz + 2 * qw * qy,
        ],
        [
            2 * qx * qy + 2 * qw * qz,
            1 - 2 * qx * qx - 2 * qz * qz,
            2 * qy * qz - 2 * qw * qx,
        ],
        [
            2 * qx * qz - 2 * qw * qy,
            2 * qy * qz + 2 * qw * qx,
            1 - 2 * qx * qx - 2 * qy * qy,
        ],
    ], dtype=np.float64)


def load_colmap_image_extrinsic(images_txt, image_name, allow_nearest=True):
    """
    Load COLMAP world-to-camera extrinsic for image_name.

    If exact image_name is missing and allow_nearest=True, use the closest
    timestamp for the same camera folder, e.g. cam01/00001.jpg for cam01/00002.jpg.

    Returns:
        T_cam_from_colmap: 4x4 world-to-camera matrix
        used_image_name: the COLMAP image actually used
    """
    images_txt = Path(images_txt).expanduser()

    target_cam = image_name.split("/")[0]
    target_frame_str = Path(image_name).stem
    target_frame = int(target_frame_str)

    exact_match = None
    candidates = []

    with open(images_txt, "r") as f:
        lines = f.readlines()

    # COLMAP images.txt has image info on non-comment odd lines,
    # followed by a 2D-points line. We can simply parse every non-comment
    # line with at least 10 fields.
    for line in lines:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split()

        # Image metadata line:
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        if len(parts) < 10:
            continue

        try:
            image_id = int(parts[0])
            qvec = np.array([float(x) for x in parts[1:5]], dtype=np.float64)
            tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
            camera_id = int(parts[8])
            name = parts[9]
        except ValueError:
            # This is probably a 2D point line, skip it.
            continue

        if name == image_name:
            exact_match = (qvec, tvec, name)
            break

        cam_name = name.split("/")[0]
        if cam_name == target_cam:
            try:
                frame_num = int(Path(name).stem)
            except ValueError:
                continue

            candidates.append((abs(frame_num - target_frame), frame_num, qvec, tvec, name))

    if exact_match is not None:
        qvec, tvec, used_name = exact_match
    else:
        if not allow_nearest or len(candidates) == 0:
            raise FileNotFoundError(f"Could not find {image_name} in {images_txt}")

        candidates.sort(key=lambda x: x[0])
        _, used_frame, qvec, tvec, used_name = candidates[0]

        print(
            f"[Warning] Exact COLMAP image pose not found: {image_name}. "
            f"Using nearest pose instead: {used_name}"
        )

    R = qvec_to_rotmat(qvec)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec

    return T, used_name


def load_colmap_from_aria(data_root):
    pkl_path = Path(data_root).expanduser() / "colmap" / "workplace" / "colmap_from_aria_transforms.pkl"
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    return {k: np.asarray(v, dtype=np.float64) for k, v in d.items()}


def read_aria_calibration_like_egohumans(calib_txt):
    """
    This follows EgoHumans aria_human.py exactly:

      lines = lines[1:]
      data = lines[idx*7:(idx+1)*7]
      data[0] = person id
      data[1] = rgb intrinsics
      data[2] = rgb extrinsics, parsed by reshape(4, 3).T
      data[3] = left intrinsics
      data[4] = left extrinsics
      data[5] = right intrinsics
      data[6] = right extrinsics
    """
    calib_txt = Path(calib_txt).expanduser()

    with open(calib_txt, "r") as f:
        lines = f.readlines()

    lines = lines[1:]  # drop header
    lines = [line.strip() for line in lines if line.strip()]

    if len(lines) % 7 != 0:
        raise ValueError(f"Expected 7 lines per person after header, got {len(lines)} lines")

    data = lines[:7]

    person_id = data[0]

    rgb_intr = np.asarray([float(x) for x in data[1].split()], dtype=np.float64)
    rgb_ext = np.asarray([float(x) for x in data[2].split()], dtype=np.float64).reshape(4, 3).T

    left_intr = np.asarray([float(x) for x in data[3].split()], dtype=np.float64)
    left_ext = np.asarray([float(x) for x in data[4].split()], dtype=np.float64).reshape(4, 3).T

    right_intr = np.asarray([float(x) for x in data[5].split()], dtype=np.float64)
    right_ext = np.asarray([float(x) for x in data[6].split()], dtype=np.float64).reshape(4, 3).T

    def to_4x4(T_3x4):
        T = np.eye(4, dtype=np.float64)
        T[:3, :] = T_3x4
        return T

    return {
        "person_id": person_id,
        "rgb": {
            "intrinsics": rgb_intr,
            "extrinsics": to_4x4(rgb_ext),
        },
        "left": {
            "intrinsics": left_intr,
            "extrinsics": to_4x4(left_ext),
        },
        "right": {
            "intrinsics": right_intr,
            "extrinsics": to_4x4(right_ext),
        },
    }


def transform_point(T, p):
    p_h = np.ones(4, dtype=np.float64)
    p_h[:3] = np.asarray(p, dtype=np.float64)
    out = T @ p_h
    return out[:3] / out[3]

def scale_from_similarity_transform(T):
    """
    If the 3x3 block is A = sR, recover s from det(A).
    """
    A = np.asarray(T, dtype=np.float64)[:3, :3]
    return float(np.cbrt(abs(np.linalg.det(A))))

def normalize_vector(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return v / n


def rotation_from_similarity_transform(T):
    """
    Extract a proper rotation matrix from the 3x3 block of a transform.

    Some EgoHumans transforms contain similarity scale, so the 3x3 block
    can be A = sR instead of pure R. We remove scale/shear by SVD.
    """
    A = np.asarray(T, dtype=np.float64)[:3, :3]

    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt

    # Fix possible reflection.
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt

    return R


def direction_from_aria_to_exo(
    T_aria_from_anchor_world,
    T_exo_from_anchor_world,
    aria_axis=np.array([0.0, 0.0, 1.0]),
):
    """
    Convert an Aria camera-local direction into exo/cam01 positive-Z coordinates.

    T_aria_from_anchor_world:
        x_aria = R_aria_world x_world + t

    T_exo_from_anchor_world:
        x_exo = R_exo_world x_world + t

    For a direction:
        d_world = R_aria_world.T @ d_aria
        d_exo   = R_exo_world @ d_world
    """
    R_aria_from_world = rotation_from_similarity_transform(T_aria_from_anchor_world)
    R_exo_from_world = rotation_from_similarity_transform(T_exo_from_anchor_world)

    aria_axis = normalize_vector(aria_axis)

    d_world = R_aria_from_world.T @ aria_axis
    d_exo = R_exo_from_world @ d_world

    return normalize_vector(d_exo)


def posz_vector_to_mesh_cam(v):
    """
    Direction vector conversion:
      positive-Z camera vector: [x, y, z]
      mesh_cam/export vector:   [x, -y, -z]
    """
    v = np.asarray(v, dtype=np.float64).copy()
    v[1] *= -1
    v[2] *= -1
    return normalize_vector(v)


def camera_center_from_world_to_cam(T_cam_from_world):
    """
    Given x_cam = R x_world + t,
    camera center in world is inv(T) @ [0,0,0,1].
    """
    return transform_point(np.linalg.inv(T_cam_from_world), [0, 0, 0])


def posz_to_mesh_cam(p):
    """
    Your exported mesh_cam convention:
      positive-Z camera: [x, y, z]
      mesh_cam/export:  [x, -y, -z]
    """
    p = np.asarray(p, dtype=np.float64).copy()
    p[1] *= -1
    p[2] *= -1
    return p


def load_obj_vertices(obj_path):
    verts = []
    with open(Path(obj_path).expanduser(), "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(verts, dtype=np.float32)


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


def load_smpl_regressor(smpl_model_dir):
    patch_numpy_for_old_smpl()

    smpl_path = Path(smpl_model_dir).expanduser() / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    with open(smpl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    J = data["J_regressor"]
    if hasattr(J, "toarray"):
        J = J.toarray()

    return np.asarray(J, dtype=np.float32)


def mesh_cam_to_posz(p):
    p = np.asarray(p, dtype=np.float64).copy()
    p[..., 1] *= -1
    p[..., 2] *= -1
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="~/DigitalHumans/data/01_tagging/001_tagging")
    parser.add_argument("--aria", default="aria03")
    parser.add_argument("--frame", default="00001")
    parser.add_argument("--exo_cam", default="cam01")
    parser.add_argument("--anchor_aria", default="aria01")
    parser.add_argument("--smpl_model_dir", default="~/DigitalHumans/4D-Humans/data")
    parser.add_argument("--save_target", default=None)
    parser.add_argument(
        "--save_orientation_target",
        default=None,
        help="Save Aria forward direction in cam01 positive-Z coordinates.",
    )
    parser.add_argument(
        "--save_forward_target",
        default=None,
        help="Save Aria forward direction in cam01 positive-Z coordinates. Same purpose as --save_orientation_target.",
    )

    parser.add_argument(
        "--save_up_target",
        default=None,
        help="Save Aria up direction in cam01 positive-Z coordinates.",
    )

    parser.add_argument(
        "--up_axis",
        choices=["+z", "-z", "+x", "-x", "+y", "-y"],
        default="-y",
        help=(
            "Camera-local up axis. If Aria follows normal camera convention "
            "(+X right, +Y down, +Z forward), then up is -Y."
        ),
    )

    parser.add_argument(
        "--orientation_camera",
        choices=["rgb", "left", "right", "avg_left_right"],
        default="rgb",
        help=(
            "Which Aria camera orientation to use. "
            "For head orientation, rgb or avg_left_right are usually reasonable."
        ),
    )

    parser.add_argument(
        "--forward_axis",
        choices=["+z", "-z", "+x", "-x", "+y", "-y"],
        default="+z",
        help=(
            "Camera-local forward axis. Try +z first. "
            "If visualization shows the direction is backward, try -z."
        ),
    )
    parser.add_argument("--gt_obj", default=None)
    parser.add_argument("--head_index", type=int, default=15)
    parser.add_argument("--root_index", type=int, default=0)
    parser.add_argument(
        "--remove_colmap_scale",
        action="store_true",
        help=(
            "Remove the Aria-to-COLMAP similarity scale from the exo-camera "
            "transform. Use this when targets should match mesh_cam_unscaled."
        ),
    )
    args = parser.parse_args()
    axis_map = {
        "+z": np.array([0.0, 0.0, 1.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
        "+x": np.array([1.0, 0.0, 0.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "+y": np.array([0.0, 1.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
    }

    aria_forward_axis = axis_map[args.forward_axis]
    aria_up_axis = axis_map[args.up_axis]

    data_root = Path(args.data_root).expanduser()

    frame_int = int(args.frame)
    frame_name = f"{frame_int:05d}"

    calib_txt = data_root / "ego" /args.aria / "calib" / f"{frame_name}.txt"
    images_txt = data_root / "colmap" / "workplace" / "images.txt"
    image_name = f"{args.exo_cam}/{frame_name}.jpg"

    colmap_from_aria = load_colmap_from_aria(data_root)

    primary_transform = colmap_from_aria[args.anchor_aria]

    # Same as EgoExoScene:
    # coordinate_transform = inv(colmap_from_aria[aria_name]) @ primary_transform
    coordinate_transform = np.linalg.inv(colmap_from_aria[args.aria]) @ primary_transform

    # Read Aria calibration exactly like EgoHumans.
    calib = read_aria_calibration_like_egohumans(calib_txt)

    # Same as AriaHuman.update:
    # rgb_extrinsics = rgb_extrinsics @ coordinate_transform
    T_rgb_from_anchor_world = calib["rgb"]["extrinsics"] @ coordinate_transform
    T_left_from_anchor_world = calib["left"]["extrinsics"] @ coordinate_transform
    T_right_from_anchor_world = calib["right"]["extrinsics"] @ coordinate_transform

    # Camera centers in anchor/world coordinate.
    rgb_center_world = camera_center_from_world_to_cam(T_rgb_from_anchor_world)
    left_center_world = camera_center_from_world_to_cam(T_left_from_anchor_world)
    right_center_world = camera_center_from_world_to_cam(T_right_from_anchor_world)

    # Same as AriaHuman.update:
    # self.location = (left_cam.location + right_cam.location) / 2
    glasses_center_world = 0.5 * (left_center_world + right_center_world)

    # Same as ExoCamera.update:
    # raw_extrinsics = cam01_colmap_from_colmap_world
    # exo_extrinsics = raw_extrinsics @ primary_transform
    T_exo_from_colmap, used_colmap_image = load_colmap_image_extrinsic(
        images_txt,
        image_name,
        allow_nearest=True,
    )

    print(f"Requested COLMAP image: {image_name}")
    print(f"Used COLMAP image:      {used_colmap_image}")

    # EgoHumans original exo transform:
    #   cam_from_anchor_world_scaled = cam_from_colmap @ colmap_from_anchor_aria
    #
    # primary_transform contains a similarity scale s ~= 1.25.
    # If we want targets compatible with mesh_cam_unscaled, remove this scale.
    T_exo_from_anchor_world = T_exo_from_colmap @ primary_transform

    colmap_scale = scale_from_similarity_transform(primary_transform)

    print("\n========== Scale debug ==========")
    print("primary_transform scale:", colmap_scale)
    print("scaled exo 3x3 column norms:",
          np.linalg.norm(T_exo_from_anchor_world[:3, :3], axis=0))

    if args.remove_colmap_scale:
        T_exo_from_anchor_world_unscaled = T_exo_from_anchor_world.copy()
        T_exo_from_anchor_world_unscaled[:3, :] /= colmap_scale
        T_exo_from_anchor_world = T_exo_from_anchor_world_unscaled

        print("[remove_colmap_scale] Enabled.")
        print("unscaled exo 3x3 column norms:",
              np.linalg.norm(T_exo_from_anchor_world[:3, :3], axis=0))
    else:
        print("[remove_colmap_scale] Disabled. Using original EgoHumans scaled target.")
    
    # ------------------------------------------------------------
    # Aria orientation in cam01 positive-Z coordinate
    # ------------------------------------------------------------
    rgb_forward_cam01 = direction_from_aria_to_exo(
        T_rgb_from_anchor_world,
        T_exo_from_anchor_world,
        aria_axis=aria_forward_axis,
    )

    left_forward_cam01 = direction_from_aria_to_exo(
        T_left_from_anchor_world,
        T_exo_from_anchor_world,
        aria_axis=aria_forward_axis,
    )

    right_forward_cam01 = direction_from_aria_to_exo(
        T_right_from_anchor_world,
        T_exo_from_anchor_world,
        aria_axis=aria_forward_axis,
    )

    avg_left_right_forward_cam01 = normalize_vector(
        left_forward_cam01 + right_forward_cam01
    )

    # ------------------------------------------------------------
    # Aria UP direction in cam01 positive-Z coordinate
    # ------------------------------------------------------------
    rgb_up_cam01 = direction_from_aria_to_exo(
        T_rgb_from_anchor_world,
        T_exo_from_anchor_world,
        aria_axis=aria_up_axis,
    )

    left_up_cam01 = direction_from_aria_to_exo(
        T_left_from_anchor_world,
        T_exo_from_anchor_world,
        aria_axis=aria_up_axis,
    )

    right_up_cam01 = direction_from_aria_to_exo(
        T_right_from_anchor_world,
        T_exo_from_anchor_world,
        aria_axis=aria_up_axis,
    )

    avg_left_right_up_cam01 = normalize_vector(
        left_up_cam01 + right_up_cam01
    )

    if args.orientation_camera == "rgb":
        aria_up_cam01 = rgb_up_cam01
    elif args.orientation_camera == "left":
        aria_up_cam01 = left_up_cam01
    elif args.orientation_camera == "right":
        aria_up_cam01 = right_up_cam01
    elif args.orientation_camera == "avg_left_right":
        aria_up_cam01 = avg_left_right_up_cam01
    else:
        raise ValueError(args.orientation_camera)

    if args.orientation_camera == "rgb":
        aria_forward_cam01 = rgb_forward_cam01
    elif args.orientation_camera == "left":
        aria_forward_cam01 = left_forward_cam01
    elif args.orientation_camera == "right":
        aria_forward_cam01 = right_forward_cam01
    elif args.orientation_camera == "avg_left_right":
        aria_forward_cam01 = avg_left_right_forward_cam01
    else:
        raise ValueError(args.orientation_camera)

    # Express Aria points in cam01 positive-Z camera coordinate.
    rgb_center_cam01 = transform_point(T_exo_from_anchor_world, rgb_center_world)
    left_center_cam01 = transform_point(T_exo_from_anchor_world, left_center_world)
    right_center_cam01 = transform_point(T_exo_from_anchor_world, right_center_world)
    glasses_center_cam01 = transform_point(T_exo_from_anchor_world, glasses_center_world)

    print("\n========== EgoHumans-style Aria camera extraction ==========")
    print("data_root:", data_root)
    print("frame:", frame_name)
    print("aria:", args.aria)
    print("exo_cam:", args.exo_cam)
    print("anchor_aria:", args.anchor_aria)
    print("calib:", calib_txt)
    print("COLMAP image:", image_name)

    print("\n--- coordinate_transform = inv(colmap_from_aria[aria]) @ colmap_from_aria[anchor] ---")
    print(coordinate_transform)

    print("\n--- centers in EgoHumans anchor/world coordinate ---")
    print("rgb_center_world:    ", rgb_center_world)
    print("left_center_world:   ", left_center_world)
    print("right_center_world:  ", right_center_world)
    print("glasses_center_world:", glasses_center_world)

    print("\n--- centers in cam01 positive-Z camera coordinate ---")
    print("rgb_center_cam01:    ", rgb_center_cam01)
    print("left_center_cam01:   ", left_center_cam01)
    print("right_center_cam01:  ", right_center_cam01)
    print("glasses_center_cam01:", glasses_center_cam01)

    print("\n--- centers in mesh_cam/export coordinate ---")
    print("rgb_center_mesh_cam:    ", posz_to_mesh_cam(rgb_center_cam01))
    print("left_center_mesh_cam:   ", posz_to_mesh_cam(left_center_cam01))
    print("right_center_mesh_cam:  ", posz_to_mesh_cam(right_center_cam01))
    print("glasses_center_mesh_cam:", posz_to_mesh_cam(glasses_center_cam01))

    print("\n--- Aria forward directions in cam01 positive-Z coordinate ---")
    print("forward_axis in Aria camera:", args.forward_axis)
    print("rgb_forward_cam01:           ", rgb_forward_cam01)
    print("left_forward_cam01:          ", left_forward_cam01)
    print("right_forward_cam01:         ", right_forward_cam01)
    print("avg_left_right_forward_cam01:", avg_left_right_forward_cam01)
    print("SELECTED aria_forward_cam01: ", aria_forward_cam01)

    print("\n--- Aria UP directions in cam01 positive-Z coordinate ---")
    print("up_axis in Aria camera:", args.up_axis)
    print("rgb_up_cam01:           ", rgb_up_cam01)
    print("left_up_cam01:          ", left_up_cam01)
    print("right_up_cam01:         ", right_up_cam01)
    print("avg_left_right_up_cam01:", avg_left_right_up_cam01)
    print("SELECTED aria_up_cam01: ", aria_up_cam01)

    print("\n--- Aria forward directions in mesh_cam/export vector convention ---")
    print("rgb_forward_mesh_cam:           ", posz_vector_to_mesh_cam(rgb_forward_cam01))
    print("left_forward_mesh_cam:          ", posz_vector_to_mesh_cam(left_forward_cam01))
    print("right_forward_mesh_cam:         ", posz_vector_to_mesh_cam(right_forward_cam01))
    print("avg_left_right_forward_mesh_cam:", posz_vector_to_mesh_cam(avg_left_right_forward_cam01))
    print("SELECTED aria_forward_mesh_cam: ", posz_vector_to_mesh_cam(aria_forward_cam01))

    print("\n--- Aria UP directions in mesh_cam/export vector convention ---")
    print("rgb_up_mesh_cam:           ", posz_vector_to_mesh_cam(rgb_up_cam01))
    print("left_up_mesh_cam:          ", posz_vector_to_mesh_cam(left_up_cam01))
    print("right_up_mesh_cam:         ", posz_vector_to_mesh_cam(right_up_cam01))
    print("avg_left_right_up_mesh_cam:", posz_vector_to_mesh_cam(avg_left_right_up_cam01))
    print("SELECTED aria_up_mesh_cam: ", posz_vector_to_mesh_cam(aria_up_cam01))

    if args.gt_obj is not None:
        gt_obj = Path(args.gt_obj).expanduser()
        verts_mesh = load_obj_vertices(gt_obj)
        J = load_smpl_regressor(args.smpl_model_dir)
        joints_mesh = J @ verts_mesh
        joints_posz = mesh_cam_to_posz(joints_mesh)

        gt_head = joints_posz[args.head_index]
        gt_root = joints_posz[args.root_index]
        gt_center = mesh_cam_to_posz(verts_mesh.mean(axis=0))

        print("\n========== GT comparison ==========")
        print("GT head positive-Z:", gt_head)
        print("GT root positive-Z:", gt_root)
        print("GT mesh center positive-Z:", gt_center)

        for name, p in [
            ("rgb", rgb_center_cam01),
            ("left", left_center_cam01),
            ("right", right_center_cam01),
            ("glasses_avg_left_right", glasses_center_cam01),
        ]:
            print(f"\n{name}:")
            print("  distance to GT head:  ", np.linalg.norm(p - gt_head), "m")
            print("  distance to GT root:  ", np.linalg.norm(p - gt_root), "m")
            print("  distance to GT center:", np.linalg.norm(p - gt_center), "m")

    if args.save_target is not None:
        out_path = Path(args.save_target).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # I recommend using glasses_center_cam01 first, because EgoHumans uses
        # human.location = average(left_cam, right_cam) as the head/glasses proxy.
        np.savetxt(out_path, glasses_center_cam01.reshape(1, 3), fmt="%.8f")
        print("\n[Saved] glasses center target in cam01 positive-Z coordinate:")
        print(out_path)
        if args.remove_colmap_scale:
            print("[Target convention] UN SCALED cam01 coordinate, compatible with mesh_cam_unscaled.")
        else:
            print("[Target convention] Original scaled EgoHumans cam01 coordinate, compatible with mesh_cam.")
    # Backward-compatible: old argument still saves forward direction.
    if args.save_orientation_target is not None:
        out_path = Path(args.save_orientation_target).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        np.savetxt(out_path, aria_forward_cam01.reshape(1, 3), fmt="%.8f")

        print("\n[Saved] Aria forward direction in cam01 positive-Z coordinate:")
        print(out_path)
        print("orientation_camera:", args.orientation_camera)
        print("forward_axis:", args.forward_axis)
        print("value:", aria_forward_cam01)

    # Explicit forward output.
    if args.save_forward_target is not None:
        out_path = Path(args.save_forward_target).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        np.savetxt(out_path, aria_forward_cam01.reshape(1, 3), fmt="%.8f")

        print("\n[Saved] Aria FORWARD direction in cam01 positive-Z coordinate:")
        print(out_path)
        print("orientation_camera:", args.orientation_camera)
        print("forward_axis:", args.forward_axis)
        print("value:", aria_forward_cam01)

    # Explicit up output.
    if args.save_up_target is not None:
        out_path = Path(args.save_up_target).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        np.savetxt(out_path, aria_up_cam01.reshape(1, 3), fmt="%.8f")

        print("\n[Saved] Aria UP direction in cam01 positive-Z coordinate:")
        print(out_path)
        print("orientation_camera:", args.orientation_camera)
        print("up_axis:", args.up_axis)
        print("value:", aria_up_cam01)


if __name__ == "__main__":
    main()