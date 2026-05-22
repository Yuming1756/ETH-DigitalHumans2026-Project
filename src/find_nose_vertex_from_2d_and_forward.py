#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np


def load_obj_vertices(obj_path):
    vertices = []
    with open(Path(obj_path).expanduser(), "r") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                vertices.append([float(p[1]), float(p[2]), float(p[3])])

    vertices = np.asarray(vertices, dtype=np.float32)

    if vertices.shape != (6890, 3):
        raise ValueError(f"Expected (6890,3), got {vertices.shape}")

    return vertices


def mesh_cam_to_camera(points):
    points = np.asarray(points, dtype=np.float32).copy()
    points[..., 1] *= -1.0
    points[..., 2] *= -1.0
    return points


def load_vec3(path):
    vals = []
    with open(Path(path).expanduser(), "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals.extend([float(x) for x in line.replace(",", " ").split()])

    if len(vals) < 3:
        raise ValueError(f"Could not read xyz from {path}")

    v = np.asarray(vals[:3], dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise ValueError(f"Near-zero vector in {path}: {v}")

    return v / n


def project_points(points_camera, fx, fy, cx, cy, eps=1e-6):
    z = np.maximum(points_camera[:, 2], eps)
    u = fx * points_camera[:, 0] / z + cx
    v = fy * points_camera[:, 1] / z + cy
    return np.stack([u, v], axis=1)


def load_poses2d_for_aria(poses2d_path, aria):
    data = np.load(Path(poses2d_path).expanduser(), allow_pickle=True)

    if isinstance(data, np.ndarray) and data.dtype == object:
        data = list(data)

    for item in data:
        if isinstance(item, dict) and item.get("human_name") == aria:
            return np.asarray(item["keypoints"], dtype=np.float32)

    available = [
        item.get("human_name")
        for item in data
        if isinstance(item, dict) and "human_name" in item
    ]

    raise KeyError(f"Could not find {aria}. Available: {available}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mesh", required=True, help="GT mesh_cam_unscaled OBJ")
    parser.add_argument("--poses2d", required=True, help="EgoHumans poses2d npy")
    parser.add_argument("--aria", required=True)
    parser.add_argument("--aria_forward_file", required=True)

    parser.add_argument("--left_eye_vertex", type=int, required=True)
    parser.add_argument("--right_eye_vertex", type=int, required=True)

    parser.add_argument("--topk_2d", type=int, default=200)
    parser.add_argument("--show", type=int, default=30)

    parser.add_argument("--fx", type=float, default=1915.346061677637)
    parser.add_argument("--fy", type=float, default=1915.5916704165572)
    parser.add_argument("--cx", type=float, default=1920.0)
    parser.add_argument("--cy", type=float, default=1080.0)

    args = parser.parse_args()

    vertices_mesh = load_obj_vertices(args.mesh)
    vertices_camera = mesh_cam_to_camera(vertices_mesh)

    aria_forward = load_vec3(args.aria_forward_file)

    kpts = load_poses2d_for_aria(args.poses2d, args.aria)

    # COCO-17:
    # 0 = nose
    nose = kpts[0, :3]
    nose_uv = nose[:2]
    nose_conf = nose[2]

    proj_uv = project_points(
        vertices_camera,
        args.fx,
        args.fy,
        args.cx,
        args.cy,
    )

    dist_px = np.linalg.norm(proj_uv - nose_uv[None, :], axis=1)

    eye_center = 0.5 * (
        vertices_camera[args.left_eye_vertex]
        + vertices_camera[args.right_eye_vertex]
    )

    # Positive means vertex is in front of the eye center along Aria forward.
    protrusion = (vertices_camera - eye_center[None, :]) @ aria_forward

    # First restrict to vertices that project near the 2D nose.
    candidates = np.argsort(dist_px)[: args.topk_2d]

    # Among 2D-near candidates, choose the most forward/protruding ones.
    candidates_sorted = sorted(
        candidates,
        key=lambda i: (-protrusion[i], dist_px[i])
    )

    print("============================================================")
    print("Find nose vertex using 2D nose + Aria forward")
    print("mesh:", args.mesh)
    print("poses2d:", args.poses2d)
    print("aria:", args.aria)
    print("aria_forward:", aria_forward)
    print("left_eye_vertex:", args.left_eye_vertex)
    print("right_eye_vertex:", args.right_eye_vertex)
    print("eye_center_camera:", eye_center)
    print("COCO nose uv/conf:", nose_uv, nose_conf)
    print("topk_2d:", args.topk_2d)
    print("============================================================")

    print("\nBest candidates by forward protrusion among 2D-near vertices:")
    for rank, vidx in enumerate(candidates_sorted[: args.show], start=1):
        print(
            f"rank {rank:02d}: "
            f"vertex {int(vidx):5d}, "
            f"proj_dist={dist_px[vidx]:7.2f}px, "
            f"protrusion={protrusion[vidx]*1000.0:8.2f}mm, "
            f"proj_uv=({proj_uv[vidx,0]:.2f}, {proj_uv[vidx,1]:.2f}), "
            f"mesh_xyz={vertices_mesh[vidx]}"
        )

    print("\nBest candidates by pure 2D projection:")
    idxs_2d = np.argsort(dist_px)[: args.show]
    for rank, vidx in enumerate(idxs_2d, start=1):
        print(
            f"rank {rank:02d}: "
            f"vertex {int(vidx):5d}, "
            f"proj_dist={dist_px[vidx]:7.2f}px, "
            f"protrusion={protrusion[vidx]*1000.0:8.2f}mm"
        )


if __name__ == "__main__":
    main()