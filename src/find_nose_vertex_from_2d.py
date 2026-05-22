#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np


def load_obj_vertices(obj_path):
    verts = []
    with open(Path(obj_path).expanduser(), "r") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
    return np.asarray(verts, dtype=np.float32)


def mesh_cam_to_camera(points):
    points = np.asarray(points, dtype=np.float32).copy()
    points[..., 1] *= -1.0
    points[..., 2] *= -1.0
    return points


def project(points_camera, fx, fy, cx, cy, eps=1e-6):
    z = np.maximum(points_camera[:, 2], eps)
    u = fx * points_camera[:, 0] / z + cx
    v = fy * points_camera[:, 1] / z + cy
    return np.stack([u, v], axis=1)


def load_aria_keypoints(poses2d_path, aria):
    data = np.load(Path(poses2d_path).expanduser(), allow_pickle=True)
    data = list(data) if isinstance(data, np.ndarray) and data.dtype == object else data

    for item in data:
        if isinstance(item, dict) and item.get("human_name") == aria:
            return np.asarray(item["keypoints"], dtype=np.float32)

    raise KeyError(f"Could not find {aria} in {poses2d_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="GT mesh_cam_unscaled OBJ")
    parser.add_argument("--poses2d", required=True, help="EgoHumans poses2d npy")
    parser.add_argument("--aria", required=True)
    parser.add_argument("--topk", type=int, default=30)

    parser.add_argument("--fx", type=float, default=1915.346061677637)
    parser.add_argument("--fy", type=float, default=1915.5916704165572)
    parser.add_argument("--cx", type=float, default=1920.0)
    parser.add_argument("--cy", type=float, default=1080.0)

    args = parser.parse_args()

    V_mesh = load_obj_vertices(args.mesh)
    V_cam = mesh_cam_to_camera(V_mesh)

    kpts = load_aria_keypoints(args.poses2d, args.aria)
    coco17 = kpts[:17]

    nose = coco17[0, :3]  # COCO nose
    nose_uv = nose[:2]
    nose_conf = nose[2]

    proj_uv = project(V_cam, args.fx, args.fy, args.cx, args.cy)
    dist_px = np.linalg.norm(proj_uv - nose_uv[None, :], axis=1)

    idxs = np.argsort(dist_px)[:args.topk]

    print("============================================================")
    print("Find nose vertex candidates from 2D nose keypoint")
    print("mesh:", args.mesh)
    print("poses2d:", args.poses2d)
    print("aria:", args.aria)
    print("COCO nose uv/conf:", nose_uv, nose_conf)
    print("============================================================")

    for rank, vidx in enumerate(idxs, start=1):
        print(
            f"rank {rank:02d}: vertex {int(vidx):5d}, "
            f"proj_dist={dist_px[vidx]:7.2f}px, "
            f"proj_uv=({proj_uv[vidx,0]:.2f}, {proj_uv[vidx,1]:.2f}), "
            f"mesh={V_mesh[vidx]}"
        )


if __name__ == "__main__":
    main()