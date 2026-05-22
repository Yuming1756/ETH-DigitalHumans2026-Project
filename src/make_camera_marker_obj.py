#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np


def load_target_camera(path):
    vals = []
    with open(Path(path).expanduser(), "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals.extend([float(x) for x in line.replace(",", " ").split()])

    if len(vals) < 3:
        raise ValueError(f"Could not read xyz from {path}")

    return np.asarray(vals[:3], dtype=np.float32)


def camera_to_mesh_cam(p):
    p = np.asarray(p, dtype=np.float32).copy()
    p[..., 1] *= -1.0
    p[..., 2] *= -1.0
    return p


def make_uv_sphere(center, radius=0.02, rings=12, sectors=24):
    verts = []
    faces = []

    for r in range(rings + 1):
        theta = np.pi * r / rings
        for s in range(sectors):
            phi = 2 * np.pi * s / sectors
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            verts.append(center + np.array([x, y, z], dtype=np.float32))

    for r in range(rings):
        for s in range(sectors):
            a = r * sectors + s
            b = r * sectors + (s + 1) % sectors
            c = (r + 1) * sectors + s
            d = (r + 1) * sectors + (s + 1) % sectors
            faces.append([a, c, b])
            faces.append([b, c, d])

    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def make_camera_frustum(center, scale=0.08):
    """
    A small pyramid/frustum marker around the camera/glasses point.
    This is only a visual marker, not the true camera orientation.
    """
    c = np.asarray(center, dtype=np.float32)

    verts = np.array([
        c,
        c + np.array([-scale, -scale, -scale], dtype=np.float32),
        c + np.array([ scale, -scale, -scale], dtype=np.float32),
        c + np.array([ scale,  scale, -scale], dtype=np.float32),
        c + np.array([-scale,  scale, -scale], dtype=np.float32),
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        [1, 2, 3],
        [1, 3, 4],
    ], dtype=np.int32)

    return verts, faces


def save_obj(path, vertices, faces, color=(0.45, 0.20, 0.05)):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for v in vertices:
            f.write(
                f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} "
                f"{color[0]:.8f} {color[1]:.8f} {color[2]:.8f}\n"
            )

        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"[Saved] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_file", required=True)
    parser.add_argument("--out_obj", required=True)
    parser.add_argument("--radius", type=float, default=0.02)
    parser.add_argument(
        "--mode",
        choices=["sphere", "frustum"],
        default="sphere",
    )
    args = parser.parse_args()

    target_camera = load_target_camera(args.target_file)
    target_mesh_cam = camera_to_mesh_cam(target_camera)

    print("target positive-Z camera:", target_camera)
    print("target mesh_cam:", target_mesh_cam)

    if args.mode == "sphere":
        V, F = make_uv_sphere(target_mesh_cam, radius=args.radius)
    else:
        V, F = make_camera_frustum(target_mesh_cam, scale=args.radius)

    save_obj(args.out_obj, V, F)


if __name__ == "__main__":
    main()