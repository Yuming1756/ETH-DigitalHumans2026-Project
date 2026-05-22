#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np


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

    return np.asarray(vals[:3], dtype=np.float32)


def normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return v / n


def camera_to_mesh_cam_point(p):
    p = np.asarray(p, dtype=np.float32).copy()
    p[1] *= -1.0
    p[2] *= -1.0
    return p


def camera_to_mesh_cam_vector(v):
    v = np.asarray(v, dtype=np.float32).copy()
    v[1] *= -1.0
    v[2] *= -1.0
    return normalize(v)


def make_cylinder_between(p0, p1, radius=0.01, sectors=16):
    """
    Create a cylinder from p0 to p1.
    """
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    axis = p1 - p0
    length = np.linalg.norm(axis)

    if length < 1e-8:
        raise ValueError("Cylinder length too small.")

    z = axis / length

    # Create arbitrary perpendicular basis x, y.
    tmp = np.array([0, 0, 1], dtype=np.float32)
    if abs(np.dot(tmp, z)) > 0.9:
        tmp = np.array([0, 1, 0], dtype=np.float32)

    x = np.cross(tmp, z)
    x = normalize(x)
    y = np.cross(z, x)
    y = normalize(y)

    verts = []
    faces = []

    for i in range(sectors):
        theta = 2 * np.pi * i / sectors
        circle = radius * (np.cos(theta) * x + np.sin(theta) * y)
        verts.append(p0 + circle)
        verts.append(p1 + circle)

    for i in range(sectors):
        j = (i + 1) % sectors

        a = 2 * i
        b = 2 * i + 1
        c = 2 * j
        d = 2 * j + 1

        faces.append([a, c, b])
        faces.append([b, c, d])

    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def make_cone(p_base, direction, height=0.06, radius=0.025, sectors=20):
    """
    Create a cone with base center p_base and tip p_base + direction * height.
    """
    direction = normalize(direction)
    p_base = np.asarray(p_base, dtype=np.float32)
    tip = p_base + direction * height

    tmp = np.array([0, 0, 1], dtype=np.float32)
    if abs(np.dot(tmp, direction)) > 0.9:
        tmp = np.array([0, 1, 0], dtype=np.float32)

    x = normalize(np.cross(tmp, direction))
    y = normalize(np.cross(direction, x))

    verts = [tip]
    faces = []

    for i in range(sectors):
        theta = 2 * np.pi * i / sectors
        circle = radius * (np.cos(theta) * x + np.sin(theta) * y)
        verts.append(p_base + circle)

    for i in range(sectors):
        a = 0
        b = 1 + i
        c = 1 + ((i + 1) % sectors)
        faces.append([a, b, c])

    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def save_obj(path, vertices, faces, color=(0.0, 0.0, 1.0)):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for v in vertices:
            f.write(
                f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} "
                f"{color[0]:.8f} {color[1]:.8f} {color[2]:.8f}\n"
            )

        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    print("[Saved]", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point_file", required=True, help="Aria/glasses target point, positive-Z camera coordinate")
    parser.add_argument("--direction_file", required=True, help="Aria forward direction, positive-Z camera coordinate")
    parser.add_argument("--out_obj", required=True)
    parser.add_argument("--length", type=float, default=0.25)
    parser.add_argument("--shaft_radius", type=float, default=0.008)
    parser.add_argument("--head_length", type=float, default=0.06)
    parser.add_argument("--head_radius", type=float, default=0.025)
    parser.add_argument("--flip", action="store_true", help="Flip direction by multiplying by -1")
    args = parser.parse_args()

    p_cam = load_vec3(args.point_file)
    d_cam = normalize(load_vec3(args.direction_file))

    if args.flip:
        d_cam = -d_cam

    p = camera_to_mesh_cam_point(p_cam)
    d = camera_to_mesh_cam_vector(d_cam)

    arrow_end = p + d * args.length
    cone_base = p + d * max(args.length - args.head_length, 0.01)

    shaft_v, shaft_f = make_cylinder_between(
        p,
        cone_base,
        radius=args.shaft_radius,
    )

    cone_v, cone_f = make_cone(
        cone_base,
        d,
        height=args.head_length,
        radius=args.head_radius,
    )

    vertices = np.concatenate([shaft_v, cone_v], axis=0)
    faces = np.concatenate([shaft_f, cone_f + len(shaft_v)], axis=0)

    print("point positive-Z camera:", p_cam)
    print("direction positive-Z camera:", d_cam)
    print("point mesh_cam:", p)
    print("direction mesh_cam:", d)
    print("arrow end mesh_cam:", arrow_end)

    save_obj(args.out_obj, vertices, faces, color=(0.0, 0.0, 1.0))


if __name__ == "__main__":
    main()