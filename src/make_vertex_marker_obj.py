#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np


def load_obj(obj_path):
    vertices = []
    faces = []

    with open(Path(obj_path).expanduser(), "r") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                vertices.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                p = line.split()[1:]
                face = [int(x.split("/")[0]) - 1 for x in p]
                if len(face) == 3:
                    faces.append(face)
                elif len(face) > 3:
                    for i in range(1, len(face) - 1):
                        faces.append([face[0], face[i], face[i + 1]])

    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def save_obj(path, vertices, faces, color=(0.75, 0.75, 0.75)):
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


def make_uv_sphere(center, radius=0.025, rings=8, sectors=16):
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


def save_marker_obj(path, marker_vertices, marker_faces, color=(1.0, 0.0, 0.0)):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for v in marker_vertices:
            f.write(
                f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} "
                f"{color[0]:.8f} {color[1]:.8f} {color[2]:.8f}\n"
            )
        for face in marker_faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--vertices", nargs="+", type=int, required=True)
    parser.add_argument("--out_mesh", required=True)
    parser.add_argument("--out_markers", required=True)
    parser.add_argument("--radius", type=float, default=0.025)
    args = parser.parse_args()

    V, F = load_obj(args.mesh)

    save_obj(args.out_mesh, V, F)

    all_marker_v = []
    all_marker_f = []
    offset = 0

    print("Selected vertices:")
    for vidx in args.vertices:
        p = V[vidx]
        print(f"  vertex {vidx}: {p}")

        sv, sf = make_uv_sphere(p, radius=args.radius)
        all_marker_v.append(sv)
        all_marker_f.append(sf + offset)
        offset += len(sv)

    marker_v = np.concatenate(all_marker_v, axis=0)
    marker_f = np.concatenate(all_marker_f, axis=0)

    save_marker_obj(args.out_markers, marker_v, marker_f)

    print("Saved mesh:", args.out_mesh)
    print("Saved markers:", args.out_markers)


if __name__ == "__main__":
    main()