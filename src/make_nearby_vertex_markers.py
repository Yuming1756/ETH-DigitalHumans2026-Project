#!/usr/bin/env python3

import argparse
from pathlib import Path
from collections import deque

import numpy as np


def load_obj_vertices_faces(obj_path):
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


def build_adjacency(num_vertices, faces):
    adj = [set() for _ in range(num_vertices)]

    for a, b, c in faces:
        adj[a].add(b)
        adj[a].add(c)
        adj[b].add(a)
        adj[b].add(c)
        adj[c].add(a)
        adj[c].add(b)

    return adj


def bfs_rings(seed, adj, max_ring):
    """
    Return dict:
        ring_id -> list of vertex ids
    """
    visited = {seed}
    q = deque([(seed, 0)])
    rings = {0: [seed]}

    while q:
        v, d = q.popleft()

        if d >= max_ring:
            continue

        for nb in sorted(adj[v]):
            if nb in visited:
                continue

            visited.add(nb)
            rings.setdefault(d + 1, []).append(nb)
            q.append((nb, d + 1))

    return rings


def save_mesh_obj(path, vertices, faces, color=(0.75, 0.75, 0.75)):
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


def make_uv_sphere(center, radius=0.003, rings=8, sectors=16):
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


def save_marker_obj(path, vertices, faces, color):
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


def make_markers_for_vertices(vertices, vertex_ids, radius):
    all_v = []
    all_f = []
    offset = 0

    for vid in vertex_ids:
        sv, sf = make_uv_sphere(vertices[vid], radius=radius)
        all_v.append(sv)
        all_f.append(sf + offset)
        offset += len(sv)

    if len(all_v) == 0:
        return None, None

    return np.concatenate(all_v, axis=0), np.concatenate(all_f, axis=0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mesh", required=True)
    parser.add_argument("--seed", type=int, required=True, help="Seed vertex, e.g. nose tip 332")
    parser.add_argument("--rings", type=int, default=4)
    parser.add_argument("--radius", type=float, default=0.003)

    parser.add_argument("--out_mesh", required=True)
    parser.add_argument("--out_prefix", required=True)

    parser.add_argument(
        "--separate_each_vertex",
        action="store_true",
        help="Also create one marker OBJ per vertex, useful because layer name shows vertex id.",
    )

    args = parser.parse_args()

    V, F = load_obj_vertices_faces(args.mesh)

    if args.seed < 0 or args.seed >= len(V):
        raise ValueError(f"seed {args.seed} outside vertex range 0..{len(V)-1}")

    adj = build_adjacency(len(V), F)
    rings = bfs_rings(args.seed, adj, args.rings)

    save_mesh_obj(args.out_mesh, V, F)

    print("============================================================")
    print("Nearby vertices around seed")
    print("mesh:", args.mesh)
    print("seed:", args.seed)
    print("seed coordinate:", V[args.seed])
    print("rings:", args.rings)
    print("============================================================")

    # Colors per ring, OBJ vertex colors.
    ring_colors = {
        0: (1.0, 0.0, 0.0),      # red
        1: (1.0, 0.5, 0.0),      # orange
        2: (1.0, 1.0, 0.0),      # yellow
        3: (0.0, 1.0, 0.0),      # green
        4: (0.0, 1.0, 1.0),      # cyan
        5: (0.0, 0.0, 1.0),      # blue
        6: (1.0, 0.0, 1.0),      # magenta
    }

    out_prefix = Path(args.out_prefix).expanduser()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary_path = out_prefix.with_suffix(".txt")

    with open(summary_path, "w") as summary:
        for ring_id in sorted(rings.keys()):
            vids = rings[ring_id]
            color = ring_colors.get(ring_id, (1.0, 0.0, 1.0))

            print()
            print(f"Ring {ring_id}: {len(vids)} vertices")
            print(" ".join(str(v) for v in vids))

            summary.write(f"Ring {ring_id}: {len(vids)} vertices\n")
            summary.write(" ".join(str(v) for v in vids) + "\n\n")

            marker_v, marker_f = make_markers_for_vertices(V, vids, args.radius)
            marker_path = out_prefix.parent / f"{out_prefix.name}_ring{ring_id}.obj"
            save_marker_obj(marker_path, marker_v, marker_f, color=color)

            print("saved:", marker_path)

            if args.separate_each_vertex:
                per_vertex_dir = out_prefix.parent / f"{out_prefix.name}_per_vertex"
                per_vertex_dir.mkdir(parents=True, exist_ok=True)

                for vid in vids:
                    mv, mf = make_markers_for_vertices(V, [vid], args.radius)
                    one_path = per_vertex_dir / f"vertex_{vid}.obj"
                    save_marker_obj(one_path, mv, mf, color=color)

    print()
    print("Saved mesh:", args.out_mesh)
    print("Saved summary:", summary_path)
    print()
    print("Open in MeshLab, for example:")
    print(f"meshlab {args.out_mesh} {out_prefix.parent}/{out_prefix.name}_ring*.obj")


if __name__ == "__main__":
    main()