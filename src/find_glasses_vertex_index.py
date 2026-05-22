#!/usr/bin/env python3

import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import plotly.graph_objects as go


def load_obj_vertices_faces(obj_path):
    obj_path = Path(obj_path).expanduser()

    vertices = []
    faces = []

    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                p = line.strip().split()
                vertices.append([float(p[1]), float(p[2]), float(p[3])])

            elif line.startswith("f "):
                p = line.strip().split()[1:]
                face = [int(x.split("/")[0]) - 1 for x in p]

                if len(face) == 3:
                    faces.append(face)
                elif len(face) > 3:
                    for i in range(1, len(face) - 1):
                        faces.append([face[0], face[i], face[i + 1]])

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    if vertices.shape != (6890, 3):
        raise ValueError(f"Expected vertices shape (6890,3), got {vertices.shape}: {obj_path}")

    return vertices, faces


def load_target_camera(path):
    """
    head_targets_unscaled/*.txt stores positive-Z camera coordinate:
        [x, y, z]
    """
    path = Path(path).expanduser()

    vals = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            vals.extend([float(x) for x in line.replace(",", " ").split()])

    if len(vals) < 3:
        raise ValueError(f"Could not read xyz from {path}")

    return np.asarray(vals[:3], dtype=np.float32)


def camera_to_mesh_cam(p):
    """
    positive-Z camera:
        [x, y, z]

    mesh_cam/export:
        [x, -y, -z]
    """
    p = np.asarray(p, dtype=np.float32).copy()
    p[..., 1] *= -1.0
    p[..., 2] *= -1.0
    return p


def topk_nearest_vertices(vertices, target_mesh_cam, k=10):
    dist = np.linalg.norm(vertices - target_mesh_cam[None, :], axis=1)
    idxs = np.argsort(dist)[:k]

    return [(int(i), float(dist[i])) for i in idxs]


def add_mesh(fig, vertices, faces, name, opacity=0.35):
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name=name,
            color="lightblue",
            opacity=opacity,
            flatshading=False,
            showscale=False,
        )
    )


def add_point(fig, p, name, color="red", size=6):
    fig.add_trace(
        go.Scatter3d(
            x=[p[0]],
            y=[p[1]],
            z=[p[2]],
            mode="markers+text",
            marker=dict(size=size, color=color),
            text=[name],
            textposition="top center",
            name=name,
        )
    )


def set_bounds(fig, points):
    pts = np.concatenate([np.asarray(x).reshape(-1, 3) for x in points], axis=0)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = 0.5 * (mn + mx)
    radius = max(0.5 * np.max(mx - mn), 0.5)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[center[0] - radius, center[0] + radius], title="X"),
            yaxis=dict(range=[center[1] - radius, center[1] + radius], title="Y"),
            zaxis=dict(range=[center[2] - radius, center[2] + radius], title="Z"),
            aspectmode="cube",
        )
    )


def save_debug_html(out_html, vertices, faces, target_mesh_cam, nearest_list, title):
    fig = go.Figure()

    add_mesh(fig, vertices, faces, "GT mesh", opacity=0.35)
    add_point(fig, target_mesh_cam, "Aria/glasses target", "brown", size=8)

    for rank, (vidx, dist) in enumerate(nearest_list[:10]):
        p = vertices[vidx]
        add_point(
            fig,
            p,
            f"rank{rank+1}: v{vidx}, d={dist*1000:.1f}mm",
            "red" if rank == 0 else "orange",
            size=7 if rank == 0 else 5,
        )

    set_bounds(fig, [vertices, target_mesh_cam.reshape(1, 3)])

    fig.update_layout(
        title=title,
        width=1300,
        height=900,
        showlegend=True,
    )

    out_html = Path(out_html).expanduser()
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    print(f"[Saved debug HTML] {out_html}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gt_root", default="data/mesh_cam_unscaled/cam01/rgb")
    parser.add_argument("--target_dir", default="head_targets_unscaled")
    parser.add_argument("--exo_cam", default="cam01")

    parser.add_argument(
        "--frames",
        nargs="+",
        default=["00001"],
        help="Frames to use, e.g. --frames 00001 00002 00003",
    )
    parser.add_argument(
        "--arias",
        nargs="+",
        default=["aria01", "aria02", "aria03", "aria04"],
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--save_debug_html", default=None)

    args = parser.parse_args()

    gt_root = Path(args.gt_root).expanduser()
    target_dir = Path(args.target_dir).expanduser()

    counter = Counter()
    dist_by_vertex = defaultdict(list)
    all_records = []

    for frame in args.frames:
        for aria in args.arias:
            gt_obj = gt_root / frame / f"mesh_{aria}.obj"
            target_file = target_dir / f"{frame}_{aria}_{args.exo_cam}_egohumans_style.txt"

            if not gt_obj.exists():
                print(f"[Skip] missing GT: {gt_obj}")
                continue

            if not target_file.exists():
                print(f"[Skip] missing target: {target_file}")
                continue

            vertices, faces = load_obj_vertices_faces(gt_obj)
            target_camera = load_target_camera(target_file)
            target_mesh_cam = camera_to_mesh_cam(target_camera)

            nearest = topk_nearest_vertices(vertices, target_mesh_cam, k=args.topk)

            print()
            print("============================================================")
            print(f"frame={frame}, aria={aria}")
            print("GT:", gt_obj)
            print("target camera:", target_camera)
            print("target mesh_cam:", target_mesh_cam)
            print("Top nearest vertices:")
            for rank, (vidx, dist) in enumerate(nearest):
                print(f"  rank {rank+1:02d}: vertex {vidx:5d}, distance {dist*1000.0:7.2f} mm")

            best_vidx, best_dist = nearest[0]
            counter[best_vidx] += 1
            dist_by_vertex[best_vidx].append(best_dist)
            all_records.append((frame, aria, best_vidx, best_dist))

            if args.save_debug_html is not None and len(all_records) == 1:
                save_debug_html(
                    args.save_debug_html,
                    vertices,
                    faces,
                    target_mesh_cam,
                    nearest,
                    title=f"{frame} {aria}: nearest SMPL vertices to Aria/glasses target",
                )

    print()
    print("############################################################")
    print("Aggregate result")
    print("############################################################")

    if len(counter) == 0:
        raise RuntimeError("No valid frame/aria pairs were processed.")

    print("Most frequent nearest vertex indices:")
    for vidx, count in counter.most_common(20):
        dists = np.asarray(dist_by_vertex[vidx], dtype=np.float32)
        print(
            f"vertex {vidx:5d}: "
            f"count={count:3d}, "
            f"mean_dist={dists.mean()*1000.0:7.2f} mm, "
            f"median_dist={np.median(dists)*1000.0:7.2f} mm"
        )

    print()
    print("Per-sample best vertex:")
    for frame, aria, vidx, dist in all_records:
        print(f"{frame} {aria}: vertex {vidx}, dist={dist*1000.0:.2f}mm")


if __name__ == "__main__":
    main()