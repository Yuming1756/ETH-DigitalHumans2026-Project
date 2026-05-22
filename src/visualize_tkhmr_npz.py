#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def camera_to_mesh_cam(points):
    """
    TokenHMR positive-Z camera coordinate:
        [x, y, z]

    Exported OBJ / EgoHumans mesh_cam-like coordinate:
        [x, -y, -z]
    """
    points = np.asarray(points, dtype=np.float32).copy()
    points[..., 1] *= -1.0
    points[..., 2] *= -1.0
    return points


def load_obj_faces(template_obj):
    """
    Load triangular faces from an existing OBJ.

    TokenHMR NPZ stores vertices but not faces, so we copy faces from
    the corresponding exported OBJ, e.g. 00001_0.obj.
    """
    template_obj = Path(template_obj).expanduser()

    faces = []

    with open(template_obj, "r") as f:
        for line in f:
            if line.startswith("f "):
                parts = line.strip().split()[1:]
                face = [int(x.split("/")[0]) - 1 for x in parts]

                if len(face) == 3:
                    faces.append(face)
                elif len(face) > 3:
                    # Triangulate polygon face if needed
                    for i in range(1, len(face) - 1):
                        faces.append([face[0], face[i], face[i + 1]])

    faces = np.asarray(faces, dtype=np.int32)

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Bad face shape from {template_obj}: {faces.shape}")

    return faces


def summarize_array(name, arr):
    arr = np.asarray(arr)

    print(f"\n{name}")
    print("-" * len(name))
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)

    if arr.ndim >= 2 and arr.shape[-1] == 3:
        pts = arr.reshape(-1, 3)
        print("center:", pts.mean(axis=0))
        print("min:   ", pts.min(axis=0))
        print("max:   ", pts.max(axis=0))
        print("extent:", pts.max(axis=0) - pts.min(axis=0))
    else:
        print("values:", arr.reshape(-1)[:20])


def add_mesh(fig, vertices, faces, name, color, opacity):
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name=name,
            color=color,
            opacity=opacity,
            flatshading=False,
            showscale=False,
        )
    )


def add_point(fig, point, name, color, size=5):
    point = np.asarray(point, dtype=np.float32)

    fig.add_trace(
        go.Scatter3d(
            x=[point[0]],
            y=[point[1]],
            z=[point[2]],
            mode="markers+text",
            marker=dict(size=size, color=color),
            text=[name],
            textposition="top center",
            name=name,
        )
    )


def add_axis(fig, length=1.0):
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    axes = [
        ("X", np.array([length, 0.0, 0.0]), "red"),
        ("Y", np.array([0.0, length, 0.0]), "green"),
        ("Z", np.array([0.0, 0.0, length]), "blue"),
    ]

    for name, end, color in axes:
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], end[0]],
                y=[origin[1], end[1]],
                z=[origin[2], end[2]],
                mode="lines+text",
                line=dict(color=color, width=6),
                text=["", name],
                textposition="top center",
                name=f"axis {name}",
            )
        )


def set_scene_bounds(fig, all_points):
    pts = np.concatenate([np.asarray(p).reshape(-1, 3) for p in all_points], axis=0)

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)

    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 1.0)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[center[0] - radius, center[0] + radius], title="X"),
            yaxis=dict(range=[center[1] - radius, center[1] + radius], title="Y"),
            zaxis=dict(range=[center[2] - radius, center[2] + radius], title="Z"),
            aspectmode="cube",
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--npz",
        required=True,
        help="TokenHMR NPZ, e.g. TokenHMR/demo_out/my_image_with_smpl_params/00001_0_tkhmr.npz",
    )
    parser.add_argument(
        "--template_obj",
        required=True,
        help="Corresponding TokenHMR OBJ used to copy faces, e.g. TokenHMR/demo_out/my_image_with_smpl_params/00001_0.obj",
    )
    parser.add_argument(
        "--save_html",
        default=None,
        help="Output HTML path. Default: same stem as NPZ + _visualize.html",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--axis",
        action="store_true",
    )

    args = parser.parse_args()

    npz_path = Path(args.npz).expanduser()
    template_obj = Path(args.template_obj).expanduser()

    if args.save_html is None:
        save_html = npz_path.with_name(npz_path.stem + "_visualize.html")
    else:
        save_html = Path(args.save_html).expanduser()

    data = np.load(npz_path)

    print("============================================================")
    print("TokenHMR NPZ inspection")
    print("NPZ:", npz_path)
    print("Template OBJ:", template_obj)
    print("Available keys:")
    for k in data.files:
        print(" ", k, data[k].shape, data[k].dtype)
    print("============================================================")

    required = ["pred_vertices_local", "pred_cam_t_full"]
    for k in required:
        if k not in data.files:
            raise KeyError(f"Missing required key: {k}")

    pred_vertices_local = data["pred_vertices_local"].astype(np.float32)
    pred_cam_t_full = data["pred_cam_t_full"].astype(np.float32).reshape(3)

    # Full positive-Z camera vertices.
    if "pred_vertices_full_camera" in data.files:
        pred_vertices_full_camera = data["pred_vertices_full_camera"].astype(np.float32)
    else:
        pred_vertices_full_camera = pred_vertices_local + pred_cam_t_full[None, :]

    # Export / mesh_cam convention.
    if "pred_vertices_export" in data.files:
        pred_vertices_export = data["pred_vertices_export"].astype(np.float32)
    else:
        pred_vertices_export = camera_to_mesh_cam(pred_vertices_full_camera)

    summarize_array("pred_vertices_local", pred_vertices_local)
    summarize_array("pred_cam_t_full", pred_cam_t_full)
    summarize_array("pred_vertices_full_camera", pred_vertices_full_camera)
    summarize_array("pred_vertices_export", pred_vertices_export)

    for k in ["global_orient", "body_pose", "betas", "pred_cam", "box_center", "box_size", "img_size", "focal_length", "person_id"]:
        if k in data.files:
            summarize_array(k, data[k])

    faces = load_obj_faces(template_obj)

    fig = go.Figure()
    all_points = []

    # Visualize the exported mesh because it matches the OBJ / mesh_cam convention.
    add_mesh(
        fig,
        pred_vertices_export,
        faces,
        name="pred_vertices_export / OBJ convention",
        color="lightblue",
        opacity=args.opacity,
    )

    center_export = pred_vertices_export.mean(axis=0)
    add_point(fig, center_export, "mesh center export", "black", size=5)

    # Also show the camera translation converted to mesh_cam convention.
    cam_t_export = camera_to_mesh_cam(pred_cam_t_full.reshape(1, 3))[0]
    add_point(fig, cam_t_export, "camera translation export", "red", size=6)

    all_points.append(pred_vertices_export)
    all_points.append(center_export.reshape(1, 3))
    all_points.append(cam_t_export.reshape(1, 3))

    if args.axis:
        add_axis(fig, length=1.0)

    set_scene_bounds(fig, all_points)

    title = f"TokenHMR NPZ visualization: {npz_path.name}"

    fig.update_layout(
        title=title,
        width=1300,
        height=900,
        showlegend=True,
    )

    save_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_html))

    print("\nSaved visualization:")
    print(save_html)


if __name__ == "__main__":
    main()