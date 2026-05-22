#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


# ------------------------------------------------------------
# Coordinate convention
# ------------------------------------------------------------
# This script assumes all OBJ meshes are already in the same
# mesh_cam/export coordinate convention:
#
#   mesh_cam/export: [x, y, z] = [x, -y_camera, -z_camera]
#
# Your GT mesh_cam_unscaled OBJ and exported HMR OBJ should already
# be in this convention.
# ------------------------------------------------------------


COLORS = {
    "gt": "rgba(80, 140, 255, 0.35)",          # blue
    "baseline": "rgba(245, 245, 245, 0.45)",  # white
    "v1": "rgba(255, 80, 180, 0.45)",         # pink
    "v2": "rgba(60, 220, 100, 0.45)",         # green
    "target": "rgba(130, 70, 20, 1.0)",       # brown
}


def parse_mapping(mapping_str):
    """
    Example:
        "0:aria03,1:aria02,2:aria01,3:aria04"

    Returns:
        {"0": "aria03", "1": "aria02", ...}
    """
    out = {}

    for pair in mapping_str.split(","):
        pair = pair.strip()
        if not pair:
            continue

        if ":" not in pair:
            raise ValueError(f"Bad mapping item: {pair}")

        det_id, aria = pair.split(":", 1)
        det_id = det_id.strip()
        aria = aria.strip()

        if not aria.startswith("aria"):
            aria = f"aria{aria}"

        out[det_id] = aria

    return out


def load_obj(obj_path):
    obj_path = Path(obj_path).expanduser()

    vertices = []
    faces = []

    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append(
                    [
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    ]
                )

            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face = []

                for p in parts:
                    # Supports f v, f v/vt, f v/vt/vn
                    idx = int(p.split("/")[0]) - 1
                    face.append(idx)

                if len(face) == 3:
                    faces.append(face)
                elif len(face) > 3:
                    # Triangulate polygon fan
                    for i in range(1, len(face) - 1):
                        faces.append([face[0], face[i], face[i + 1]])

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Bad vertices shape from {obj_path}: {vertices.shape}")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Bad faces shape from {obj_path}: {faces.shape}")

    return vertices, faces


def load_target_mesh_cam(target_file):
    """
    Head target files are saved in positive-Z camera coordinates:
        [x, y, z]

    Convert to mesh_cam/export coordinate:
        [x, -y, -z]
    """
    target_file = Path(target_file).expanduser()

    vals = []
    with open(target_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals.extend([float(x) for x in line.replace(",", " ").split()])

    if len(vals) < 3:
        raise ValueError(f"Could not read xyz target from {target_file}")

    p = np.asarray(vals[:3], dtype=np.float32)
    p[1] *= -1.0
    p[2] *= -1.0
    return p


def centroid(vertices):
    return vertices.mean(axis=0)


def add_mesh(fig, vertices, faces, name, color, opacity=None):
    if opacity is None:
        # Plotly opacity can be set separately, but rgba color already has alpha.
        opacity = 1.0

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
            lighting=dict(
                ambient=0.6,
                diffuse=0.5,
                specular=0.2,
                roughness=0.8,
                fresnel=0.1,
            ),
            hoverinfo="name",
        )
    )


def add_point(fig, point, name, color="brown", size=5):
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
            hoverinfo="text",
        )
    )


def add_line(fig, p0, p1, name, color="black", width=3):
    fig.add_trace(
        go.Scatter3d(
            x=[p0[0], p1[0]],
            y=[p0[1], p1[1]],
            z=[p0[2], p1[2]],
            mode="lines",
            line=dict(color=color, width=width),
            name=name,
            hoverinfo="name",
        )
    )


def find_pred_obj(pred_dir, frame, det_id):
    """
    Expected:
        00001_0.obj

    Also tolerates extra suffixes less aggressively by first trying exact.
    """
    pred_dir = Path(pred_dir).expanduser()

    exact = pred_dir / f"{frame}_{det_id}.obj"
    if exact.exists():
        return exact

    candidates = sorted(pred_dir.glob(f"{frame}_{det_id}*.obj"))
    candidates = [p for p in candidates if "_all" not in p.stem]

    if len(candidates) > 0:
        return candidates[0]

    return None


def make_legend_note():
    return (
        "<b>Legend</b><br>"
        "Blue transparent = EgoHumans GT<br>"
        "White transparent = model baseline<br>"
        "Pink transparent = SMPLify-v1<br>"
        "Green transparent = SMPLify-v2<br>"
        "Brown point = Aria/glasses/head target"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--frame", required=True, help="Frame ID, e.g. 00001")

    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--v1_dir", required=True)
    parser.add_argument("--v2_dir", required=True)
    parser.add_argument("--gt_root", required=True)
    parser.add_argument("--target_dir", default=None)

    parser.add_argument(
        "--mapping",
        required=True,
        help='Detection-to-Aria mapping, e.g. "0:aria03,1:aria02,2:aria01,3:aria04"',
    )

    parser.add_argument("--save_html", required=True)
    parser.add_argument("--draw_lines", action="store_true")
    parser.add_argument("--axis", action="store_true")

    parser.add_argument(
        "--show_targets",
        action="store_true",
        default=True,
        help="Show Aria/head target points if target_dir exists.",
    )

    args = parser.parse_args()

    frame = args.frame
    mapping = parse_mapping(args.mapping)

    baseline_dir = Path(args.baseline_dir).expanduser()
    v1_dir = Path(args.v1_dir).expanduser()
    v2_dir = Path(args.v2_dir).expanduser()
    gt_root = Path(args.gt_root).expanduser()
    target_dir = Path(args.target_dir).expanduser() if args.target_dir else None
    save_html = Path(args.save_html).expanduser()
    save_html.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()

    all_points = []

    print("============================================================")
    print("Visualize baseline vs SMPLify-v1 vs SMPLify-v2 vs GT")
    print("frame:", frame)
    print("baseline_dir:", baseline_dir)
    print("v1_dir:", v1_dir)
    print("v2_dir:", v2_dir)
    print("gt_root:", gt_root)
    print("target_dir:", target_dir)
    print("mapping:", mapping)
    print("save_html:", save_html)
    print("============================================================")

    for det_id, aria in sorted(mapping.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        print()
        print("------------------------------------------------------------")
        print(f"det {det_id} -> {aria}")

        baseline_obj = find_pred_obj(baseline_dir, frame, det_id)
        v1_obj = find_pred_obj(v1_dir, frame, det_id)
        v2_obj = find_pred_obj(v2_dir, frame, det_id)
        gt_obj = gt_root / frame / f"mesh_{aria}.obj"

        print("baseline:", baseline_obj)
        print("v1:", v1_obj)
        print("v2:", v2_obj)
        print("gt:", gt_obj)

        if not gt_obj.exists():
            print(f"[Warning] Missing GT OBJ: {gt_obj}")
            continue

        if baseline_obj is None or not baseline_obj.exists():
            print(f"[Warning] Missing baseline OBJ for det {det_id}")
        if v1_obj is None or not v1_obj.exists():
            print(f"[Warning] Missing v1 OBJ for det {det_id}")
        if v2_obj is None or not v2_obj.exists():
            print(f"[Warning] Missing v2 OBJ for det {det_id}")

        # GT
        V_gt, F_gt = load_obj(gt_obj)
        add_mesh(
            fig,
            V_gt,
            F_gt,
            name=f"GT {aria}",
            color=COLORS["gt"],
        )
        all_points.append(V_gt)

        c_gt = centroid(V_gt)
        add_point(fig, c_gt, f"GT center {aria}", color="blue", size=3)

        # Baseline
        if baseline_obj is not None and baseline_obj.exists():
            V_b, F_b = load_obj(baseline_obj)
            add_mesh(
                fig,
                V_b,
                F_b,
                name=f"baseline det{det_id}->{aria}",
                color=COLORS["baseline"],
            )
            all_points.append(V_b)

            c_b = centroid(V_b)
            add_point(fig, c_b, f"baseline center det{det_id}", color="white", size=3)

            if args.draw_lines:
                add_line(
                    fig,
                    c_b,
                    c_gt,
                    name=f"baseline→GT det{det_id}",
                    color="gray",
                    width=4,
                )

        # v1
        if v1_obj is not None and v1_obj.exists():
            V_v1, F_v1 = load_obj(v1_obj)
            add_mesh(
                fig,
                V_v1,
                F_v1,
                name=f"SMPLify-v1 det{det_id}->{aria}",
                color=COLORS["v1"],
            )
            all_points.append(V_v1)

            c_v1 = centroid(V_v1)
            add_point(fig, c_v1, f"v1 center det{det_id}", color="deeppink", size=3)

            if args.draw_lines:
                add_line(
                    fig,
                    c_v1,
                    c_gt,
                    name=f"v1→GT det{det_id}",
                    color="deeppink",
                    width=4,
                )

        # v2
        if v2_obj is not None and v2_obj.exists():
            V_v2, F_v2 = load_obj(v2_obj)
            add_mesh(
                fig,
                V_v2,
                F_v2,
                name=f"SMPLify-v2 det{det_id}->{aria}",
                color=COLORS["v2"],
            )
            all_points.append(V_v2)

            c_v2 = centroid(V_v2)
            add_point(fig, c_v2, f"v2 center det{det_id}", color="green", size=3)

            if args.draw_lines:
                add_line(
                    fig,
                    c_v2,
                    c_gt,
                    name=f"v2→GT det{det_id}",
                    color="green",
                    width=4,
                )

        # Aria/glasses/head target
        if args.show_targets and target_dir is not None and target_dir.exists():
            target_file = target_dir / f"{frame}_{aria}_cam01_egohumans_style.txt"

            if target_file.exists():
                p_target = load_target_mesh_cam(target_file)
                add_point(
                    fig,
                    p_target,
                    f"Aria target {aria}",
                    color="saddlebrown",
                    size=6,
                )
                all_points.append(p_target.reshape(1, 3))
            else:
                print(f"[Warning] Missing target file: {target_file}")

    if len(all_points) == 0:
        raise RuntimeError("No meshes/points were loaded; cannot visualize.")

    P = np.concatenate(all_points, axis=0)

    center = P.mean(axis=0)
    extent = P.max(axis=0) - P.min(axis=0)
    radius = float(max(extent.max() * 0.55, 0.5))

    x_range = [center[0] - radius, center[0] + radius]
    y_range = [center[1] - radius, center[1] + radius]
    z_range = [center[2] - radius, center[2] + radius]

    title = f"Frame {frame}: baseline vs SMPLify-v1 vs SMPLify-v2 vs EgoHumans GT"

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X", range=x_range, visible=args.axis),
            yaxis=dict(title="Y", range=y_range, visible=args.axis),
            zaxis=dict(title="Z", range=z_range, visible=args.axis),
            aspectmode="cube",
        ),
        width=1400,
        height=950,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.6)",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        annotations=[
            dict(
                text=make_legend_note(),
                align="left",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.01,
                y=0.01,
                bordercolor="black",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.7)",
            )
        ],
    )

    fig.write_html(str(save_html))
    print()
    print("[Saved]", save_html)


if __name__ == "__main__":
    main()