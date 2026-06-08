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
# GT mesh_cam_unscaled OBJ and exported HMR/SMPLify OBJ should
# already be in this convention.
# ------------------------------------------------------------


# ------------------------------------------------------------
# OBJ / target loading
# ------------------------------------------------------------
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
                    idx = int(p.split("/")[0]) - 1
                    face.append(idx)

                if len(face) == 3:
                    faces.append(face)
                elif len(face) > 3:
                    for i in range(1, len(face) - 1):
                        faces.append([face[0], face[i], face[i + 1]])

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Bad vertices shape from {obj_path}: {vertices.shape}")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Bad faces shape from {obj_path}: {faces.shape}")

    print(f"{obj_path.name}: vertices={vertices.shape}, faces={faces.shape}")
    print("  center:", vertices.mean(axis=0))
    print("  min:   ", vertices.min(axis=0))
    print("  max:   ", vertices.max(axis=0))

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


# ------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------
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


def find_pred_obj(pred_dir, frame, det_id):
    """
    Expected:
        00001_0.obj

    Also tolerates extra suffixes by first trying exact.
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


def centroid(vertices):
    return vertices.mean(axis=0)


# ------------------------------------------------------------
# Plotly drawing helpers
# ------------------------------------------------------------
def add_mesh(fig, vertices, faces, name, color, opacity=0.55):
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
            hoverinfo="name",
            showlegend=False,
            lighting=dict(
                ambient=0.6,
                diffuse=0.5,
                specular=0.2,
                roughness=0.8,
                fresnel=0.1,
            ),
        )
    )


def add_line(fig, p1, p2, name, color="black", width=5):
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)

    fig.add_trace(
        go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[p1[2], p2[2]],
            mode="lines",
            line=dict(color=color, width=width),
            name=name,
            hoverinfo="name",
            showlegend=False,
        )
    )


def add_point(fig, point, name, color="black", size=5, show_text=False):
    point = np.asarray(point, dtype=np.float32)
    mode = "markers+text" if show_text else "markers"

    fig.add_trace(
        go.Scatter3d(
            x=[point[0]],
            y=[point[1]],
            z=[point[2]],
            mode=mode,
            marker=dict(size=size, color=color),
            text=[name] if show_text else None,
            textposition="top center",
            name=name,
            hoverinfo="text",
            showlegend=False,
        )
    )


def add_centroid_line(fig, p1, p2, name, color="gray", width=4):
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    dist = np.linalg.norm(p1 - p2)

    add_line(
        fig,
        p1,
        p2,
        name=f"{name}: {dist:.3f} m",
        color=color,
        width=width,
    )


def add_axis(fig, length=1.0):
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    axes = [
        ("X", np.array([length, 0.0, 0.0], dtype=np.float32), "red"),
        ("Y", np.array([0.0, length, 0.0], dtype=np.float32), "green"),
        ("Z", np.array([0.0, 0.0, length], dtype=np.float32), "blue"),
    ]

    for name, end, color in axes:
        add_line(fig, origin, end, name=f"axis {name}", color=color, width=6)
        add_point(fig, end, name=name, color=color, size=4, show_text=False)


def add_camera_frustum_meshcam(fig, scale=0.25):
    """
    Visualize cam01 in mesh_cam/export coordinates.

    In mesh_cam:
        positive-Z camera [x, y, z] -> mesh_cam [x, -y, -z]

    Therefore:
        cam01 center is at origin
        optical axis points roughly toward -Z in mesh_cam

    Returns:
        camera_points: (N, 3), points used for scene bounds.
    """
    O = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    depth = scale
    half_w = 0.7 * scale
    half_h = 0.4 * scale

    p1 = np.array([-half_w, half_h, -depth], dtype=np.float32)
    p2 = np.array([half_w, half_h, -depth], dtype=np.float32)
    p3 = np.array([half_w, -half_h, -depth], dtype=np.float32)
    p4 = np.array([-half_w, -half_h, -depth], dtype=np.float32)

    add_point(fig, O, "cam01 center", color="black", size=6, show_text=False)

    add_line(fig, O, p1, "cam01 frustum", color="black", width=4)
    add_line(fig, O, p2, "cam01 frustum", color="black", width=4)
    add_line(fig, O, p3, "cam01 frustum", color="black", width=4)
    add_line(fig, O, p4, "cam01 frustum", color="black", width=4)

    add_line(fig, p1, p2, "cam01 image plane", color="black", width=4)
    add_line(fig, p2, p3, "cam01 image plane", color="black", width=4)
    add_line(fig, p3, p4, "cam01 image plane", color="black", width=4)
    add_line(fig, p4, p1, "cam01 image plane", color="black", width=4)

    optical_end = np.array([0.0, 0.0, -1.5 * depth], dtype=np.float32)
    add_line(fig, O, optical_end, "cam01 optical axis", color="cyan", width=6)

    return np.stack([O, p1, p2, p3, p4, optical_end], axis=0)


# ------------------------------------------------------------
# Scene layout and auto-rotation
# ------------------------------------------------------------
def compute_scene_stats(all_points, padding=1.30):
    pts = np.concatenate(all_points, axis=0)

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)

    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)

    radius = radius * padding
    radius = max(radius, 1.0)

    return center, radius


def set_scene_layout(
    fig,
    center,
    radius,
    show_axis=False,
    show_legend=False,
    camera_eye_scale=1.0,
):
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="",
                range=[center[0] - radius, center[0] + radius],
                visible=show_axis,
                showgrid=show_axis,
                zeroline=show_axis,
                showticklabels=show_axis,
                showbackground=False,
            ),
            yaxis=dict(
                title="",
                range=[center[1] - radius, center[1] + radius],
                visible=show_axis,
                showgrid=show_axis,
                zeroline=show_axis,
                showticklabels=show_axis,
                showbackground=False,
            ),
            zaxis=dict(
                title="",
                range=[center[2] - radius, center[2] + radius],
                visible=show_axis,
                showgrid=show_axis,
                zeroline=show_axis,
                showticklabels=show_axis,
                showbackground=False,
            ),
            aspectmode="cube",
            # mesh_cam Y is the standing vertical direction.
            # camera.up = Y keeps the ground horizontal on screen.
            camera=dict(
                eye=dict(
                    x=1.2 * camera_eye_scale,
                    y=0.7 * camera_eye_scale,
                    z=1.2 * camera_eye_scale,
                ),
                up=dict(x=0.0, y=1.0, z=0.0),
            ),
        ),
        width=1500,
        height=950,
        showlegend=show_legend,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )


def make_auto_rotate_post_script(rotation_degrees=360, steps=360, interval_ms=50):
    """
    Orbit around mesh_cam Y axis.

    This means:
        Y is vertical / up direction
        camera moves in X-Z horizontal plane
        ground stays horizontal on screen
    """
    return f"""
var gd = document.getElementById('{{plot_id}}');

var isRotating = true;
var internalRelayout = false;

var totalSteps = {steps};
var totalAngle = {rotation_degrees} * Math.PI / 180.0;
var step = 0;

function getCurrentEye() {{
    var cam = (gd.layout.scene && gd.layout.scene.camera) ? gd.layout.scene.camera : {{}};
    var eye = cam.eye || {{x: 1.2, y: 0.7, z: 1.2}};
    return eye;
}}

var eye0 = getCurrentEye();

// Horizontal orbit:
// rotate in X-Z plane, keep Y fixed.
var radius = Math.sqrt(eye0.x * eye0.x + eye0.z * eye0.z);
if (!isFinite(radius) || radius < 0.1) radius = 1.5;

var fixedY = eye0.y;
var startAngle = Math.atan2(eye0.z, eye0.x);

var btn = document.createElement('button');
btn.innerHTML = 'Pause rotation';
btn.style.position = 'absolute';
btn.style.top = '10px';
btn.style.left = '10px';
btn.style.zIndex = 9999;
btn.style.padding = '8px 12px';
btn.style.background = 'white';
btn.style.border = '1px solid #999';
btn.style.borderRadius = '6px';
btn.style.cursor = 'pointer';

btn.onclick = function() {{
    if (isRotating) {{
        isRotating = false;
        btn.innerHTML = 'Resume rotation';
    }} else {{
        var eye = getCurrentEye();

        radius = Math.sqrt(eye.x * eye.x + eye.z * eye.z);
        if (!isFinite(radius) || radius < 0.1) radius = 1.5;

        fixedY = eye.y;
        startAngle = Math.atan2(eye.z, eye.x);
        step = 0;

        isRotating = true;
        btn.innerHTML = 'Pause rotation';
    }}
}};

gd.parentElement.style.position = 'relative';
gd.parentElement.appendChild(btn);

gd.on('plotly_relayout', function(e) {{
    if (internalRelayout) return;

    if (
        e['scene.camera'] ||
        e['scene.camera.eye'] ||
        e['scene.camera.center'] ||
        e['scene.camera.up']
    ) {{
        isRotating = false;
        btn.innerHTML = 'Resume rotation';
    }}
}});

setInterval(function() {{
    if (!isRotating) return;

    var theta = startAngle + (step / totalSteps) * totalAngle;

    var newEye = {{
        x: radius * Math.cos(theta),
        y: fixedY,
        z: radius * Math.sin(theta)
    }};

    internalRelayout = true;

    Plotly.relayout(gd, {{
        'scene.camera.eye': newEye,
        'scene.camera.up': {{x: 0, y: 1, z: 0}}
    }}).then(function() {{
        internalRelayout = false;
    }}).catch(function() {{
        internalRelayout = false;
    }});

    step += 1;
    if (step > totalSteps) {{
        step = 0;
    }}
}}, {interval_ms});
"""


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--frame", required=True)
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

    parser.add_argument("--baseline_opacity", type=float, default=0.50)
    parser.add_argument("--v1_opacity", type=float, default=0.55)
    parser.add_argument("--v2_opacity", type=float, default=0.55)
    parser.add_argument("--gt_opacity", type=float, default=0.55)

    parser.add_argument("--show_camera", action="store_true")
    parser.add_argument("--hide_camera", dest="show_camera", action="store_false")
    parser.set_defaults(show_camera=True)

    parser.add_argument("--camera_scale", type=float, default=0.25)

    parser.add_argument("--scene_padding", type=float, default=1.30)
    parser.add_argument("--camera_eye_scale", type=float, default=1.0)

    parser.add_argument("--auto_rotate", action="store_true")
    parser.add_argument("--no_auto_rotate", dest="auto_rotate", action="store_false")
    parser.set_defaults(auto_rotate=True)

    parser.add_argument("--rotation_degrees", type=float, default=360.0)
    parser.add_argument("--rotation_steps", type=int, default=360)
    parser.add_argument("--rotation_interval_ms", type=int, default=50)

    parser.add_argument("--show_axis", action="store_true")
    parser.add_argument("--hide_axis", dest="show_axis", action="store_false")
    parser.set_defaults(show_axis=False)

    parser.add_argument("--show_legend", action="store_true")
    parser.add_argument("--hide_legend", dest="show_legend", action="store_false")
    parser.set_defaults(show_legend=False)

    parser.add_argument("--draw_lines", action="store_true")
    parser.add_argument("--show_centers", action="store_true")
    parser.add_argument("--show_targets", action="store_true")
    parser.add_argument("--hide_targets", dest="show_targets", action="store_false")
    parser.set_defaults(show_targets=True)

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
    bounds_points = []

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

    for det_id, aria in sorted(
        mapping.items(),
        key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]),
    ):
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

        V_gt, F_gt = load_obj(gt_obj)
        bounds_points.append(V_gt)
        c_gt = centroid(V_gt)

        add_mesh(
            fig,
            V_gt,
            F_gt,
            name=f"GT {aria}",
            color="royalblue",
            opacity=args.gt_opacity,
        )

        if args.show_centers:
            add_point(fig, c_gt, f"GT center {aria}", color="blue", size=4, show_text=False)

        # Baseline
        if baseline_obj is not None and baseline_obj.exists():
            V_b, F_b = load_obj(baseline_obj)
            bounds_points.append(V_b)
            c_b = centroid(V_b)

            add_mesh(
                fig,
                V_b,
                F_b,
                name=f"baseline det{det_id}->{aria}",
                color="lightgray",
                opacity=args.baseline_opacity,
            )

            if args.show_centers:
                add_point(fig, c_b, f"baseline center det{det_id}", color="gray", size=4, show_text=False)

            if args.draw_lines:
                add_centroid_line(
                    fig,
                    c_b,
                    c_gt,
                    name=f"baseline→GT det{det_id}",
                    color="gray",
                    width=4,
                )
        else:
            print(f"[Warning] Missing baseline OBJ for det {det_id}")

        # v1
        if v1_obj is not None and v1_obj.exists():
            V_v1, F_v1 = load_obj(v1_obj)
            bounds_points.append(V_v1)
            c_v1 = centroid(V_v1)

            add_mesh(
                fig,
                V_v1,
                F_v1,
                name=f"SMPLify-v1 det{det_id}->{aria}",
                color="limegreen",
                opacity=args.v1_opacity,
            )

            if args.show_centers:
                add_point(fig, c_v1, f"v1 center det{det_id}", color="green", size=4, show_text=False)

            if args.draw_lines:
                add_centroid_line(
                    fig,
                    c_v1,
                    c_gt,
                    name=f"v1→GT det{det_id}",
                    color="green",
                    width=4,
                )
        else:
            print(f"[Warning] Missing v1 OBJ for det {det_id}")

        # v2
        if v2_obj is not None and v2_obj.exists():
            V_v2, F_v2 = load_obj(v2_obj)
            bounds_points.append(V_v2)
            c_v2 = centroid(V_v2)

            add_mesh(
                fig,
                V_v2,
                F_v2,
                name=f"SMPLify-v2 det{det_id}->{aria}",
                color="deeppink",
                opacity=args.v2_opacity,
            )

            if args.show_centers:
                add_point(fig, c_v2, f"v2 center det{det_id}", color="deeppink", size=4, show_text=False)

            if args.draw_lines:
                add_centroid_line(
                    fig,
                    c_v2,
                    c_gt,
                    name=f"v2→GT det{det_id}",
                    color="deeppink",
                    width=4,
                )
        else:
            print(f"[Warning] Missing v2 OBJ for det {det_id}")

        # Optional Aria/glasses/head target
        if args.show_targets and target_dir is not None and target_dir.exists():
            target_file = target_dir / f"{frame}_{aria}_cam01_egohumans_style.txt"

            if target_file.exists():
                p_target = load_target_mesh_cam(target_file)
                bounds_points.append(p_target.reshape(1, 3))

                add_point(
                    fig,
                    p_target,
                    f"Aria target {aria}",
                    color="black",
                    size=6,
                    show_text=False,
                )
            else:
                print(f"[Warning] Missing target file: {target_file}")

    if args.show_camera:
        camera_points = add_camera_frustum_meshcam(fig, scale=args.camera_scale)
        bounds_points.append(camera_points)

    if len(bounds_points) == 0:
        raise RuntimeError("No meshes/points were loaded; cannot visualize.")

    center, radius = compute_scene_stats(
        bounds_points,
        padding=args.scene_padding,
    )

    if args.show_axis:
        add_axis(fig, length=max(0.5, radius * 0.25))

    set_scene_layout(
        fig,
        center=center,
        radius=radius,
        show_axis=args.show_axis,
        show_legend=args.show_legend,
        camera_eye_scale=args.camera_eye_scale,
    )

    fig.update_layout(
        title=f"Frame {frame}: baseline vs SMPLify-v1 vs SMPLify-v2 vs EgoHumans GT",
    )

    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "scrollZoom": True,
        "responsive": True,
    }

    post_script = None
    if args.auto_rotate:
        post_script = make_auto_rotate_post_script(
            rotation_degrees=args.rotation_degrees,
            steps=args.rotation_steps,
            interval_ms=args.rotation_interval_ms,
        )

    html = fig.to_html(
        full_html=True,
        include_plotlyjs=True,
        config=config,
        post_script=post_script,
    )

    with open(save_html, "w", encoding="utf-8") as f:
        f.write(html)

    print()
    print("[Saved]", save_html)


if __name__ == "__main__":
    main()