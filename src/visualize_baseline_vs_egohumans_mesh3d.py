#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import plotly.graph_objects as go


# ------------------------------------------------------------
# OBJ loading
# ------------------------------------------------------------
def load_obj_mesh(obj_path):
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

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Bad vertices shape for {obj_path}: {vertices.shape}")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Bad faces shape for {obj_path}: {faces.shape}")

    print(f"{obj_path.name}: vertices={vertices.shape}, faces={faces.shape}")
    print("  center:", vertices.mean(axis=0))
    print("  min:   ", vertices.min(axis=0))
    print("  max:   ", vertices.max(axis=0))

    return vertices, faces


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def parse_mapping(mapping_str):
    """
    Example:
        "0:aria03,1:aria02,2:aria01,3:aria04"
    """
    if mapping_str is None or mapping_str.strip() == "":
        return {}

    mapping = {}

    for pair in mapping_str.split(","):
        left, right = pair.split(":")
        det_id = int(left.strip())
        aria_id = right.strip()

        if not aria_id.startswith("aria"):
            aria_id = "aria" + aria_id

        mapping[det_id] = aria_id

    return mapping


def det_id_from_path(path):
    """
    For file:
        00001_0.obj
    returns:
        0
    """
    stem = Path(path).stem
    parts = stem.split("_")

    for part in reversed(parts):
        if part.isdigit():
            return int(part)

    raise ValueError(f"Could not parse detection id from filename: {path}")


def normalize_model_name(model):
    m = model.lower()

    if m in ["tokenhmr", "tkhmr", "token"]:
        return "tokenhmr", "TokenHMR"

    if m in ["4dhumans", "4dh", "4d-humans"]:
        return "4dhumans", "4DHumans"

    raise ValueError(f"Unknown model: {model}. Use tokenhmr or 4dhumans.")


def default_pred_dir(model_key):
    if model_key == "tokenhmr":
        return Path("~/DigitalHumans/Project/TokenHMR/demo_out/my_image").expanduser()

    if model_key == "4dhumans":
        return Path("~/DigitalHumans/Project/4D-Humans/demo_out/my_image").expanduser()

    raise ValueError(model_key)


def default_save_html(model_key, frame):
    out_dir = Path("~/DigitalHumans/Project/visualizations").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_key == "tokenhmr":
        return out_dir / f"{frame}_TokenHMR_baseline_vs_EgoHumans_GT.html"

    if model_key == "4dhumans":
        return out_dir / f"{frame}_4DHumans_baseline_vs_EgoHumans_GT.html"

    raise ValueError(model_key)


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


def add_centroid_line(fig, c_gt, c_pred, name, color="gray"):
    c_gt = np.asarray(c_gt, dtype=np.float32)
    c_pred = np.asarray(c_pred, dtype=np.float32)

    diff = c_pred - c_gt
    dist = np.linalg.norm(diff)

    add_line(
        fig,
        c_gt,
        c_pred,
        name=f"{name}: {dist:.3f} m",
        color=color,
        width=4,
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

    p1 = np.array([-half_w,  half_h, -depth], dtype=np.float32)
    p2 = np.array([ half_w,  half_h, -depth], dtype=np.float32)
    p3 = np.array([ half_w, -half_h, -depth], dtype=np.float32)
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
# Scene layout
# ------------------------------------------------------------
def compute_scene_stats(all_points, padding=1.25):
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

            # Important:
            # mesh_cam Y is the standing vertical direction.
            # Keep camera.up = Y so the ground stays horizontal on screen.
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
    Orbit camera around mesh_cam Y axis.

    This means:
        - Y is vertical / up direction
        - camera moves in X-Z plane
        - ground stays horizontal on screen
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

// Correct horizontal orbit:
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
        // Resume from current manual camera view
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

// Pause when user manually rotates / zooms / pans
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

        // Critical: Y is screen-up / human standing direction.
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

    parser.add_argument(
        "--model",
        required=True,
        choices=["tokenhmr", "TokenHMR", "tkhmr", "4dhumans", "4DHumans", "4dh"],
    )
    parser.add_argument("--frame", required=True)
    parser.add_argument("--pred_dir", default=None)
    parser.add_argument(
        "--gt_root",
        default="~/DigitalHumans/Project/data/mesh_cam_unscaled/cam01/rgb",
    )
    parser.add_argument(
        "--mapping",
        default="0:aria03,1:aria02,2:aria01,3:aria04",
    )
    parser.add_argument("--save_html", default=None)

    parser.add_argument("--pred_opacity", type=float, default=0.50)
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

    parser.add_argument("--draw_centroid_lines", action="store_true")

    args = parser.parse_args()

    model_key, model_label = normalize_model_name(args.model)

    pred_dir = default_pred_dir(model_key) if args.pred_dir is None else Path(args.pred_dir).expanduser()
    gt_root = Path(args.gt_root).expanduser()
    frame_dir = gt_root / args.frame

    if args.save_html is None:
        save_path = default_save_html(model_key, args.frame)
    else:
        save_path = Path(args.save_html).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)

    det_to_gt = parse_mapping(args.mapping)

    pred_files = sorted(pred_dir.glob(f"{args.frame}_*.obj"))
    pred_files = [p for p in pred_files if "_all" not in p.stem]

    if len(pred_files) == 0:
        raise FileNotFoundError(f"No {model_label} OBJ files found: {pred_dir}/{args.frame}_*.obj")

    if not frame_dir.exists():
        raise FileNotFoundError(f"EgoHumans frame directory does not exist: {frame_dir}")

    fig = go.Figure()
    bounds_points = []

    pred_data = {}
    gt_data = {}

    print(f"\nLoading {model_label} prediction meshes:")
    for p in pred_files:
        det_id = det_id_from_path(p)
        v, f = load_obj_mesh(p)

        pred_data[det_id] = {
            "path": p,
            "vertices": v,
            "faces": f,
            "center": v.mean(axis=0),
        }

        bounds_points.append(v)

    print("\nLoading EgoHumans ground-truth meshes:")
    for det_id, aria_id in det_to_gt.items():
        gt_path = frame_dir / f"mesh_{aria_id}.obj"

        if not gt_path.exists():
            print(f"[Warning] mapped GT missing: {gt_path}")
            continue

        v, f = load_obj_mesh(gt_path)

        gt_data[aria_id] = {
            "path": gt_path,
            "vertices": v,
            "faces": f,
            "center": v.mean(axis=0),
        }

        bounds_points.append(v)

    if len(gt_data) == 0:
        raise FileNotFoundError(f"No mapped EgoHumans GT OBJ files found in: {frame_dir}")

    for det_id, aria_id in det_to_gt.items():
        if aria_id in gt_data:
            add_mesh(
                fig,
                gt_data[aria_id]["vertices"],
                gt_data[aria_id]["faces"],
                name=f"GT {aria_id}",
                color="royalblue",
                opacity=args.gt_opacity,
            )
            add_point(
                fig,
                gt_data[aria_id]["center"],
                name=f"GT center {aria_id}",
                color="blue",
                size=4,
                show_text=False,
            )

        if det_id in pred_data:
            add_mesh(
                fig,
                pred_data[det_id]["vertices"],
                pred_data[det_id]["faces"],
                name=f"{model_label} baseline det{det_id}->{aria_id}",
                color="lightgray",
                opacity=args.pred_opacity,
            )
            add_point(
                fig,
                pred_data[det_id]["center"],
                name=f"baseline center det{det_id}",
                color="gray",
                size=4,
                show_text=False,
            )

        if args.draw_centroid_lines and det_id in pred_data and aria_id in gt_data:
            add_centroid_line(
                fig,
                gt_data[aria_id]["center"],
                pred_data[det_id]["center"],
                name=f"baseline→GT det{det_id}->{aria_id}",
                color="gray",
            )

    if args.show_camera:
        camera_points = add_camera_frustum_meshcam(fig, scale=args.camera_scale)
        bounds_points.append(camera_points)

    if len(bounds_points) == 0:
        raise RuntimeError("No meshes loaded.")

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
        title=f"{model_label} baseline vs EgoHumans GT - Frame {args.frame}",
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

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nSaved interactive Mesh3D visualization to:\n{save_path}")


if __name__ == "__main__":
    main()