from pathlib import Path
import argparse
import numpy as np
import plotly.graph_objects as go


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
    return int(stem.split("_")[-1])


def normalize_model_name(model):
    m = model.lower()

    if m in ["tokenhmr", "tkhmr", "token"]:
        return "tokenhmr", "TokenHMR"

    if m in ["4dhumans", "4dh", "4d-humans"]:
        return "4dhumans", "4DHumans"

    raise ValueError(f"Unknown model: {model}. Use tokenhmr or 4dhumans.")


def default_pred_dir(model_key):
    if model_key == "tokenhmr":
        return Path("~/DigitalHumans/TokenHMR/demo_out/my_image").expanduser()

    if model_key == "4dhumans":
        return Path("~/DigitalHumans/4D-Humans/demo_out/my_image_real_intrinsics").expanduser()

    raise ValueError(model_key)


def default_save_html(model_key, frame):
    out_dir = Path("~/DigitalHumans/visualizations").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_key == "tokenhmr":
        return out_dir / f"{frame}_TokenHMR_vs_EgoHumans_mesh3d.html"

    if model_key == "4dhumans":
        return out_dir / f"{frame}_4DHumans_vs_EgoHumans_mesh3d.html"

    raise ValueError(model_key)


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


def add_centroid(fig, center, name, color):
    fig.add_trace(
        go.Scatter3d(
            x=[center[0]],
            y=[center[1]],
            z=[center[2]],
            mode="markers+text",
            marker=dict(size=6, color=color),
            text=[name],
            textposition="top center",
            name=name,
        )
    )


def add_centroid_line(fig, c_gt, c_pred, name):
    diff = c_pred - c_gt
    dist = np.linalg.norm(diff)

    label = (
        f"{dist:.2f} m<br>"
        f"dx={diff[0]:.2f}, dy={diff[1]:.2f}, dz={diff[2]:.2f}"
    )

    fig.add_trace(
        go.Scatter3d(
            x=[c_gt[0], c_pred[0]],
            y=[c_gt[1], c_pred[1]],
            z=[c_gt[2], c_pred[2]],
            mode="lines+text",
            line=dict(color="black", width=5),
            text=["", label],
            textposition="middle center",
            name=name,
        )
    )


def add_axis(fig, length=1.0):
    origin = np.array([0.0, 0.0, 0.0])

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


def set_scene_bounds(fig, all_vertices):
    pts = np.concatenate(all_vertices, axis=0)
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
        "--model",
        required=True,
        choices=["tokenhmr", "TokenHMR", "tkhmr", "4dhumans", "4DHumans", "4dh"],
        help="Which prediction source to visualize.",
    )

    parser.add_argument(
        "--frame",
        required=True,
        help="Frame id, e.g. 00001",
    )

    parser.add_argument(
        "--pred_dir",
        default=None,
        help="Optional override for prediction OBJ directory.",
    )

    parser.add_argument(
        "--gt_root",
        default="~/DigitalHumans/data/01_tagging/001_tagging/processed_data/mesh_cam/cam01/rgb",
        help="EgoHumans mesh_cam GT root.",
    )

    parser.add_argument(
        "--mapping",
        default="0:aria03,1:aria02,2:aria01,3:aria04",
        help='Detection-to-GT mapping, e.g. "0:aria03,1:aria02,2:aria01,3:aria04"',
    )

    parser.add_argument(
        "--save_html",
        default=None,
        help="Optional output html path.",
    )

    parser.add_argument(
        "--opacity",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--draw_centroid_lines",
        action="store_true",
    )

    parser.add_argument(
        "--axis",
        action="store_true",
        help="Draw world coordinate axes.",
    )

    args = parser.parse_args()

    model_key, model_label = normalize_model_name(args.model)

    if args.pred_dir is None:
        pred_dir = default_pred_dir(model_key)
    else:
        pred_dir = Path(args.pred_dir).expanduser()

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
        raise FileNotFoundError(
            f"No {model_label} OBJ files found: {pred_dir}/{args.frame}_*.obj"
        )

    if not frame_dir.exists():
        raise FileNotFoundError(
            f"EgoHumans frame directory does not exist: {frame_dir}"
        )

    if det_to_gt:
        gt_files = []
        for det_id, aria_id in det_to_gt.items():
            gt_path = frame_dir / f"mesh_{aria_id}.obj"
            if gt_path.exists():
                gt_files.append(gt_path)
            else:
                print(f"Warning: mapped GT missing: {gt_path}")
    else:
        gt_files = sorted(frame_dir.glob("mesh_aria*.obj"))

    if len(gt_files) == 0:
        raise FileNotFoundError(f"No EgoHumans GT OBJ files found in: {frame_dir}")

    fig = go.Figure()
    all_vertices = []

    pred_data = {}
    gt_data = {}

    print(f"\nLoading {model_label} prediction meshes:")
    for p in pred_files:
        det_id = det_id_from_path(p)

        v, f = load_obj_mesh(p)
        c = v.mean(axis=0)

        pred_data[det_id] = {
            "path": p,
            "vertices": v,
            "faces": f,
            "center": c,
        }

        all_vertices.append(v)

        add_mesh(
            fig,
            v,
            f,
            name=f"{model_label} det{det_id}: {p.stem}",
            color="red",
            opacity=args.opacity,
        )
        add_centroid(fig, c, f"P{det_id}", "red")

    print("\nLoading EgoHumans ground-truth meshes:")
    for g in gt_files:
        aria_id = g.stem.replace("mesh_", "")

        v, f = load_obj_mesh(g)
        c = v.mean(axis=0)

        gt_data[aria_id] = {
            "path": g,
            "vertices": v,
            "faces": f,
            "center": c,
        }

        all_vertices.append(v)

        add_mesh(
            fig,
            v,
            f,
            name=f"EgoHumans GT {aria_id}",
            color="blue",
            opacity=args.opacity,
        )
        add_centroid(fig, c, f"G:{aria_id}", "blue")

    print("\n================ Center / Drift Summary ================")

    if det_to_gt:
        for det_id, aria_id in det_to_gt.items():
            if det_id not in pred_data:
                print(f"det {det_id}: missing {model_label} prediction")
                continue

            if aria_id not in gt_data:
                print(f"det {det_id} -> {aria_id}: missing EgoHumans GT")
                continue

            c_pred = pred_data[det_id]["center"]
            c_gt = gt_data[aria_id]["center"]
            diff = c_pred - c_gt
            dist = np.linalg.norm(diff)

            print(f"det {det_id} -> {aria_id}")
            print(f"  {model_label} center:", c_pred)
            print("  EgoHumans center:", c_gt)
            print("  diff pred-gt:    ", diff)
            print(f"  distance:        {dist:.4f} m")

            if args.draw_centroid_lines:
                add_centroid_line(
                    fig,
                    c_gt,
                    c_pred,
                    name=f"drift det{det_id}->{aria_id}",
                )
    else:
        print("No explicit mapping provided.")
        print("Use --mapping if you want correct detection id to EgoHumans aria id lines.")

    if args.axis:
        add_axis(fig, length=1.0)

    set_scene_bounds(fig, all_vertices)

    fig.update_layout(
        title=f"{model_label} vs EgoHumans Mesh3D - Frame {args.frame}",
        width=1300,
        height=900,
        showlegend=True,
    )

    fig.write_html(str(save_path))

    print(f"\nSaved interactive Mesh3D visualization to:\n{save_path}")


if __name__ == "__main__":
    main()