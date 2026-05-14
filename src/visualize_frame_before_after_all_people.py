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
        raise ValueError(f"Bad vertices shape: {obj_path}, {vertices.shape}")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Bad faces shape: {obj_path}, {faces.shape}")

    print(f"\nLoaded: {obj_path}")
    print("  center:", vertices.mean(axis=0))
    print("  min:", vertices.min(axis=0))
    print("  max:", vertices.max(axis=0))

    return vertices, faces


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


def add_line(fig, p1, p2, name, color="black", width=4):
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    dist = np.linalg.norm(p1 - p2)

    fig.add_trace(
        go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[p1[2], p2[2]],
            mode="lines+text",
            line=dict(color=color, width=width),
            text=["", f"{dist:.2f}m"],
            textposition="middle center",
            name=f"{name}: {dist:.2f}m",
        )
    )


def parse_point_file(path):
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

def parse_mapping(mapping_str):
    mapping = {}
    for pair in mapping_str.split(","):
        det, aria = pair.split(":")
        det = int(det.strip())
        aria = aria.strip()
        if not aria.startswith("aria"):
            aria = "aria" + aria
        mapping[det] = aria
    return mapping


def camera_to_mesh_cam(points):
    points = np.asarray(points, dtype=np.float32).copy()
    points[..., 1] *= -1.0
    points[..., 2] *= -1.0
    return points


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

    parser.add_argument("--frame", default="00001")
    parser.add_argument("--before_dir", required=True)
    parser.add_argument("--after_dir", required=True)
    parser.add_argument("--gt_root", required=True)
    parser.add_argument("--target_dir", default=None)
    parser.add_argument("--save_html", default="frame_before_after_all_people.html")
    parser.add_argument("--opacity", type=float, default=0.35)
    parser.add_argument("--draw_lines", action="store_true")
    parser.add_argument("--mapping", default="0:aria03,1:aria02,2:aria01,3:aria04",
        help='Detection-to-GT mapping, e.g. "0:aria03,1:aria02,2:aria01,3:aria04"',
    )

    args = parser.parse_args()

    frame = args.frame
    before_dir = Path(args.before_dir).expanduser()
    after_dir = Path(args.after_dir).expanduser()
    gt_root = Path(args.gt_root).expanduser()
    save_html = Path(args.save_html).expanduser()

    # detection id -> EgoHumans identity
    mapping = parse_mapping(args.mapping)

    # Same color per identity, different shade by before/after/GT
    identity_colors = {
        "aria01": dict(before="rgba(255,0,0,0.35)", after="red", gt="darkred"),
        "aria02": dict(before="rgba(0,180,0,0.35)", after="green", gt="darkgreen"),
        "aria03": dict(before="rgba(0,0,255,0.35)", after="blue", gt="darkblue"),
        "aria04": dict(before="rgba(255,165,0,0.35)", after="orange", gt="darkorange"),
    }

    fig = go.Figure()
    all_vertices = []

    for det_id, aria in mapping.items():
        before_obj = before_dir / f"{frame}_{det_id}.obj"
        after_obj = after_dir / f"{frame}_{det_id}.obj"
        gt_obj = gt_root / frame / f"mesh_{aria}.obj"

        print("\n============================================================")
        print(f"det {det_id} -> {aria}")
        print("before:", before_obj)
        print("after: ", after_obj)
        print("gt:    ", gt_obj)
        print("============================================================")

        if not before_obj.exists():
            print(f"[Warning] missing before OBJ: {before_obj}")
            continue

        if not after_obj.exists():
            print(f"[Warning] missing after OBJ: {after_obj}")
            continue

        if not gt_obj.exists():
            print(f"[Warning] missing GT OBJ: {gt_obj}")
            continue

        colors = identity_colors[aria]

        before_v, before_f = load_obj_mesh(before_obj)
        after_v, after_f = load_obj_mesh(after_obj)
        gt_v, gt_f = load_obj_mesh(gt_obj)

        before_center = before_v.mean(axis=0)
        after_center = after_v.mean(axis=0)
        gt_center = gt_v.mean(axis=0)

        all_vertices.extend([before_v, after_v, gt_v])

        add_mesh(
            fig,
            before_v,
            before_f,
            name=f"{aria} / det{det_id} BEFORE TokenHMR",
            color=colors["before"],
            opacity=args.opacity,
        )

        add_mesh(
            fig,
            after_v,
            after_f,
            name=f"{aria} / det{det_id} AFTER SMPLify",
            color=colors["after"],
            opacity=args.opacity,
        )

        add_mesh(
            fig,
            gt_v,
            gt_f,
            name=f"{aria} GT",
            color=colors["gt"],
            opacity=args.opacity,
        )

        add_point(fig, before_center, f"{aria} before center", colors["before"], size=4)
        add_point(fig, after_center, f"{aria} after center", colors["after"], size=5)
        add_point(fig, gt_center, f"{aria} GT center", colors["gt"], size=5)

        if args.draw_lines:
            add_line(
                fig,
                before_center,
                gt_center,
                name=f"{aria} before→GT center",
                color="red",
                width=3,
            )
            add_line(
                fig,
                after_center,
                gt_center,
                name=f"{aria} after→GT center",
                color="green",
                width=3,
            )

        # Optional Aria/head target
        if args.target_dir is not None:
            target_dir = Path(args.target_dir).expanduser()
            target_file = target_dir / f"{frame}_{aria}_cam01_egohumans_style.txt"

            if target_file.exists():
                target_camera = parse_point_file(target_file)
                target_mesh_cam = camera_to_mesh_cam(target_camera)
                all_vertices.append(target_mesh_cam.reshape(1, 3))

                add_point(
                    fig,
                    target_mesh_cam,
                    f"{aria} Aria/head target",
                    "black",
                    size=7,
                )
            else:
                print(f"[Warning] missing target file: {target_file}")

    if len(all_vertices) == 0:
        raise RuntimeError("No meshes were loaded. Check paths.")

    set_scene_bounds(fig, all_vertices)

    fig.update_layout(
        title=f"Frame {frame}: TokenHMR Before vs SMPLify After vs EgoHumans GT",
        width=1500,
        height=950,
        showlegend=True,
    )

    fig.write_html(str(save_html))
    print(f"\nSaved visualization to:\n{save_html}")


if __name__ == "__main__":
    main()