#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import plotly.graph_objects as go


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--vertices", nargs="+", type=int, required=True)
    parser.add_argument("--out_html", required=True)
    args = parser.parse_args()

    V, F = load_obj(args.mesh)
    ids = args.vertices
    P = V[ids]

    fig = go.Figure()

    fig.add_trace(
        go.Mesh3d(
            x=V[:, 0],
            y=V[:, 1],
            z=V[:, 2],
            i=F[:, 0],
            j=F[:, 1],
            k=F[:, 2],
            color="lightblue",
            opacity=0.35,
            name="mesh",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=P[:, 0],
            y=P[:, 1],
            z=P[:, 2],
            mode="markers+text",
            marker=dict(size=6, color="red"),
            text=[str(i) for i in ids],
            textposition="top center",
            name="candidate vertices",
        )
    )

    center = P.mean(axis=0)
    radius = max(np.max(np.linalg.norm(P - center[None, :], axis=1)), 0.08)

    fig.update_layout(
        width=1200,
        height=900,
        scene=dict(
            xaxis=dict(range=[center[0] - radius, center[0] + radius]),
            yaxis=dict(range=[center[1] - radius, center[1] + radius]),
            zaxis=dict(range=[center[2] - radius, center[2] + radius]),
            aspectmode="cube",
        ),
    )

    out = Path(args.out_html).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out))
    print("Saved:", out)


if __name__ == "__main__":
    main()