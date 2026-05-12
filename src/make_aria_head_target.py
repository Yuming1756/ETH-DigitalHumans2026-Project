import argparse
import pickle
from pathlib import Path

import numpy as np


def load_transform_dict(path):
    path = Path(path).expanduser()
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aria_from_colmap_pkl", required=True)
    parser.add_argument("--aria_id", required=True, help="e.g. aria03")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    aria_from_colmap = load_transform_dict(args.aria_from_colmap_pkl)

    if args.aria_id not in aria_from_colmap:
        raise KeyError(f"{args.aria_id} not found. Available keys: {list(aria_from_colmap.keys())}")

    T_aria_from_colmap = np.asarray(aria_from_colmap[args.aria_id], dtype=np.float32)

    print("T_aria_from_colmap:")
    print(T_aria_from_colmap)

    # Camera/head center is the translation column.
    # This assumes the transform maps colmap/world coordinates into the Aria coordinate frame
    # or stores Aria pose in the cam01/world coordinate system.
    aria_position = T_aria_from_colmap[:3, 3]

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        f.write(f"{aria_position[0]:.8f} {aria_position[1]:.8f} {aria_position[2]:.8f}\n")

    print(f"Saved Aria camera/head proxy position to: {out}")
    print("position:", aria_position)


if __name__ == "__main__":
    main()