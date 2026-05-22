from pathlib import Path
import argparse
import numpy as np


def describe(x, indent=0):
    pad = " " * indent

    if isinstance(x, np.ndarray):
        print(pad + f"ndarray shape={x.shape}, dtype={x.dtype}")

        if x.dtype == object:
            if x.shape == ():
                describe(x.item(), indent + 2)
            else:
                for i, item in enumerate(x[:5]):
                    print(pad + f"[{i}]")
                    describe(item, indent + 2)
        else:
            print(pad + "min/max:", np.nanmin(x), np.nanmax(x))
            flat = x.reshape(-1)
            print(pad + "first values:", flat[:20])

    elif isinstance(x, dict):
        print(pad + f"dict keys={list(x.keys())}")
        for k, v in x.items():
            print(pad + f"key: {k}")
            describe(v, indent + 2)

    elif isinstance(x, list):
        print(pad + f"list len={len(x)}")
        for i, item in enumerate(x[:5]):
            print(pad + f"[{i}]")
            describe(item, indent + 2)

    else:
        print(pad + f"{type(x)}: {x}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    p = Path(args.file).expanduser()
    data = np.load(p, allow_pickle=True)

    print("file:", p)
    describe(data)


if __name__ == "__main__":
    main()