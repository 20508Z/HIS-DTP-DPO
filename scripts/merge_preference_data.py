"""合并多GPU生成的偏好对数据分片"""
import argparse
import glob
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pattern", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    all_data = []
    files = sorted(glob.glob(args.input_pattern))
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            print(f"  {f}: {len(data)} pairs")
            all_data.extend(data)

    with open(args.output_path, 'w') as fh:
        json.dump(all_data, fh, indent=2)
    print(f"Merged {len(all_data)} total pairs -> {args.output_path}")


if __name__ == "__main__":
    main()
