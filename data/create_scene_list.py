#!/usr/bin/env python3
"""
Create scene list file for Long-LRM evaluation.

This script reads the SparseSplat evaluation index JSON and generates a text file
listing paths to all scene JSON files for Long-LRM evaluation.

Usage:
    python create_scene_list.py \
        --index_file /path/to/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
        --data_dir /path/to/converted_scenes \
        --output_file /path/to/dl3dv_eval_scenes.txt

Author: Generated for Long-LRM evaluation alignment with SparseSplat
"""

import os
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Create scene list file for Long-LRM evaluation')
    parser.add_argument('--index_file', type=str, required=True,
                        help='Path to SparseSplat evaluation index JSON file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing converted scene data')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output text file listing scene JSON paths')
    parser.add_argument('--verify', action='store_true',
                        help='Verify that all scenes exist before creating list')

    args = parser.parse_args()

    # Read index file
    print(f"Reading index file: {args.index_file}")
    with open(args.index_file, 'r') as f:
        index_data = json.load(f)

    # Get scene IDs
    scene_ids = list(index_data.keys())
    print(f"Found {len(scene_ids)} scenes in index file")

    # Build paths to scene JSON files
    data_dir = Path(args.data_dir)
    scene_paths = []
    missing_scenes = []

    for scene_id in scene_ids:
        scene_json_path = data_dir / scene_id / 'opencv_cameras.json'

        if args.verify:
            if scene_json_path.exists():
                scene_paths.append(str(scene_json_path))
            else:
                missing_scenes.append(scene_id)
                print(f"Warning: Scene not found: {scene_id}")
        else:
            scene_paths.append(str(scene_json_path))

    # Report status
    print(f"\nScene status:")
    print(f"  Total in index: {len(scene_ids)}")
    print(f"  Found: {len(scene_paths)}")
    if args.verify and missing_scenes:
        print(f"  Missing: {len(missing_scenes)}")
        print(f"\nMissing scenes:")
        for scene_id in missing_scenes[:10]:  # Show first 10
            print(f"    {scene_id}")
        if len(missing_scenes) > 10:
            print(f"    ... and {len(missing_scenes) - 10} more")

    # Write output file
    if len(scene_paths) == 0:
        print("\nError: No valid scene paths found!")
        return

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for path in scene_paths:
            f.write(path + '\n')

    print(f"\nScene list file created: {args.output_file}")
    print(f"Total scenes: {len(scene_paths)}")

    # Print example usage
    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print(f"1. Use this scene list in your Long-LRM config:")
    print(f"   data_eval:")
    print(f"     data_path: \"{args.output_file}\"")
    print(f"\n2. Run evaluation:")
    print(f"   bash scripts/eval_dl3dv_aligned.sh")
    print("=" * 60)


if __name__ == '__main__':
    main()
