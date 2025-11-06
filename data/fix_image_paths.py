#!/usr/bin/env python3
"""
Fix image paths in already-converted opencv_cameras.json files.

This script updates file_path entries to use the correct image subdirectory
(e.g., images_8 instead of images).

Usage:
    python fix_image_paths.py \
        --data_dir /path/to/converted_scenes \
        --image_subdir images_8

Author: Fix for Long-LRM data conversion issue
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def fix_scene(scene_json_path, image_subdir):
    """Fix file paths in a single scene's opencv_cameras.json file."""
    # Read the JSON file
    with open(scene_json_path, 'r') as f:
        data = json.load(f)

    # Get scene directory
    scene_dir = Path(scene_json_path).parent

    # Check if image directory exists
    image_dir = scene_dir / image_subdir
    if not image_dir.exists():
        print(f"Warning: Image directory not found: {image_dir}")
        return False

    # Update file paths
    updated = False
    for frame in data['frames']:
        original_path = frame['file_path']

        # Extract filename
        if '/' in original_path or '\\' in original_path:
            filename = original_path.split('/')[-1].split('\\')[-1]
        else:
            filename = original_path

        # Construct new path
        new_path = f"{image_subdir}/{filename}"

        # Update if different
        if original_path != new_path:
            frame['file_path'] = new_path
            updated = True

    # Write back if updated
    if updated:
        with open(scene_json_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    else:
        return False


def main():
    parser = argparse.ArgumentParser(description='Fix image paths in converted Long-LRM data')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing converted scenes')
    parser.add_argument('--image_subdir', type=str, default='images_8',
                        help='Correct image subdirectory name (default: images_8)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be changed without actually changing files')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Find all opencv_cameras.json files
    json_files = list(data_dir.glob('*/opencv_cameras.json'))

    if len(json_files) == 0:
        print(f"Error: No opencv_cameras.json files found in {data_dir}")
        return

    print(f"Found {len(json_files)} scenes to fix")
    print(f"Image subdirectory: {args.image_subdir}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be modified ***\n")

    # Fix each scene
    fixed_count = 0
    error_count = 0

    for json_file in tqdm(json_files, desc="Fixing scenes"):
        try:
            if args.dry_run:
                # Just check, don't modify
                with open(json_file, 'r') as f:
                    data = json.load(f)
                needs_fix = any(
                    not frame['file_path'].startswith(f"{args.image_subdir}/")
                    for frame in data['frames']
                )
                if needs_fix:
                    fixed_count += 1
                    tqdm.write(f"Would fix: {json_file.parent.name}")
            else:
                # Actually fix
                if fix_scene(json_file, args.image_subdir):
                    fixed_count += 1
        except Exception as e:
            error_count += 1
            tqdm.write(f"Error processing {json_file}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Fix Summary")
    print(f"{'='*60}")
    print(f"Total scenes: {len(json_files)}")

    if args.dry_run:
        print(f"Scenes that would be fixed: {fixed_count}")
    else:
        print(f"Scenes fixed: {fixed_count}")

    print(f"Errors: {error_count}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\nTo actually fix the files, run without --dry_run flag")


if __name__ == '__main__':
    main()
