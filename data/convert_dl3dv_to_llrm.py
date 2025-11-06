#!/usr/bin/env python3
"""
Convert DL3DV dataset format (Blender coordinate system) to Long-LRM format (OpenCV coordinate system).

Usage:
    python convert_dl3dv_to_llrm.py --input_dir /path/to/dl3dv_scenes --output_dir /path/to/output

Author: Generated for Long-LRM evaluation alignment with SparseSplat
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def blender_to_opencv_matrix():
    """
    Transformation matrix from Blender coordinate system to OpenCV coordinate system.
    Blender: Y-up, Z-forward, X-right
    OpenCV: Y-down, Z-forward, X-right
    """
    return np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ], dtype=np.float64)


def convert_scene(input_scene_dir, output_scene_dir, image_subdir='images_8'):
    """
    Convert a single DL3DV scene from Blender to OpenCV format.

    Args:
        input_scene_dir: Path to input scene directory containing transforms.json
        output_scene_dir: Path to output scene directory
        image_subdir: Subdirectory name containing images (default: 'images_8')

    Returns:
        bool: True if conversion successful, False otherwise
    """
    input_scene_dir = Path(input_scene_dir)
    output_scene_dir = Path(output_scene_dir)

    # Read input transforms.json
    transforms_file = input_scene_dir / 'transforms.json'
    if not transforms_file.exists():
        print(f"Warning: transforms.json not found in {input_scene_dir}")
        return False

    with open(transforms_file, 'r') as f:
        transforms_data = json.load(f)

    # Extract scene name (use directory name)
    scene_name = input_scene_dir.name

    # Create output directory
    output_scene_dir.mkdir(parents=True, exist_ok=True)

    # Copy images to output directory
    input_image_dir = input_scene_dir / image_subdir
    output_image_dir = output_scene_dir / image_subdir

    if input_image_dir.exists():
        # Create symbolic link instead of copying to save space
        if not output_image_dir.exists():
            output_image_dir.symlink_to(input_image_dir.absolute())
    else:
        print(f"Warning: Image directory {input_image_dir} not found")
        return False

    # Get transformation matrix
    blender_to_opencv = blender_to_opencv_matrix()

    # Process frames
    frames_output = []
    frames_input = transforms_data.get('frames', [])

    if len(frames_input) == 0:
        print(f"Warning: No frames found in {transforms_file}")
        return False

    for frame in frames_input:
        # Get camera parameters
        # DL3DV uses camera_angle_x or direct intrinsics
        w = frame.get('w', transforms_data.get('w', 1920))
        h = frame.get('h', transforms_data.get('h', 1080))

        # Get focal length
        if 'fl_x' in frame:
            fx = frame['fl_x']
            fy = frame.get('fl_y', fx)
        elif 'camera_angle_x' in transforms_data:
            camera_angle_x = transforms_data['camera_angle_x']
            fx = 0.5 * w / np.tan(0.5 * camera_angle_x)
            fy = fx
        else:
            print(f"Warning: Cannot determine focal length for frame {frame.get('file_path', 'unknown')}")
            fx = fy = w  # fallback

        # Get principal point
        cx = frame.get('cx', w / 2.0)
        cy = frame.get('cy', h / 2.0)

        # Get transform matrix (c2w in Blender coordinates)
        transform_matrix = np.array(frame['transform_matrix'], dtype=np.float64)

        # Convert to OpenCV coordinate system
        c2w_opencv = blender_to_opencv @ transform_matrix

        # Convert c2w to w2c (Long-LRM uses w2c)
        w2c_opencv = np.linalg.inv(c2w_opencv)

        # Create output frame
        frame_output = {
            'file_path': frame['file_path'],
            'w': int(w),
            'h': int(h),
            'fx': float(fx),
            'fy': float(fy),
            'cx': float(cx),
            'cy': float(cy),
            'w2c': w2c_opencv.tolist()
        }

        # Optional: keep colmap_im_id if available
        if 'colmap_im_id' in frame:
            frame_output['colmap_im_id'] = frame['colmap_im_id']

        frames_output.append(frame_output)

    # Create output JSON
    output_data = {
        'scene_name': scene_name,
        'frames': frames_output
    }

    # Write opencv_cameras.json
    output_json = output_scene_dir / 'opencv_cameras.json'
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Converted {scene_name}: {len(frames_output)} frames")
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert DL3DV dataset to Long-LRM format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Root directory containing DL3DV scenes')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for converted scenes')
    parser.add_argument('--image_subdir', type=str, default='images_8',
                        help='Subdirectory name containing images (default: images_8)')
    parser.add_argument('--scene_list', type=str, default=None,
                        help='Optional: text file with list of scene IDs to convert (one per line)')

    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    # Get list of scenes to convert
    if args.scene_list:
        with open(args.scene_list, 'r') as f:
            scene_ids = [line.strip() for line in f if line.strip()]
        scene_dirs = [input_root / scene_id for scene_id in scene_ids]
    else:
        # Convert all subdirectories
        scene_dirs = [d for d in input_root.iterdir() if d.is_dir()]

    print(f"Found {len(scene_dirs)} scenes to convert")

    # Convert each scene
    success_count = 0
    fail_count = 0

    for scene_dir in tqdm(scene_dirs, desc="Converting scenes"):
        scene_name = scene_dir.name
        output_scene_dir = output_root / scene_name

        if convert_scene(scene_dir, output_scene_dir, args.image_subdir):
            success_count += 1
        else:
            fail_count += 1

    print(f"\nConversion complete!")
    print(f"Success: {success_count} scenes")
    print(f"Failed: {fail_count} scenes")

    if success_count > 0:
        print(f"\nConverted scenes saved to: {output_root}")
        print(f"Next step: Run create_scene_list.py to generate scene list file")


if __name__ == '__main__':
    main()
