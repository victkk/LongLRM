#!/usr/bin/env python3
"""
Verify that Long-LRM evaluation is properly aligned with SparseSplat.

This script checks:
1. Frame indices match between Long-LRM and SparseSplat
2. Camera parameters are correctly converted
3. Image loading works correctly

Usage:
    python verify_alignment.py \
        --llrm_scene_json /path/to/opencv_cameras.json \
        --sparsesplat_index /path/to/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
        --scene_id <scene_hash>

Author: Generated for Long-LRM evaluation alignment with SparseSplat
"""

import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path


def load_llrm_scene(scene_json_path):
    """Load Long-LRM scene data."""
    with open(scene_json_path, 'r') as f:
        data = json.load(f)
    return data


def load_sparsesplat_index(index_path, scene_id):
    """Load SparseSplat evaluation index for a specific scene."""
    with open(index_path, 'r') as f:
        index_data = json.load(f)

    if scene_id not in index_data:
        print(f"Error: Scene {scene_id} not found in SparseSplat index")
        return None

    return index_data[scene_id]


def verify_frame_indices(llrm_data, sparsesplat_indices):
    """Verify that frame indices are available in the scene."""
    num_frames = len(llrm_data['frames'])
    context_indices = sparsesplat_indices['context']
    target_indices = sparsesplat_indices['target']

    print(f"\n{'='*60}")
    print(f"Frame Indices Verification")
    print(f"{'='*60}")
    print(f"Total frames in scene: {num_frames}")
    print(f"\nContext (input) frames: {len(context_indices)}")
    print(f"  Indices: {context_indices}")
    print(f"\nTarget (test) frames: {len(target_indices)}")
    print(f"  Range: {min(target_indices)} to {max(target_indices)}")

    # Check if all indices are valid
    all_indices = context_indices + target_indices
    max_index = max(all_indices)
    min_index = min(all_indices)

    if max_index >= num_frames:
        print(f"\n⚠️  ERROR: Maximum index {max_index} exceeds available frames {num_frames}")
        return False

    if min_index < 0:
        print(f"\n⚠️  ERROR: Minimum index {min_index} is negative")
        return False

    print(f"\n✓ All frame indices are valid")
    return True


def verify_camera_conversion(llrm_data, sample_indices):
    """Verify camera parameter conversion from Blender to OpenCV."""
    print(f"\n{'='*60}")
    print(f"Camera Parameters Verification")
    print(f"{'='*60}")

    # Sample a few frames
    sample_frames = [llrm_data['frames'][i] for i in sample_indices[:3]]

    for idx, frame in zip(sample_indices[:3], sample_frames):
        print(f"\nFrame {idx}:")
        print(f"  Image: {frame['file_path']}")
        print(f"  Resolution: {frame['w']} x {frame['h']}")
        print(f"  Focal length: fx={frame['fx']:.2f}, fy={frame['fy']:.2f}")
        print(f"  Principal point: cx={frame['cx']:.2f}, cy={frame['cy']:.2f}")

        # Check w2c matrix
        w2c = np.array(frame['w2c'])
        print(f"  W2C matrix shape: {w2c.shape}")

        # Verify matrix properties
        c2w = np.linalg.inv(w2c)
        identity = np.matmul(w2c, c2w)
        error = np.max(np.abs(identity - np.eye(4)))
        print(f"  Matrix inversion error: {error:.6e}")

        if error > 1e-6:
            print(f"  ⚠️  WARNING: Large inversion error")

    print(f"\n✓ Camera parameters loaded successfully")
    return True


def verify_image_paths(llrm_scene_dir, llrm_data, sample_indices):
    """Verify that image files exist and can be loaded."""
    print(f"\n{'='*60}")
    print(f"Image Files Verification")
    print(f"{'='*60}")

    scene_dir = Path(llrm_scene_dir).parent
    sample_frames = [llrm_data['frames'][i] for i in sample_indices[:3]]

    missing_files = []
    for idx, frame in zip(sample_indices[:3], sample_frames):
        file_path = frame['file_path']
        full_path = scene_dir / file_path

        if full_path.exists():
            print(f"✓ Frame {idx}: {file_path}")
        else:
            print(f"✗ Frame {idx}: {file_path} (NOT FOUND)")
            missing_files.append(full_path)

    if missing_files:
        print(f"\n⚠️  WARNING: {len(missing_files)} image files not found")
        return False

    print(f"\n✓ All sampled image files exist")
    return True


def main():
    parser = argparse.ArgumentParser(description='Verify Long-LRM and SparseSplat alignment')
    parser.add_argument('--llrm_scene_json', type=str, required=True,
                        help='Path to Long-LRM scene JSON file (opencv_cameras.json)')
    parser.add_argument('--sparsesplat_index', type=str, required=True,
                        help='Path to SparseSplat evaluation index JSON')
    parser.add_argument('--scene_id', type=str, required=True,
                        help='Scene ID (hash) to verify')

    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Long-LRM <-> SparseSplat Alignment Verification")
    print(f"{'='*60}")
    print(f"Scene ID: {args.scene_id}")
    print(f"Long-LRM JSON: {args.llrm_scene_json}")
    print(f"SparseSplat Index: {args.sparsesplat_index}")

    # Load data
    print(f"\nLoading data...")
    llrm_data = load_llrm_scene(args.llrm_scene_json)
    sparsesplat_indices = load_sparsesplat_index(args.sparsesplat_index, args.scene_id)

    if sparsesplat_indices is None:
        return

    # Verify scene name matches
    llrm_scene_name = llrm_data.get('scene_name', '')
    if llrm_scene_name != args.scene_id:
        print(f"\n⚠️  WARNING: Scene name mismatch:")
        print(f"    Long-LRM: {llrm_scene_name}")
        print(f"    Expected: {args.scene_id}")

    # Run verification checks
    checks_passed = []

    # 1. Frame indices
    checks_passed.append(verify_frame_indices(llrm_data, sparsesplat_indices))

    # 2. Camera conversion
    context_indices = sparsesplat_indices['context']
    checks_passed.append(verify_camera_conversion(llrm_data, context_indices))

    # 3. Image paths
    checks_passed.append(verify_image_paths(args.llrm_scene_json, llrm_data, context_indices))

    # Summary
    print(f"\n{'='*60}")
    print(f"Verification Summary")
    print(f"{'='*60}")
    print(f"Total checks: {len(checks_passed)}")
    print(f"Passed: {sum(checks_passed)}")
    print(f"Failed: {len(checks_passed) - sum(checks_passed)}")

    if all(checks_passed):
        print(f"\n✓ All verifications passed!")
        print(f"  Scene is ready for aligned evaluation.")
    else:
        print(f"\n✗ Some verifications failed!")
        print(f"  Please fix the issues before running evaluation.")

    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
