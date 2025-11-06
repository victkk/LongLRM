#!/usr/bin/env python3
"""
Compare Long-LRM and SparseSplat evaluation results.

This script reads evaluation results from both methods and generates
comparison tables and visualizations.

Usage:
    python compare_with_sparsesplat.py \
        --llrm_results /path/to/llrm/summary.csv \
        --sparsesplat_results /path/to/sparsesplat/results \
        --output_dir /path/to/output

Author: Generated for Long-LRM evaluation alignment with SparseSplat
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_llrm_results(summary_csv):
    """Load Long-LRM evaluation results from summary.csv."""
    df = pd.read_csv(summary_csv)
    # The last row is the mean
    mean_row = df.iloc[-1]
    scene_rows = df.iloc[:-1]

    results = {
        'scenes': scene_rows['scene_name'].tolist(),
        'psnr': scene_rows['psnr'].tolist(),
        'ssim': scene_rows['ssim'].tolist(),
        'lpips': scene_rows['lpips'].tolist(),
        'num_gaussians_total': scene_rows.get('num_gaussians_total', [None] * len(scene_rows)).tolist(),
        'num_gaussians_active': scene_rows.get('num_gaussians_active', [None] * len(scene_rows)).tolist(),
        'mean_psnr': mean_row['psnr'],
        'mean_ssim': mean_row['ssim'],
        'mean_lpips': mean_row['lpips'],
    }

    return results


def load_sparsesplat_results(results_dir):
    """Load SparseSplat evaluation results."""
    # This function needs to be adapted based on SparseSplat's output format
    # Placeholder implementation
    print("Note: SparseSplat results loading needs to be implemented based on actual format")
    return None


def create_comparison_table(llrm_results, sparsesplat_results, output_path):
    """Create comparison table of metrics."""
    data = {
        'Metric': ['PSNR ↑', 'SSIM ↑', 'LPIPS ↓'],
        'Long-LRM': [
            f"{llrm_results['mean_psnr']:.3f}",
            f"{llrm_results['mean_ssim']:.4f}",
            f"{llrm_results['mean_lpips']:.4f}",
        ]
    }

    if sparsesplat_results:
        data['SparseSplat'] = [
            f"{sparsesplat_results['mean_psnr']:.3f}",
            f"{sparsesplat_results['mean_ssim']:.4f}",
            f"{sparsesplat_results['mean_lpips']:.4f}",
        ]
        data['Difference'] = [
            f"{llrm_results['mean_psnr'] - sparsesplat_results['mean_psnr']:+.3f}",
            f"{llrm_results['mean_ssim'] - sparsesplat_results['mean_ssim']:+.4f}",
            f"{llrm_results['mean_lpips'] - sparsesplat_results['mean_lpips']:+.4f}",
        ]

    df = pd.DataFrame(data)

    # Save to CSV
    csv_path = output_path / 'comparison_table.csv'
    df.to_csv(csv_path, index=False)

    # Create formatted markdown table
    md_path = output_path / 'comparison_table.md'
    with open(md_path, 'w') as f:
        f.write("# Long-LRM vs SparseSplat Comparison\n\n")
        f.write("## Mean Metrics\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        f.write("↑ indicates higher is better, ↓ indicates lower is better\n")

    print(f"\nComparison table saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  Markdown: {md_path}")

    return df


def create_gaussian_statistics(llrm_results, output_path):
    """Create Gaussian statistics visualization."""
    scenes = llrm_results['scenes']
    num_total = llrm_results['num_gaussians_total']
    num_active = llrm_results['num_gaussians_active']

    # Filter out None values
    valid_indices = [i for i, (t, a) in enumerate(zip(num_total, num_active)) if t is not None and a is not None]
    if not valid_indices:
        print("Warning: No Gaussian statistics available")
        return

    scenes = [scenes[i] for i in valid_indices]
    num_total = [num_total[i] for i in valid_indices]
    num_active = [num_active[i] for i in valid_indices]

    # Statistics
    stats = {
        'mean_total': np.mean(num_total),
        'mean_active': np.mean(num_active),
        'median_total': np.median(num_total),
        'median_active': np.median(num_active),
        'usage_ratio': np.mean([a/t for a, t in zip(num_active, num_total)])
    }

    # Save statistics
    stats_path = output_path / 'gaussian_statistics.txt'
    with open(stats_path, 'w') as f:
        f.write("Gaussian Statistics (Long-LRM)\n")
        f.write("="*50 + "\n\n")
        f.write(f"Mean total Gaussians: {stats['mean_total']:.0f}\n")
        f.write(f"Mean active Gaussians (opacity > threshold): {stats['mean_active']:.0f}\n")
        f.write(f"Median total Gaussians: {stats['median_total']:.0f}\n")
        f.write(f"Median active Gaussians: {stats['median_active']:.0f}\n")
        f.write(f"Mean usage ratio: {stats['usage_ratio']:.2%}\n")

    print(f"\nGaussian statistics saved to: {stats_path}")

    # Create histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(num_total, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(stats['mean_total'], color='red', linestyle='--', label=f'Mean: {stats["mean_total"]:.0f}')
    axes[0].set_xlabel('Number of Gaussians (Total)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Total Gaussians')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(num_active, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(stats['mean_active'], color='red', linestyle='--', label=f'Mean: {stats["mean_active"]:.0f}')
    axes[1].set_xlabel('Number of Gaussians (Active)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Active Gaussians')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / 'gaussian_distribution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Gaussian distribution plot saved to: {plot_path}")


def create_per_scene_comparison(llrm_results, output_path):
    """Create per-scene metrics visualization."""
    scenes = llrm_results['scenes']
    psnr = llrm_results['psnr']
    ssim = llrm_results['ssim']
    lpips = llrm_results['lpips']

    # Create DataFrame
    df = pd.DataFrame({
        'Scene': scenes,
        'PSNR': psnr,
        'SSIM': ssim,
        'LPIPS': lpips,
    })

    # Sort by PSNR
    df_sorted = df.sort_values('PSNR', ascending=False)

    # Save detailed results
    csv_path = output_path / 'per_scene_metrics.csv'
    df_sorted.to_csv(csv_path, index=False)
    print(f"\nPer-scene metrics saved to: {csv_path}")

    # Create box plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics = ['PSNR', 'SSIM', 'LPIPS']
    colors = ['skyblue', 'lightgreen', 'salmon']

    for ax, metric, color in zip(axes, metrics, colors):
        data = df[metric]
        ax.boxplot([data], labels=[metric], patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax.set_ylabel('Value')
        ax.set_title(f'{metric} Distribution')
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics text
        mean_val = data.mean()
        median_val = data.median()
        ax.text(1.15, ax.get_ylim()[1] * 0.95, f'Mean: {mean_val:.3f}\nMedian: {median_val:.3f}',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plot_path = output_path / 'metrics_distribution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Metrics distribution plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare Long-LRM and SparseSplat results')
    parser.add_argument('--llrm_results', type=str, required=True,
                        help='Path to Long-LRM summary.csv')
    parser.add_argument('--sparsesplat_results', type=str, default=None,
                        help='Path to SparseSplat results directory (optional)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for comparison results')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Long-LRM vs SparseSplat Comparison")
    print("="*60)

    # Load results
    print("\nLoading Long-LRM results...")
    llrm_results = load_llrm_results(args.llrm_results)
    print(f"  Loaded {len(llrm_results['scenes'])} scenes")

    sparsesplat_results = None
    if args.sparsesplat_results:
        print("\nLoading SparseSplat results...")
        sparsesplat_results = load_sparsesplat_results(args.sparsesplat_results)

    # Create comparisons
    print("\nGenerating comparison reports...")

    # 1. Comparison table
    create_comparison_table(llrm_results, sparsesplat_results, output_path)

    # 2. Gaussian statistics
    create_gaussian_statistics(llrm_results, output_path)

    # 3. Per-scene comparison
    create_per_scene_comparison(llrm_results, output_path)

    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)
    print(f"All results saved to: {output_path}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
