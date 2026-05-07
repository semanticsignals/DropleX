#!/usr/bin/env python3
"""
Common utilities for calculating and displaying region statistics.
Used by both measure.py and load_regions.py
"""

import numpy as np
from scipy import ndimage as ndi


def compute_region_area(region_mask, pixel_spacing_mm=None):
    """
    Compute the area of a region in pixels and optionally in mm².

    Parameters:
    -----------
    region_mask : np.ndarray
        Binary mask of the region
    pixel_spacing_mm : float or tuple, optional
        Physical spacing between pixels in mm. If tuple, (y_spacing, x_spacing).
        If None, returns pixel count only.

    Returns:
    --------
    area_info : dict
        Dictionary with area information including dimensions
    """
    area_pixels = np.sum(region_mask)

    # Get bounding box dimensions
    y_coords, x_coords = np.where(region_mask)
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()

    # Dimensions in pixels (inclusive)
    width_pixels = x_max - x_min + 1
    height_pixels = y_max - y_min + 1

    # Compute perimeter using edge detection
    # Find boundary pixels (pixels that have at least one non-region neighbor)
    eroded = ndi.binary_erosion(region_mask)
    boundary = region_mask & ~eroded
    perimeter_pixels = np.sum(boundary)

    # Compute perimeter roughness (ratio of actual perimeter to ideal circular perimeter)
    # For a circle: perimeter = 2 * pi * sqrt(area / pi) = 2 * sqrt(pi * area)
    if area_pixels > 0:
        ideal_perimeter = 2 * np.sqrt(np.pi * area_pixels)
        perimeter_roughness = perimeter_pixels / ideal_perimeter
    else:
        ideal_perimeter = 0
        perimeter_roughness = 0

    area_info = {
        'area_pixels': int(area_pixels),
        'width_pixels': int(width_pixels),
        'height_pixels': int(height_pixels),
        'perimeter_pixels': int(perimeter_pixels),
        'perimeter_roughness': float(perimeter_roughness),
        'bbox': (y_min, y_max, x_min, x_max)
    }

    if pixel_spacing_mm is not None:
        if isinstance(pixel_spacing_mm, (int, float)):
            # Uniform spacing
            area_mm2 = area_pixels * (pixel_spacing_mm ** 2)
            width_mm = width_pixels * pixel_spacing_mm
            height_mm = height_pixels * pixel_spacing_mm
            perimeter_mm = perimeter_pixels * pixel_spacing_mm
        else:
            # Different spacing in x and y - use average for perimeter
            y_spacing, x_spacing = pixel_spacing_mm
            area_mm2 = area_pixels * y_spacing * x_spacing
            width_mm = width_pixels * x_spacing
            height_mm = height_pixels * y_spacing
            perimeter_mm = perimeter_pixels * np.mean([y_spacing, x_spacing])

        area_info['area_mm2'] = area_mm2
        area_info['width_mm'] = width_mm
        area_info['height_mm'] = height_mm
        area_info['perimeter_mm'] = perimeter_mm
        area_info['pixel_spacing'] = pixel_spacing_mm

    return area_info


def calculate_region_stats(region_mask, data, pixel_spacing_mm=None):
    """
    Calculate statistics for a single region.

    Parameters:
    -----------
    region_mask : np.ndarray
        Binary mask of the region
    data : np.ndarray
        2D data array
    pixel_spacing_mm : float or tuple, optional
        Physical spacing between pixels in mm

    Returns:
    --------
    stats : dict
        Dictionary containing all region statistics
    """
    region_values = data[region_mask]
    y_coords, x_coords = np.where(region_mask)

    # Special handling for 2-pixel regions: bias towards pixel with largest absolute value
    if len(y_coords) == 2:
        val0 = data[y_coords[0], x_coords[0]]
        val1 = data[y_coords[1], x_coords[1]]
        if abs(val0) > abs(val1):
            centroid_y = y_coords[0]
            centroid_x = x_coords[0]
        else:
            centroid_y = y_coords[1]
            centroid_x = x_coords[1]
    else:
        # Normal centroid calculation with rounding
        centroid_y = int(np.mean(y_coords) + 0.5)
        centroid_x = int(np.mean(x_coords) + 0.5)

    centroid_value = data[centroid_y, centroid_x]

    # Compute area info
    area_info = compute_region_area(region_mask, pixel_spacing_mm)

    stats = {
        'pixel_count': np.sum(region_mask),
        'centroid_y': centroid_y,
        'centroid_x': centroid_x,
        'centroid_value': centroid_value,
        'sum': np.sum(region_values),
        'mean': np.mean(region_values),
        'std': np.std(region_values),
        'min': np.min(region_values),
        'max': np.max(region_values),
        'area_pixels': area_info['area_pixels'],
        'width_pixels': area_info['width_pixels'],
        'height_pixels': area_info['height_pixels'],
        'perimeter_pixels': area_info['perimeter_pixels'],
        'perimeter_roughness': area_info['perimeter_roughness'],
        'bbox': area_info['bbox']
    }

    # Add mm measurements if available
    if 'area_mm2' in area_info:
        stats['area_mm2'] = area_info['area_mm2']
        stats['width_mm'] = area_info['width_mm']
        stats['height_mm'] = area_info['height_mm']
        stats['perimeter_mm'] = area_info['perimeter_mm']
        stats['pixel_spacing'] = area_info['pixel_spacing']

    return stats


def calculate_summary_stats(individual_regions, combined_mask, data, pixel_spacing_mm=None):
    """
    Calculate summary statistics for all regions combined and per-region averages.

    Parameters:
    -----------
    individual_regions : list of np.ndarray
        List of binary masks for individual regions
    combined_mask : np.ndarray
        Combined binary mask of all regions
    data : np.ndarray
        2D data array
    pixel_spacing_mm : float or tuple, optional
        Physical spacing between pixels in mm

    Returns:
    --------
    summary : dict
        Dictionary containing combined and average statistics
    """
    num_regions = len(individual_regions)
    all_selected_values = data[combined_mask]

    # Combined statistics
    y_coords_all, x_coords_all = np.where(combined_mask)
    centroid_y_all = int(np.mean(y_coords_all) + 0.5)
    centroid_x_all = int(np.mean(x_coords_all) + 0.5)
    centroid_value_all = data[centroid_y_all, centroid_x_all]

    area_info_all = compute_region_area(combined_mask, pixel_spacing_mm)

    # Per-region averages
    avg_perimeter_pixels = np.mean([compute_region_area(r, pixel_spacing_mm)['perimeter_pixels']
                                     for r in individual_regions])
    avg_roughness = np.mean([compute_region_area(r, pixel_spacing_mm)['perimeter_roughness']
                             for r in individual_regions])
    avg_centroid_value = np.mean([data[int(np.mean(np.where(r)[0]) + 0.5),
                                       int(np.mean(np.where(r)[1]) + 0.5)]
                                  for r in individual_regions])
    avg_sum = np.mean([np.sum(data[r]) for r in individual_regions])

    summary = {
        'num_regions': num_regions,
        'combined': {
            'pixel_count': int(np.sum(combined_mask)),
            'centroid_y': centroid_y_all,
            'centroid_x': centroid_x_all,
            'centroid_value': centroid_value_all,
            'sum': np.sum(all_selected_values),
            'mean': np.mean(all_selected_values),
            'std': np.std(all_selected_values),
            'min': np.min(all_selected_values),
            'max': np.max(all_selected_values),
            'area_pixels': area_info_all['area_pixels'],
            'width_pixels': area_info_all['width_pixels'],
            'height_pixels': area_info_all['height_pixels'],
            'perimeter_pixels': area_info_all['perimeter_pixels'],
            'perimeter_roughness': area_info_all['perimeter_roughness']
        },
        'averages': {
            'perimeter_pixels': avg_perimeter_pixels,
            'perimeter_roughness': avg_roughness,
            'centroid_value': avg_centroid_value,
            'sum': avg_sum
        }
    }

    # Add mm measurements if available
    if 'area_mm2' in area_info_all:
        summary['combined']['area_mm2'] = area_info_all['area_mm2']
        summary['combined']['width_mm'] = area_info_all['width_mm']
        summary['combined']['height_mm'] = area_info_all['height_mm']
        summary['combined']['perimeter_mm'] = area_info_all['perimeter_mm']

    return summary


def print_region_stats(stats, region_idx, total_regions):
    """
    Print statistics for a single region.

    Parameters:
    -----------
    stats : dict
        Statistics dictionary from calculate_region_stats()
    region_idx : int
        Region index (1-based for display)
    total_regions : int
        Total number of regions
    """
    print(f"\n--- Region {region_idx} of {total_regions} ---")
    print(f"Pixel count: {stats['pixel_count']}")

    # Display dimensions
    print(f"Dimensions: {stats['width_pixels']} x {stats['height_pixels']} pixels", end='')
    if 'width_mm' in stats:
        print(f" ({stats['width_mm']:.1f} x {stats['height_mm']:.1f} mm)")
    else:
        print()

    # Display area
    print(f"Area: {stats['area_pixels']} pixels", end='')
    if 'area_mm2' in stats:
        print(f" ({stats['area_mm2']:.2f} mm²)")
    else:
        print()

    # Display perimeter
    print(f"Perimeter: {stats['perimeter_pixels']} pixels", end='')
    if 'perimeter_mm' in stats:
        print(f" ({stats['perimeter_mm']:.2f} mm)")
    else:
        print()
    print(f"Perimeter roughness: {stats['perimeter_roughness']:.3f}")

    print(f"Centroid: ({stats['centroid_x']}, {stats['centroid_y']})")
    print(f"Centroid value: {stats['centroid_value']:.2f}")
    print(f"Sum: {stats['sum']:.2f}")
    print(f"Average value: {stats['mean']:.2f}")
    print(f"Std dev: {stats['std']:.2f}")
    print(f"Min value: {stats['min']:.2f}")
    print(f"Max value: {stats['max']:.2f}")


def print_summary_stats(summary):
    """
    Print summary statistics for all regions.

    Parameters:
    -----------
    summary : dict
        Summary statistics dictionary from calculate_summary_stats()
    """
    combined = summary['combined']
    averages = summary['averages']

    print(f"\n--- All Selected Regions Combined ---")
    print(f"Number of regions: {summary['num_regions']}")
    print(f"Pixel count: {combined['pixel_count']}")

    # Display combined dimensions
    print(f"Dimensions: {combined['width_pixels']} x {combined['height_pixels']} pixels", end='')
    if 'width_mm' in combined:
        print(f" ({combined['width_mm']:.1f} x {combined['height_mm']:.1f} mm)")
    else:
        print()

    # Display combined area
    print(f"Area: {combined['area_pixels']} pixels", end='')
    if 'area_mm2' in combined:
        print(f" ({combined['area_mm2']:.2f} mm²)")
    else:
        print()

    # Note: Perimeter, roughness, and centroid are not meaningful for disconnected regions
    print(f"Sum: {combined['sum']:.2f}")
    print(f"Average value: {combined['mean']:.2f}")
    print(f"Std dev: {combined['std']:.2f}")
    print(f"Min value: {combined['min']:.2f}")
    print(f"Max value: {combined['max']:.2f}")

    # Print per-region averages
    print(f"\n--- Per-Region Averages ---")
    print(f"Average perimeter: {averages['perimeter_pixels']:.2f} pixels")
    print(f"Average perimeter roughness: {averages['perimeter_roughness']:.3f}")
    print(f"Average centroid value: {averages['centroid_value']:.2f}")
    print(f"Stdev centroid value: {np.std([combined['centroid_value'] for _ in range(summary['num_regions'])]):.2f}")
    print(f"Average sum: {averages['sum']:.2f}")
