#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 15:15:33 2025

@author: Hisham Eldardiry
"""



import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal_array
from utils import expand_mask, find_dem_pixel


def compute_dem_hypsometric_curve(dem_path, mask_path, max_water_level, seed_point=None, expansion_radius=3):
    """
    Computes a DEM-derived elevation–area–storage curve for a reservoir.

    Parameters:
        dem_path (str): Path to the DEM raster (e.g. "DEM.tif").
        mask_path (str): Path to the binary reservoir mask raster (e.g. "reservoir_mask.tif").
        max_water_level (float): Maximum expected water level (used to define curve extent).
        seed_point (tuple, optional): (lon, lat) coordinates of dam point. If provided, elevation at this location is used as curve base.
        expansion_radius (int): Number of pixels to buffer the reservoir mask for DEM cropping.

    Returns:
        tuple:
            - min_elev (int): Minimum elevation (start of curve).
            - curve (list of lists): Curve data with header, each row [Elevation (m), Area (sq.km), Storage (mcm)].
    """
    dem = gdal_array.LoadFile(dem_path).astype(np.float32)
    mask = gdal_array.LoadFile(mask_path).astype(np.float32)

    # Expand the reservoir mask and apply it to DEM
    dem[expand_mask(mask, expansion_radius) != 1] = np.nan

    # Get the starting elevation either from the DEM value at a point or the DEM minimum
    if seed_point:
        col, row = find_dem_pixel(seed_point, dem_path)
        min_elev = int(dem[row, col])
    else:
        min_elev = int(np.nanmin(dem))

    max_elev = int(max_water_level + 20)
    # Adjust if max_elev is lower than min_elev
    if max_elev <= min_elev:
        print(f"Warning: max_elev ({max_elev}) <= min_elev ({min_elev}). Adjusting max_elev....")
        max_elev = min_elev + 50
    
    curve = [["Elevation (m)", "Area (sq.km)", "Storage (mcm)"]]
    prev_area = 0
    cumulative_storage = 0

    # Loop through elevation levels to compute area and cumulative storage
    for elev in range(min_elev, max_elev):
        mask_tmp = np.copy(dem)
        mask_tmp[dem > elev] = 0
        mask_tmp[mask_tmp > 0] = 1
        area_km2 = np.nansum(mask_tmp) * 9 / 10000
        storage = (area_km2 + prev_area) / 2
        cumulative_storage += storage
        prev_area = area_km2
        curve.append([elev, round(area_km2, 4), round(cumulative_storage, 4)])

    return min_elev, curve


def merge_grdl_and_dem_curves(grdl_curve_np, dem_curve_np):
    """
    Merges a GRDL-derived and a DEM-derived elevation–area (or elevation–area–storage) curve
    into a continuous, non-overlapping elevation–area(-storage) curve.

    This function follows the following logic:
    1. Comparing the maximum area from the GRDL curve to the starting area in the DEM curve.
    2. If the GRDL curve overlaps significantly (i.e., its max area is larger than the DEM's starting area),
       it trims the GRDL curve just before the overlap, and removes overlapping values from the DEM curve.
    3. If the GRDL curve ends below the DEM curve (its max area is smaller than the DEM's),
       it trims the last row of GRDL and removes redundant low-elevation rows from the DEM.
    4. The trimmed GRDL and DEM segments are then vertically stacked to form the merged curve.

    Parameters:
        grdl_curve_np (np.ndarray): GRDL curve in format [Elevation, Area].
                                    Elevation is inferred from depth using (bottom_elev + depth).
        dem_curve_np (np.ndarray): DEM-based curve in format [Elevation, Area] or [Elevation, Area, Storage].

    Returns:
        np.ndarray: Merged curve combining GRDL and DEM data.
                    The returned array preserves the same number of columns as the DEM input
                    (i.e., will include Storage if DEM input had it).
    """
    
    # Split GRDL and DEM curves into elevation and area for comparison
    grdl_elev = grdl_curve_np[:, 0]
    grdl_area = grdl_curve_np[:, 1]
    
    dem_elev = dem_curve_np[:, 0]
    dem_area = dem_curve_np[:, 1]

    # ----------------------------
    # CASE 1: GRDL area >= DEM area at overlap
    # This means GRDL curve goes higher in area than the start of DEM curve
    # → We need to trim GRDL's top rows and DEM's bottom rows
    # ----------------------------
    if grdl_area[-1] >= dem_area[-1]:
        # Find first DEM point with area above a reasonable threshold
        # (To avoid noisy low-area DEM pixels)
        small_area_thresh = 10  # km²
        dem_pos = np.argmax(dem_area > small_area_thresh)
        
        # Match GRDL area closest to this DEM area value
        overlap_val = dem_area[dem_pos]
        grdl_diff = np.abs(grdl_area - overlap_val)
        grdl_pos = np.argmin(grdl_diff)
        
        # Gets the elevation at the DEM point where area first exceeds the overlap threshold.
        dem_overlap_elev = dem_curve_np[dem_pos, 0]
        
        
        # Trim GRDL up to the overlap point (keep low-elevation portion) and drop grdl elevations that conflict with DEM
        grdl_trimmed = grdl_curve_np[grdl_curve_np[:, 0] < dem_overlap_elev - 0.5]
        
        # Trim DEM after the overlap point (keep higher elevations)
        dem_trimmed = dem_curve_np[dem_pos + 1:]  # ABOVE
        
        # Merge both parts together
        merged = np.vstack((grdl_trimmed, dem_trimmed))
        _, idx = np.unique(merged[:, 0], return_index=True)
        merged_clean = merged[np.sort(idx)]

    # ----------------------------
    # CASE 2: GRDL area < DEM area at overlap
    # GRDL ends before DEM begins — less overlap
    # → Keep all of GRDL except last row, then start DEM after overlap
    # ----------------------------
    else:
        # Get the final area value from the GRanD curve (used to find overlap with DEM)
        overlap_val = grdl_area[-1]
    
        # Compute the absolute difference between DEM areas and the GRanD final area
        dem_diff = np.abs(dem_area - overlap_val)
    
        # Find the DEM index where the area is closest to GRanD’s final area (overlap point)
        dem_pos = np.argmin(dem_diff)
        
        # Get elevation at the DEM point matching GRanD's last area
        overlap_elev = dem_curve_np[dem_pos, 0]
        
        # Remove GRanD rows at or above the overlapping elevation
        grdl_trimmed = grdl_curve_np[grdl_curve_np[:, 0] < overlap_elev - 0.5]
        
        # Remove DEM rows at or below the overlap point
        dem_trimmed = dem_curve_np[dem_curve_np[:, 0] > overlap_elev + 0.5]
        
        # Merge and clean
        merged = np.vstack((grdl_trimmed, dem_trimmed))
        _, idx = np.unique(merged[:, 0], return_index=True)
        merged_clean = merged[np.sort(idx)]

    return merged_clean




def apply_bias_correction(merged_curve, reservoir_area_km2, reference_capacity, max_water_level):
    """
    Applies bias correction to a merged elevation–area curve by aligning both area and storage
    with known reference values (area from mask, volume from metadata) at the max water level.

    Recomputes cumulative storage using trapezoidal integration (with 1m elevation steps),
    adjusts area values to match the DEM-derived reservoir area, and scales storage to match
    known volume at max water level.

    Parameters:
        merged_curve (np.ndarray): Array of merged [Elevation, Area] or [Elevation, Area, Volume].
        reservoir_area_km2 (float): Known surface area (e.g., from DEM mask).
        reference_capacity (float): Known total volume at max water level (e.g., from GRanD).
        max_water_level (float): Full supply elevation for alignment.

    Returns:
        tuple:
            - corrected_curve (np.ndarray): Array with [Elevation, Area, Corrected Storage].
            - merged_with_storage (np.ndarray): Same as above but with uncorrected storage.
    """

    # STEP 1: Integrate uncorrected storage using trapezoidal rule
    prev_area = 0
    cumulative_storage = 0
    uncorrected_storage = []
    
    
    for i in range(len(merged_curve)):
        elev = merged_curve[i, 0]
        area = merged_curve[i, 1]
        dz = merged_curve[i, 0] - merged_curve[i - 1, 0] if i > 0 else 0
        storage = (area + prev_area) / 2 * dz
        cumulative_storage += storage
        uncorrected_storage.append(round(cumulative_storage, 3))
        prev_area = area

    merged_with_storage = np.column_stack((merged_curve[:, 0], merged_curve[:, 1], uncorrected_storage))

    # STEP 2: Area correction at max_water_level
    area_at_max_wl = np.interp(max_water_level, merged_with_storage[:, 0], merged_with_storage[:, 1])
    area_ratio = area_at_max_wl / reservoir_area_km2
    area_corrected = merged_with_storage[:, 1] / area_ratio

    # STEP 3: Re-integrate storage with corrected area
    corrected_storage = []
    cumulative_storage = 0
    prev_area = 0

    for i in range(len(area_corrected)):
        dz = merged_with_storage[i, 0] - merged_with_storage[i - 1, 0] if i > 0 else 0
        storage = (area_corrected[i] + prev_area) / 2 * dz
        cumulative_storage += storage
        corrected_storage.append(round(cumulative_storage, 3))
        prev_area = area_corrected[i]

    # STEP 4: Volume correction at max_water_level
    volume_at_max_wl = np.interp(max_water_level, merged_with_storage[:, 0], corrected_storage)
    volume_ratio = volume_at_max_wl / reference_capacity
    storage_final = np.array(corrected_storage) / volume_ratio

    # Final output [Elevation, Area, Storage]
    corrected_curve = np.column_stack((
        merged_with_storage[:, 0],
        area_corrected,
        storage_final
    ))

    return corrected_curve, merged_with_storage



def generate_curve_pre_srtm(reservoir_name, seed_point, bounding_box, max_water_level,
                            baselayers_dir, output_dir, reference_curves_dir,
                            grand_id, reference_capacity, bias_correction=True,plot_curve=False):
    """
    Generates a pre-SRTM elevation–area–storage curve by merging DEM and GRDL curves,
    applying volume correction, and exporting to CSV and PNG.

    This version computes min_elevation from max_water_level - max_GRDL_depth.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load GRDL reference CSV (GRanD)
    grdl_path = os.path.join(reference_curves_dir)
    matching_files = [
    f for f in os.listdir(grdl_path)
    if f.endswith(".csv") and int(''.join(filter(str.isdigit, f))) == int(grand_id)
    ]
    if not matching_files:
        raise FileNotFoundError(f"No GRDL curve found for GRAND_ID {grand_id}")    
        
    df_grdl = pd.read_csv(os.path.join(grdl_path, matching_files[0]), parse_dates=True)
    headers = df_grdl.iloc[3, 0].split(';')
    grdl_rows = [row.iloc[0].split(';') for _, row in df_grdl.iloc[4:].iterrows()]
    grdl_curve = pd.DataFrame(grdl_rows, columns=headers).astype(np.float32)
    grdl_np_raw = grdl_curve.to_numpy()
    
    # Convert depth → elevation using estimated bottom_elev
    # ➤ Estimate bottom_elev using: max_wl - max depth in GRDL
    max_grdl_depth = np.max(grdl_np_raw[:, 0])
    bottom_elev = max_water_level - max_grdl_depth
    grdl_elevs = bottom_elev + grdl_np_raw[:, 0]  # Elevation = base + depth
    grdl_areas = grdl_np_raw[:, 1]
    grdl_volumes = grdl_np_raw[:, 2]
    grdl_curve_np = np.column_stack((grdl_elevs, grdl_areas, grdl_volumes))

    # Compute DEM-based curve
    dem_path = os.path.join(baselayers_dir, "DEM.tif")
    mask_path = os.path.join(baselayers_dir, "reservoir_mask.tif")
    min_elev_dem, dem_curve = compute_dem_hypsometric_curve(dem_path, mask_path, max_water_level, seed_point)
    dem_curve_np = np.array(dem_curve[1:], dtype=np.float32)
    
    # Merge DEM and GRDL using corrected min_elev
    merged_curve = merge_grdl_and_dem_curves(grdl_curve_np, dem_curve_np)

    # Estimate max surface area from mask
    mask = gdal_array.LoadFile(mask_path).astype(np.float32)
    reservoir_area_km2 = round(np.count_nonzero(mask == 1) * 0.0009, 2)

    # Bias correction to match known area & storage
    if bias_correction:
        corrected_curve, merged_before_correction = apply_bias_correction(
            merged_curve, reservoir_area_km2, reference_capacity,max_water_level
        )
    else:
        corrected_curve=merged_curve
        merged_before_correction=merged_curve
    # Trim elevations beyond max water level buffer
    # wl_extra = np.max(corrected_curve[:, 0]) - max_water_level
    # trim_n = int(round(wl_extra / 2))
    # if trim_n > 0:
    #     corrected_curve = corrected_curve[:-trim_n]

    # Save to CSV
    header = ["Elevation (m)", "Area (sq.km)", "Storage (mcm)"]
    output_data = [header] + corrected_curve.astype(str).tolist()
    csv_path = os.path.join(output_dir, "reservoir_hypsometry.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(output_data)

    # Optional plotting
    if plot_curve:
        plt.figure()
        plt.scatter(merged_before_correction[:, 0], merged_before_correction[:, 2], s=8, c='gray', label="Before Correction")
        plt.scatter(corrected_curve[:, 0], corrected_curve[:, 2], s=8, c='blue', label="After Correction")
        plt.xlabel("Elevation (m)")
        plt.ylabel("Storage (mcm)")
        plt.title(f"{reservoir_name} Storage–Elevation Curve")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{reservoir_name}_storage_curve.png"), dpi=600, bbox_inches='tight')

    return int(corrected_curve[0, 0])


def generate_curve_post_srtm(reservoir_name, max_water_level, baselayers_dir, output_dir, plot_curve=False):
    """
    Generates a hypsometric curve (Elevation–Area–Storage) from DEM only (no reference correction).

    Suitable for post-SRTM reservoirs where no GRDL (reference) curve is available.

    Parameters:
        reservoir_name (str): Reservoir name (used in plots and filenames).
        max_water_level (float): Maximum elevation level for the curve.
        baselayers_dir (str): Directory containing 'DEM.tif' and 'reservoir_mask.tif'.
        output_dir (str): Output directory for CSV and plots.
        plot_curve (bool): If True, saves a PNG of the storage–elevation plot.

    Returns:
        int: Minimum elevation (start of curve).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load DEM and mask file paths
    dem_path = os.path.join(baselayers_dir, "DEM.tif")
    mask_path = os.path.join(baselayers_dir, "reservoir_mask.tif")

    # Compute hypsometric curve from DEM
    min_elev, curve_data = compute_dem_hypsometric_curve(
        dem_path=dem_path,
        mask_path=mask_path,
        max_water_level=max_water_level
    )

    # Save the resulting elevation-area-storage curve to CSV
    csv_path = os.path.join(output_dir, "reservoir_hypsometry.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(curve_data)

    # Optional: plot and save the curve
    if plot_curve:
        data_array = np.array(curve_data[1:], dtype=np.float32)  # Skip header
        plt.figure()
        plt.scatter(data_array[:, 0], data_array[:, 2], s=8, c='red')
        plt.xlabel("Elevation (m)")
        plt.ylabel("Storage (mcm)")
        plt.title(f"{reservoir_name} Storage–Elevation Curve")
        plt.savefig(
            os.path.join(output_dir, f"{reservoir_name}_storage_curve.png"),
            dpi=600, bbox_inches='tight'
        )

    return min_elev
