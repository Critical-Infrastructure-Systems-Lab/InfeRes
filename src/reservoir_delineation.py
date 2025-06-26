#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 15:15:33 2025

@author: Hisham Eldardiry
"""

# reservoir_delineation.py

"""
Reservoir Delineation Module

This module isolates the reservoir extent using DEM and water occurrence data.
It generates a binary mask raster (reservoir_mask.tif) representing the reservoir area
based on MaxExtent and a seed point.

Outputs:
- reservoir_mask.tif: binary mask of the delineated reservoir
- Diagnostic plots: DEM and frequency overlays

Functions:
- delineate_reservoir(...): main interface

Dependencies: numpy, osgeo, matplotlib, utils.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array
from utils import expand_mask, find_dem_pixel, flood_fill

def delineate_reservoir(reservoir_name, max_water_level, seed_point, bounding_box, baselayers_dir, plot=False):
    """
    Isolates a reservoir extent based on DEM and water occurrence raster.

    Parameters:
        reservoir_name (str): Name of the reservoir.
        max_water_level (float): Max observed water level (m).
        seed_point (list): [longitude, latitude] of a seed point.
        bounding_box (list): Bounding box [xmin, ymin, xmax, ymax].
        baselayers_dir (str): Base directory with supporting rasters.
        plot (bool): If True, generates diagnostic plots. Default is False.
    """

    dem_array = gdal_array.LoadFile(os.path.join(baselayers_dir,"DEM.tif")).astype(np.float32)
    freq_array = gdal_array.LoadFile(os.path.join(baselayers_dir,"frequency.tif")).astype(np.float32)
    max_extent_array = gdal_array.LoadFile(os.path.join(baselayers_dir,"max_extent.tif")).astype(np.float32)

    lat, lon = seed_point[1], seed_point[0]
    bounds = [bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]]
    dem_path = os.path.join(baselayers_dir, "DEM.tif")
    col, row = find_dem_pixel(seed_point, dem_path)
    
    # Ensure max_extent_array is binary (1 = valid, 0 = invalid)
    binary_max_extent = (max_extent_array > 0).astype(np.uint8)
    
    # Validate and adjust seed point if needed
    row, col = find_nearest_valid_pixel(binary_max_extent, row, col)
    
    # Restrict DEM to likely flooded areas (below max_wl + buffer)
    dem_mask = np.copy(dem_array)
    dem_mask[dem_mask == 0] = np.nan
    dem_mask[dem_mask > max_water_level + 10] = np.nan
    dem_mask[dem_mask > 0] = 1  # binary mask

    # Extract reservoir extent using flood fill on max extent
    flood_mask = flood_fill(col, row, max_extent_array)
    expanded_mask = expand_mask(flood_mask, 1)

    # Save final mask
    gdal_array.SaveArray(expanded_mask.astype(np.uint8), os.path.join(baselayers_dir, "reservoir_mask.tif"), format="GTiff", prototype=dem_path)

    if plot:
        plt.figure()
        plt.imshow(dem_mask, cmap='viridis')
        plt.scatter([col], [row], c='red', s=20)
        plt.title('DEM-derived Mask')
        plt.savefig(os.path.join(baselayers_dir, f"{reservoir_name}_dem_mask.png"), dpi=600)
        
        # Visualize isolated reservoir mask
        plt.figure()
        plt.imshow(flood_mask, cmap='viridis')
        plt.scatter([col], [row], c='red', s=20)
        plt.title('Initial Reservoir Mask (Flood Fill)')
        plt.savefig(os.path.join(baselayers_dir, f"{reservoir_name}_flood_mask.png"), dpi=600)
        
        plt.figure()
        plt.imshow(expanded_mask, cmap='viridis')
        plt.scatter([col], [row], c='red', s=20)
        plt.title('Initial Reservoir Mask (Flood Fill)')
        plt.savefig(os.path.join(baselayers_dir, f"{reservoir_name}_expanded_mask.png"), dpi=600)


        # Overlay reservoir mask on frequency map for inspection
        freq_overlay = np.copy(freq_array)
        freq_overlay[expanded_mask != 1] = np.nan
        cmap = plt.cm.jet
        cmap.set_bad('white')

        plt.figure()
        plt.imshow(freq_overlay, cmap=cmap)
        plt.colorbar()
        plt.title('Water Occurrence Frequency in Reservoir Area')
        plt.savefig(os.path.join(baselayers_dir,f"{reservoir_name}_frequency_overlay.png"), dpi=600)


def find_nearest_valid_pixel(mask, row, col):
    """
    Finds the nearest valid (non-zero) pixel to a given (row, col) in a binary mask.

    Parameters:
        mask (np.ndarray): Binary mask (1 = valid, 0 = invalid).
        row (int): Row index of the seed.
        col (int): Column index of the seed.

    Returns:
        (int, int): Tuple of (new_row, new_col) representing nearest valid pixel.
    """
    if mask[row, col] == 1:
        return row, col

    import scipy.ndimage

    if not np.any(mask == 1):
        raise ValueError("No valid pixels in the mask.")

    distance, (rows_idx, cols_idx) = scipy.ndimage.distance_transform_edt(
        mask == 0, return_indices=True
    )
    new_row, new_col = rows_idx[row, col], cols_idx[row, col]
    return new_row, new_col
