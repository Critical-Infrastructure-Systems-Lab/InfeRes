#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 14:18:02 2025

@author: Hisham Eldardiry
"""


# utils.py

"""
General and Earth Engine Utility Functions

This module contains shared helper functions for:
- Raster and array operations
- Coordinate transformations
- Earth Engine image downloading and handling

Dependencies: numpy, osgeo, geemap, rasterio, ee
"""

import numpy as np
from osgeo import gdal, osr, gdal_array
import geemap
import ee
import pandas as pd
import os
import ast



# -----------------------------
# General Utilities
# -----------------------------

def parse_reservoir_ids(value):
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        value = value.strip()
        if value.endswith('.txt'):
            if not os.path.exists(value):
                raise FileNotFoundError(f"File not found: {value}")
            with open(value) as f:
                return [int(line.strip()) for line in f if line.strip().isdigit()]
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
            elif isinstance(parsed, int):
                return [parsed]
        except Exception:
            raise ValueError(f"Invalid RESERVOIR_GRAND_ID: {value}")
    raise TypeError(f"Unsupported type for RESERVOIR_GRAND_ID: {type(value)}")


def setup_reservoir_folders(res_name, input_base="input", output_base="output"):
    """
    Creates full directory structure for reservoir processing.

    Structure:
    output/{res_name}/
        ├── baselayers/        # DEM, extent, frequency
        ├── curve.csv          # Hypsometric curve
        └── reservoir_area_storage.csv

    Returns:
        str: Absolute path to baselayers directory
        str: Absolute path to reservoir output root
    """
    output_res_dir = os.path.abspath(os.path.join(output_base, res_name))
    baselayers_dir = os.path.join(output_res_dir, "baselayers")

    os.makedirs(baselayers_dir, exist_ok=True)
    return baselayers_dir, output_res_dir

def load_reservoir_metadata(csv_path, active_only=True):
    """
    Loads reservoir metadata from a CSV file.

    Parameters:
        csv_path (str): Path to the CSV file.
        active_only (bool): If True, filters rows with status == 1.

    Returns:
        pd.DataFrame: Filtered reservoir metadata.
    """
    df = pd.read_csv(csv_path)
    if active_only and "run_flag" in df.columns:
        df = df[df["run_flag"] == 1]
    return df.reset_index(drop=True)

def flood_fill(seed_x, seed_y, binary_mask):
    """
    Performs a custom flood fill on a binary mask starting from a seed point.

    Parameters:
        seed_x (int): Column index (x).
        seed_y (int): Row index (y).
        binary_mask (np.ndarray): Binary array (1 = valid, 0 = invalid).

    Returns:
        np.ndarray: Binary flood-filled mask with same shape as input.
    """
    filled = set()
    fill = set()
    fill.add((seed_x, seed_y))
    width = binary_mask.shape[1] - 1
    height = binary_mask.shape[0] - 1
    flood_mask = np.zeros_like(binary_mask, dtype=np.int8)

    while fill:
        x, y = fill.pop()
        if y == height or x == width or x < 0 or y < 0:
            continue
        if binary_mask[y][x] == 1:
            flood_mask[y][x] = 1
            filled.add((x, y))
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            for nx, ny in neighbors:
                if (nx, ny) not in filled:
                    fill.add((nx, ny))

    return flood_mask


def expand_mask(array, n):
    """
    Expands a binary mask array by n pixels in all directions.

    Parameters:
        array (np.ndarray): Binary array of 0s and 1s.
        n (int): Expansion radius in pixels.

    Returns:
        np.ndarray: Expanded binary array.
    """
    expanded_mask = array-array
    for i in range(len(array)):
        for j in range(len(array[i])):
            if array[i][j] == 1:
                for k in range(max(0, i - n), min(i + n, len(array)-1)):
                    for l in range(max(0, j - n), min(j + n, len(array[i])-1)):
                        expanded_mask[k][l] = 1
    return expanded_mask



def find_dem_pixel(point, dem_path):
    """
    Finds the pixel coordinates in a DEM corresponding to a lat/lon point.

    Parameters:
        point (list): [longitude, latitude].
        dem_path (str): Path to the DEM GeoTIFF.

    Returns:
        (int, int): (column, row) indices in the raster.
    """
    dem_dataset = gdal.Open(dem_path)
    if dem_dataset is None:
        raise FileNotFoundError(f"DEM file not found: {dem_path}")

    geotransform = dem_dataset.GetGeoTransform()
    x_origin = geotransform[0]
    y_origin = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]

    input_srs = osr.SpatialReference()
    input_srs.ImportFromEPSG(4326)
    output_srs = osr.SpatialReference()
    output_srs.ImportFromWkt(dem_dataset.GetProjection())

    transform = osr.CoordinateTransformation(input_srs, output_srs)
    latitude = point[1]
    longitude = point[0]

    x_proj, y_proj, _ = transform.TransformPoint(longitude, latitude)
    column = int((x_proj - x_origin) / pixel_width)
    row = int((y_proj - y_origin) / pixel_height)

    return column, row

# %%  -----------------------------
#       Earth Engine Utilities
#     -----------------------------

def process_large_image(image, region, band=None):
    """
    Converts a large Earth Engine image into a 2D NumPy array by dividing it into 4 quadrants.

    Parameters:
        image (ee.Image): Earth Engine image to convert.
        region (ee.Geometry): Bounding region.
        band (str or None): Specific band to extract.

    Returns:
        np.ndarray: 2D image array stitched from four subregions.
    """
    coords = region['coordinates'][0]
    xmin = min(pt[0] for pt in coords)
    xmax = max(pt[0] for pt in coords)
    ymin = min(pt[1] for pt in coords)
    ymax = max(pt[1] for pt in coords)

    lon_mid = (xmin + xmax) / 2
    lat_mid = (ymin + ymax) / 2

    quadrants_geom = [
        ee.Geometry.BBox(xmin, lat_mid, lon_mid, ymax),  # Top-left
        ee.Geometry.BBox(lon_mid, lat_mid, xmax, ymax),  # Top-right
        ee.Geometry.BBox(xmin, ymin, lon_mid, lat_mid),  # Bottom-left
        ee.Geometry.BBox(lon_mid, ymin, xmax, lat_mid),  # Bottom-right
    ]

    sub_images = []
    for i, quad in enumerate(quadrants_geom):
        try:
            img = image.select(band) if band else image
            img_data = geemap.ee_to_numpy(img, region=quad, scale=30)
            img_data = np.squeeze(img_data).astype(np.float32)
            sub_images.append(img_data)
        except Exception as e:
            print(f"Skipping quadrant {i+1} due to error: {e}")
            sub_images.append(None)

    if all(img is not None for img in sub_images):
        q1, q2, q3, q4 = sub_images
        h_min = min(q.shape[0] for q in sub_images)
        w_min = min(q.shape[1] for q in sub_images)
        q1, q2, q3, q4 = [q[:h_min, :w_min] for q in sub_images]

        top = np.hstack([q1, q2])
        bottom = np.hstack([q3, q4])
        final_image = np.vstack([top, bottom])
        return final_image
    else:
        print("One or more quadrants failed. Returning None.")
        return None


def boundary_to_geometry(boundary):
    """
    Convert bounding box [xmin, ymin, xmax, ymax] to ee.Geometry.BBox.

    Parameters:
        boundary (list): [xmin, ymin, xmax, ymax]

    Returns:
        ee.Geometry.BBox
    """
    return ee.Geometry.BBox(boundary[0], boundary[1], boundary[2], boundary[3])
