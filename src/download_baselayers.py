#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 16:17:35 2025

@author: Hisham Eldardiry
"""

# satellite_image_downloader.py

"""
A general-purpose Earth Engine satellite image downloader for reservoir analysis.

Provides functions to fetch:
- DEM (SRTM)
- Global Surface Water (GSW) layers: frequency, max extent
- (Extensible to Landsat, Sentinel, etc.)

Dependencies: ee, geemap, numpy, rasterio, utils
"""

import ee
import geemap
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import os
from utils import process_large_image

def download_dem(region, output_path):
    """
    Downloads and saves a DEM (SRTM) image.

    Parameters:
        region (ee.Geometry): Earth Engine geometry (bounding box).
        output_path (str): Output GeoTIFF file path

    Returns:
        tuple: (np.ndarray DEM array, ee.Image DEM image for projection reference)
    """
    dem = ee.Image('USGS/SRTMGL1_003').clip(region)
    try:
        target_proj = dem.projection().atScale(30)
        dem_reprojected = dem.reproject(target_proj)
        dem_reprojected_clipped = dem_reprojected.clip(region)
        array = geemap.ee_to_numpy(dem_reprojected_clipped.select("elevation"), region=region, scale=30)
    except Exception:
        print("Using fallback for DEM quadrant stitching...")
        array = process_large_image(dem, region, band="elevation")

    array = np.squeeze(array).astype(np.float32)
    save_geotiff(output_path, array, region)
    return array, dem
 
def download_gsw_frequency(region, output_path, reference_image):
    """
    Downloads and saves Global Surface Water frequency layer.

    Parameters:
        region (ee.Geometry): Region of interest
        output_path (str): Output file path
        reference_image (ee.Image): Image to copy projection from

    Returns:
        np.ndarray: Frequency array
    """
    target_proj = reference_image.projection().atScale(30)
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    gsw = gsw.reproject(target_proj).clip(region)
    try:
        array = geemap.ee_to_numpy(gsw, region=region, scale=30)
    except Exception:
        print("Using fallback for frequency quadrant stitching...")
        array = process_large_image(gsw, region, band="occurrence")

    array = np.squeeze(array).astype(np.float32)
    array[array == -128] = 0
    save_geotiff(output_path, array, region)
    return array

def download_gsw_extent(region, output_path, reference_image):
    """
    Downloads and saves Global Surface Water maximum extent layer.

    Parameters:
        region (ee.Geometry): Region of interest
        output_path (str): Output file path
        reference_image (ee.Image): Image to copy projection from

    Returns:
        np.ndarray: Max extent array
    """
    target_proj = reference_image.projection().atScale(30)
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent")
    gsw = gsw.reproject(target_proj).clip(region)
    try:
        array = geemap.ee_to_numpy(gsw, region=region, scale=30)
    except Exception:
        print("Using fallback for extent quadrant stitching...")
        array = process_large_image(gsw, region, band="max_extent")

    array = np.squeeze(array).astype(np.float32)
    save_geotiff(output_path, array, region)
    return array

def save_geotiff(path, array, region_geom):
    """
    Saves a NumPy array as a single-band GeoTIFF.

    Parameters:
        path (str): Output path
        array (np.ndarray): Raster data
        region_geom (ee.Geometry): Region (used for geotransform)
    """
    coords = region_geom.getInfo()['coordinates'][0]
    xmin = min(pt[0] for pt in coords)
    xmax = max(pt[0] for pt in coords)
    ymin = min(pt[1] for pt in coords)
    ymax = max(pt[1] for pt in coords)
    transform = from_bounds(xmin, ymin, xmax, ymax, array.shape[1], array.shape[0])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=array.shape[0], width=array.shape[1],
        count=1, dtype=array.dtype, crs="EPSG:4326",
        transform=transform
    ) as dst:
        dst.write(array, 1)
    print(f"Saved: {os.path.basename(path)}")
