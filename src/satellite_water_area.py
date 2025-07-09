#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 15:55:15 2025
@author: Hisham Eldardiry
"""

# satellite_water_area.py

"""
Satellite-Based Water Surface Area Estimation Module

This module estimates water surface area from satellite-derived NDWI images.
It applies unsupervised classification and zone-based filtering.

Outputs:
- Estimated water surface area (in sq. km) per NDWI image.

Functions:
- estimate_water_area(...): main interface
- classify_water_kmeans(...): clustering classification
- zone_based_filtering(...): regional mask filtering
- extract_area_from_composite(...): EE composite to area

Dependencies: numpy, sklearn, scipy.ndimage, osgeo, utils
"""

import numpy as np
from osgeo import gdal_array
import configparser
import geemap
import os

from ndwi_processing import filter_ndwi_by_mask,zone_based_filtering,apply_local_filtering,cluster_filtering

# Load configuration
config = configparser.ConfigParser()
config.read("config.ini")
PIXEL_AREA_KM2 = float(config["Simulation Settings"].get("PIXEL_AREA_KM2", 900 / 1e6))






def estimate_water_area(composite_image, region, baselayers_dir, reservoir_name=None, sensor_id="LS"):
    """
    Processes an EE NDWI composite image to estimate surface water area using a multi-stage pipeline.

    Steps:
    1. Convert Earth Engine image to NumPy
    2. Filter by cloud percentage
    3. Estimate water area at each step:
       - Raw NDWI > 0
       - CLAHE + cloud masking
       - Zone-based filtering
       - Local spatial refinement

    Returns:
        tuple: (cloud_pct, raw_area, clahe_area, zone_filtered_area, local_filtered_area, quality_flag)
    """
    try:
        # Step 1: Convert EE image to NumPy
        ndwi_array = geemap.ee_to_numpy(composite_image, region=region, scale=30)
        if ndwi_array is None or ndwi_array.size == 0 or ndwi_array.ndim < 2:
            print(f"[WARNING] NDWI image for {reservoir_name} is empty or invalid.")
            return None, None, None, None, None,None, 0
        ndwi_array = np.squeeze(ndwi_array).astype(np.float32)
        
        
        # Step 2: Compute cloud % and apply CLAHE mask
        filtered_ndwi, cloud_pct = filter_ndwi_by_mask(ndwi_array, sensor_id, baselayers_dir)
        if cloud_pct is not None and cloud_pct > 85:
            print(f"[INFO] Skipping {reservoir_name} due to high cloud coverage ({cloud_pct}%)")
            return cloud_pct, None, None, None, None,None, 0

        # Step 3: Load masks and frequency data
        mask_path = os.path.join(baselayers_dir, "reservoir_mask.tif")
        res_mask = gdal_array.LoadFile(mask_path).astype(np.float32)
        res_max_area = np.sum(res_mask == 1) * PIXEL_AREA_KM2

        frequency_path = os.path.join(baselayers_dir, "frequency.tif")
        freq_array = gdal_array.LoadFile(frequency_path).astype(np.float32)

        # Step 4: Area from raw NDWI > 0
        raw_ndwi_mask = (ndwi_array > 0).astype(np.uint8)
        raw_area = np.sum(raw_ndwi_mask == 1) * PIXEL_AREA_KM2
        
        
        # Step 5: Area after CLAHE/cloud-masked NDWI > 0
        clahe_mask = (filtered_ndwi > 0).astype(np.uint8)
        clahe_area = np.sum(clahe_mask == 1) * PIXEL_AREA_KM2
        
        # Step 6: Extract dominant water cluster (Level 1 product)
        water_cluster = cluster_filtering(filtered_ndwi)
        # Standardize zone map to range 1–50 and replace NaNs with 0 (non-zone)
        zone_map=freq_array.copy()
        zone_map = np.ceil(zone_map / 2).astype(np.float32)
        zone_map[np.isnan(zone_map)] = 0
        # Mask out non-reservoir areas
        water_cluster_mask=water_cluster.copy()
        water_cluster_mask[zone_map == 0] = 0
        water_cluster_area = np.sum(water_cluster_mask == 1) * PIXEL_AREA_KM2
        
        # Step 7: Refine using zone-based filtering (Level 2 product)
        zone_refined_mask, zone_quality_flag = zone_based_filtering(water_cluster, freq_array)
        zone_filtered_area = np.sum(zone_refined_mask == 1) * PIXEL_AREA_KM2
        
        # Step 8: Local filtering
        local_filtered_mask = apply_local_filtering(zone_refined_mask, freq_array)
        local_filtered_area = np.sum(local_filtered_mask == 1) * PIXEL_AREA_KM2

        # Step 9: Quality check (multi-criteria)
        quality_flag = 1  # assume valid
        
        if (
            raw_area == 0 or
            local_filtered_area > res_max_area or
            zone_filtered_area < 0.01 * res_max_area or
            local_filtered_area < 0.15 * res_max_area
        ):
            quality_flag = 0
            print(f"[WARNING] Low-quality estimate for {reservoir_name}. Using raw_area as fallback.")
            
        

        return (
            cloud_pct,
            round(raw_area, 2),
            round(clahe_area, 2),
            round(water_cluster_area, 2),
            round(zone_filtered_area, 2),
            round(local_filtered_area, 2),
            zone_quality_flag
        )

    except Exception as e:
        print(f"[ERROR] Failed to process composite for {reservoir_name}: {e}")
        return None, None, None, None, None,None, 0
    



