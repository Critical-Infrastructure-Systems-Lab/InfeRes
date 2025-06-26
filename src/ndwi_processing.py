#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 16:39:36 2025

@author: Hisham Eldardiry
"""

# ndwi_processing.py

"""
NDWI Calculation, Cloud Masking, and Preprocessing for Landsat and Sentinel Imagery.

This module includes:
- Scale factor application
- NDWI index calculations
- Cloud/QA pixel masking
- NDWI preprocessing and enhancement (CLAHE)

Dependencies: ee, numpy, cv2, gdal_array, utils
"""

import ee
import os
import numpy as np
import cv2
from osgeo import gdal_array
from utils import expand_mask as expand
from sklearn.cluster import KMeans
from scipy.ndimage import label

# -----------------------------
# Landsat 8/9
# -----------------------------

def mask_qa_pixels_L8_L9(image):
    """
    Masks clouds and shadows in Landsat 8 and 9 imagery using the QA_PIXEL band.
    Keeps pixels with QA_PIXEL values below 22280. Fills masked areas with -2.
    """
    qa = image.select('QA_PIXEL')
    mask = qa.lt(22280)
    return image.updateMask(mask).unmask(-2)


def calculate_ndwi_L8_L9(image):
    """
    Calculates the Normalized Difference Water Index (NDWI) for Landsat 8 and 9.
    NDWI = (Green - SWIR) / (Green + SWIR) using bands SR_B3 (Green) and SR_B5 (SWIR).
    Adds the NDWI as a new band named 'NDWI'.
    """
    ndwi = image.normalizedDifference(['SR_B3', 'SR_B5'])
    ndwi = ndwi.unmask(-2) 
    return image.addBands(ndwi.rename('NDWI'))


# -----------------------------
# Landsat 5/7
# -----------------------------
def mask_qa_pixels_L5_L7(image):
    """
    Masks clouds and shadows in Landsat 5 and 7 imagery using the QA_PIXEL band.
    Keeps pixels with QA_PIXEL values below 5896. Fills masked areas with -2.
    """
    qa = image.select('QA_PIXEL')
    mask = qa.lt(5896)
    return image.updateMask(mask).unmask(-2)


def calculate_ndwi_L5_L7(image):
    """
    Calculates the Normalized Difference Water Index (NDWI) for Landsat 5 and 7.
    NDWI = (Green - SWIR) / (Green + SWIR) using bands SR_B2 (Green) and SR_B4 (SWIR).
    Adds the NDWI as a new band named 'NDWI'.
    """
    ndwi = image.normalizedDifference(['SR_B2', 'SR_B4'])
    ndwi = ndwi.unmask(-2) 
    return image.addBands(ndwi.rename('NDWI'))


# -----------------------------
# Landsat 4
# -----------------------------
def mask_qa_pixels_L4(image):
    """
    Masks clouds and shadows in Landsat 4 imagery using the QA_PIXEL band.
    Uses the same mask threshold as Landsat 5/7 (qa < 5896).
    Keeps pixels with QA_PIXEL values below 5896. Fills masked areas with -2.
    """
    qa = image.select('QA_PIXEL')
    mask = qa.lt(5896)
    return image.updateMask(mask).unmask(-2)


def calculate_ndwi_L4(image):
    """
    Calculates the Normalized Difference Water Index (NDWI) for Landsat 4.
    NDWI = (Green - SWIR) / (Green + SWIR) using bands SR_B1 (Green) and SR_B3 (SWIR).
    Adds the NDWI as a new band named 'NDWI'.
    """
    ndwi = image.normalizedDifference(['B1', 'B3'])
    ndwi = ndwi.unmask(-2) 
    return image.addBands(ndwi.rename('NDWI'))

# -----------------------------
# QA Cloud and Fill Masking for Landsat Imagery 
# -----------------------------

# For Mukti-Spectral Scanner Images (i.e., Landsat 1 to 4)
def mask_clouds_LS_MSS(image): 
    qa = image.select('QA_PIXEL')
    fill = 1 << 0        
    cloud = 1 << 3       
    cloud_conf = qa.rightShift(8).bitwiseAnd(3)   # Bits 8 and 9
    mask = (
        qa.bitwiseAnd(fill).eq(0)                 
        .And(qa.bitwiseAnd(cloud).eq(0))         
        .And(cloud_conf.neq(3))                   
    )
    return image.updateMask(mask)

# For all other images (i.e., Landsat 4-TM to 9)
def mask_clouds_LS(image):
    qa = image.select('QA_PIXEL')
    cloud = 1 << 3              # Bit 3
    cloud_conf = qa.rightShift(8).bitwiseAnd(3)        # Bits 8-9
    mask = (
        qa.bitwiseAnd(cloud).eq(0)
        .And(cloud_conf.lt(3))    
    )
    return image.updateMask(mask)

# -----------------------------
# Scaling Factor for Landsat Images
# -----------------------------

def apply_scale_factors(image):
    """
    Applies radiometric scale factors to Landsat Collection 2 Level-2 imagery (Landsat 5, 7, 8, and 9).

    Scales:
    - Optical surface reflectance bands (SR_B*) using: reflectance = DN * 0.0000275 - 0.2
    - Thermal band (ST_B*) using: temperature = DN * 0.00341802 + 149.0

    Only bands matching the patterns are scaled. The scaled bands replace the originals.
    """
    # Scale optical bands (e.g., SR_B1 through SR_B7)
    optical_bands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)

    # Scale thermal bands (e.g., ST_B6, ST_B10 depending on the sensor)
    thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)

    # Replace the original bands with the scaled versions
    return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)



# -----------------------------
# Sentinel-2
# -----------------------------
def mask_clouds_S2(image):
    """
    Cloud masking for Sentinel-2 using QA60 bitmask.
    """
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = (qa.bitwiseAnd(cloud_bit_mask).eq(0)
              .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0)))
    return image.updateMask(mask).divide(10000)


def calculate_ndwi_S2(image):
    """
    Calculates NDWI for Sentinel-2.
    NDWI = (Green - NIR) / (Green + NIR)
    """
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndwi = ndwi.unmask(-2)
    return image.addBands(ndwi)


# -----------------------------
# NDWI filtering
# -----------------------------
def apply_clahe_rescaling(ndwi, sensor_id):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) normalization and rescaling to NDWI data.

    Parameters:
        ndwi (np.ndarray): NDWI image
        sensor_id (str): 'LS' or 'S2' to adjust rescaling threshold

    Returns:
        np.ndarray: Preprocessed NDWI
    """
    ndwi_norm = ((ndwi + 1) * 127.5).astype(np.uint8)
    ndwi_norm[ndwi_norm<=0]=0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))   # operates on 8x8 pixel tiles
    clahe_image = clahe.apply(ndwi_norm)
    
    if sensor_id == 'LS':
        clahe_image[clahe_image > 100] = np.max(clahe_image)
    elif sensor_id == 'S2':
        clahe_image[clahe_image > 125] = np.max(clahe_image)

    ndwi_rescaled = ((clahe_image - np.nanmin(clahe_image)) /
                     (np.nanmax(clahe_image) - np.nanmin(clahe_image))) * 2 - 1
    
    return ndwi_rescaled.astype(np.float32)


def filter_ndwi_by_mask(ndwi_array, sensor_id,baselayers_dir):
    """
    Applies mask (masking out cloud pixels) and CLAHE rescaling to NDWI data.

    Parameters:
        ndwi_array (np.ndarray): NDWI image (raw)
        sensor_id (str): 'LS' or 'S2'

    Returns:
        tuple: (processed_ndwi, cloud_percentage)
    """
    try:
        # Load masks
        mask_path = os.path.join(baselayers_dir, "reservoir_mask.tif")
        res_mask = gdal_array.LoadFile(mask_path).astype(np.float32)
        res_expanded = expand(res_mask, 3)
        
        freq_path = os.path.join(baselayers_dir, "frequency.tif")
        freq_array = gdal_array.LoadFile(freq_path).astype(np.float32)
        
        # Step 1: Initial NDWI masking
        ndwi_cloud = np.copy(ndwi_array).astype(np.float32)
        ndwi_cloud[res_expanded == 0] = -1
        ndwi_cloud[np.isnan(ndwi_cloud)] = -1
        if not np.sum(ndwi_cloud > 0) > 0:
            ndwi_cloud[:, :] = -2    # Fully cloud-covered or invalid
        
        # Step 2: Zone-based cloud filling
        # Identify cloudy pixels (-2) in trusted zones (freq >= 80)
        ndwi_1 = ((freq_array >= 80) & (ndwi_cloud == -2)).astype(np.uint8)
        # Identify valid water pixels (NDWI > 0.1) in trusted zones (freq >= 80)
        ndwi_2 = ((freq_array >= 80) & (ndwi_cloud > 0.1)).astype(np.uint8)
        # Extract NDWI values from valid water pixels; set others to NaN
        ndwi_3 = np.where(ndwi_2 == 1, ndwi_cloud, np.nan)
        
        if not np.isnan(ndwi_3).all():
            fill_val = round(np.nanmean(ndwi_3), 3)
            ndwi_cloud[ndwi_1 == 1] = fill_val
            
        # Step 3: Calculate cloud percentage
        cloud_pct = round(
            len(np.where(ndwi_cloud == -2)[0]) / np.nansum(res_expanded == 1) * 100, 2
        )

        
        # Ensures all non-valid pixels (NaN, cloud, outside ROI) are set to -1
        ndwi_cloud[ndwi_cloud <= -1] = -1
        
        # Step 4: Apply CLAHE enhancement
        if cloud_pct < 100:
            processed_ndwi = apply_clahe_rescaling(ndwi_cloud, sensor_id)
        else:
            processed_ndwi = ndwi_array  # fallback if full cloud

        return processed_ndwi, cloud_pct

    except Exception as e:
        print(f"[ERROR] CLAHE masking failed: {e}")
        return None, None, None
    

    


def zone_based_filtering(ndwi_array, zone_map, method="predefined_zones", num_clusters=2, zone_threshold=0.1):
    """
    Performs water area filtering using either:
    - 'blobs': NDWI-connected components filtered by frequency clustering
    - 'predefined_zones': Zone-based KMeans filtering + water cluster fusion

    Parameters:
        ndwi_array (np.ndarray): NDWI image (preprocessed)
        zone_map (np.ndarray): Frequency image (0–100) or zone index image (1–50)
        method (str): 'blobs' or 'predefined_zones'
        num_clusters (int): KMeans cluster count (used in predefined_zones)
        zone_threshold (float): Minimum ratio to consider zone "active" (used for quality check)

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Refined water mask (binary)
            - Initial water cluster mask (binary)
    """
    # Zones are essentially spatial partitions or regions within a reservoir's area that help in refining water detection. 
    # Zone 50 → areas that are always covered with water (core of the reservoir).
    # Zone 1 → areas rarely covered with water (fringe, floodplains).
    # A fixed rule is used: if zone ratio ≥10%, keep the zone.


    if method == "blobs":
        # --- Method 1: Blob filtering (NDWI > 0, then group by frequency)
        water_mask = (ndwi_array > 0).astype(np.uint8)
        labeled_zones, num_zones = label(water_mask)
        zone_ratios = []
        valid_mask = np.zeros_like(water_mask)

        for zone_id in range(1, num_zones + 1):
            zone = labeled_zones == zone_id
            ratio = np.mean(zone_map[zone])
            zone_ratios.append(ratio)

        if len(zone_ratios) >= 2:
            kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
            zone_labels = kmeans.fit_predict(np.array(zone_ratios).reshape(-1, 1))
            group_means = [np.mean([r for r, l in zip(zone_ratios, zone_labels) if l == g]) for g in range(num_clusters)]
            water_group = int(np.argmax(group_means))

            for zone_id in range(1, num_zones + 1):
                if zone_labels[zone_id - 1] == water_group:
                    valid_mask[labeled_zones == zone_id] = 1

        return valid_mask.astype(np.uint8)

    elif method == "predefined_zones":
                
        num_zones = 50
        zone_map = np.ceil(zone_map / 2).astype(np.float32)
        zone_map[np.isnan(zone_map)] = 0

        # Step 1: Cluster NDWI to extract strongest water blob
        ndwi_flat = ndwi_array.ravel()
        # Mask NaNs for KMeans fitting
        valid_mask = ~np.isnan(ndwi_flat)
        valid_ndwi = ndwi_flat[valid_mask]
        km = KMeans(n_clusters=3, random_state=0)
        km.fit(valid_ndwi.reshape(-1, 1))
        cluster_centers = km.cluster_centers_
        cluster_labels = km.predict(ndwi_array.reshape(-1, 1))
        cluster_image = cluster_labels.reshape(ndwi_array.shape)
        water_cluster_label = np.argmax(np.sum(cluster_centers, axis=1))
        water_cluster = (cluster_image == water_cluster_label).astype(np.float32)
        
        # Step 2: Compute water pixel ratios for each zone
        pixels_per_zone = np.array([np.count_nonzero(zone_map == i + 1) for i in range(num_zones)])
        water_pixels_per_zone = np.array([
            np.count_nonzero((zone_map == i + 1) & (water_cluster == 1)) for i in range(num_zones)
        ])
        water_ratio_per_zone = water_pixels_per_zone / (pixels_per_zone + 1e-20)

        # Step 3: Quality assessment
        num_zones_meeting_threshold = np.sum(water_ratio_per_zone >= zone_threshold)
        zone_quality_flag = int(num_zones_meeting_threshold >= 20)
        
        # Step 4: KMeans clustering of zone water ratios
        normalized_ratios = water_ratio_per_zone * 100 / (np.max(water_ratio_per_zone) + 1e-20)
        zone_indices = np.arange(1, num_zones + 1)
        zone_features = np.vstack((zone_indices, normalized_ratios)).T
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(zone_features)
        zone_labels = kmeans.labels_

        # Identify the dominant water group
        min_zone_label_0 = np.min(zone_indices[zone_labels == 0], initial=num_zones)
        min_zone_label_1 = np.min(zone_indices[zone_labels == 1], initial=num_zones)
        min_accepted_zone_id = max(min_zone_label_0, min_zone_label_1)
        dominant_group_id = 0 if min_zone_label_0 == min_accepted_zone_id else 1

        # Step 5: Enhance classification by adding selected zones
        selected_zone_mask = np.copy(zone_map)
        selected_zone_mask[selected_zone_mask < min_accepted_zone_id] = 0
        refined_mask = water_cluster + selected_zone_mask
        refined_mask[refined_mask > 1] = 1
        
        # Mask out pixels outside valid zones
        water_cluster[zone_map == 0] = 0
        
        
        
        
        return refined_mask.astype(np.uint8), water_cluster.astype(np.uint8),zone_quality_flag

    
    else:
        raise ValueError(f"Invalid method: {method}. Choose 'blobs' or 'predefined_zones'.")




def apply_local_filtering(mask, frequency_array, window_size=10, method='window',stride=None, min_ratio=0.3):
    """
    Applies local spatial filtering to refine a binary water mask using a frequency map.

    This function supports two methods:
    
    - 'pixel': For each pixel, considers a surrounding neighborhood (defined by `win_size`).
               If the mean frequency of water pixels in the neighborhood exceeds a threshold
               (`min_ratio`), the pixel is retained as water. This helps remove isolated
               or spurious detections in low-confidence areas.

    - 'window': Applies a sliding window across the mask. Within each window, it computes
                the mean frequency of pixels already labeled as water. All pixels in the window
                with a frequency above that mean are reclassified as water. This method can
                densify detections in high-confidence zones by filling in missed pixels.

    Parameters:
        mask (np.ndarray): Binary water mask (values 0 and 1).
        frequency_array (np.ndarray): Frequency map showing water occurrence (0–100 scale).
        win_size (int): Window size for filtering. Interpreted as radius in 'pixel' mode,
                        and full window size in 'window' mode.
        method (str): Filtering method to use. Must be either 'pixel' or 'window'.
        stride (int, optional): Step size for the sliding window in 'window' mode.
                                Defaults to half the window size.
        min_ratio (float, optional): Minimum required average frequency (0–1) in the local 
                                     neighborhood for retaining a water pixel in 'pixel' mode.

    Returns:
        np.ndarray: Filtered binary water mask with improved spatial consistency.
    """
    if method=='pixel':
        padded_mask = np.pad(mask, window_size, mode='constant')
        padded_freq = np.pad(frequency_array, window_size, mode='constant')
        filtered_mask = np.zeros_like(mask)
    
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                window_mask = padded_mask[i:i + 2 * window_size + 1, j:j + 2 * window_size + 1]
                window_freq = padded_freq[i:i + 2 * window_size + 1, j:j + 2 * window_size + 1]
                if np.sum(window_mask) > 0:
                    ratio = np.mean(window_freq[window_mask == 1])
                    if ratio >= min_ratio * 100:
                        filtered_mask[i, j] = 1
    
        return filtered_mask  
    
    elif method=='window':
        if stride is None:
            stride = window_size // 2
    
        refined_mask = np.copy(mask).astype(np.uint8)
        frequency_array = np.ceil(frequency_array/2)
        zone_mask = np.copy(frequency_array).astype(np.uint8)
        zone_mask[np.isnan(zone_mask) ==1] = 0
    
        for y in range(0, zone_mask.shape[0] - window_size + 1, stride):
            for x in range(0, zone_mask.shape[1] - window_size + 1, stride):
                window_freq = zone_mask[y:y+window_size, x:x+window_size]
                window_mask = refined_mask[y:y+window_size, x:x+window_size]
    
                if np.count_nonzero(window_mask) > 0:
                    mean_freq_water = np.mean(window_freq[window_mask == 1])
                    new_window = np.copy(window_mask)
                    new_window[window_freq > mean_freq_water] = 1
                    refined_mask[y:y+window_size, x:x+window_size] = new_window
                    
        # Mask out pixels outside valid zones
        refined_mask[zone_mask == 0] = 0
        return refined_mask



