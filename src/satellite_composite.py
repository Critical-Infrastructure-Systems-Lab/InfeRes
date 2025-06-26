#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 16:45:19 2025

@author: dardiry
"""

# satellite_composites.py

"""
Construct Landsat and Sentinel NDWI-based composites for a given date range and region.

Dependencies: ee
"""

import ee
from ndwi_processing import (
    apply_scale_factors, calculate_ndwi_L8_L9, mask_qa_pixels_L8_L9,
    calculate_ndwi_L5_L7, calculate_ndwi_L4,mask_qa_pixels_L5_L7,mask_clouds_LS_MSS,mask_clouds_LS,
    calculate_ndwi_S2, mask_clouds_S2
)


def get_landsat_composite(start_date, end_date, region,reference_image=None):
    """
    Builds a Landsat composite using NDWI across L5, L8, and L9 imagery.

    Parameters:
        start_date (str): YYYY-MM-DD
        end_date (str): YYYY-MM-DD
        region (ee.Geometry): Region of interest

    Returns:
        ee.Image: NDWI composite (or None if no images found)
    """
    # Landsat 9-OLI2/TIRS2 (Operational Land Imager/Thermal Infrared Sensor)
    L9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
          .filterBounds(region)
          .filterDate(start_date, end_date)
          .map(apply_scale_factors)
          .map(mask_clouds_LS)
          .map(calculate_ndwi_L8_L9))

    # Landsat 8-OLI/TIRS (Operational Land Imager/Thermal Infrared Sensor)
    L8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .filterBounds(region)
          .filterDate(start_date, end_date)
          .map(apply_scale_factors)
          .map(mask_clouds_LS)
          .map(calculate_ndwi_L8_L9))

    # Landsat 7-ETM+ (Enhanced Thematic Mapper Plus)
    L7 = (ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
          .filterBounds(region)
          .filterDate(start_date, end_date)
          .map(apply_scale_factors)
          .map(mask_clouds_LS)
          .map(calculate_ndwi_L5_L7))
     
    # Landsat 5-TM (Thematic Mapper)
    L5 = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
          .filterBounds(region)
          .filterDate(start_date, end_date)
          .map(apply_scale_factors)
          .map(mask_clouds_LS)
          .map(calculate_ndwi_L5_L7))
    
    # Landsat 5-MSS (Multispectral Scanner System)
    L5_MSS = (ee.ImageCollection("LANDSAT/LM05/C02/T1")
          .filterBounds(region)
          .filterDate(start_date, end_date)
          .map(apply_scale_factors)
          .map(mask_clouds_LS_MSS)
          .map(calculate_ndwi_L5_L7))
    
    # Landsat 4-MSS (Multispectral Scanner System)
    L4_MSS = (ee.ImageCollection("LANDSAT/LM04/C02/T1")
          .filterBounds(region)
          .filterDate(start_date, end_date)
          .map(apply_scale_factors)
          .map(mask_clouds_LS_MSS)
          .map(calculate_ndwi_L4))    
    # -----------------------------------------------
    # Check availability of images for each Landsat mission
    # -----------------------------------------------
    L9_size = L9.size().getInfo()  # Number of images in Landsat 9 collection
    L8_size = L8.size().getInfo()  # Number of images in Landsat 8 collection
    L7_size = L7.size().getInfo()  # Number of images in Landsat 7 collection
    L5_size = L5.size().getInfo()  # Number of images in Landsat 5-TM collection
    L5_MSS_size = L5_MSS.size().getInfo()  # Number of images in Landsat 5-MSS collection
    L4_MSS_size = L4_MSS.size().getInfo()  # Number of images in Landsat 4-MSS collection
    
    # Initialize list to store composite NDWI images
    images = []
    
    # -----------------------------------------------
    # For each Landsat mission, if images are available:
    # - Create a quality mosaic using NDWI
    # - Append the NDWI band from the composite to the list
    # -----------------------------------------------
    # ImageCollection.qualityMosaic(band_name) selects, for each pixel, the image in the collection that has the highest value of the given band (NDWI in our case). 
    # For every pixel location:
    # 1. Look across all images in the collection
    # 2. Find the image where the NDWI value is highest
    # 3. Use that pixel's value in the final composite
    # -----------------------------------------------
    
    if L9_size > 0:
        # Composite Landsat 9 using pixels with max NDWI per location
        composite = L9.qualityMosaic('NDWI')
        images.append(composite.select('NDWI').clip(region))
    
    if L8_size > 0:
        # Composite Landsat 8 using pixels with max NDWI
        composite = L8.qualityMosaic('NDWI')
        images.append(composite.select('NDWI').clip(region))
    
    if L7_size > 0:
        # Composite Landsat 7 using pixels with max NDWI
        composite = L7.qualityMosaic('NDWI')
        images.append(composite.select('NDWI').clip(region))
    
    if L5_size > 0:
        # Composite Landsat 5 using pixels with max NDWI
        composite = L5.qualityMosaic('NDWI')
        images.append(composite.select('NDWI').clip(region))
        
    if L5_MSS_size > 0:
        # Composite Landsat 5 using pixels with max NDWI
        composite = L5_MSS.qualityMosaic('NDWI')
        images.append(composite.select('NDWI').clip(region))
        
    if L4_MSS_size > 0:
        # Composite Landsat 5 using pixels with max NDWI
        composite = L4_MSS.qualityMosaic('NDWI')
        images.append(composite.select('NDWI').clip(region))    
    
    # If no images are found, return a zero-filled NDWI image clipped to region
    if images:
        # Combine the results using ee.ImageCollection(...).reduce(ee.Reducer.max()) to produce a max NDWI composite.
        LS_collection = ee.ImageCollection(images)
        reduced = LS_collection.reduce(ee.Reducer.max())
        if reference_image:
            target_proj = reference_image.projection().atScale(30)
            reduced = reduced.reproject(target_proj)
        return reduced.clip(region)
    else:
        print(f"No Landsat images found between {start_date} and {end_date}.")
        dummy_image = ee.Image.constant(0).rename('NDWI').clip(region)
        if reference_image:
            target_proj = reference_image.projection().atScale(30)
            dummy_image = dummy_image.reproject(target_proj)
        return dummy_image
        

def get_sentinel_composite(start_date, end_date, region,reference_image=None):
    """
    Builds a Sentinel-2 composite using NDWI.

    Parameters:
        start_date (str): YYYY-MM-DD
        end_date (str): YYYY-MM-DD
        region (ee.Geometry): Region of interest

    Returns:
        ee.Image: NDWI composite (or None if no images found)
    """
    S2 = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
          .filterBounds(region)
          .filterDate(start_date, end_date)
          .map(mask_clouds_S2)
          .map(calculate_ndwi_S2))

    if S2.size().getInfo() > 0:
        # Uses qualityMosaic to reduce composite, which selects, for each pixel, the entire image in which the pixel has the maximum NDWI value.
        composite = S2.qualityMosaic('NDWI').select("NDWI")
        if reference_image:
            target_proj = reference_image.projection().atScale(30)
            composite = composite.reproject(target_proj)
        return composite
    else:
        print(f"No Sentinel-2 images found between {start_date} and {end_date}.")
        return None
    
    
   