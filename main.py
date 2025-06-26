#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 16:48:31 2025

@author: Hisham Eldardiry
"""


# main.py

"""
Main driver script for InfeRes: Inference of Reservoir Storage using remote sensing.
Orchestrates the full workflow:
- Download static base layers (DEM, GSW)
- Isolate reservoir from DEM
- Generate hypsometric (EAS) curve
- Extract NDWI composites
- Estimate water surface area
- Convert to storage/elevation

Dependencies: ee, pandas, modules in src/
"""

# %% Load Packages
import sys
import os
# Add the src directory to the system path
# project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.getcwd()
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)


import ee
import configparser
import pandas as pd
import geemap
import calendar

from utils import (boundary_to_geometry,setup_reservoir_folders,load_reservoir_metadata,parse_reservoir_ids)
from metadata_builder import generate_reservoir_metadata
from download_baselayers import (download_dem,download_gsw_frequency,download_gsw_extent)
from reservoir_delineation import delineate_reservoir
from reservoir_curve import generate_curve_pre_srtm, generate_curve_post_srtm
from satellite_composite import get_landsat_composite, get_sentinel_composite
from satellite_water_area import estimate_water_area
from area_to_storage import convert_area_to_storage


# %% Load config.ini
config = configparser.ConfigParser()
config.read("config.ini")

grand_ids_raw = config["Reservoir Selection"]["RESERVOIR_GRAND_ID"]
grand_ids = parse_reservoir_ids(grand_ids_raw)
    
ref_curves = config["Data Sources"]["REFERENCE_CURVES"]
ee_project = config["Data Sources"]["EE_PROJECT"]
start_year = int(config["Simulation Period"]["START_YEAR"])
end_year = int(config["Simulation Period"]["END_YEAR"])

# Initialize Earth Engine
# ee.Authenticate()
ee.Initialize(project=ee_project)




# %% Process InfeRes for Active Reservoirs
count_res=0
for gid in grand_ids:
    count_res=count_res+1
    print("=========================================================================================================")
    print(f"[RES {count_res}] Processing Reservoir (GRAND ID: {gid}) for the period  {start_year}-{end_year}")
    print("=========================================================================================================")
    metadata_path = os.path.join("input/reservoir_metadata")

    # Step 1: Generate metadata CSV
    print(f"[Step 1] Generating Reservoir Metadata (InfeRes Input) for GRanD ID {gid}...")
    generate_reservoir_metadata(
        grand_id=gid,
        grand_shapefile_dir=project_root+"/input/grand_dataset/",
        metadata_dir=metadata_path,
        save_plot=True
    )
    
    # Load reservoir metadata
    reservoirs_df = load_reservoir_metadata(metadata_path+f'/inferes_input_{gid:04d}.csv')
    res = reservoirs_df.iloc[0]  # direct access
    res_name = res.Name
    
    res_year = res.Year
    grand_id = res.GRAND_ID
    grand_capacity = res.GRAND_Capacity
    max_water_level = round(res.MAX_WL)
    point = [float(x) for x in res.Point.split(",")]
    boundary = [float(x) for x in res.Boundary.split(",")]
    region = boundary_to_geometry(boundary)
    
    
    
    # Step 2: Download DEM, Frequency, MaxExtent
    print(f"[Step 2] Downloading base layers for reservoir: {res_name} ...")
    # Create folder structure
    baselayers_dir, output_res_dir = setup_reservoir_folders(res_name)
    dem_array, dem_image = download_dem(region, os.path.join(baselayers_dir, "DEM.tif"))
    frequency =download_gsw_frequency(region, os.path.join(baselayers_dir, "frequency.tif"),dem_image)
    extent=download_gsw_extent(region, os.path.join(baselayers_dir, "max_extent.tif"),dem_image)
    
    
    # Step 3: Reservoir isolation
    print("[Step 3] Delineating reservoir boundaries...")
    delineate_reservoir(res_name, max_water_level, point, boundary, baselayers_dir,plot=False)

    # Step 4: Generate Reservoir Hypsometric Curve (Elevation–Area–Storage Relationship)
    print("[Step 4] Generating Reservoir Hypsometric Curve (Elevation–Area–Storage Relationship) ...")
    
    reference_curves_dir = os.path.join(project_root, ref_curves)
    if res_year <= 2000:
        generate_curve_pre_srtm(
            res_name,
            point,
            boundary,
            max_water_level,
            baselayers_dir,
            output_res_dir,
            reference_curves_dir,
            grand_id,
            grand_capacity,
            plot_curve=False
        )
    else:
        generate_curve_post_srtm(
            res_name,
            max_water_level,
            baselayers_dir,
            output_res_dir,
            plot_curve=False
        )

    # Step 5: Estimate surface area from NDWI composites & append results to output file
    print("[Step 5] Estimating Water Surface Area...")
    # Define output CSV path early so we can append during processing
    output_csv = os.path.join(output_res_dir, "inferes_area_storage.csv")
    if os.path.exists(output_csv):
        os.remove(output_csv)
    # Initialize CSV file with header if it doesn't exist
    with open(output_csv, "w") as f:
        f.write("sensor,date,cloud_percentage,quality_flag,raw_area,pre_filtering_area_km2,post_filtering_area_km2,water_level_m,post_filtering_storage_mcm\n")

    curve_path = os.path.join(output_res_dir, "reservoir_hypsometry.csv")
    count_img=0
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            days_in_month = calendar.monthrange(year, month)[1]
            for start_day in range(1, days_in_month, 11):
                count_img=count_img+1
                end_day = min(start_day + 10, days_in_month)
                start_date = f"{year}-{month:02d}-{start_day:02d}"
                end_date = f"{year}-{month:02d}-{end_day:02d}"
                print(count_img,f'----- Processing Satellite Images between {start_date} and {end_date}')
                landsat_comp = get_landsat_composite(start_date,end_date, region,reference_image=dem_image)
                sentinel_comp = get_sentinel_composite(start_date,end_date, region,reference_image=dem_image)
    
                for sensor_label, comp in zip(["landsat", "sentinel"], [landsat_comp, sentinel_comp]):
                    if comp:
                        sensor_id = "LS" if sensor_label == "landsat" else "S2"
                        cloud_pct,raw_area,clahe_area,water_cluster_area,zone_filtered_area, local_filtered_area,quality_flag = estimate_water_area(comp, region, baselayers_dir, res_name, sensor_id)
                        
                        # Set pre_filtering_area 
                        pre_filtering_area=water_cluster_area
                        post_filtering_area = local_filtered_area 
                        
                        if cloud_pct is None or pre_filtering_area is None or post_filtering_area is None:
                            print("Skipping due to invalid image or too much cloud.")
                        else:
                            # Create temporary DataFrame for the current row
                            
                            row_data = {
                                "sensor": sensor_label,
                                "date": f"{year}-{month:02d}-{end_day:02d}",
                                "cloud_percentage": round(cloud_pct, 2) if cloud_pct is not None else None,
                                "quality_flag": quality_flag,
                                "raw_area":raw_area,
                                "pre_filtering_area_km2": pre_filtering_area,
                                "post_filtering_area_km2": post_filtering_area,
                            }

                            #  Convert area to storage using post-filtering area
                            area_df = pd.DataFrame([row_data])   # create dataframe from results
                            elev_storage = convert_area_to_storage(area_df["post_filtering_area_km2"], curve_path)
                            elev_storage.rename(columns={'storage_mcm': 'post_filtering_storage_mcm'}, inplace=True)
                            elev_storage = elev_storage.round(2)  # apply rounding here
                            area_df[["water_level_m", "post_filtering_storage_mcm"]] = elev_storage
                            # Append to CSV
                            area_df.to_csv(output_csv, mode="a", index=False, header=False)
                    
    
    print(f"InfeRes results saved: {os.path.basename(output_csv)}")
    print(f"InfeRes completed for: {res_name}")

print("\n ✅ InfeRes completed for all reservoirs.")
    
  
