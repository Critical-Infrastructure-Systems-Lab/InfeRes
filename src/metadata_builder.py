#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 15:15:33 2025

@author: Hisham Eldardiry
"""


import os
import re
import csv
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt


def abbreviate_name(name):
    """Simplifies reservoir name by removing generic words and spaces."""
    name = re.sub(r'\b(?:Dam|Lake|Impoundment|Tailings)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\b[A-Z]\.\s*', '', name)
    name = re.sub(r'\s+', '', name)
    return name.strip()



def estimate_shape_factor(depth_m, dam_height, area_skm=None, cap_mcm=None):
    """
    Estimates the reservoir's cross-sectional shape factor (K), used to approximate the
    maximum water level and bottom elevation more realistically than average depth alone.

    This function combines two complementary geometric indicators:

    - Shape Ratio (dam_height / depth_m): Reflects the steepness of the dam basin and 
      represents local vertical geometry at the dam site.

    - Area-to-Volume Ratio (AVR = area_skm / cap_mcm): Reflects how broadly water spreads 
      for each unit of storage volume, capturing the global or whole-reservoir flatness.

    Both metrics are used together because they tell different parts of the story:

    - Shape Ratio reflects dam structure vs. water depth — a local-scale measure of steepness.
    - AVR reflects the water's horizontal spread per unit volume — a global-scale indicator 
      of whether the reservoir is flat and shallow or narrow and deep.

    The shape factor K is first estimated from the shape ratio, then fine-tuned using AVR:
    - A high AVR suggests a shallow, wide reservoir → K is reduced.
    - A low AVR suggests a deep, narrow basin → K is increased.

    Parameters:
        depth_m (float): Average reservoir depth in meters.
        dam_height (float): Height of the dam in meters.
        area_skm (float): Surface area of the reservoir in square kilometers.
        cap_mcm (float): Total capacity of the reservoir in million cubic meters.

    Returns:
        float: Estimated shape factor K.
    """

    if depth_m <= 0 or dam_height <= 0:
        return 2.0  # Fallback for invalid data

    shape_ratio = dam_height / depth_m
    area_cap_ratio = (area_skm / cap_mcm) if area_skm and cap_mcm and cap_mcm > 0 else 1 / depth_m

    # Base K from shape ratio
    if shape_ratio < 1.5:
        K = 1.5             # Very shallow reservoir
    elif shape_ratio < 3.0:
        K = 2.0             # Moderate slope (like Shasta)
    else:
        K = 3.0             # Steep or tall dams

    # Adjust K using AVR
    if area_cap_ratio > 0.09:    # large surface area for a small volume ⇒ flat, shallow reservoir
        K = max(1.0, K - 0.2)
    elif area_cap_ratio < 0.01:  # small surface area for large volume ⇒ deep, narrow basin
        K = min(3.5, K + 0.2)

    return round(K, 2)

def generate_reservoir_metadata(
    grand_id,
    grand_shapefile_dir,
    metadata_dir,
    utm_epsg=32612,
    buffer_km=5,
    shape_factor=False,
    free_board=5,
    save_plot=True
):
    """
    Generates geospatial metadata for a reservoir using GRanD shapefiles.
    
    Parameters:
        grand_id (int): The GRanD reservoir ID to process.
        grand_shapefile_dir (str): Directory path containing GRanD shapefile (.shp).
        metadata_dir (str): Output directory where metadata files will be saved.
        utm_epsg (int, optional): EPSG code for UTM projection; default is 32612 (Zone 12N).
        buffer_km (float, optional): Buffer distance (in kilometers) applied around the reservoir
                                     polygon to define the bounding box. This is used to set the spatial 
                                     extent for DEM downloads, image clipping, etc. Default is 5 km.
        save_plot (bool, optional): If True, saves a preview plot of the reservoir and bounding box.
    
    Returns:
        dict: A metadata dictionary containing:
              - centroid (lon, lat)
              - bounding box (minx, miny, maxx, maxy)
              - reservoir polygon (GeoJSON format)
              - projection details
              - buffer used
    """

    os.makedirs(metadata_dir, exist_ok=True)
    grand_dams = gpd.read_file(os.path.join(grand_shapefile_dir, "GRanD_dams_v1_3.shp"))
    reservoirs = gpd.read_file(os.path.join(grand_shapefile_dir, "GRanD_reservoirs_v1_3.shp"))

    try:
        row = grand_dams[grand_dams["GRAND_ID"] == grand_id].iloc[0]
        res_poly = reservoirs[reservoirs['GRAND_ID'] == grand_id]

        if res_poly.empty:
            print(f"❌ Reservoir polygon not found for GRAND_ID {grand_id}")
            return

        res_row = res_poly.iloc[0]
        name = row["DAM_NAME"]
        year = int(row["YEAR"])
        dam_height = row["DAM_HGT_M"]
        depth_m = row["DEPTH_M"]
        elev_masl = row["ELEV_MASL"]
        area_skm = row.get("AREA_SKM", None)
        cap_mcm = row.get("CAP_MCM", None)
        grand_capacity = cap_mcm
        
        if shape_factor:
            # Estimate cross-section shape factor
            K = estimate_shape_factor(depth_m, dam_height, area_skm, cap_mcm)
            
            # Estimate bottom elevation and max water level
            bottom_elevation = elev_masl - K * depth_m
            max_wl = bottom_elevation + dam_height-free_board
        else:
            K=1
            # Estimate bottom elevation and max water level
            bottom_elevation = elev_masl - K * depth_m
            max_wl = bottom_elevation + dam_height-free_board
            
        # === Buffer logic ===
        if pd.notna(res_row["AREA_SKM"]) and res_row["AREA_SKM"] > 0:
            buffer_km = min(max(0.2 * (res_row["AREA_SKM"] ** 0.5), 2), 15)
        elif pd.notna(res_row["CAP_MCM"]) and res_row["CAP_MCM"] > 0:
            buffer_km = min(max(0.005 * (res_row["CAP_MCM"] ** 0.5), 2), 20)

        buffer_m = buffer_km * 1000
        res_poly_m = res_poly.to_crs(epsg=utm_epsg)
        buffered_m = res_poly_m.buffer(buffer_m)
        buffered = buffered_m.to_crs(epsg=4326)

        bounds = buffered.total_bounds
        res_geom = res_poly.geometry.unary_union
        res_centroid = res_geom.centroid
        if not res_geom.contains(res_centroid):
            res_centroid = res_geom.representative_point()

        centroid = res_centroid
        point_str = f"{centroid.x:.6f}, {centroid.y:.6f}"
        boundary = f"{bounds[0]:.6f}, {bounds[3]:.6f}, {bounds[2]:.6f}, {bounds[1]:.6f}"
        short_name = abbreviate_name(name)

        # === Output CSV ===
        output_path = os.path.join(metadata_dir, f"inferes_input_{grand_id:04d}.csv")
        with open(output_path, mode="w", newline="") as csvfile:
            fieldnames = [
                "GRAND_ID", "Name", "Year", "GRAND_Capacity",
                "MAX_WL", "Bottom_Elev", "Shape_Factor",
                "Point", "Boundary"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                "GRAND_ID": int(grand_id),
                "Name": short_name,
                "Year": year,
                "GRAND_Capacity": round(grand_capacity, 2),
                "MAX_WL": round(max_wl, 2),
                "Bottom_Elev": round(bottom_elevation, 2),
                "Shape_Factor": K,
                "Point": point_str,
                "Boundary": boundary
            })

        print(f"Saved: {os.path.basename(output_path)}")

        # === Optional plotting ===
        fig, ax = plt.subplots(figsize=(8, 8))
        res_poly.plot(ax=ax, color='blue', alpha=1, label="Reservoir")
        buffered.plot(ax=ax, edgecolor='red', facecolor='none', label=f"Buffered Area ({buffer_km:.1f} km)")
        gpd.GeoSeries(box(*bounds)).plot(ax=ax, edgecolor='green', facecolor='none', linewidth=3, label="Bounding Box")
        gpd.GeoSeries(centroid).plot(ax=ax, color='orange', marker='*', markersize=100, label="Centroid")
        plt.legend()
        plt.grid(True)
        plt.title(f"{short_name} | GRAND ID: {grand_id}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        if save_plot:
            # Save figure as PNG
            plot_path = os.path.join(metadata_dir, f"res_{grand_id:04d}_boundary.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

    except Exception as e:
        print(f"❌ Error processing GRAND_ID {grand_id}: {e}")

