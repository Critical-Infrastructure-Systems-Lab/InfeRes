#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 16:34:15 2025

@author: Hisham Eldardiry
"""

# area_to_storage.py

"""
Converts surface water area estimates into elevation and storage (volume)
using an elevation–area–storage (EAS) curve derived from DEM or reference data.

Dependencies: numpy, pandas, scipy.interpolate
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess


def load_curve(csv_path):
    """
    Loads an elevation–area–storage curve from CSV.

    Parameters:
        csv_path (str): Path to the curve CSV file

    Returns:
        tuple: (area_array, elevation_array, storage_array)
    """
    df = pd.read_csv(csv_path)
    elevation = df.iloc[:, 0].values  # Elevation (m)
    area = df.iloc[:, 1].values       # Area (sq. km)
    storage = df.iloc[:, 2].values    # Storage (mcm)
    return area, elevation, storage


def build_interpolators(area, elevation, storage):
    """
    Builds interpolation functions for area → elevation and area → storage.

    Parameters:
        area (array): Surface area values (sq. km)
        elevation (array): Elevation values (m)
        storage (array): Storage values (mcm)

    Returns:
        tuple: (elevation_interp, storage_interp)
    """
    elevation_interp = interp1d(area, elevation, kind='linear', fill_value='extrapolate')
    storage_interp = interp1d(area, storage, kind='linear', fill_value='extrapolate')
    return elevation_interp, storage_interp


def convert_area_to_storage(area_series, curve_csv_path):
    """
    Converts a series of surface area values into estimated elevation and storage.

    Parameters:
        area_series (pd.Series): Water surface area values (sq. km)
        curve_csv_path (str): Path to EAS CSV file

    Returns:
        pd.DataFrame: DataFrame with columns: ['water_level_m', 'storage_mcm']
    """
    area, elevation, storage = load_curve(curve_csv_path)
    elevation_interp, storage_interp = build_interpolators(area, elevation, storage)
    
    
    levels = elevation_interp(area_series)
    volumes = storage_interp(area_series)

    return pd.DataFrame({
        'water_level_m': levels,
        'storage_mcm': volumes
    })

