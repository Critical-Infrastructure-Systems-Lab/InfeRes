InfeRes: A Python Package for Inferring Reservoir Water Surface Area, Level, and Storage Volume
==============================================================================================

.. image:: https://img.shields.io/pypi/l/sciris.svg
 :target: https://github.com/ssmahto/InfeRes_test/blob/main/LICENSE

``InfeRes`` is a python package that is designed to help automatic extraction of reservoir characteristics (water surface area, level, and storage volume) time-series by taking leverage
of the Google Earth Engine data collection (`Landsat series <https://developers.google.com/earth-engine/datasets/catalog/landsat/>`_, `Sentinel-2 <https://developers.google.com/earth-engine/datasets/catalog/sentinel-2/>`_), and high resolution `DEM (30m) <https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1/>`_.
It built on top of `GDAL <https://gdal.org/>`_, `Scikit-Learn <https://scikit-learn.org/>`_, `NumPy <https://numpy.org/>`_ and `Matplotlib <https://matplotlib.org/>`_,
and other popular python packages. ``InfeRes`` is developed with a novel algorithm which helps inferring reservoir characteristics even from the partially cloudy images.
``InfeRes`` can be applied to monitor water surface area in any reservoir or waterbody; whereas, storage-volume can be obtained for the large reservoirs (storage >= 0.1m:sup:`3`) listed in the `GRanD <https://www.globaldamwatch.org/directory/>`_ database.

The InfeRes workflow is fully modularized, with scripts organized by function:

1- main.py: Central orchestration script for running the full InfeRes pipeline over multiple reservoirs.

2- utils.py: Generic helper functions for raster operations, coordinate transformations, and basic math.

3- metadata_builder.py: Generates geospatial metadata from the GRanD database for input preparation.

4- download_baselayers.py: Downloads foundational datasets (e.g., DEM, GSW frequency, and maximum extent layers) from Google Earth Engine.

5- reservoir_delineation.py: Identifies reservoir extents based on DEMs and frequency maps.

6- reservoir_curve.py: Builds hypsometric curves from DEM data and reference curves (e.g., GRDL dataset).

7- satellite_composite.py: Constructs Landsat and Sentinel-based NDWI composites over specified periods and regions.

8- ndwi_processing.py: Preprocesses Landsat and Sentinel imagery, including NDWI calculation and cloud masking.

9- satellite_water_area.py: Extracts water surface area from satellite NDWI using filtering and clustering methods.

10- area_to_storage.py: Converts surface water area estimates into elevation and storage (volume) using reservoir hypsometric curve.


Dependencies
----------------

 - Python version-3.8 and above (we used Anaconda3, which is an open-source distribution of the Python)
 - Python standard library (os, numpy, pandas, matplotlib, csv)
 - Python advanced library (ee, osgeo, rasterio, sklearn.cluster, scipy.ndimage, skimage.morphology)


References 
---------------------

- Vu, T.D., Dang, T.D., Galelli, S., Hossain, F. (2022) `Satellite observations reveal 13 years of reservoir filling strategies, operating rules, and hydrological alterations in the Upper Mekong River basin. <https://hess.copernicus.org/articles/26/2345/2022/>` Hydrol. Earth Syst. Sci., 26, 2345–2364.

- Mahto, S., Fatichi, S., Galelli, S. (2025) `A 1985–2023 time series dataset of absolute reservoir storage in Mainland Southeast Asia (MSEA-Res). <https://doi.org/10.5194/essd-17-2693-2025>` Earth Syst. Sci. Data, 17, 2693–2712.


Acknowledgement 
---------------------

We have acquired the reference GRAND_Curves (reservoir's reconstructed bathymetry) form `Hao et al., (2024) <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023WR035781>`_, which is available for the list of reservoirs in the Global Reservoir and Dam (GRanD) Database (`Lehner et al., 2011 <https://esajournals.onlinelibrary.wiley.com/doi/10.1890/100125>`_).  

 - Hao, Z., Chen, F., Jia, X., Cai, X., Yang, C., Du, Y., & Ling, F. (2024). GRDL: A New Global Reservoir Area‐Storage‐Depth Data Set Derived Through Deep Learning‐Based Bathymetry Reconstruction. Water Resources Research, 60(1), e2023WR035781.

 - Lehner, B., C. Reidy Liermann, C. Revenga, C. Vörösmarty, B. Fekete, P. Crouzet, P. Döll, M. Endejan, K. Frenken, J. Magome, C. Nilsson, J.C. Robertson, R. Rodel, N. Sindorf, and D. Wisser. 2011. High-resolution mapping of the world’s reservoirs and dams for sustainable river-flow management. Frontiers in Ecology and the Environment 9 (9): 494-502.





















