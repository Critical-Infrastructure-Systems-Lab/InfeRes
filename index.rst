.. reservoir monitoring documentation master file, created by
   sphinx-quickstart on Fri Aug 11 02:51:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to documentation of InfeRes: A Python package for inferring reservoir water extent, level and storage volume
=====================================================================================================================

This page provides the documentation and guidelines for the **InfeRes** python package. This package has two main modules, 
``data_download.py`` and ``data_processing.py``.

First module is the data downloading using Google Earth Engine [**Normalized Difference Water Index (NDWI) and Quality Assessment (QA_PIXEL) bands**] from *Landsat-5,7 and 8* satellite.
You can also use this module to download other datasets and collections (https://developers.google.com/earth-engine/datasets/catalog) avaliable on the earth engine platform. 
The second module is the processing of downloaded data (NDWI and BandQuality in this case) to get the reservoir characteristics such as **water surface area** and **volume**.
The modules/functions used in the package are described in detail in the following sections.  

.. note:: 
   Digital Elevation Model (DEM) used in the second module needs to be downloaded separately!

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: docs

   docs/DataDownload
   docs/DataProcessing


