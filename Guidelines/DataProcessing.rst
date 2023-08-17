Data Processing
++++++++++++++++

Main Script Structure
======================

The main script consists of four major sub-modules:

#. ``CURVE.py``
#. ``MASK.py``
#. ``WSA.py``
#. ``CURVE_Tile.py``

Detailed Description
---------------------

1. ``CURVE.py``
---------------------
This section provides step-by-step instructions on effectively using the ``CURVE.py`` script. The code calculates the Area-Elevation-Volume relationship using a Digital Elevation Model (DEM). The script is called from the main function and uses the GDAL library for image processing and analysis.

Usage
~~~~~

1. Open the "Reservoir_Extraction" project folder in your Python development environment.
2. Import the necessary libraries and helper functions from the script.
3. Set the input parameters in the script as follows:
   - ``res_name``: Name of the reservoir.
   - ``max_wl``: Maximum water level of the reservoir.
   - ``point_coordinates``: Coordinates of a point within the reservoir (longitude, latitude).
   - ``boundary_coordinates``: Upper-left and lower-right coordinates of the reservoir boundary (longitude, latitude).
   - ``dem_file_path``: Path to the DEM (Digital Elevation Model) file.
4. Execute the ``curve()`` function by passing the input parameters to it.
5. The script will process the Landsat data, create an Elevation-Area-Storage relationship, and save the results in a CSV file named "Curve.csv" inside the "Outputs" folder.

Functions
~~~~~~~~~

The Reservoir Extraction Project script includes the following function:
- ``curve(res_name, max_wl, point_coordinates, boundary_coordinates, dem_file_path)``: This function calculates the DEM-based Area-Elevation-Storage curve for the reservoir. It takes the reservoir name, maximum water level, point within the reservoir, boundary coordinates, and DEM file path as inputs.

2. ``MASK.py``
---------------------
This section provides step-by-step instructions on effectively using the ``MASK.py`` script. The script processes Landsat images and generates masks and zoning maps for a given reservoir area using the GDAL library.

Usage
~~~~~

1. Import the necessary libraries and helper functions from the script.
2. Set the input parameters in the script as follows:
   - ``res_name``: Name of the reservoir.
   - ``max_wl``: Maximum water level of the reservoir.
   - ``point_coordinates``: Coordinates of a point within the reservoir (longitude, latitude).
   - ``boundary_coordinates``: Upper-left and lower-right coordinates of the reservoir boundary (longitude, latitude).
   - ``dem_file_path``: Path to the DEM (Digital Elevation Model) file.
   - ``res_directory``: Directory path for storing outputs.
3. The script will process the Landsat data, create masks and zoning maps, and save the results in the "Outputs" folder.

Functions
~~~~~~~~~

The Reservoir Masking and Zoning script includes the following helper functions:
- ``pick(c, r, mask)``: Picks a region from an array based on the provided column and row indices.
- ``expand(array, n)``: Expands the area of the water mask by a specified number of additional pixels.
The main function is as follows:
- ``mask(res_name, max_wl, point_coordinates, boundary_coordinates, dem_file_path, res_directory)``: This function creates masks and zoning maps for the reservoir based on Landsat data. It takes various input parameters.

3. ``WSA.py``
---------------------
This section provides step-by-step instructions on effectively using the ``WSA.py`` script to estimate water surface area using Landsat NDWI images.

Prerequisites
~~~~~~~~~~~~~

- Landsat NDWI images (Normalized Difference Water Index) for processing.
- Output images from the ``MASK.py`` script (zone mask, DEM-based and Landsat-based expanded mask images).

Usage
~~~~~

1. Import the necessary libraries and helper functions from the script.
2. Set the input parameters in the script as follows:
   - ``res_name``: Name of the reservoir.
   - ``res_directory``: Directory path for storing outputs.
3. K-means Clustering and Threshold Calculation:
   - Applies K-means clustering to NDWI values, grouping them into clusters representing water, wet non-water, dry non-water, and no-data regions.
   - Determines an optimal threshold for classifying water pixels based on clustering results.
4. Image Classification Improvement:
   - Uses zone-based information to improve image classification.
   - Identifies additional water pixels based on the zone mask and the threshold derived from K-means clustering.

Functions
~~~~~~~~~

The main function is as follows:
- ``wsa(res_name, res_directory)``: This function computes water surface area before and after improvement using pixel counts and area conversion factors. It generates estimates for both original and improved image classifications.

4. ``CURVE_Tile.py``
---------------------
This section provides step-by-step instructions on using the ``CURVE_Tile.py`` script to calculate total surface area and storage for the complete reservoir, especially useful for reservoirs containing multiple Landsat tiles.

Usage
~~~~~

1. Open the "Reservoir_Extraction" project folder in your Python development environment.
2. Import the necessary libraries and helper function from the script.
3. Set the input parameters in the script as follows:
   - ``res_name``: Name of the reservoir.
   - ``max_wl``: Maximum water level of the reservoir.
   - ``dead_wl``: Minimum water level of the reservoir.
   - ``res_directory``: Directory path for storing outputs.
   - ``point_coordinates``: Coordinates of a point within the reservoir (longitude, latitude).
   - ``boundary_coordinates``: Upper-left and lower-right coordinates of the reservoir boundary (longitude, latitude).
   - ``dem_file_path``: Path to the DEM (Digital Elevation Model) file.
4. Execute the ``one_tile()`` and ``two_tile()`` functions by passing input parameters.
5. The script will process the DEM file, Curve.csv, and create an updated Elevation-Area-Storage relationship, saving the results in a CSV file named "Curve_complete_res.csv" inside the "Outputs" folder.

Functions
~~~~~~~~~

The Reservoir Extraction Project script includes the following function:
- ``pick(c, r, mask)``: Picks a region from an array based on provided column and row indices.
- ``one_tile(res_name, max_wl, dead_wl, res_directory)``: This function calculates the DEM-based Area-Elevation-Storage curve for the complete reservoir when using a single tile. It takes various input parameters.
- ``two_tile(res_name, max_wl, dead_wl, point_coordinates, complete_res_boundary, dem_file_path, res_directory)``: This function calculates the DEM-based Area-Elevation-Storage curve for the complete reservoir when using multiple tiles. It takes various input parameters.

License and Credits
--------------------

This script is released under the [insert license name here]. Credits and acknowledgments go to [insert any external libraries, data sources, or contributors].
