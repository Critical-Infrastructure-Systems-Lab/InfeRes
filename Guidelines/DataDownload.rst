Data Download
==============

Introduction
------------

This user guide provides step-by-step instructions on how to use the Landsat data downloading script to download Landsat 8, Landsat 7, and Landsat 5 satellite images for a specific location. The script utilizes Google Earth Engine to access and download the data, enabling you to obtain Normalized Difference Water Index (NDWI) and Band Quality Assessment (BQA) images for a given geographic area.

Prerequisites and Installation
-------------------------------

Before using the Landsat data downloading script, make sure you have the following prerequisites:

#. Python installed on your system (we used Python version 3.8)
   
   First install ``anaconda3`` in your local machine or server and create an environment with
   python version 3.8. This will create a seperate python envoronment not overlapping
   with the default python that comes with ``anaconda3``. You can create an environment
   by running the following code::

      (base) C:/User/UserName/conda create -n environment_name python=3.8
   
   .. note::
      Unfortunatly, the GDAL package used in **InfeRes** did not work with python 3.9 and above.
      Whereas, it is GDAL is comfortable with python version 3.8.
   
   Activate the python environment:: 

      (base) C:/User/UserName/conda activate environment_name
     
#. Required Python packages are installed::

        (environment_name) C:/User/UserName/conda install -c conda-forge rasterio
        (environment_name) C:/User/UserName/conda install -c conda-forge gdal
        (environment_name) C:/User/UserName/conda install -c conda-forge spyder
        (environment_name) C:/User/UserName/conda install -c conda-forge earthengine-api

   Likewise, you install all the required packages.

   .. note:: You can use any IDE (we used Spyder IDE).

#. Google Earth Engine authentication::

        import ee
        ee.Authenticate()
        ee.Initialize()
    
   .. note:: Once the Earth Engine is authenticated, disable it. 

Usage Instructions
-------------------

Set up the Directory
~~~~~~~~~~~~~~~~~~~~~

Before running the script, specify the directory where you want to store the downloaded data. Modify the ``parent_directory`` variable in the script to the desired location. By default, the script will create a new folder named after the reservoir (example: Nuozhadu) in the parent directory. It will create a folder called “LandsatData” to download the data.

Define the Image Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the script, you can customize the image specifications based on your requirements. Each Landsat collection (Landsat 8, Landsat 7, and Landsat 5) has its own section in the script. For each collection, modify the following parameters:

- ``row`` and ``path``: Landsat row and path numbers for your region of interest.
- ``point_coordinates``: The coordinates of a point within the reservoir (longitude, latitude).
- ``boundary_coordinates``: The bounding box coordinates of the area you want to download (longitude1, latitude1, longitude2, latitude2).
- ``collection_id``: The Landsat collection ID (e.g., "LANDSAT/LC08/C02/T1_TOA" for Landsat 8).
- ``start_date`` and ``end_date``: The date range for image filtering.

Run the Script
~~~~~~~~~~~~~~~

Open a terminal or command prompt, navigate to the directory containing the script, and run the script using the following command::

    python data_download.py

Download Progress
~~~~~~~~~~~~~~~~~

The script will print the status of the download tasks. Once completed, the downloaded GeoTIFF files will be saved in the *LandsatData* folder within the specified parent directory.

Examples
--------

The script contains three sections, one for each Landsat collection (Landsat 8, Landsat 7, and Landsat 5). You can customize the image specifications for each section and run the script independently for each Landsat collection.

Customization
~~~~~~~~~~~~~

You can modify the script to download data for different locations and time periods by updating the image specifications as explained in the *Usage Instructions* section.

Output
------

The downloaded Landsat images will be saved as GeoTIFF files in the *LandsatData* folder within the specified parent directory inside a folder created with the name of the reservoir.

Conclusion
-----------

Congratulations! You have successfully downloaded Landsat 8, Landsat 7, and Landsat 5 images for your specified location. The downloaded images are available in the "LandsatData" folder. You can use these images for further analysis and processing.