############+++++++++++ PLEASE MAKE SURE OF THE FOLLOWING POINTS BEFORE RUNNING THE CODE +++++++++++############
# [1]. Data are already downloaded (Satellite images and DEM)
# [2]. Use the same coordinates that you have used in "data_download.py"
# [3]. All the python(scripts) files are inside ".../ReservoirExtraction/codes"
# [4]. NOTE: Each set of input data (DEM and Landsat images) is treated as an individual reservoir
############++++++++++++++++++++++++++++++++++++++++ END ++++++++++++++++++++++++++++++++++++++++#############

# IMPORTING LIBRARY

import os 
os.chdir("G:/My Drive/NUSproject/ReservoirExtraction/")
from codes.PREPROCESING import preprocessing
from codes.CURVE import res_isolation
from codes.CURVE import one_tile
from codes.CURVE import curve_preDEM
from codes.CURVE import curve_postDEM
from codes.MASK import mask
from codes.WSA import wsa



if __name__ == "__main__":

    #====================================>> USER INPUT PARAMETERS 
    parent_directory = "G:/My Drive/NUSproject/ReservoirExtraction/Reservoirs/"
    os.chdir(parent_directory)
    res_name = "Chulabhorn" 
    res_built_year = 1972
    dem_acquisition_year = 2000                             #SRTM DEM (30m) acquired in Feb 2000
    
    # Other Reservaoir information
    res_directory = parent_directory + res_name
    # A point within the reservoir [longitude, latitude]
    point = [101.636, 16.539]
    # Upper-Left and Lower-right coordinates. Example coordinates [longitude, latitude]
    boundary = [101.590, 16.617, 101.660, 16.524] 
    max_wl = 768                            
    os.makedirs(res_name, exist_ok=True)                  
    os.chdir(parent_directory + res_name)
    # Create a new folder within the working directory to download the data
    os.makedirs("Outputs", exist_ok=True)
    # Path to DEM (SouthEastAsia_DEM30m.tif), PCS: WGS1984
    # We have a bigger DEM file that is being used to clip the reservoir-DEM
    dem_file_path = "G:/My Drive/NUSproject/ReservoirExtraction/SEAsia_DEM/SouthEastAsia_DEM30m.tif"
    
    #------------------->> FUNCTION CALLING -1
    # [1]. Data pre-processing (reprojection and clipping)
    preprocessing(res_name, boundary, parent_directory, dem_file_path)
    
    #------------------->> FUNCTION CALLING -2
    # [2]. DEM-based reservoir isolation
    # Px and Py are the coordinates (in number of pixels) the point on the reservoir from top-left grid 
    os.chdir(parent_directory + res_name + '/Outputs')
    [Px, Py] =res_isolation(res_name, max_wl, point, boundary, res_directory)
    point_loc = [Px, Py]
    
    #------------------->> FUNCTION CALLING -3
    # [3]. Creating mask/intermediate files
    os.chdir(parent_directory + res_name + '/Outputs')
    mask(res_name, max_wl, point, boundary, res_directory)
    
        
    # CASE1- Reservoir built before DEM acquisition ==============================================
    if res_built_year < dem_acquisition_year:      
        print('Name of the reservoir: ' + res_name)
        print('Reservoir has built before the acquisition of DEM')
            
        #------------------->> FUNCTION CALLING -4
        # [4]. DEM-Landsat-based updated Area-Elevation-Storage curve
        os.chdir(parent_directory + res_name + '/Outputs')
        res_minElev = curve_preDEM(res_name, point_loc, res_directory)
         
        #------------------->> FUNCTION CALLING -5
        # [5]. Calculating the water surface area
        os.chdir(res_directory)
        wsa(res_name, res_directory)
        
        #------------------->> FUNCTION CALLING -6
        # [6]. Calculating the reservoir restorage (1 tiles)
        os.chdir(res_directory)
        one_tile(res_name, max_wl, res_minElev, res_directory)

    # CASE2- Reservoir built after DEM acquisition ==============================================
    if res_built_year > dem_acquisition_year:        
        print('Name of the reservoir: ' + res_name)
        print('Reservoir has built after the acquisition of DEM')
        
        #------------------->> FUNCTION CALLING -4
        # [4]. DEM-Landsat-based updated Area-Elevation-Storage curve
        os.chdir(parent_directory + res_name + '/Outputs')
        res_minElev = curve_postDEM(res_name, res_directory)
         
        #------------------->> FUNCTION CALLING -5
        # [5]. Calculating the water surface area
        os.chdir(res_directory)
        wsa(res_name, res_directory)
        
        #------------------->> FUNCTION CALLING -6
        # [6]. Calculating the reservoir restorage (1 tiles)
        os.chdir(res_directory)
        one_tile(res_name, max_wl, res_minElev, res_directory)


# ================================================ Finally  moving all .png/jpg files in a seperate folder for better organisation   
import shutil
# Create a folder to store the pictures if it doesn't exist
pictures_folder = "intermediate_pictures"
os.makedirs(pictures_folder, exist_ok=True)
# List all files in the current directory
files = os.listdir()
# Move all PNG files to the 'pictures' directory
for file in files:
    if file.lower().endswith(".png"):
        file_path = os.path.join(os.getcwd(), file)
        shutil.move(file_path, os.path.join(pictures_folder, file))
                   
        
    
    
    
    
    
    
    
    
    
    
    


