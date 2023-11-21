############+++++++++++ PLEASE MAKE SURE OF THE FOLLOWING POINTS BEFORE RUNNING THE CODE +++++++++++############
# [1]. Data are already downloaded (Satellite images and DEM)
# [2]. Use the same coordinates that you have used in "data_download.py"
# [3]. All the python(scripts) files are inside ".../ReservoirExtraction/codes"
############++++++++++++++++++++++++++++++++++++++++ END ++++++++++++++++++++++++++++++++++++++++#############

# IMPORTING LIBRARY

import os 
os.chdir("G:/My Drive/NUSproject/ReservoirExtraction/")
from codes.CURVE import curve
from codes.CURVE import res_isolation
from codes.MASK import mask
from codes.WSA import wsa
from codes.PREPROCESING import preprocessing
from codes.CURVE_Tile import one_tile

if __name__ == "__main__":

    #====================================>> USER INPUT PARAMETERS 
    parent_directory = "G:/My Drive/NUSproject/ReservoirExtraction/Reservoirs/"
    os.chdir(parent_directory)
    res_name = "Xiaowan" 
    res_built_year = 2010
    dem_acquisition_year = 2000                             #SRTM DEM (30m) acquired in Feb 2000
    
    if res_built_year > dem_acquisition_year:
        res_directory = parent_directory + res_name
        # A point within the reservoir [longitude, latitude]
        point = [99.95, 24.745]
        # Upper-Left and Lower-right coordinates. Example coordinates [longitude, latitude]
        boundary = [99.20, 25.60, 100.25, 24.65] 
        max_wl = 1236                            
        os.makedirs(res_name, exist_ok=True)                  
        os.chdir(parent_directory + res_name)
        # Create a new folder within the working directory to download the data
        os.makedirs("Outputs", exist_ok=True)
        # Path to DEM (SouthEastAsia_DEM30m.tif), PCS: WGS1984
        # We have a bigger DEM file that is being used to clip the reservoir-DEM
        dem_file_path = "G:/My Drive/NUSproject/ReservoirExtraction/SEAsia_DEM/SouthEastAsia_DEM30m.tif"
        
        #====================================>> FUNCTION CALLING -1
        # [1]. Data pre-processing (reprojection and clipping)
        preprocessing(res_name, point, boundary, parent_directory, dem_file_path)
        
        #====================================>> FUNCTION CALLING -2
        # [2]. DEM-based reservoir isolation
        os.chdir(parent_directory + res_name + '/Outputs')
        res_isolation(res_name, max_wl, point, boundary, res_directory)
        
        #====================================>> FUNCTION CALLING -3
        # [3]. Creating mask/intermediate files
        os.chdir(parent_directory + res_name + '/Outputs')
        mask(res_name, max_wl, point, boundary, res_directory)
        
        #====================================>> FUNCTION CALLING -4
        # [4]. DEM-Landsat-based updated Area-Elevation-Storage curve
        os.chdir(parent_directory + res_name + '/Outputs')
        res_minElev = curve(res_name, res_directory)
         
        #====================================>> FUNCTION CALLING -5
        # [5]. Calculating the water surface area
        os.chdir(res_directory)
        wsa(res_name, res_directory)
        
        #====================================>> FUNCTION CALLING -6
        # [6]. Calculating the reservoir restorage (1 tiles)
        os.chdir(res_directory)
        one_tile(res_name, max_wl, res_minElev, res_directory)


# Finally  moving all .png files in a seperate folder for better organisation   
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
                   
        
    
    
    
    
    
    
    
    
    
    
    


