############+++++++++++ PLEASE MAKE SURE OF THE FOLLOWING POINTS BEFORE RUNNING THE CODE +++++++++++############
# [1]. Data are already downloaded (Satellite images and DEM)
# [2]. DEM should be in the projected coordinate system (unit: meters)
# [3]. Use the same coordinates that you have used in "first_script_data_download.py"
# [4]. All the python(scripts) files are inside ".../ReservoirExtraction/codes"
# [5]. "Number_of_tiles" = Number of Landsat tiles to cover the entire reservoir. It is recommended to download the Landsat tile covering the maximum reservoir area 
############++++++++++++++++++++++++++++++++++++++++ END ++++++++++++++++++++++++++++++++++++++++#############

# IMPORTING LIBRARY

import os
parent_directory = "H:/My Drive/NUSproject/ReservoirExtraction/"
os.chdir(parent_directory)
from codes.CURVE import curve
from codes.MASK import mask
from codes.WSA import wsa
from codes.CURVE_Tile import one_tile
from codes.CURVE_Tile import two_tile


if __name__ == "__main__":

    #====================================>> USER INPUT PARAMETERS  
    # A point within the reservoir [longitude, latitude]
    point = [100.373, 22.676]
    # Upper-Left and Lower-right coordinates. Example coordinates [longitude, latitude]
    boundary = [100.30, 23, 100.40, 22.54]
    res_name = "Nuozhadu"                        # Name of the reservoir 
    max_wl = 812 
    dead_wl = 750
    Number_of_tiles = 2                             
    os.makedirs(res_name, exist_ok=True)                  
    os.chdir(parent_directory + res_name)
    # Create a new folder within the working directory to download the data
    os.makedirs("Outputs", exist_ok=True)
    dem_file_path = "H:/My Drive/NUSproject/ReservoirExtraction/SEAsia_DEM/SouthEastAsia_DEM30m_PCS.tif"
    
    #====================================>> FUNCTION CALLING (1)
    # [1]. DEM-based Area-Elevation-Storage curve
    curve(res_name, max_wl, point, boundary, dem_file_path)
    
    #====================================>> FUNCTION CALLING (2)
    # [2]. Creating mask/intermediate files
    res_directory = parent_directory + res_name
    os.chdir(res_directory)
    os.makedirs("LandsatData_Clip", exist_ok=True)
    mask(res_name, max_wl, point, boundary, dem_file_path, res_directory)
    
    #====================================>> FUNCTION CALLING (3)
    # [3]. Calculating the water surface area
    res_directory = parent_directory + res_name
    os.chdir(res_directory)
    wsa(res_name, res_directory)
    
    #====================================>> FUNCTION CALLING (4)
    # [4]. Calculating the reservoir restorage (1 tiles)
    if Number_of_tiles==1:
        res_directory = parent_directory + res_name
        os.chdir(res_directory)
        print("One tile reservoir")
        one_tile(res_name, max_wl, dead_wl, res_directory)
               
    # Calculation of water surface area for the complete reservoir (2 tiles) and corresponding reservoir restorage 
    if Number_of_tiles==2:
        res_directory = parent_directory + res_name
        os.chdir(res_directory)
        print("Two tiles reservoir")
        # Upper-Left and Lower-right coordinates of the complete reservoir
        complete_res_boundary = [100.2, 23, 100.40, 22.54]
        two_tile(res_name, max_wl, dead_wl, point, complete_res_boundary, dem_file_path, res_directory)
        
    
    
    
    
    
    
    
    
    
    
    
    
    


