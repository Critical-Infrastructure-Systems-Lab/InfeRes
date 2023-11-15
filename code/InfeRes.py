############+++++++++++ PLEASE MAKE SURE OF THE FOLLOWING POINTS BEFORE RUNNING THE CODE +++++++++++############
# [1]. Data are already downloaded (Satellite images and DEM)
# [2]. DEM should be in the projected coordinate system (unit: meters)
# [3]. Use the same coordinates that you have used in "data_download.py"
# [4]. All the python(scripts) files are inside ".../ReservoirExtraction/codes"
# [5]. "Number_of_tiles" = Number of Landsat tiles to cover the entire reservoir. It is recommended to download the Landsat tile covering the maximum reservoir area 
############++++++++++++++++++++++++++++++++++++++++ END ++++++++++++++++++++++++++++++++++++++++#############

# IMPORTING LIBRARY

import os 
os.chdir("H:/My Drive/NUSproject/ReservoirExtraction/")
from codes.CURVE import curve
from codes.CURVE import res_iso
from codes.MASK import mask
from codes.WSA import wsa
from codes.PREPROCESING import preprocessing
from codes.CURVE_Tile import one_tile
from codes.CURVE_Tile import two_tile


if __name__ == "__main__":

    #====================================>> USER INPUT PARAMETERS 
    parent_directory = "H:/My Drive/NUSproject/ReservoirExtraction/Reservoirs/"
    os.chdir(parent_directory)
    res_name = "PleiKrong"                        # Name of the reservoir
    res_directory = parent_directory + res_name
    # A point within the reservoir [longitude, latitude]
    point = [107.872, 14.422]
    # Upper-Left and Lower-right coordinates. Example coordinates [longitude, latitude]
    boundary = [107.763, 14.672, 107.924, 14.392] #[107.763, 14.672, 107.924, 14.392]
    max_wl = 575 
    yearOFcommission = 2008
    Number_of_tiles = 1                             
    os.makedirs(res_name, exist_ok=True)                  
    os.chdir(parent_directory + res_name)
    # Create a new folder within the working directory to download the data
    os.makedirs("Outputs", exist_ok=True)
    # Path to DEM (SouthEastAsia_DEM30m.tif), PCS: WGS1984
    dem_file_path = "H:/My Drive/NUSproject/ReservoirExtraction/SEAsia_DEM/SouthEastAsia_DEM30m.tif"
    
    #====================================>> FUNCTION CALLING -1
    # [1]. Data pre-processing (reprojection and clipping)
    preprocessing(res_name, point, boundary, parent_directory, dem_file_path)
    
    #====================================>> FUNCTION CALLING -2
    # [2]. DEM-based reservoir isolation
    res_iso(res_name, max_wl, point, boundary, res_directory)
    
    #====================================>> FUNCTION CALLING -3
    # [3]. Creating mask/intermediate files
    mask(res_name, yearOFcommission, max_wl, point, boundary, res_directory)
    
    #====================================>> FUNCTION CALLING -4
    # [4]. DEM-Landsat-based updated Area-Elevation-Storage curve
    res_minElev = curve(res_name, res_directory)
     
    #====================================>> FUNCTION CALLING -5
    # [5]. Calculating the water surface area
    res_directory = parent_directory + res_name
    os.chdir(res_directory)
    wsa(res_name, res_directory)
    
    #====================================>> FUNCTION CALLING -6
    # [6]. Calculating the reservoir restorage (1 tiles)
    if Number_of_tiles==1:
        res_directory = parent_directory + res_name
        os.chdir(res_directory)
        print("One tile reservoir")
        one_tile(res_name, max_wl, res_minElev, res_directory)
               
    # Calculation of water surface area for the complete reservoir (2 tiles) and corresponding reservoir restorage 
    if Number_of_tiles==2:
        res_directory = parent_directory + res_name
        os.chdir(res_directory)
        print("Two tiles reservoir")
        # Upper-Left and Lower-right coordinates of the complete reservoir
        complete_res_boundary = [101.693, 20.007, 102.331, 19.240]
        two_tile(res_name, max_wl, res_minElev, point, complete_res_boundary, dem_file_path, res_directory)
        
    
    
    
    
    
    
    
    
    
    
    


