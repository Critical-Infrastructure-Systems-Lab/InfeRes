############+++++++++++ PLEASE MAKE SURE OF THE FOLLOWING POINTS BEFORE RUNNING THE CODE +++++++++++############
# [1]. Data are already downloaded (DEM.tif, Maximum_extent.tif, Frequency.tif, and Satellite images)
# [2]. Use the same coordinates that you have used in "data_download.py"
# [3]. All the python(scripts) files are inside ".../ReservoirExtraction/codes"
# [4]. NOTE: Each set of input data (DEM and Landsat images) is treated as an individual reservoir
# [5]. If you have a multi-tile reservoir then please save different parts of the reservoir in folders 'ReservoirNAME_tile1', 'ReservoirNAME_tile2'...and so on.
############++++++++++++++++++++++++++++++++++++++++ END ++++++++++++++++++++++++++++++++++++++++#############

# IMPORTING LIBRARY

import os 
parent_directory = "G:/My Drive/NUSproject/ReservoirExtraction/InfeRes/"    # Path/to/InfeRes/
os.chdir(parent_directory)
from PREPROCESING import preprocessing
from CURVE import curve_preDEM
from CURVE import curve_postDEM
from WSA import wsa 
import pandas as pd
df = pd.read_csv('inputs_InfeRes.csv', parse_dates=True)

if __name__ == "__main__":

    #====================================>> USER INPUT PARAMETERS 
    i=0
    #for i in range(0, np.size(df,0)):
    os.chdir(parent_directory)
    res_name = df.Name[i] 
    res_built_year = df.Year[i]
    dem_acquisition_year = 2000            # SRTM DEM (30m) acquired in Feb 2000
    grandID = df.GRAND_ID[i]
    # A point within the reservoir [longitude, latitude]
    point = [float(value) for value in df.Point[i].split(',')]
    # Upper-Left and Lower-right coordinates. Example coordinates [longitude, latitude]
    boundary = [float(value) for value in df.Boundary[i].split(',')] 
    max_wl = df.Max_wl[i]                                               
    print('Name of the reservoir: ' + res_name)
    os.chdir(parent_directory)
    res_directory = "../Reservoirs/" + res_name
    os.chdir(res_directory)
    
    # [[A]] Data Pre-processing  ==============
    preprocessing(res_name, max_wl, res_built_year, point, boundary)

    # Case1- Reservoir built before DEM acquisition (i.e. before 2000) ==============
    if res_built_year <= dem_acquisition_year:      
        print('Name of the reservoir: ' + res_name)
        print('Reservoir has built before the acquisition of DEM')
        
        # [[B.1]]. Area-Elevation-Storage curve 
        os.chdir(parent_directory)
        os.chdir(res_directory)
        res_minElev = curve_preDEM(res_name, max_wl, parent_directory, grandID)
        
        # [[C.1]]. Calculating the water surface area
        os.chdir(parent_directory)
        os.chdir(res_directory)
        wsa(res_name)

    # Case2- Reservoir built after DEM acquisition (i.e. after 2000) ==============
    if res_built_year > dem_acquisition_year:        
        print('Name of the reservoir: ' + res_name)
        print('Reservoir has built after the acquisition of DEM')
        
        # [[B.2]]. Area-Elevation-Storage curve
        os.chdir(parent_directory)
        os.chdir(res_directory)
        res_minElev = curve_postDEM(res_name, max_wl)
        
        # [[C.2]]. Calculating the water surface area
        os.chdir(parent_directory)
        os.chdir(res_directory)
        wsa(res_name)

                   
        
    
    
    
    
    
    
    
    
    
    
    


