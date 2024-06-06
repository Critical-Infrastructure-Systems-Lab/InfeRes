############----------------- Note (START)----------------------############

#[1]. Please make sure that all the data has been already downloaded (DEM.tif, Maximum_extent.tif, Frequency.tif, and Satellite images)

############----------------- Note (START)----------------------############

# IMPORTING LIBRARY

import os 
parent_directory = "G:/My Drive/InfeRes_Version1/"    # Path/to/InfeRes/
os.chdir(parent_directory)
from PREPROCESING import preprocessing
from CURVE import curve_preDEM
from CURVE import curve_postDEM
from WSA import wsa_latest 
import pandas as pd
import numpy as np
df = pd.read_csv('inputs_InfeRes.csv', parse_dates=True)

if __name__ == "__main__":

    #====================================>> USER INPUT PARAMETERS 
    i=0
    for i in range(0, np.size(df,0)):
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
            res_minElev = curve_preDEM(res_name, max_wl, parent_directory, grandID, point, boundary)
            
            # [[C.1]]. Calculating the water surface area
            os.chdir(parent_directory)
            os.chdir(res_directory)
            wsa_latest(res_name)
    
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
            wsa_latest(res_name)

                   
        
    
    
    
    
    
    
    
    
    
    
    


