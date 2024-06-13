
# IMPORTING LIBRARY

import csv
import os
import numpy as np
import pandas as pd
from osgeo import gdal, gdal_array, osr
import matplotlib.pyplot as plt


# HELPER FUNCTION
def expand(array, n): # (an array of 1 and 0, number of additional pixels)
    expand = array - array
    for i in range(len(array)):
        for j in range(len(array[i])):
            if array[i][j] == 1:
                for k in range(max(0, i-n), min(i+n, len(array)-1)):
                    for l in range(max(0, j-n), min(j+n, len(array[i])-1)):
                        expand[k][l] = 1
                continue
            else:
                continue
    return expand

# ============================Function definition-3 =============================    
def res_isolation(res_name, max_wl, point, boundary): 
    from osgeo import gdal, osr    
    # os.chdir(res_directory + "/Outputs")
    res_dem_file = "DEM_ResIso.tif"
    dem_dataset = gdal.Open(res_dem_file)
    
    # Get the geotransform (georeferencing information)
    geotransform = dem_dataset.GetGeoTransform()
    
    # Extract necessary geotransform parameters
    x_origin = geotransform[0]
    y_origin = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    
    # Convert latitude and longitude to projected coordinates (assuming WGS84)
    input_srs = osr.SpatialReference()
    input_srs.ImportFromEPSG(4326)  # EPSG code for WGS84
    output_srs = osr.SpatialReference()
    output_srs.ImportFromWkt(dem_dataset.GetProjection())
    
    # Create a coordinate transformation object
    transform = osr.CoordinateTransformation(input_srs, output_srs)
    
    # Define the latitude and longitude of your point
    latitude = point[1]
    longitude = point[0]
    
    # Transform latitude and longitude to projected coordinates
    transformed_point = transform.TransformPoint(longitude, latitude)
    x_proj, y_proj, _ = transformed_point
    
    # Convert projected coordinates to row and column indices
    column = int((x_proj - x_origin) / pixel_width)
    row = int((y_proj - y_origin) / pixel_height)
    
    # Close the DEM dataset
    dem_dataset = None
    return column, row 

#============================================================== E-A relationship
def curve_preDEM(res_name, point, boundary, max_wl, parent_directory, grandID, grandCapacity): 
    # E-A-S Curve import from GRAND (WRR paper)
    dtdr = os.getcwd()
    os.chdir(parent_directory)
    os.chdir('GRAND_Curves')
    ID = [file for file in os.listdir() if str(int(grandID)) in file]
    dfA = pd.read_csv(ID[0], parse_dates=True)
    column_names = dfA.iloc[3,0].split(';')
    rows = [row.iloc[0].split(';') for idx, row in dfA.iloc[4:].iterrows()]
    # Creating the new DataFrame with the appropriate columns and rows
    curve_below = pd.DataFrame(rows, columns=column_names).astype(np.float32)
    
    # caculating ABOVE reservoir surface area and storage volume coresponding to each water level
    Outputs_directory = dtdr + '/Outputs'
    os.chdir(Outputs_directory)                   
    res_dem_file = ("DEMclip.tif")
    res_dem = gdal_array.LoadFile(res_dem_file).astype(np.float32)  
    res_dem[res_dem == 0] = np.nan
    res_area = gdal_array.LoadFile('ResIso.tif').astype(np.float32)
    res_areaN = expand(res_area, 3)
    res_areaN1 = expand(res_area, 1)
    res_dem[np.where(res_areaN==0)] = np.nan
 
    [column, row] = res_isolation(res_name, max_wl, point, boundary)      # 'res_isolation' function calling
    min_dem = int(res_dem[row, column])
    curve_ext = max_wl+20            
    curve_temp = [["Level (m)", "Area (sq.km)", "Storage (mcm)"]]
    pre_area = 0
    tot_storage = 0 
    for i in range(min_dem, curve_ext): 
        level = i
        water_px = np.copy(res_dem)
        water_px[res_dem > i] = 0 
        water_px[water_px > 0] = 1
        area = np.nansum(water_px)*9/10000
        storage = (area + pre_area)/2
        tot_storage += storage
        pre_area = area   
        curve_temp = np.append(curve_temp, [[level, round(area,4), round(tot_storage,4)]], 
                            axis=0) 
    
    curve_above = pd.DataFrame(curve_temp[1:], columns=column_names).astype(np.float32)
    
    if (curve_below.iloc[-1,1] >= curve_above.iloc[-1,1]):
        aa = np.array(curve_above.iloc[:,1]).astype(np.float32)
        pos = len(np.array(np.where(aa < 10))[0])
        pos += 1
        valA = aa[pos]
        bb = np.array(curve_below.iloc[:,1]).astype(np.float32)
        bb1 = abs(bb-valA)
        pos1 = np.where(bb1==min(bb1))[0]
        
        df_below = curve_below.copy()
        df_below = df_below.drop(index=range(int(pos1), len(df_below)))
        df_above = curve_above.copy()
        df_above = df_above.drop(index=range(0, int(pos)))

        curve_total = pd.concat([df_below, df_above])
        curve_total = curve_total.reset_index(drop=True)
        
        max_elev = curve_total.iloc[len(curve_total)-1,0]
        elevation_values = np.arange(max_elev, max_elev-len(curve_total), -1)[::-1]
        curve_total['Depth(m)'] = elevation_values.astype(np.float32)
        curve_total = round(curve_total,3)
    
    if (curve_below.iloc[-1,1] < curve_above.iloc[-1,1]):
        df_below = curve_below.copy()
        df_below = df_below.drop(len(df_below)-1)
        
        aa = np.array(curve_above.iloc[:,1]).astype(np.float32)
        aa1 = abs(aa-df_below.iloc[-1,1])
        pos1 = np.where(aa1==min(aa1))[0]

        df_above = curve_above.copy()
        df_above = df_above.drop(index=range(0, int(pos1)+1))

        curve_total = pd.concat([df_below, df_above])
        curve_total = curve_total.reset_index(drop=True)
        
        max_elev = curve_total.iloc[len(curve_total)-1,0]
        elevation_values = np.arange(max_elev, max_elev-len(curve_total), -1)[::-1]
        curve_total['Depth(m)'] = elevation_values.astype(np.float32)
        curve_total = round(curve_total,3)
        
    curve_final = [["Level (m)", "Area (sq.km)", "Storage (mcm)"]]
    pre_area = 0
    tot_storage = 0 
    for i in range(0, len(curve_total)): 
        level = curve_total.iloc[i,0]
        area = curve_total.iloc[i,1]
        storage = (area + pre_area)/2
        tot_storage += storage
        pre_area = area   
        curve_final = np.append(curve_final, [[level, np.round(area,3), np.round(tot_storage,3)]], 
                            axis=0)   
          
    data = curve_final[1:, :]
    data = np.array(data, dtype=np.float32)
    area_km2 = round(np.count_nonzero(res_areaN1 == 1)*0.0009,2)
    wrong_storage = round(np.interp(area_km2, data[:, 1], data[:, 2]),2)
    correct_storage = grandCapacity
    bias = round((wrong_storage - correct_storage)/wrong_storage,2)
    
    if bias>0:
        correct_storage_curve = data[:, 2] - (data[:, 2]*bias)
    if bias<=0:
        correct_storage_curve = data[:, 2] + (data[:, 2]*bias)
    
    data1 = np.column_stack((data[:, 0], data[:, 1], correct_storage_curve))
    corrected_curve_final = [["Level (m)", "Area (sq.km)", "Storage (mcm)"]]
    result = data1.astype(str)
    corrected_curve_final = np.append(corrected_curve_final, result, axis=0)
    
    # saving output as a csv file
    with open('Curve.csv',"w", newline='') as my_csv:
        csvWriter = csv.writer(my_csv)
        csvWriter.writerows(corrected_curve_final)

    # ==================== Plot the DEM-based Level-Storage curve   
    plt.figure()
    plt.scatter(data[:, 0], data[:, 2], s=8, c='red', label='Before storage adjustment')
    plt.scatter(data[:, 0], correct_storage_curve, s=8, c='blue', label='After storage adjustment')
    plt.xlabel('Level (m)')
    plt.ylabel('Storage (mcm)')
    plt.title(res_name + ' (Minimum DEM level= '+ str(round(data[0,0]))+'m)')
    plt.legend()
    plt.savefig(res_name+'_storageVSelevation.png', dpi=600, bbox_inches='tight')
    
    return round(data[0,0]) 
        
#============================================================== E-A relationship
def curve_postDEM(res_name, max_wl): 
    # caculating reservoir surface area and storage volume coresponding to each water level
    dtdr = os.getcwd()
    Outputs_directory = dtdr + '/' + 'Outputs'
    os.chdir(Outputs_directory)                   
    res_dem_file = ("DEMclip.tif")
    res_dem = gdal_array.LoadFile(res_dem_file).astype(np.float32)  
    res_dem[res_dem == 0] = np.nan
    
    res_area = gdal_array.LoadFile('ResIso.tif').astype(np.float32)
    res_areaN = expand(res_area, 3)
    
    res_dem[np.where(res_areaN==0)] = np.nan
 
    min_dem = int(np.nanmin(res_dem))
    curve_ext = max_wl+20            
    results = [["Level (m)", "Area (sq.km)", "Storage (mcm)"]]
    pre_area = 0
    tot_storage = 0 
    for i in range(min_dem, curve_ext): 
        level = i
        water_px = np.copy(res_dem)
        water_px[res_dem > i] = 0 
        water_px[water_px > 0] = 1
        area = np.nansum(water_px)*9/10000
        storage = (area + pre_area)/2
        tot_storage += storage
        pre_area = area   
        results = np.append(results, [[level, round(area,4), round(tot_storage,4)]], 
                            axis=0)
    
    # saving output as a csv file
    with open('Curve.csv',"w", newline='') as my_csv:
        csvWriter = csv.writer(my_csv)
        csvWriter.writerows(results)
        
    # ==================== Plot the DEM-based Level-Storage curve   
    data = results[1:, :]
    data = np.array(data, dtype=np.float32)
    # Create the scatter plot
    plt.figure()
    plt.scatter(data[:, 0], data[:, 2], s=8, c='red')
    # Set labels and title
    plt.xlabel('Level (m)')
    plt.ylabel('Storage (mcm)')
    plt.title(res_name + ' (Minimum DEM level= '+ str(round(data[0,0]))+'m)')
    plt.savefig(res_name+'_storageVSelevation.png', dpi=600, bbox_inches='tight')
    
    return round(data[0,0])
    
