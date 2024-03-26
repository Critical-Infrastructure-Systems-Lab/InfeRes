
# IMPORTING LIBRARY

import csv
import os
import numpy as np
import pandas as pd
from osgeo import gdal, gdal_array, osr
import matplotlib.pyplot as plt


# HELPER FUNCTION
def pick(c, r, mask): # (c_number, r_number, an array of 1 amd 0) 
    filled = set()
    fill = set()
    fill.add((c, r))
    width = mask.shape[1]-1
    height = mask.shape[0]-1
    picked = np.zeros_like(mask, dtype=np.int8)
    while fill:
        x, y = fill.pop()
        if y == height or x == width or x < 0 or y < 0:
            continue
        if mask[y][x] == 1:
            picked[y][x] = 1
            filled.add((x, y))
            west = (x-1, y)
            east = (x+1, y)
            north = (x, y-1)
            south = (x, y+1)
            if west not in filled:
                fill.add(west)
            if east not in filled:
                fill.add(east)
            if north not in filled:
                fill.add(north)
            if south not in filled:
                fill.add(south)
    return picked

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


#============================================================== E-A relationship
def curve_preDEM(res_name, max_wl, parent_directory, grandID): 
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
    res_dem[np.where(res_areaN==0)] = np.nan
 
    min_dem = int(np.nanmin(res_dem))
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
        
    # saving output as a csv file
    with open('Curve.csv',"w", newline='') as my_csv:
        csvWriter = csv.writer(my_csv)
        csvWriter.writerows(curve_final)
        
    # ==================== Plot the DEM-based Level-Storage curve   
    data = curve_final[1:, :]
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
    
    




