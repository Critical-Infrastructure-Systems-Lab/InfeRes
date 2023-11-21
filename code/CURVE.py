
# IMPORTING LIBRARY

import csv
import os
import utm
import numpy as np
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



def res_isolation(res_name, max_wl, point, boundary, res_directory): 
    # =====================================================  INPUT PARAMETERS
    os.chdir(res_directory + "/Outputs")
    res_dem_file = (res_name + "_DEM_UTM_CLIP.tif")
    
    try: # Converting point and boundary coordinates from CGS to UTM ===========    
        #30m nearly equal to 0.00027777778 decimal degree
        xp = abs(round((point[0]-boundary[0])/0.00027777778))
        yp = abs(round((point[1]-boundary[1])/0.00027777778)) 
        
        dem_bin = gdal_array.LoadFile(res_dem_file).astype(np.float32)
        dem_bin[dem_bin == 32767] = np.nan    
        dem_bin[np.where(dem_bin > max_wl+10)] = 0        #to expand the reservoir extent for accounting uncertainity in max_wl
        dem_bin[np.where(dem_bin > 0)] = 1
        res_iso = pick(xp, yp, dem_bin)
        aa=sum(sum(res_iso))
        
        if aa == 0:    
            dem_ds = gdal.Open(res_dem_file)   
            geotransform = dem_ds.GetGeoTransform()   
            # Calculate the bounding box coordinates
            left = geotransform[0]
            top = geotransform[3]
            right = left + geotransform[1] * dem_ds.RasterXSize
            bottom = top + geotransform[5] * dem_ds.RasterYSize 
            # Bounding box of the reservoir [ulx, uly, lrx, lry]        
            bbox = [left, top, right, bottom]
            
            utm_coords = np.array([utm.from_latlon(point[i + 1], point[i]) for i in range(0, len(point), 2)])
            res_point = np.array([utm_coords[0,0], utm_coords[0,1]], dtype=np.float32)
            xp = round(abs(res_point[0]-bbox[0])/30)
            yp = round(abs(res_point[1]-bbox[1])/30)    
            
            dem_bin = gdal_array.LoadFile(res_dem_file).astype(np.float32)
            dem_bin[dem_bin == 32767] = np.nan    
            dem_bin[np.where(dem_bin > max_wl+10)] = 0        #to expand the reservoir extent for accounting uncertainity in max_wl
            dem_bin[np.where(dem_bin > 0)] = 1
            res_iso = pick(xp, yp, dem_bin)
    
    except Exception as e:
          # Handle the exception or perform actions to handle the error gracefully
          print(f"An error occurred: {str(e)}")
    
    #------------------ Visualization <Start>
    plt.figure()
    plt.imshow(res_iso, cmap='viridis')
    plt.scatter([xp], [yp], c='r', s=10)
    plt.imshow(dem_bin, cmap='viridis')
    plt.scatter([xp], [yp], c='r', s=20)
    plt.title('DEM-based reservoir isolation')
    plt.savefig(res_name+"_DEM_res_iso.png", dpi=600, bbox_inches='tight')
    #------------------ Visualization <End>
    
    gdal_array.SaveArray(res_iso.astype(gdal_array.numpy.float32), 
                                  "res_iso.tif", format="GTiff", 
                                  prototype = res_dem_file)   

#============================================================== E-A relationship
def curve(res_name, res_directory): 
    # caculating reservoir surface area and storage volume coresponding to each water level
    os.chdir(res_directory + "/Outputs")                    
    res_dem_file = (res_name + "_DEM_UTM_CLIP.tif")
    res_dem = gdal_array.LoadFile(res_dem_file).astype(np.float32)
    res_dem[res_dem == 32767] = np.nan    
    
    exp_mask = gdal_array.LoadFile("Expanded_Mask.tif").astype(np.float32)
    res_dem[np.where(exp_mask == 0)] = np.nan
    gdal_array.SaveArray(res_dem.astype(gdal_array.numpy.float32), 
                                  "DEM_Landsat_res_iso.tif", 
                                  format="GTiff", prototype = res_dem_file)
    # plt.figure()
    # plt.imshow(res_dem, cmap='jet')
    # plt.colorbar()
    # 
    min_dem = int(np.nanmin(res_dem))
    curve_ext = int(np.nanmax(res_dem))            
    res_dem_updated = ("DEM_Landsat_res_iso.tif")
        
    results = [["Level (m)", "Area (skm)", "Storage (mcm)"]]
    pre_area = 0
    tot_stor = 0 
    for i in range(min_dem, curve_ext): 
        level = i
        water_px = gdal_array.LoadFile(res_dem_updated)
        water_px[np.where(res_dem > i)] = 0 
        water_px[np.where(water_px > 0)] = 1
        area = np.nansum(water_px)*9/10000
        storage = (area + pre_area)/2
        tot_stor += storage
        pre_area = area   
        results = np.append(results, [[level, round(area,4), round(tot_stor,4)]], 
                            axis=0)
    
    # saving output as a csv file
    with open('Curve_' + res_name + '.csv',"w", newline='') as my_csv:
        csvWriter = csv.writer(my_csv)
        csvWriter.writerows(results)
    
    # ==================== Plot the DEM-based Level-Storage curve   
    data = results[1:, :]
    data = np.array(data, dtype=np.float32)
    # Extract column 2 and 3 from the array
    column_1 = data[:, 0]
    column_3 = data[:, 2]
    # Create the scatter plot
    plt.figure()
    plt.scatter(column_1, column_3, s=5, c='red')
    # Set labels and title
    plt.xlabel('Level (m)')
    plt.ylabel('Storage (mcm)')
    plt.title(res_name)
    plt.savefig(res_name+'_storageVSelevation.png', dpi=600, bbox_inches='tight')
    
    return min_dem
    




