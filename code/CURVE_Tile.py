
# IMPORTING LIBRARY
import csv
import os
import numpy as np
import pandas as pd
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt


# # HELPER FUNCTION
# def pick(c, r, mask): # (c_number, r_number, an array of 1 amd 0) 
#     filled = set()
#     fill = set()
#     fill.add((c, r))
#     width = mask.shape[1]-1
#     height = mask.shape[0]-1
#     picked = np.zeros_like(mask, dtype=np.int8)
#     while fill:
#         x, y = fill.pop()
#         if y == height or x == width or x < 0 or y < 0:
#             continue
#         if mask[y][x] == 1:
#             picked[y][x] = 1
#             filled.add((x, y))
#             west = (x-1, y)
#             east = (x+1, y)
#             north = (x, y-1)
#             south = (x, y+1)
#             if west not in filled:
#                 fill.add(west)
#             if east not in filled:
#                 fill.add(east)
#             if north not in filled:
#                 fill.add(north)
#             if south not in filled:
#                 fill.add(south)
#     return picked

def one_tile(res_name, max_wl, dead_wl, res_directory):
    #========== ESTIMATING TOTAL RESERVOIR AREA USING PREVIOUSLY GENERATED CURVE FOR THE BIGGER TILE
    
    os.chdir(res_directory +  "/Outputs") 
    curve = pd.read_csv('Curve_' + res_name + '.csv')
    landsat_wsa = pd.read_csv('WSA_' + res_name + '.csv')
       
    Wsa= landsat_wsa
    Wsa["dem_value_m"] = None
    Wsa["Tot_res_volume_mcm"] = None
    for i in range(0,len(Wsa)):
        diff = np.abs(curve.iloc[:, 1] - Wsa.Fn_area[i])    
        closest_index = np.argmin(diff)
        closest_elev = curve.iloc[closest_index, 0]
        closest_vol = curve.iloc[closest_index, 2]
        Wsa.dem_value_m[i] = closest_elev
        Wsa.Tot_res_volume_mcm[i] = closest_vol
    
    delete_id = Wsa[(Wsa['Quality'] == 0) | (Wsa['dem_value_m'] < dead_wl) | (Wsa['dem_value_m'] > max_wl+20)].index
    Wsa = Wsa.drop(delete_id)
    # ========================================== EXPORT RESULTS AS A CSV FILE
    Wsa.to_csv('WSA_updated_' + res_name + '.csv', index=False)
    print("Done")    
    print("  ")




# def two_tile(res_name, max_wl, dead_wl, point, complete_res_boundary, dem_file_path, res_directory):
#    # ================================================== (Complete reservoir)
#    # =====================================================  INPUT PARAMETERS
#     dem = gdal.Open(dem_file_path) 
#     os.chdir(os.getcwd() + "/Outputs")
#     res_dem_file = res_name + "DEMcomplete.tif"
#     dem = gdal.Translate(res_dem_file, dem, projWin = complete_res_boundary)
#     dem = None
    
#     #------------------ Visualization <Start>
#     # image = gdal_array.LoadFile(res_dem_file)
#     # plt.figure()
#     # plt.imshow(image, cmap='jet')
#     # plt.colorbar()
    
# #============================================================ GCS to UTM         
#     # Input and output file paths
#     ref_file = "DEM_Landsat_res_iso.tif"
#     target_file = res_dem_file
#     output_file = res_name + "DEMcomplete_UTM.tif"     
         
#     # Open the reference raster (bigger dem)
#     ref_ds = gdal.Open(ref_file)
    
#     # Get the projection and geotransform from the reference raster
#     ref_proj = ref_ds.GetProjection()
#     ref_transform = ref_ds.GetGeoTransform()
    
#     # Open the raster to be reprojected (complete dem)
#     input_ds = gdal.Open(target_file)
    
#     # Create a new raster with the same projection as the reference raster
#     output_ds = gdal.Warp(output_file, input_ds, dstSRS=ref_proj, xRes=ref_transform[1], 
#                           yRes=abs(ref_transform[5]), 
#                           outputBounds=[ref_transform[0],
#                                         ref_transform[3] + ref_transform[5] * input_ds.RasterYSize,
#                                         ref_transform[0] + ref_transform[1] * input_ds.RasterXSize,
#                                         ref_transform[3]],
#                           format='GTiff')
    
#     # Close the input and output datasets
#     input_ds = None
#     output_ds = None
#     ref_ds = None
    
# # =====================================================  INPUT PARAMETERS   
#     os.chdir(res_directory + "/Outputs")
#     res_dem_file = (res_name + "DEMcomplete_UTM.tif")
#     dem_ds = gdal.Open(res_dem_file)   
#     geotransform = dem_ds.GetGeoTransform() 
    
#     # 30m nearly equal to 0.00027777778 decimal degree
#     xp = abs(round((point[0]-complete_res_boundary[0])/0.00027777778))
#     yp = abs(round((point[1]-complete_res_boundary[1])/0.00027777778))     
#     dem_ds = None
                             
#     # CREATING E-A-S RELATIONSHOP   
#     # isolating the reservoir
#     dem_bin = gdal_array.LoadFile(res_dem_file).astype(np.float32)
#     dem_bin[dem_bin == 32767] = np.nan
#     #------------------ Visualization <Start>
#     plt.figure()
#     plt.imshow(dem_bin, cmap='jet')
#     plt.colorbar()
    
#     dem_bin[np.where(dem_bin > max_wl+10)] = 0        #to expand the reservoir extent for accounting uncertainity in max_wl
#     dem_bin[np.where(dem_bin > 0)] = 1
#     res_iso = pick(xp, yp, dem_bin)
#     plt.figure()
#     plt.imshow(res_iso, cmap='jet')
#     plt.colorbar()
#     output = gdal_array.SaveArray(res_iso.astype(gdal_array.numpy.float32), 
#                                   "res_iso.tif", format="GTiff", 
#                                   prototype = res_dem_file)
#     output = None
    
#     # finding the lowest DEM value in the reservoir extent
#     res_dem = gdal_array.LoadFile(res_dem_file)
#     res_dem[res_dem == 32767] = np.nan
#     res_dem[np.where(res_iso == 0)] = 9999           # 9999 is any arbitrary unrealistice value
#     min_dem = int(np.nanmin(res_dem))   
   
#=======================================================================================================   
   # bc = complete_res_boundary
   # pc = point_coordinates
   # coords = bc + pc
   # utm_coords = np.array([utm.from_latlon(coords[i + 1], coords[i]) for i in range(0, len(coords), 2)])
   # # Bounding box of the reservoir [ulx, uly, lrx, lry]
   # bbox = np.array([utm_coords[0,0], utm_coords[0,1], utm_coords[1,0], utm_coords[1,1]], dtype=np.float32) 
   # res_point = np.array([utm_coords[2,0], utm_coords[2,1]], dtype=np.float32)
   # xp = round(abs(res_point[0]-bbox[0])/30)
   # yp = round(abs(res_point[1]-bbox[1])/30)
   # # Maximum reservoir water level                            
   # curve_ext = max_wl + 10 # to expand the curve  
   
   # CREATING E-A-S RELATIONSHOP
   # clipping DEM by the bounding box
   # print("Clipping DEM by the bounding box ...") 
   # dem = gdal.Open(dem_file_path) 
   
   # # Changing path to the desired reservoir
   # os.chdir(os.getcwd() + "/Outputs")
   # res_dem_file = "Complete_" + res_name+"DEM.tif"
   # dem = gdal.Translate(res_dem_file, dem, projWin = bbox)
   # dem = None 
   
   # # isolating the reservoir
   # dem_bin = gdal_array.LoadFile(res_dem_file)
   # dem_bin[np.where(dem_bin > curve_ext)] = 0
   # dem_bin[np.where(dem_bin > 0)] = 1
   # res_iso = pick(xp, yp, dem_bin)
   
   # # finding the lowest DEM value in the reservoir extent
   # res_dem = gdal_array.LoadFile(res_dem_file)
   # res_dem[np.where(res_iso == 0)] = 9999
   # min_dem = np.nanmin(res_dem)
   
   # # caculating reservoir surface area and storage volume 
   # # coresponding to each water level
   # results = [["Level (m)", "Area (skm)", "Storage (mcm)"]]
   # pre_area = 0
   # tot_stor = 0 
   # for i in range(min_dem, curve_ext): 
   #     level = i
   #     water_px = gdal_array.LoadFile(res_dem_file)
   #     water_px[np.where(res_iso == 0)] = 9999
   #     water_px[np.where(res_dem > i)] = 0 
   #     water_px[np.where(water_px > 0)] = 1
   #     area = np.nansum(water_px)*9/10000
   #     storage = (area + pre_area)/2
   #     tot_stor += storage
   #     pre_area = area   
   #     results = np.append(results, [[level, round(area,4), round(tot_stor,4)]], 
   #                         axis=0)
   
   #  # saving output as a csv file
   # with open("Curve_complete_res.csv","w", newline='') as my_csv:
   #      csvWriter = csv.writer(my_csv)
   #      csvWriter.writerows(results)
        
   #  #========== ESTIMATING TOTAL RESERVOIR AREA USING PREVIOUSLY GENERATED CURVE FOR THE BIGGER TILE
    
   #  # Import E-A curve for complete reservoir
   # complete = pd.read_csv("Curve_complete_res.csv")
    
   #  # Import E-A curve for bigger portion of the reservoir
   # bigger = pd.read_csv("Curve.csv")
    
   #  # Import landsat-derived water surface area for bigger portion of the reservoir
   # bigger_landsat_wsa = pd.read_csv("WSA.csv")
    
   # Wsa= bigger_landsat_wsa
   # Wsa["dem_value_m"] = None
   # Wsa["Tot_res_area_skm"] = None
   # Wsa["Tot_res_volume_mcm"] = None
   # for i in range(0,len(Wsa)):
   #     diff = np.abs(bigger.iloc[:, 1] - Wsa.Fn_area[i])    
   #     closest_index = np.argmin(diff)
   #     closest_elev = bigger.iloc[closest_index, 0]
   #     Wsa.dem_value_m[i] = closest_elev
        
   #     # Select the corresponding row in the complete DataFrame
   #     selected_row = complete.iloc[np.where(complete.iloc[:, 0] == closest_elev)]
   #     Wsa.Tot_res_area_skm[i] = selected_row.iloc[0, 1]
   #     Wsa.Tot_res_volume_mcm[i] = selected_row.iloc[0, 2]
    
   # delete_id = Wsa[(Wsa['Quality'] == 0) | (Wsa['dem_value_m'] < dead_wl) | (Wsa['dem_value_m'] > max_wl)].index
   # Wsa = Wsa.drop(delete_id)
   # # ========================================== EXPORT RESULTS AS A CSV FILE
   # Wsa.to_csv('WSA_Complete_' + res_name + '.csv', index=False)
   # print("Done")    
   # print("  ")
    