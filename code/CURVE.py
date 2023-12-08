
# IMPORTING LIBRARY

import csv
import os
import utm
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

# Check for best fit curve 
def best_fit_degree (xdata, ydata):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    
    # Maximum degree for polynomial regression
    max_degree = 4
    
    # Lists to store R-squared values and RMSEs for each degree
    r_squared_values = []
    rmse_values = []
    
    for degree in range(1, max_degree + 1):
        # Perform polynomial regression
        coefficients = np.polyfit(xdata, ydata, degree)
        p = np.poly1d(coefficients)
    
        # Calculate fitted values and residuals
        ydata_fit = p(xdata)
        residuals = ydata - ydata_fit
    
        # Calculate R-squared using NumPy
        r_squared = r2_score(ydata, ydata_fit)
        r_squared_values.append(r_squared)
    
        # Calculate RMSE
        rmse = np.sqrt(np.mean(residuals ** 2))
        rmse_values.append(rmse)
        
    # # Plotting R-squared values and RMSEs for different degrees
    # plt.figure(figsize=(10, 4))
    
    # # Plot R-squared values
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, max_degree + 1), r_squared_values, marker='o')
    # plt.xlabel('Polynomial Degree')
    # plt.ylabel('R-squared')
    # plt.title('R-squared vs Polynomial Degree')
    
    # # Plot RMSE values
    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, max_degree + 1), rmse_values, marker='o', color='orange')
    # plt.xlabel('Polynomial Degree')
    # plt.ylabel('RMSE')
    # plt.title('RMSE vs Polynomial Degree')
    
    # plt.tight_layout()
    # plt.show()
    
    # score1 = np.diff((rmse_values[0]-rmse_values)/rmse_values[0]*100)
    # score2 = -np.diff((r_squared_values[0]-r_squared_values)/r_squared_values[0]*100)
    # rmse_pos = int(np.where(score1 == min(score1))[0]) + 1
    # r_squared_pos = int(np.where(score2 == min(score2))[0]) + 1
    
    rmse_pos = int(np.where(rmse_values == min(rmse_values))[0]) + 1
    r_squared_pos = int(np.where(r_squared_values == max(r_squared_values))[0]) + 1
    
    return round((rmse_pos + r_squared_pos)/2)-1

    
# Curve fitting ========================>      
def curve_fit(xdata, ydata, bf_degree):    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Perform polynomial regression (adjust the degree as needed)
    degree = bf_degree  # Example degree of the polynomial
    coefficients = np.polyfit(xdata, ydata, degree)
    p = np.poly1d(coefficients)
    
    # Generate area values for smoother curve plotting
    xdata_values = np.linspace(min(xdata), max(xdata), 1000)
    
    # Calculate corresponding elevation values using the fitted polynomial
    ydata_fit = p(xdata_values)
    
    # # Plotting the original data and the fitted curve
    # plt.figure()
    # plt.scatter(ydata, xdata, label='Original Data')
    # plt.plot(ydata_fit, xdata_values, label='Fitted Curve', color='red')
    # plt.xlabel('Elevation')
    # plt.ylabel('Area')
    # plt.title('Area vs Elevation with Fitted Curve')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # plt.savefig('Fitted_areaVSelevation.png', dpi=600, bbox_inches='tight')
    
    out = np.column_stack((ydata_fit, xdata_values))
    return out

# Curve extrapolation (area tends to zero) =================================================================
def curve_extrapolate(xdata, ydata, bf_degree):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Perform polynomial regression (using a 4th-degree polynomial as an example)
    degree = bf_degree
    coefficients = np.polyfit(xdata, ydata, degree)
    p = np.poly1d(coefficients)
    
    # Generate area values approaching 30% of the largest area for extrapolation
    extrapolated_xdata_values = np.linspace(0.2*min(xdata), max(xdata), 1000)  
    #extrapolated_area_values = np.linspace(max(area), 1.5*max(area), 1000)
    # Use the polynomial function to extrapolate elevation values for small area values
    extrapolated_ydata_values = p(extrapolated_xdata_values)
    
    # Plotting the fitted curve and extrapolated values
    plt.figure()
    plt.plot(ydata, xdata, 'o', label='Original Data')
    plt.plot(extrapolated_ydata_values, extrapolated_xdata_values, label='Extrapolated Curve', color='red')
    plt.ylabel('Area')
    plt.xlabel('Elevation')
    plt.title('Extrapolation of Elevation for Small Area Values')
    plt.legend()
    plt.grid(True)
    plt.show()
    #plt.savefig('Extrapolated_areaVSelevation.png', dpi=600, bbox_inches='tight')
    
    out = np.column_stack((extrapolated_ydata_values, extrapolated_xdata_values))
    return out
    
# DEM-based reservoir isolation =============================    
def res_isolation(res_name, max_wl, point, boundary, res_directory): 
    
    os.chdir(res_directory + "/Outputs")
    res_dem_file = (res_name + "_DEM_UTM_CLIP.tif")
    
    try: # Converting point and boundary coordinates from CGS to UTM ===========    
        #30m nearly equal to 0.00027777778 decimal degree
        xp = abs(round((point[0]-boundary[0])/0.00027777778))
        yp = abs(round((point[1]-boundary[1])/0.00027777778)) 
        
        dem_bin = gdal_array.LoadFile(res_dem_file).astype(np.float32)
        dem_bin[dem_bin == 32767] = np.nan    
        dem_bin[np.where(dem_bin > max_wl+10)] = 0        #to expand the reservoir extent (10m extra) for accounting uncertainity in max_wl
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
    # plt.imshow(res_iso, cmap='viridis')
    # plt.scatter([xp], [yp], c='r', s=10)
    plt.imshow(dem_bin, cmap='viridis')
    plt.scatter([xp], [yp], c='r', s=20)
    plt.title('DEM-based reservoir isolation')
    plt.savefig(res_name+"_DEM_res_iso.png", dpi=600, bbox_inches='tight')
    #------------------ Visualization <End>
    
    gdal_array.SaveArray(res_iso.astype(gdal_array.numpy.float32), 
                                  "res_iso.tif", format="GTiff", 
                                  prototype = res_dem_file)   

    return xp, yp


#============================================================== E-A relationship
def curve_preDEM(res_name, point_loc, res_directory): 
    # caculating reservoir surface area and storage volume coresponding to each water level
    os.chdir(res_directory + "/Outputs")                    
    res_dem_file = (res_name + "_DEM_UTM_CLIP.tif")  
    dem_bin = gdal_array.LoadFile(res_dem_file).astype(np.float32)
    dem_bin[dem_bin == 32767] = np.nan 
    [xp, yp]=  point_loc
    dem_val = int(dem_bin[yp,xp])
    
    results = [["Level (m)", "Area (skm)", "Storage (mcm)"]]
    pre_area = 0
    tot_stor = 0
    for i in range(dem_val, dem_val+30): 
        level = i
        water_px = gdal_array.LoadFile(res_dem_file).astype(np.float32)
        water_px[water_px == 32767] = np.nan
        water_px[np.where(water_px < dem_val)] = np.nan 
        water_px[np.where(water_px > i)] = 0 
        water_px[np.where(water_px > 0)] = 1
        res_iso = pick(xp, yp, water_px)
        area = np.nansum(res_iso)*9/10000
        storage = (area + pre_area)/2
        tot_stor += storage
        pre_area = area   
        results = np.append(results, [[level, round(area,4), round(tot_stor,4)]], 
                            axis=0)
        
    # Extract column 2 and 3 from the array    
    data = results[1:, :]
    data = np.array(data, dtype=np.float32)
    area = data[2:, 1]    # area 
    elevation = data[2:, 0]    # elevation (ydata)
    
    # Function calling to get best-fit degree of polinomial
    bf_degree = best_fit_degree(area, elevation)
    
    # Function calling to get best-fit E-A-S curve values
    #EA_curve = curve_fit(area, elevation, bf_degree)            # This curve represents the characteristics of DEM outside the reservoir 
    
    # Function calling to extrapolate the best-fit to get E-A-S values
    EA_curve_extrapolated = curve_extrapolate(area, elevation, bf_degree)
    
    updated_results = [["Level (m)", "Area (skm)", "Storage (mcm)"]]
    pre_area = 0
    tot_stor = 0
    initial_elevation = EA_curve_extrapolated[0,0]
    for i in range(0, len(EA_curve_extrapolated)): 
        level = EA_curve_extrapolated[i,0]
        area = EA_curve_extrapolated[i,1]
        effective_height = EA_curve_extrapolated[i,0] - initial_elevation
        storage = (area + pre_area)/2*(effective_height)
        tot_stor += storage
        pre_area = area  
        initial_elevation = EA_curve_extrapolated[i,0]
        updated_results = np.append(updated_results, [[level, round(area,4), round(tot_stor,4)]], 
                            axis=0) 
    
    # saving output as a csv file
    with open('Curve_' + res_name + '.csv',"w", newline='') as my_csv:
        csvWriter = csv.writer(my_csv)
        csvWriter.writerows(updated_results)
    
    # ==================== Plot the DEM-based Level-Storage curve   
    data = updated_results[1:, :]
    data = np.array(data, dtype=np.float32)
    # Create the scatter plot
    plt.figure()
    plt.scatter(data[:,0], data[:,2], s=5, c='red')
    # Set labels and title
    plt.xlabel('Level (m)')
    plt.ylabel('Storage (mcm)')
    plt.title(res_name + ' (Minimum DEM level= '+ str(round(data[0,0]))+'m)')
    plt.savefig(res_name+'_storageVSelevation.png', dpi=600, bbox_inches='tight')
    
    return round(data[0,0])  
        


#============================================================== E-A relationship
def curve_postDEM(res_name, max_wl, res_directory): 
    # caculating reservoir surface area and storage volume coresponding to each water level
    os.chdir(res_directory + "/Outputs")                    
    res_dem_file = (res_name + "_DEM_UTM_CLIP.tif")
    res_dem = gdal_array.LoadFile(res_dem_file).astype(np.float32)
    res_dem[res_dem == 32767] = np.nan    
    
    exp_mask = gdal_array.LoadFile("Expanded_Mask.tif").astype(np.float32)
    res_dem[np.where(exp_mask == 0)] = np.nan
    output = gdal_array.SaveArray(res_dem.astype(gdal_array.numpy.float32), 
                                  "DEM_Landsat_res_iso.tif", 
                                  format="GTiff", prototype = res_dem_file)
    output = None
    # plt.figure()
    # plt.imshow(res_dem, cmap='jet')
    # plt.colorbar()
    # 
    min_dem = int(np.nanmin(res_dem))
    curve_ext = max_wl+20            
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
    
    # Extract column 2 and 3 from the array    
    data = results[1:, :]
    data = np.array(data, dtype=np.float32)
    idx = np.where(data[:, 1]==min(data[:, 1]))
    data = data[np.size(idx,1)-1:]
    idx = np.where((data[:, 0] > 0) & (data[:, 1] > 0) & (data[:, 2] > 0))[0]
    data = data[idx[0]:]
    area = data[2:, 1]    # area 
    elevation = data[2:, 0]    # elevation (ydata)
    
    # Function calling to get best-fit degree of polinomial
    from scipy.interpolate import interp1d
    interpolation_function = interp1d(area, elevation, kind='linear', fill_value='extrapolate')
    new_area_values = np.linspace(min(area), max(area), 500)
    interpolated_elevation_values = interpolation_function(new_area_values)
    
    # Function calling to get best-fit E-A-S curve values
    EA_curve = np.column_stack((interpolated_elevation_values[2:], new_area_values[2:]))  
    
    updated_results = [["Level (m)", "Area (skm)", "Storage (mcm)"]]
    pre_area = 0
    tot_stor = 0
    initial_elevation = EA_curve[0,0]
    for i in range(0, len(EA_curve)): 
        level = EA_curve[i,0]
        area = EA_curve[i,1]
        effective_height = EA_curve[i,0] - initial_elevation
        storage = (area + pre_area)/2*(effective_height)
        tot_stor += storage
        pre_area = area  
        initial_elevation = EA_curve[i,0]
        updated_results = np.append(updated_results, [[level, round(area,4), round(tot_stor,4)]], 
                            axis=0) 
    
    # saving output as a csv file
    with open('Curve_' + res_name + '.csv',"w", newline='') as my_csv:
        csvWriter = csv.writer(my_csv)
        csvWriter.writerows(updated_results)
    
    # ==================== Plot the DEM-based Level-Storage curve   
    data = updated_results[1:, :]
    data = np.array(data, dtype=np.float32)
    # Create the scatter plot
    plt.figure()
    plt.scatter(data[:, 0], data[:, 2], s=5, c='red')
    # Set labels and title
    plt.xlabel('Level (m)')
    plt.ylabel('Storage (mcm)')
    plt.title(res_name + ' (Minimum DEM level= '+ str(round(data[0,0]))+'m)')
    plt.savefig(res_name+'_storageVSelevation.png', dpi=600, bbox_inches='tight')
    
    return round(data[0,0])


#=================================== Update Landsat-based E-A database with DEM-based E-A-S relationship
def one_tile(res_name, max_wl, dead_wl, res_directory):
    
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
    
    delete_id = Wsa[(Wsa['Quality'] == 0) | (Wsa['dem_value_m'] > max_wl+20)].index  #(Wsa['dem_value_m'] < dead_wl)
    Wsa = Wsa.drop(delete_id)
    # ========================================== EXPORT RESULTS AS A CSV FILE
    Wsa.to_csv('WSA_updated_' + res_name + '.csv', index=False)
    print("Done")    
    print("  ")
    




