# IMPORTING LIBRARY
import os 
import ee
import csv
import cv2
import shutil
import calendar
import numpy as np
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#========================= Function definition-1=============================
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

#========================= Function definition-2 =============================
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
    res_dem_file = "DEM.tif"
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

    dem_bin = gdal_array.LoadFile(res_dem_file).astype(np.float32)
    dem_bin[dem_bin == 0] = np.nan 
    dem_bin[dem_bin > max_wl+10] = np.nan        #to expand the reservoir extent (10m extra) for accounting uncertainity in max_wl
    dem_bin[dem_bin > 0] = 1
    
    #Visualization <Start>
    plt.figure()
    plt.imshow(dem_bin, cmap='viridis')
    plt.scatter([column], [row], c='r', s=20)
    plt.title('DEM-based isolation')
    plt.savefig("DEM_based_isolation.png", dpi=600, bbox_inches='tight')
    #Visualization <End>  
    
    return column, row 

def preprocessing(res_name, max_wl, res_built_year, point, boundary):   
    
 # # [0] ============================= Generate image with common overlapping area 
    dtdr = os.getcwd()
    RawData_directory = dtdr + '/' + res_name + '_RawData'
    
    filesL9 = [file for file in os.listdir(RawData_directory) if "L9_NDWI" in file]
    filesL8 = [file for file in os.listdir(RawData_directory) if "L8_NDWI" in file]
    filesL7 = [file for file in os.listdir(RawData_directory) if "L7_NDWI" in file]
    filesL5 = [file for file in os.listdir(RawData_directory) if "L5_NDWI" in file]
    filesL0 = [file for file in os.listdir(RawData_directory) if "L0_NDWI" in file]
    filesS2 = [file for file in os.listdir(RawData_directory) if "S2_NDWI" in file]
    file_lists = [filesL9, filesL8, filesL7, filesL5, filesS2, filesL0]
    filtered_files = []
    if all(file_list for file_list in file_lists if file_list):
        for file_list in file_lists:
            if file_list:
                filtered_files.append(file_list[0])
        
    datasets = []
    for filename in filtered_files:
        ds = gdal.Open(os.path.join(RawData_directory, filename))
        datasets.append(ds)
    
    # Determine the minimum dimensions
    min_width = min(ds.RasterXSize for ds in datasets)
    min_height = min(ds.RasterYSize for ds in datasets)
    
    # Create a new raster with the minimum dimensions
    Supporting_directory = dtdr + '/' + res_name + '_Supporting'
    os.chdir(Supporting_directory)
    output_path = 'ROI.tif'
    if os.path.exists(output_path):
        os.remove(output_path)
    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_path, min_width, min_height, 1, gdal.GDT_Float32)
    output_ds.SetProjection(datasets[0].GetProjection())
    output_ds.SetGeoTransform(datasets[0].GetGeoTransform())
    output_ds = None
    for ds in datasets:
        ds = None    
    

# # [1] ============================= DEM clip based on given extent   
    os.chdir(Supporting_directory)
    dem_file_path = os.getcwd() + '/DEM.tif'
    dem = gdal_array.LoadFile(dem_file_path).astype(np.float32)
    freq = gdal_array.LoadFile('Frequency.tif').astype(np.float32)
    roi = gdal_array.LoadFile('ROI.tif').astype(np.float32)
    freq = ((freq - np.nanmin(freq)) / (np.nanmax(freq) - np.nanmin(freq))) * 100

    max_ext = gdal_array.LoadFile('MaxExtent.tif').astype(np.float32)
    max_ext[np.isnan(max_ext)==1]= np.nanmin(max_ext)
    max_ext_normalized = (max_ext + 1) * 127.5  # Normalize to range [0, 255]
    max_ext_normalized = max_ext_normalized.astype(np.uint8)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25, 25))
    # clahe_image = clahe.apply(max_ext_normalized)    
    #max_ext_Rescaled = ((clahe_image - np.nanmin(clahe_image)) / (np.nanmax(clahe_image) - np.nanmin(clahe_image))) * 2 - 1
    #max_ext_Rescaled = np.copy(max_ext_Rescaled).astype(np.float32)
    
    #-------------------- K-means clustering for maximum water extent -------------------- 
    x = max_ext_normalized.ravel()
    km = KMeans(n_clusters=2, n_init=10)
    km.fit(x.reshape(-1,1)) 
    cluster_labels = km.predict(max_ext_normalized.reshape(-1, 1))
    cluster_raster = cluster_labels.reshape(max_ext_normalized.shape)
    z = km.cluster_centers_
    max_center_index = np.argmax(np.sum(z, axis=1))
    max_cluster_mask1D = (cluster_labels == max_center_index)
    max_cluster_mask2D = max_cluster_mask1D.reshape(max_ext_normalized.shape)
    max_water_cluster = np.copy(max_cluster_mask2D).astype(np.float32)
    #-------------------- K-means clustering for maximum water extent --------------------

    os.chdir(dtdr)
    os.makedirs('Outputs', exist_ok=True)
    Outputs_directory = dtdr + '/' + 'Outputs'
    os.chdir(Outputs_directory)
    output = gdal_array.SaveArray(dem.astype(gdal_array.numpy.float32), 
                                  "DEM.tif", 
                                  format="GTiff", prototype = dem_file_path)
    output = None
    output = gdal_array.SaveArray(freq.astype(gdal_array.numpy.float32), 
                                  "frequency.tif", 
                                  format="GTiff", prototype = dem_file_path)
    output = None
    output = gdal_array.SaveArray(roi.astype(gdal_array.numpy.float32), 
                                  "ROI.tif", 
                                  format="GTiff", prototype = dem_file_path)
    output = None
    
# # [2] ============================= Reservoir isolation  
    [column, row] = res_isolation(res_name, max_wl, point, boundary)      # 'res_isolation' function calling
    Max_Extent = np.copy(max_water_cluster).astype(np.float32)
    Max_ExtentN1 = expand(Max_Extent, 1)
    res_isoN1 = pick(column, row, Max_ExtentN1)
    output = gdal_array.SaveArray(res_isoN1.astype(gdal_array.numpy.float32), 
                                  "res_iso.tif", 
                                  format="GTiff", prototype = dem_file_path)
    output = None
    res_area = gdal_array.LoadFile('res_iso.tif').astype(np.float32)

    res_area[res_area<1] = np.nan
    plt.figure()
    plt.imshow(res_area, cmap='viridis')
    plt.scatter([column], [row], c='r', s=20)
    plt.title('DEM and Landsat-based reservoir isolation')
    plt.savefig("Updated_reservoir_isolation.png", dpi=600, bbox_inches='tight')
    

# # [3] =============================   DATA RESIZE
    os.chdir(Outputs_directory)
    dem_file_path = os.getcwd() + '/DEM.tif'
    res_iso_path = os.getcwd() + '/res_iso.tif'
    freq_path = os.getcwd() + '/frequency.tif'
    ref_file = os.getcwd() + '/ROI.tif'
    
    # Open the input rasters
    dem_ds = gdal.Open(dem_file_path)
    res_ds = gdal.Open(res_iso_path)
    frq_ds = gdal.Open(freq_path)
    ndwi_ds = gdal.Open(ref_file)
    
    # Get geotransform and dimensions of both rasters
    dem_transform = dem_ds.GetGeoTransform()
    dem_cols, dem_rows = dem_ds.RasterXSize, dem_ds.RasterYSize
    
    ndwi_transform = ndwi_ds.GetGeoTransform()
    ndwi_cols, ndwi_rows = ndwi_ds.RasterXSize, ndwi_ds.RasterYSize
    
    # Determine the common extent
    xmin = max(dem_transform[0], ndwi_transform[0])
    xmax = min(dem_transform[0] + dem_cols * dem_transform[1], ndwi_transform[0] + ndwi_cols * ndwi_transform[1])
    ymin = max(dem_transform[3] + dem_rows * dem_transform[5], ndwi_transform[3] + ndwi_rows * ndwi_transform[5])
    ymax = min(dem_transform[3], ndwi_transform[3])
    
    # Calculate the subset window
    xoff = int((xmin - dem_transform[0]) / dem_transform[1])
    yoff = int((dem_transform[3] - ymax) / abs(dem_transform[5]))
    width = int((xmax - xmin) / dem_transform[1])
    height = int((ymax - ymin) / abs(dem_transform[5]))
    
    # Read the data for the subset from both rasters
    dem_subset = dem_ds.GetRasterBand(1).ReadAsArray(xoff, yoff, width, height)
    res_subset = res_ds.GetRasterBand(1).ReadAsArray(xoff, yoff, width, height)
    frq_subset = frq_ds.GetRasterBand(1).ReadAsArray(xoff, yoff, width, height)
    ndwi_subset = ndwi_ds.GetRasterBand(1).ReadAsArray(xoff, yoff, width, height)
       
    # Create the subset GeoTIFF for DEM
    os.chdir(Outputs_directory)
    output_path = os.path.join("DEMclip.tif")
    driver = gdal.GetDriverByName('GTiff')
    output_dem_ds = driver.Create(output_path, width, height, 1, gdal.GDT_Float32)
    output_dem_ds.SetProjection(dem_ds.GetProjection())
    output_dem_ds.SetGeoTransform((xmin, dem_transform[1], 0, ymax, 0, dem_transform[5]))
    output_dem_ds.GetRasterBand(1).WriteArray(dem_subset)
    output_dem_ds = None

    output_path = os.path.join("ResIso.tif")
    driver = gdal.GetDriverByName('GTiff')
    output_dem_ds = driver.Create(output_path, width, height, 1, gdal.GDT_Float32)
    output_dem_ds.SetProjection(dem_ds.GetProjection())
    output_dem_ds.SetGeoTransform((xmin, dem_transform[1], 0, ymax, 0, dem_transform[5]))
    output_dem_ds.GetRasterBand(1).WriteArray(res_subset)
    output_dem_ds = None
    
    output_path = os.path.join("FreqMap.tif")
    driver = gdal.GetDriverByName('GTiff')
    output_dem_ds = driver.Create(output_path, width, height, 1, gdal.GDT_Float32)
    output_dem_ds.SetProjection(dem_ds.GetProjection())
    output_dem_ds.SetGeoTransform((xmin, dem_transform[1], 0, ymax, 0, dem_transform[5]))
    output_dem_ds.GetRasterBand(1).WriteArray(frq_subset)
    output_dem_ds = None
    
    del dem_ds   
    os.remove('DEM.tif')
    del res_ds   
    os.remove('res_iso.tif')
    del frq_ds   
    os.remove('frequency.tif')
    dem_file_path = os.getcwd() + '/DEMclip.tif'
    dem_subset[res_subset < 1] = -1
    output = gdal_array.SaveArray(dem_subset.astype(gdal_array.numpy.float32), 
                                  "DEM_ResIso.tif", 
                                  format="GTiff", prototype = dem_file_path)
    output = None
    
    #Visualization <Start>
    os.chdir(Outputs_directory)
    res_area = gdal_array.LoadFile('ResIso.tif').astype(np.float32)
    res_areaN = expand(res_area, 3)
    freq = gdal_array.LoadFile("FreqMap.tif").astype(np.float32)
    freq[res_areaN == 0] = np.nan
    cmap = plt.cm.jet
    cmap.set_bad('white')
    plt.figure()
    plt.imshow(freq, cmap=cmap)
    plt.colorbar()
    plt.title('Frequency map(%)')
    plt.savefig('Frequency.png', dpi=600, bbox_inches='tight')
    
# # [4] ===================  RESIZING all images and saving folder (folder name is "Clip") 
    output_folder = (dtdr + '/Clip')
    os.makedirs(output_folder, exist_ok=True) 
    os.chdir(RawData_directory)
    directory = os.getcwd()
    slno = 1
    img_list = [["ID", "Date", "Cloud_percentage"]]
    filtered_files = [file for file in os.listdir(directory) if "NDWI" in file]
    for filename in filtered_files:
        try:           
           print(str(slno) + '/' + str(len(filtered_files)))
           print(filename)

           ndwi_ds = gdal.Open(filename)
           ndwi_subset = ndwi_ds.GetRasterBand(1).ReadAsArray(xoff, yoff, width, height) 
           ndwiA = np.copy(ndwi_subset).astype(np.float32)
           
           # Compute cloud percentage
           ndwiCloud = np.copy(ndwiA)
           ndwiCloud[res_areaN == 0] = -1
           ndwiCloud[np.isnan(ndwiCloud) == 1] = -2
           ndwiCloud[(np.logical_and(ndwiCloud <-0.5, ndwiCloud > -1))] = -3
           cloud_percentage1 = round(len(np.where(ndwiCloud== -2)[0])/np.nansum(res_areaN == 1)*100,2).astype(np.float32)
           cloud_percentage2 = round(len(np.where(ndwiCloud== -3)[0])/np.nansum(res_areaN == 1)*100,2).astype(np.float32)
           cloud_percentage = cloud_percentage1 + cloud_percentage2
           if int(filename[8:12]) < res_built_year:
               cloud_percentage = abs(round(cloud_percentage-20,2))
           print(str(cloud_percentage) + '% cloud')
           slno += 1
      
           if cloud_percentage>80:
               print("Neglecting")
               print('-------------------------------------------------') 
               os.remove(ndwiA)

           if cloud_percentage<=80:
               print("Adding")
               ndwiC = np.copy(ndwiCloud).astype(np.float32)
               ndwiC[ndwiC <=-1] = -1
               img_list = np.append(img_list, [[filename[0:2], filename[8:18], cloud_percentage]], axis=0)   
               clipped_file = "Clipped_" + filename # Output file name
               output_image_path = os.path.join(output_folder, clipped_file)
               output = gdal_array.SaveArray(ndwiC.astype(gdal_array.numpy.float32), 
                                             output_image_path, 
                                             format="GTiff", prototype = dem_file_path)
               output = None
               print('-------------------------------------------------')

        except:
            continue  
        
    os.chdir(Outputs_directory) 
    with open('Image_List.csv',"w", newline='') as my_csv:
        csvWriter = csv.writer(my_csv)
        csvWriter.writerows(img_list)
    print(" ") 
    
    
    
#===================== EXTRA ============== EXTRA ============== EXTRA ============== EXTRA ==============



       

    
    
    


        
    
    
    
    
    
    
    
    
    
    
    


