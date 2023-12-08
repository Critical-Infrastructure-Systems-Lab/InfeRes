############+++++++++++ PLEASE MAKE SURE OF THE FOLLOWING POINTS BEFORE RUNNING THE CODE +++++++++++############
# [1]. Data are already downloaded (Satellite images and DEM)
# [2]. DEM should be in the projected coordinate system (unit: meters)
# [3]. Use the same coordinates that you have used in "data_download.py"
# [4]. All the python(scripts) files are inside ".../ReservoirExtraction/codes"
# [5]. "Number_of_tiles" = Number of Landsat tiles to cover the entire reservoir. It is recommended to download the Landsat tile covering the maximum reservoir area 
############++++++++++++++++++++++++++++++++++++++++ END ++++++++++++++++++++++++++++++++++++++++#############

# IMPORTING LIBRARY

import os 
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt
import numpy as np

def preprocessing(res_name, boundary, parent_directory, dem_file_path):    

 # STEP1 ================================================== DEM clip based on given extent    
    # clipping DEM by the bounding box
    print("Clipping DEM by the bounding box ...") 
    dem = gdal.Open(dem_file_path) 
    
    # Changing path to the desired reservoir
    os.chdir(os.getcwd() + "/Outputs")
    res_dem_file = res_name+"_DEM_GCS.tif"
    dem = gdal.Translate(res_dem_file, dem, projWin = boundary)
    dem = None
    
    #import matplotlib.pyplot as plt
    #image = gdal_array.LoadFile(res_dem_file)
    #plt.figure()
    #plt.imshow(image, cmap='jet')
    #plt.colorbar()
    
# STEP2 ================================================== GCS to UTM 
    data_folder_path= (parent_directory + res_name + '/' + res_name + '_LandsatData')
    # List all files in the folder
    all_files = os.listdir(data_folder_path)   
    ndwi_files = [file for file in all_files if "NDWI" in file]
    first_ndwi_file = ndwi_files[0]
        
    # Input and output file paths
    ref_file = (data_folder_path + '/' + first_ndwi_file)
    target_file = res_dem_file
    output_file = res_name+"_DEM_UTM.tif"     
         
    # Open the reference raster (ndwi.tif)
    ref_ds = gdal.Open(ref_file)
    
    # Get the projection and geotransform from the reference raster
    ref_proj = ref_ds.GetProjection()
    ref_transform = ref_ds.GetGeoTransform()
    
    # Open the raster to be reprojected (dem.tif)
    input_ds = gdal.Open(target_file)
    
    # Create a new raster with the same projection as the reference raster
    output_ds = gdal.Warp(output_file, input_ds, dstSRS=ref_proj, xRes=ref_transform[1], 
                          yRes=abs(ref_transform[5]), 
                          outputBounds=[ref_transform[0],
                                        ref_transform[3] + ref_transform[5] * input_ds.RasterYSize,
                                        ref_transform[0] + ref_transform[1] * input_ds.RasterXSize,
                                        ref_transform[3]],
                          format='GTiff')
    
    # Close the input and output datasets
    input_ds = None
    output_ds = None
    ref_ds = None
    
# STEP3 =========================== Clipping with maximum overlapping area between DEM and LANDSAT-image   
    # Input raster files
    dem_path = output_file 
    ndwi_path = ref_file
    
    # Output subset folder
    output_folder = (parent_directory + res_name + '/' + 'Clip')
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the input rasters
    dem_ds = gdal.Open(dem_path)
    ndwi_ds = gdal.Open(ndwi_path)
    
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
    ndwi_subset = ndwi_ds.GetRasterBand(1).ReadAsArray(xoff, yoff, width, height)
       
    # Create the subset GeoTIFF for DEM
    output_dem_path = os.path.join(res_name+"_DEM_UTM_CLIP.tif")
    driver = gdal.GetDriverByName('GTiff')
    output_dem_ds = driver.Create(output_dem_path, width, height, 1, gdal.GDT_Float32)
    output_dem_ds.SetProjection(dem_ds.GetProjection())
    output_dem_ds.SetGeoTransform((xmin, dem_transform[1], 0, ymax, 0, dem_transform[5]))
    output_dem_ds.GetRasterBand(1).WriteArray(dem_subset)
    output_dem_ds = None
    
    # Create the subset GeoTIFF for NDWI
    output_ndwi_path = os.path.join(res_name+"_NDWI_UTM_CLIP.tif")
    output_ndwi_ds = driver.Create(output_ndwi_path, width, height, 1, gdal.GDT_Float32)
    output_ndwi_ds.SetProjection(ndwi_ds.GetProjection())
    output_ndwi_ds.SetGeoTransform((xmin, ndwi_transform[1], 0, ymax, 0, ndwi_transform[5]))
    output_ndwi_ds.GetRasterBand(1).WriteArray(ndwi_subset)
    output_ndwi_ds = None
    
# STEP4 ======================= Clipping all Landsat Images and saving in a new folder (folder name is "Clip")  
        
    os.chdir(parent_directory + res_name + '/' + res_name + '_LandsatData')
    data_directory = os.getcwd()
    clip_count = 0
    for filename in os.listdir(data_directory):
        try:
           ndwi_ds = gdal.Open(filename)
           ndwi_subset = ndwi_ds.GetRasterBand(1).ReadAsArray(xoff, yoff, width, height)       
    
           # Create the subset GeoTIFF for all Landsat images
           if 'BQA' in filename:
               clipped_file = "Clipped_" + filename[:19] + '_' + res_name + filename[19:] # Output file name
           if 'NDWI' in filename:
               clipped_file = "Clipped_" + filename[:20] + '_' + res_name + filename[20:] # Output file name
           
           output_image_path = os.path.join(output_folder, clipped_file)
           output_image_ds = driver.Create(output_image_path, width, height, 1, gdal.GDT_Float32)
           output_image_ds.SetProjection(ndwi_ds.GetProjection())
           output_image_ds.SetGeoTransform((xmin, ndwi_transform[1], 0, ymax, 0, ndwi_transform[5]))
           output_image_ds.GetRasterBand(1).WriteArray(ndwi_subset)
           output_image_ds = None
           clip_count += 1
           print(clip_count)
           print(filename)
        except:
            continue      
    
    os.chdir(parent_directory + res_name + '/Outputs')
    #------------------ Visualization <Start>
    plt.figure()
    dem = gdal_array.LoadFile(res_name+"_DEM_UTM_CLIP.tif").astype(np.float32)
    dem[dem == 32767] = np.nan
    plt.imshow(dem, cmap='jet')
    plt.colorbar()
    plt.title('Clipped DEM (UTM)')
    plt.savefig(res_name+"_DEM.png", dpi=600, bbox_inches='tight')
    #------------------ Visualization <End>

    os.remove(res_name+"_NDWI_UTM_CLIP.tif")
       

    
    
#================================= EXTRA (UTM to GCS)  ===================================        
    # # Input and output file paths
    # input_file = (data_folder_path + '/' + first_ndwi_file)
    # output_file = "H:/My Drive/NUSproject/ReservoirExtraction/Reservoirs/Xiaowan/Outputs/XiaowanDEM.tif"
    
    # # Define the target coordinate system (GCS)
    # target_crs = 'EPSG:4326'  # Example: WGS 84
    
    # # Open the input raster file
    # input_ds = gdal.Open(input_file)
    
    # # Get the input raster's projection and geotransform
    # input_proj = input_ds.GetProjection()
    # input_transform = input_ds.GetGeoTransform()
    
    # # Create a SpatialReference object for the target coordinate system
    # target_srs = osr.SpatialReference()
    # target_srs.SetFromUserInput(target_crs)
    
    # # Create a new raster in the target coordinate system
    # output_ds = gdal.Warp(output_file, input_ds, dstSRS=target_srs.ExportToWkt(), format='GTiff')
    
    # # Close the input and output datasets
    # input_ds = None
    # output_ds = None
    
    
### STEP5 ================================================== Remove bad landsat(5 and 8) images from clipped database 
   # data_folder_path= (parent_directory + res_name + '/' + res_name + '_LandsatData')
   # os.chdir(data_folder_path)
   # all_files = os.listdir(data_folder_path)  
   # filtered_filesL8 = [file for file in all_files if "Clipped_LC08_NDWI" in file]   
   # filtered_filesL5 = [file for file in all_files if "Clipped_LT05_NDWI" in file] 
   # ndwi_files = filtered_filesL8 + filtered_filesL5
   # files_removed = 0
   # count = 1
   # for filename in ndwi_files:
   #     print(count)
   #     print(filename)
   #     try:
   #        image = gdal_array.LoadFile(filename).astype(np.float32)
   #        plt.figure()
   #        plt.imshow(image, cmap='jet')
   #        plt.title(str(count))
   #        plt.colorbar()

   #         nan_row = np.zeros((np.size(image,0),2))
   #         p20 = int(0.2*np.size(image,0))
   #         p80 = int(0.8*np.size(image,0))
   #         for i in range(p20, p80):
   #             irow = image[i,:]
   #             pix = len(sum(np.where(np.isnan(irow))))
   #             nan_area_percentage = pix/np.size(image,1)*100
   #             nan_row[i,0] = nan_area_percentage
          
   #         bad_data_threshold = len(sum(np.where(nan_row[:,0]>1)))/np.size(image,0)*100
   #         if bad_data_threshold > 50: 
   #             print('Removed')
   #             BQA_filename = filename.replace('NDWI', 'BQA')
   #             os.remove(filename)
   #             os.remove(BQA_filename)
   #             files_removed +=1
              
   #        image = None
   #        count +=1
   #     except:
   #         continue 
       
   # print('files_removed=', str(files_removed))


        
    
    
    
    
    
    
    
    
    
    
    


