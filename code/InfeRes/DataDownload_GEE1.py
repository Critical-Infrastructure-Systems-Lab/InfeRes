############################################################################################################### 
# [0]============= Initialization ============== Initialization =============== Initialization =============
###############################################################################################################

import os
parent_directory = "G:/My Drive/NUSproject/ReservoirExtraction/InfeRes/"   # Path/to/your/google-drive/InfeRes/
os.chdir(parent_directory)
os.chdir("..")
os.chdir("Reservoirs")
import time
import ee
import calendar
from datetime import datetime
#ee.Authenticate()
ee.Initialize()
import pandas as pd

df = pd.read_csv('inputs_GEE.csv', parse_dates=True)
res_name = df.Name[0] 
res_built_year = df.Year[0]
point = [float(value) for value in df.Point[0].split(',')]
point = ee.Geometry.Point(point)
boundary = [float(value) for value in df.Boundary[0].split(',')] 
boundary = ee.Geometry.Rectangle(boundary)                                              
print('Name of the reservoir: ' + res_name)                                              
                                                                                                                
os.makedirs(res_name, exist_ok=True)
os.chdir(res_name)
os.makedirs(res_name + '_Supporting', exist_ok=True)
os.makedirs(res_name + '_RawData', exist_ok=True)
Supporting_directory = res_name + '_Supporting'
RawData_directory = res_name + '_RawData'

############################################################################################################### 
# [1]============= DEM download ============== DEM download =============== DEM download =============
############################################################################################################### 
dataset = ee.Image('USGS/SRTMGL1_003')
elevation = dataset.select('elevation')
dem_clip = elevation.clip(boundary)

export_params1 = {
  'image': dem_clip,
  'folder': Supporting_directory, 
  'scale': 30,           
  'description': 'DEM',
  'region': boundary,
  'fileFormat': 'GeoTIFF',
  'formatOptions': {
  'cloudOptimized': True,
  }
}  

task = ee.batch.Export.image.toDrive(**export_params1)
task.start()
print("Exporting DEM")


############################################################################################################### 
# [2]============= Functions ============== Functions =============== Functions =============
############################################################################################################### 
def mask_qa_pixels98(image):
    qa = image.select('QA_PIXEL')
    mask = qa.lt(22280)
    return image.updateMask(mask)

def mask_qa_pixels75(image):
    qa = image.select('QA_PIXEL')
    mask = qa.lt(5896)
    return image.updateMask(mask)

def apply_scale_factors(image):
    optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)

def calculate_ndwi98(image):
    ndwi = image.normalizedDifference(['SR_B3', 'SR_B5'])    
    return image.addBands(ndwi.rename('NDWI'))

def calculate_ndwi75(image):
    ndwi = image.normalizedDifference(['SR_B2', 'SR_B4'])    
    return image.addBands(ndwi.rename('NDWI'))

def mask_clouds_S2(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = (qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0)))
    return image.updateMask(mask).divide(10000)

def calculate_ndwi_S2(image):
    ndwi = image.normalizedDifference(['B3', 'B8'])  
    return image.addBands(ndwi.rename('NDWI'))


############################################################################################################### 
# [3]============= Composite ============== Composite =============== Composite =============
############################################################################################################### 
# LANDSAT-based (ndwi calculation -> then -> cloud mask)

col = (
      ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(boundary)
        .filterDate('2014-01-01', '2023-12-31')
        .filter(ee.Filter.lt('CLOUD_COVER', 80))
        .map(apply_scale_factors)
        .map(calculate_ndwi98)
        .map(mask_qa_pixels98)
    )

composite = col.qualityMosaic('NDWI')
img = composite.select('NDWI')
ndwi_clipL = img.clip(boundary)

# SENTINAL-based (cloud mask -> then -> ndwi calculation)
col_S2 = (
  ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(boundary) 
    .filterDate('2015-01-01', '2023-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
    .map(mask_clouds_S2)
    .map(calculate_ndwi_S2)
)

# Get the composite of the best pixels (cloud-free) based on NDWI
composite = col_S2.qualityMosaic('NDWI')
img = composite.select('NDWI')
ndwi_clipS = img.clip(boundary)

Max_extent = ndwi_clipL.max(ndwi_clipS)

# Data export
export_params1 = {
  'image': Max_extent,
  'folder': Supporting_directory,  
  'scale': 30,           
  'description': 'MaxExtent',
  'region': boundary,
  'fileFormat': 'GeoTIFF',
  'formatOptions': {
  'cloudOptimized': True,
  }
}  
# Export the image as a cloud-optimized GeoTIFF to Google Drive
task = ee.batch.Export.image.toDrive(**export_params1)
task.start()
print("Exporting maximum water extent")

# i=0
# while task.status()['state'] in ['READY', 'RUNNING']:
#     print(task.status())
#     print('time:'+ str(i) + 'seconds')
#     i+=30
#     time.sleep(30)
# print('Export completed:', task.status())


############################################################################################################### 
# [4]============= Frequency map ============== Frequency map =============== Frequency map =============
###############################################################################################################       

start_year = res_built_year-5
all_ndwi_images = [] 
all_ndwi_imagesS2 = [] 
frequencyS = []
frequencyL = []
n=1   
m=1

def Satellite(col):
    composite = col.qualityMosaic('NDWI')
    img = composite.select('NDWI')
    ndwi_clip = img.clip(boundary)

    water_threshold=0
    water_pixels = ndwi_clip.gt(water_threshold).reduceRegion(
        reducer=ee.Reducer.sum(), 
        geometry=boundary, 
        scale=30,
        bestEffort=True
    ).get('NDWI').getInfo()
    #print("Number of water pixels:", int(water_pixels))#*9/10000, "km2")
    
    nan_mask = ndwi_clip.mask().Not()
    cloud_pixels = nan_mask.reduceRegion(
        reducer=ee.Reducer.sum(), 
        geometry=boundary, 
        scale=30,
        bestEffort=True
    ).get('NDWI').getInfo()
    #print("Number of cloud pixels:", int(cloud_pixels))
    
    ndwi_mask = ndwi_clip.mask()
    ndwi_pixels = ndwi_mask.reduceRegion(
        reducer=ee.Reducer.sum(), 
        geometry=boundary, 
        scale=30,
        bestEffort=True
    ).get('NDWI').getInfo()
    #print("Number of non-cloud pixels:", int(ndwi_pixels))
    
    total_pixels = int(cloud_pixels + ndwi_pixels)
    cloud_masked_percentage = round((cloud_pixels / total_pixels) * 100,2)
    #print("Cloud percentage:", cloud_masked_percentage)
    return cloud_masked_percentage, ndwi_clip


if start_year<1985:
    start_year = 1985

        
for year in range(start_year, 2024):
    for month in range(1, 13):
        period_start_day = 1
        period_end_day = calendar.monthrange(year, month)[1]
        start_date = ee.Date.fromYMD(year, month, period_start_day).format('YYYY-MM-dd').getInfo()
        end_date = ee.Date.fromYMD(year, month, period_end_day).format('YYYY-MM-dd').getInfo()

# Landsat9 >>>>>>>>>>>>>>>>
        L9 = (
              ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
                .filterBounds(boundary)
                .filterDate(start_date, end_date)
                .map(apply_scale_factors)
                .map(calculate_ndwi98)
                .map(mask_qa_pixels98)
            )
        num_imagesL9 = L9.size().getInfo()
        print("Landsat-9 images available between " + start_date + ' and ' + end_date + ':', num_imagesL9)
        
        if num_imagesL9 > 0:
            cloud_masked_percentageL9, ndwi_clipL9  = Satellite(L9)
            print("Cloud percentage:", cloud_masked_percentageL9)
        
            if cloud_masked_percentageL9 < 15:          
                all_ndwi_images.append(ndwi_clipL9)
                binary_imageL9 = ndwi_clipL9.gt(0.01)
                frequencyL.append(binary_imageL9) 
                n += 1
        
# Landsat8 >>>>>>>>>>>>>>>>
        L8 = (
              ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .filterBounds(boundary)
                .filterDate(start_date, end_date)
                .map(apply_scale_factors)
                .map(calculate_ndwi98)
                .map(mask_qa_pixels98)
            )
        num_imagesL8 = L8.size().getInfo()
        print("Landsat-8 images available between " + start_date + ' and ' + end_date + ':', num_imagesL8)
        
        if num_imagesL8 > 0:
            cloud_masked_percentageL8, ndwi_clipL8  = Satellite(L8)
            print("Cloud percentage:", cloud_masked_percentageL8)
            
            if cloud_masked_percentageL8 < 15:          
                all_ndwi_images.append(ndwi_clipL8)
                binary_imageL8 = ndwi_clipL8.gt(0.01)
                frequencyL.append(binary_imageL8) 
                n += 1      
                
# Landsat7 >>>>>>>>>>>>>>>>

        L7 = (
              ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
                .filterBounds(boundary)
                .filterDate(start_date, end_date)
                .map(apply_scale_factors)
                .map(calculate_ndwi75)
                .map(mask_qa_pixels75)
            )
        # period_end_day = min(period_start_day + 14, days_in_month)
        # end_date = ee.Date.fromYMD(year, month, period_end_day).format('YYYY-MM-dd').getInfo()
        num_imagesL7 = L7.size().getInfo()
        print("Landsat-7 images available between " + start_date + ' and ' + end_date + ':', num_imagesL7)
        end_date_str = datetime.strptime(end_date, '%Y-%m-%d')
        comparison_date = datetime.strptime('2003-05-31', '%Y-%m-%d')
        
        if end_date_str <= comparison_date:
            if num_imagesL7 > 0:
                cloud_masked_percentageL7, ndwi_clipL7  = Satellite(L7)
                print("Cloud percentage:", cloud_masked_percentageL7)
            
                if cloud_masked_percentageL7 < 15:          
                    all_ndwi_images.append(ndwi_clipL7)
                    binary_imageL7 = ndwi_clipL7.gt(0.01)
                    frequencyL.append(binary_imageL7) 
                    n += 1
                
# Landsat5 >>>>>>>>>>>>>>>>
        L5 = (
              ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
                .filterBounds(boundary)
                .filterDate(start_date, end_date)
                .map(apply_scale_factors)
                .map(calculate_ndwi75)
                .map(mask_qa_pixels75)
            )
        num_imagesL5 = L5.size().getInfo()
        print("Landsat-5 images available between " + start_date + ' and ' + end_date + ':', num_imagesL5)
        
        if num_imagesL5 > 0:
            cloud_masked_percentageL5, ndwi_clipL5  = Satellite(L5)
            print("Cloud percentage:", cloud_masked_percentageL5)
        
            if cloud_masked_percentageL5 < 15:          
                all_ndwi_images.append(ndwi_clipL5)
                binary_imageL5 = ndwi_clipL5.gt(0.01)
                frequencyL.append(binary_imageL5) 
                n += 1   

# Sentinal-2 >>>>>>>>>>>>>>>>
        S2 = (
              ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(boundary)
                .filterDate(start_date, end_date)
                .map(mask_clouds_S2)
                .map(calculate_ndwi_S2)
            )
        num_imagesS2 = S2.size().getInfo()
        print("Sentinal-2 images available between " + start_date + ' and ' + end_date + ':', num_imagesS2)
        
        if num_imagesS2 > 0:
            S2composite = S2.qualityMosaic('NDWI')
            S2img = S2composite.select('NDWI')
            S2ndwi_clip = S2img.clip(boundary)
            #S2ndwi_clip_resampled = S2ndwi_clip.resample('bilinear').reproject(crs=S2ndwi_clip.projection(), scale=30)

            water_threshold=0.1
            water_pixels = S2ndwi_clip.gt(water_threshold).reduceRegion(
                reducer=ee.Reducer.sum(), 
                geometry=boundary, 
                scale=30,
                bestEffort=True
            ).get('NDWI').getInfo()
            #print("Number of water pixels:", int(water_pixels))#*9/10000, "km2")
            
            nan_mask = S2ndwi_clip.mask().Not()
            cloud_pixels = nan_mask.reduceRegion(
                reducer=ee.Reducer.sum(), 
                geometry=boundary, 
                scale=30,
                bestEffort=True
            ).get('NDWI').getInfo()
            #print("Number of cloud pixels:", int(cloud_pixels))
            
            ndwi_mask = S2ndwi_clip.mask()
            ndwi_pixels = ndwi_mask.reduceRegion(
                reducer=ee.Reducer.sum(), 
                geometry=boundary, 
                scale=30,
                bestEffort=True 
            ).get('NDWI').getInfo()
            #print("Number of non-cloud pixels:", int(ndwi_pixels))
            
            total_pixels = int(cloud_pixels + ndwi_pixels)
            cloud_masked_percentageS2 = round((cloud_pixels / total_pixels) * 100,2)
            print("Cloud percentage:", cloud_masked_percentageS2)
            
            if cloud_masked_percentageS2 < 15:          
               all_ndwi_imagesS2.append(S2ndwi_clip)
               binary_imageS2 = S2ndwi_clip.gt(0.1)
               frequencyS.append(binary_imageS2)
               m += 1
                
print("Number of Landsat images for creating frequency map:", len(all_ndwi_images))
print("Number of Sentinal images for creating frequency map:", len(all_ndwi_imagesS2))

frequencyL_cat = ee.Image.cat(frequencyL)
summed_image = frequencyL_cat.reduce(ee.Reducer.sum())
frequency_imageL = summed_image.divide(len(all_ndwi_images)).multiply(100)

frequencyS_cat = ee.Image.cat(frequencyS)
summed_image = frequencyS_cat.reduce(ee.Reducer.sum())
frequency_imageS = summed_image.divide(len(all_ndwi_imagesS2)).multiply(100)

frequency = frequency_imageL.add(frequency_imageS).divide(2)

# Data export >>>>>>>>>>>>>>>>
export_params1 = {
  'image': frequency,
  'folder': Supporting_directory, 
  'scale': 30,           
  'description': 'Frequency',
  'region': boundary,
  'fileFormat': 'GeoTIFF',
  'formatOptions': {
  'cloudOptimized': True,
  }
}  
# Export the image as a cloud-optimized GeoTIFF to Google Drive
task = ee.batch.Export.image.toDrive(**export_params1)
task.start()




#================================ Extra (if you want to track the status)
# # Data Downloading Status
# print('Exporting image to Google Drive...')
# i=0
# while task.status()['state'] in ['READY', 'RUNNING']:
#     print(task.status())
#     print('time:'+ str(i) + 'seconds')
#     i+=30
#     time.sleep(30)
# print('Export completed:', task.status())


# ============================ In case of emargency if you want to cancel the running tasks then execute the following
    
# Get a list of all tasks
# all_tasks = ee.data.getTaskList()

# # Filter out the active tasks
# active_tasks = [task for task in all_tasks if task['state'] == 'RUNNING' or task['state'] == 'READY']

# #Loop through the list of active tasks and cancel each task
# for task in active_tasks:
#     task_id = task['id']
#     ee.data.cancelTask(task_id)
#     print(f"Cancelled task: {task_id}")

# ============================ 
    