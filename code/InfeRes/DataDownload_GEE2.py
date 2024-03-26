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
# [1]============= Functions ============== Functions =============== Functions =============
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
# [3]=========== Scene-based satellite data download ========= Scene-based satellite data download ============
############################################################################################################### 
    
# ================================================  Landsat-9-OLI 
print("-----Data download start-----")       
start_year = res_built_year-5
if start_year<1985:
    start_year = 1985
count = 1               
for year in range(start_year, 2024):
    
    for month in range(1, 13):  
        days_in_month = calendar.monthrange(year, month)[1]
        
        for period_start_day in range(1, days_in_month, 11):
            print(count)
            start_date = ee.Date.fromYMD(year, month, period_start_day).format('YYYY-MM-dd').getInfo()
            period_end_day = min(period_start_day + 10, days_in_month)
            end_date = ee.Date.fromYMD(year, month, period_end_day).format('YYYY-MM-dd').getInfo()
# ========================            
            L9 = (
                  ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
                    .filterBounds(boundary)
                    .filterDate(start_date, end_date)
                    .map(apply_scale_factors)
                    .map(calculate_ndwi98)
                    .map(mask_qa_pixels98)
                )
            # Get the composite of the best pixels (cloud-free) based on NDWI
            L9num_images = L9.size().getInfo()
            if L9num_images > 0:
                composite = L9.qualityMosaic('NDWI')
                img = composite.select('NDWI')
                L9ndwi = img.clip(boundary)
            else:
                L9ndwi = L9.first().multiply(0)
                L9ndwi = L9ndwi.clip(boundary)
            
# ========================            
            L8 = (
                  ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                    .filterBounds(boundary)
                    .filterDate(start_date, end_date)
                    .map(apply_scale_factors)
                    .map(calculate_ndwi98)
                    .map(mask_qa_pixels98)
                )
            # Get the composite of the best pixels (cloud-free) based on NDWI
            L8num_images = L8.size().getInfo()
            if L8num_images > 0:
                composite = L8.qualityMosaic('NDWI')
                img = composite.select('NDWI')
                L8ndwi = img.clip(boundary)
            else:
                L8ndwi = L8.first().multiply(0)
                L8ndwi = L8ndwi.clip(boundary)
                
# ========================           
            L7 = (
                  ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
                    .filterBounds(boundary)
                    .filterDate(start_date, end_date)
                    .map(apply_scale_factors)
                    .map(calculate_ndwi75)
                    .map(mask_qa_pixels75)
                )
            
            # Get the composite of the best pixels (cloud-free) based on NDWI
            L7num_images = L7.size().getInfo()
            if L7num_images > 0:
                composite = L7.qualityMosaic('NDWI')
                img = composite.select('NDWI')
                L7ndwi = img.clip(boundary)
            else:
                L7ndwi = L7.first().multiply(0)
                L7ndwi = L7ndwi.clip(boundary)
                
# ========================                
            L5 = (
                  ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
                    .filterBounds(boundary)
                    .filterDate(start_date, end_date)
                    .map(apply_scale_factors)
                    .map(calculate_ndwi75)
                    .map(mask_qa_pixels75)
                )
            # Get the composite of the best pixels (cloud-free) based on NDWI
            L5num_images = L5.size().getInfo()
            if L5num_images > 0:
                composite = L5.qualityMosaic('NDWI')
                img = composite.select('NDWI')
                L5ndwi = img.clip(boundary)
            else:
                L5ndwi = L5.first().multiply(0)
                L5ndwi = L5ndwi.clip(boundary)
                
# ========================
            num_images =  L9.size().getInfo() +  L8.size().getInfo() +  L7.size().getInfo() + L5.size().getInfo()
            print("Total Landsat images available between " + start_date + ' and ' + end_date + ':', num_images)
            
            if num_images > 0: 
                images = [L9ndwi, L8ndwi, L7ndwi, L5ndwi]
                num_images = [L9num_images, L8num_images, L7num_images, L5num_images]
                Lcollection = ee.ImageCollection([img for img, num in zip(images, num_images) if num > 0])
                composite = Lcollection.reduce(ee.Reducer.mean())
                ndwi_clip = composite.clip(boundary)

                export_params1 = {
                  'image': ndwi_clip,
                  'folder': RawData_directory,  
                  'scale': 30,           
                  'description': f'L0_NDWI_{year}-{month:02d}-{period_start_day:02d}',
                  'region': boundary,
                  'fileFormat': 'GeoTIFF',
                  'formatOptions': {
                  'cloudOptimized': True,
                  }
                }  
                # Export the image as a cloud-optimized GeoTIFF to Google Drive
                task = ee.batch.Export.image.toDrive(**export_params1)
                task.start()
                print('Exporting')
                
                # i=0
                # while task.status()['state'] in ['READY', 'RUNNING']:
                #     print(task.status())
                #     print('time:'+ str(i) + 'seconds')
                #     i+=30
                #     time.sleep(30)
                # print('Export completed:', task.status())
                
# ========================                
            S2 = (
                  ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                    .filterBounds(boundary)
                    .filterDate(start_date, end_date)  
                    .map(mask_clouds_S2)
                    .map(calculate_ndwi_S2)
                )
            
            num_images = S2.size().getInfo()
            print("Sentinal images available between " + start_date + ' and ' + end_date + ':', num_images)
            
            if num_images > 0:
                composite = S2.qualityMosaic('NDWI')
                img = composite.select('NDWI')
                ndwi_clip = img.clip(boundary)
                
                export_params1 = {
                  'image': ndwi_clip,
                  'folder': RawData_directory,  
                  'scale': 30,           
                  'description': f'S2_NDWI_{year}-{month:02d}-{period_start_day:02d}',
                  'region': boundary,
                  'fileFormat': 'GeoTIFF',
                  'formatOptions': {
                  'cloudOptimized': True,
                  }
                }  
                # Export the image as a cloud-optimized GeoTIFF to Google Drive
                task = ee.batch.Export.image.toDrive(**export_params1)
                task.start()
                print('Exporting')
                
            print("-----------------------------------------------------")
            count += 1
            
print("-----Data download finish-----") 
                     

# ============================ In case of emargency if you want to cancel the running tasks then execute the following
    
# all_tasks = ee.data.getTaskList()
# print(len(all_tasks))
# active_tasks = [task for task in all_tasks if task['state'] == 'RUNNING' or task['state'] == 'READY']
# print(len(active_tasks))

# #Loop through the list of active tasks and cancel each task
# for task in active_tasks:
#     task_id = task['id']
#     ee.data.cancelTask(task_id)
#     print(f"Cancelled task: {task_id}")