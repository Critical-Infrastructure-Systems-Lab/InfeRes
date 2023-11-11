############+++++++++++ PLEASE MAKE SURE OF THE FOLLOWING POINTS BEFORE RUNNING THE CODE +++++++++++############
# [1]. Get Authentication from Google Earth Engine before running the code [i.e. ee.Authenticate()] and then comment it (See bwlow)
#------------  import ee
#------------  ee.Authenticate()    #Run it only one time

# [2]. Get the Landsat (or any satellite) image collection ID from (https://developers.google.com/earth-engine/datasets/catalog/landsat)
# [3]. Use a smaller bounding box and shorter time-period to test the downloading
############++++++++++++++++++++++++++++++++++++++++ END ++++++++++++++++++++++++++++++++++++++++#############


import os
import time
import ee
#ee.Authenticate()
ee.Initialize()

#============================================ FUNCTION DEFINITIONS
# Define a function to track the progress of a download task
def track_task_progress(task):
    while task.active():
        status = task.status()
        print(f"Task Status: {status['state']}")
        time.sleep(10)       
    print("Download Completed!")
    
# Define a function to download the satellite data from Google Earth Engine
def get_landsat_images(dataFolder, Satellite, row, path, point_coordinates, boundary_coordinates, collection_id, start_data, end_date):
    # Import the Landsat 8 TOA image collection
    point = ee.Geometry.Point(point_coordinates)
    boundary = ee.Geometry.Rectangle(boundary_coordinates)
    l8 = ee.ImageCollection(collection_id) \
            .filter(ee.Filter.eq('WRS_ROW', row)) \
            .filter(ee.Filter.eq('WRS_PATH', path))

    # Filter the collection for the year (# Get the least cloudy image)
    l8images = l8.filterDate(start_data, end_date) \
                   .filterBounds(point)

    # Define a function to clip an image to a given geometry
    def clip_image_to_geometry(image):
        return image.clip(boundary)

    # Clip the image collection to the geometry
    l8images_clip = l8images.map(clip_image_to_geometry)

    # Get the number of images
    num_images = l8images_clip.size().getInfo()

    print("Total number of Landsat images:", num_images)

    #================================================ [Export Landsat Images]

    slno = l8images_clip.toList(l8images_clip.size())
    count = 1
    for i in range(num_images):  # range(num_images)
        print(count)
        selected_image = ee.Image(slno.get(i))
        
        #Change the bands as per Landsat collection [Example: B3 = Green, B5 = NIR is for Landsat8]   
        if Satellite=='L8':
            ndwi = selected_image.normalizedDifference(['B3', 'B5'])
            #Thermal = selected_image.select('B10')
        if (Satellite=='L5'):
            ndwi = selected_image.normalizedDifference(['B2', 'B4'])
            #Thermal = selected_image.select('B6')
        if (Satellite=='L7'):
            ndwi = selected_image.normalizedDifference(['B2', 'B4'])
            #Thermal = selected_image.select('B6_VCID_1')            
            
        BQA = selected_image.select('QA_PIXEL')        
    
        # Get the date of acquisition
        acquisition_date = selected_image.date()
    
        # Convert the date to a string format
        date_string = acquisition_date.format('YYYY-MM-dd')
    
        # Print the date
        print("Acquisition date:", date_string.getInfo())
    
        # Retrieve the projection information from a band of the original image
        projection = ndwi.projection().getInfo()
            
        # Define the export parameters for a cloud-optimized GeoTIFF
        export_params1 = {
            'image': ndwi,
            'folder': dataFolder,  # Optional: Specify a folder in your Google Drive to save the exported image
            'scale': 30,              # Optional: Specify the scale/resolution of the exported image in meters
            'description': collection_id[8:12] + '_NDWI_' + date_string.getInfo(),
            'crs': projection['crs'],
            'crs_transform': projection['transform'],
            'region': boundary['coordinates'],
            'fileFormat': 'GeoTIFF',
            'formatOptions': {
                'cloudOptimized': True,
                }
            }
        # Export the image as a cloud-optimized GeoTIFF to Google Drive
        task = ee.batch.Export.image.toDrive(**export_params1)
        task.start()
        
        
        # Define the export parameters for a cloud-optimized GeoTIFF
        export_params2 = {
            'image': BQA,
            'folder': dataFolder,      # Optional: Specify a folder in your Google Drive to save the exported image
            'scale': 30,                  # Optional: Spatial Resolution
            'description': collection_id[8:12] + '_BQA_' + date_string.getInfo(),
            'crs': projection['crs'],
            'crs_transform': projection['transform'],
            'region': boundary['coordinates'],
            'fileFormat': 'GeoTIFF',
            'formatOptions': {
                'cloudOptimized': True,
                }
            }
        # Export the image as a cloud-optimized GeoTIFF to Google Drive
        task = ee.batch.Export.image.toDrive(**export_params2)
        task.start()
        #track_task_progress(task)
        count+=1

###################################  End of Function Definition  #######################################


if __name__ == "__main__":
    
    # Set to the current working directory
    parent_directory = "H:/My Drive/NUSproject/ReservoirExtraction/Reservoirs/"    
    os.chdir(parent_directory)
    # Name of the reservoir 
    res_name = "NamOu2"                                                        
    os.makedirs(res_name, exist_ok=True)                  
    os.chdir(parent_directory + res_name)
    # Create a new folder within the working directory to download the data
    dataFolder = res_name + '_LandsatData'
    os.makedirs(dataFolder, exist_ok=True)
    print(res_name)
    
    #====================================>> USER INPUT PARAMETERS (Landsat-8 Image Specifications)
    row = 46
    path = 129
    Satellite = 'L8'
    # A point within the reservoir [longitude, latitude]
    point_coordinates = [102.4510, 20.3926]
    # Example coordinates [longitude, latitude]
    boundary_coordinates = [102.435, 20.640, 102.688, 20.388]
    collection_id = "LANDSAT/LC08/C02/T1_TOA"      
    start_data = '2015-06-01'
    end_date = '2022-12-31'
    print("-----Landsat-8 data download start-----")
    get_landsat_images(dataFolder, Satellite, row, path, point_coordinates, boundary_coordinates, collection_id, start_data, end_date)
    print("Congratulations...all Landsat-8 files have successfully downloaded!")
    print(res_name)
    
    #====================================>> USER INPUT PARAMETERS (Landsat-7 Image Specifications)
    Satellite = 'L7'
    collection_id = "LANDSAT/LE07/C02/T1_TOA"      
    start_data = '2015-06-01'
    end_date = '2022-12-31'
    print(res_name)
    print("-----Landsat-7 data download start-----")
    get_landsat_images(dataFolder, Satellite, row, path, point_coordinates, boundary_coordinates, collection_id, start_data, end_date)
    print("Congratulations...all Landsat-7 files have successfully downloaded!")
    print(res_name)
    
    #====================================>> USER INPUT PARAMETERS (Landsat-5 Image Specifications)
    Satellite = 'L5'
    collection_id = "LANDSAT/LT05/C02/T1_TOA"      
    start_data = '2015-06-01'
    end_date = '2022-12-31'
    print(res_name)
    print("-----Landsat-5 data download start-----")
    get_landsat_images(dataFolder, Satellite, row, path, point_coordinates, boundary_coordinates, collection_id, start_data, end_date)
    print("Congratulations...all Landsat-5 files have successfully downloaded!")
    print(res_name)



# ==== In case of emargency if you want to cancel the running tasks then execute the following
    
# Get a list of all tasks
# all_tasks = ee.data.getTaskList()

# # Filter out the active tasks
# active_tasks = [task for task in all_tasks if task['state'] == 'RUNNING' or task['state'] == 'READY']

# #Loop through the list of active tasks and cancel each task
# for task in all_tasks:
#     task_id = task['id']
#     ee.data.cancelTask(task_id)
#     print(f"Cancelled task: {task_id}")




