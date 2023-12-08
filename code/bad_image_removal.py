# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 23:56:16 2023

@author: ss.mahto
"""

 ### EXTRA (run it manually) ================================================== Remove bad landsat(5 and 8) images from clipped database 
import os 
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":  
    
  parent_directory = "G:/My Drive/NUSproject/ReservoirExtraction/Reservoirs/"
  os.chdir(parent_directory)
  res_name = "Srinagarind"
  data_folder_path= (parent_directory + res_name + '/' + res_name + '_LandsatData')
  os.chdir(data_folder_path)
  all_files = os.listdir(data_folder_path)  
  filtered_filesL8 = [file for file in all_files if "LC08_NDWI" in file]   
  filtered_filesL5 = [file for file in all_files if "LT05_NDWI" in file] 
  ndwi_files = filtered_filesL8 + filtered_filesL5 
  count = 1
  print(res_name)
  for filename in ndwi_files:
      print(count)
      print(filename)
      try:
         image = gdal_array.LoadFile(filename).astype(np.float32)
         plt.figure()
         plt.imshow(image, cmap='jet')
         plt.title(str(count))
         plt.colorbar()    
         image = None
         count +=1
      except:
          continue 
      
  ##=========================== Set it manually  ================================================= IMPORTANT  
  pos = [201]  
  new_pos = [index - 1 for index in pos]
  ndwi_delete = [ndwi_files[i] for i in new_pos]
  files_removed = 0 
  for filename in ndwi_delete:
      print(filename)
      print('Removed')
      BQA_filename = filename.replace('NDWI', 'BQA')
      os.remove(filename)
      os.remove(BQA_filename)
      files_removed +=1    
  print('files_removed=', str(files_removed))