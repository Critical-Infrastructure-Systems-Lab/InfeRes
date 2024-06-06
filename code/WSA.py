# IMPORTING LIBRARY
import os
import csv
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from osgeo import gdal_array
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
#from skimage.morphology import opening, disk

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

          
def wsa_latest(res_name):
# # [5] ============================  DATA CLIPPING and CREATING FREQUENCY MAP
    dtdr = os.getcwd()
    Outputs_directory = dtdr + '/' + 'Outputs'
    os.chdir(Outputs_directory)
    res_area = gdal_array.LoadFile("ResIso.tif").astype(np.float32)
    ZONE = gdal_array.LoadFile("FreqMap.tif").astype(np.float32)
    ZONE = np.ceil(ZONE/2)
    res_areaN = expand(res_area, 3)
    ZONE[res_areaN==0] = 0
    Clip_directory = dtdr + '/Clip'
    os.chdir(Clip_directory)
    directory = os.getcwd()
    filtered_files = [file for file in os.listdir(directory) if "Clipped_" in file] #and ("2019" in file)
    no_zones = 50     #it can be user defined
    results = [["ID", "Date", "Cloud_percentage", "Quality", "Before_area", "After_area", "Final_area"]]
    slno = 1     
    for filename in filtered_files:
        try:
            print('-------------------------------------------------------------------')
            print('-------------------------------------------------------------------')
            print(str(slno) + '/' + str(len(filtered_files)))
            print(filename)
            ndwi = gdal_array.LoadFile(filename).astype(np.float32)
            
            # Local image enhancement
            ndwi_normalized = (ndwi + 1) * 127.5  # Normalize to range [0, 255]
            ndwi_normalized = ndwi_normalized.astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_image = clahe.apply(ndwi_normalized)
            if "L0" in filename:
                clahe_image[clahe_image>100] = np.max(clahe_image)
                ndwiRescaled = ((clahe_image - np.nanmin(clahe_image)) / (np.nanmax(clahe_image) - np.nanmin(clahe_image))) * 2 - 1
                ndwiRescaled = np.copy(ndwiRescaled).astype(np.float32)
            if "S2" in filename:
                clahe_image[clahe_image>125] = np.max(clahe_image)
                ndwiRescaled = ((clahe_image - np.nanmin(clahe_image)) / (np.nanmax(clahe_image) - np.nanmin(clahe_image))) * 2 - 1
                ndwiRescaled = np.copy(ndwiRescaled).astype(np.float32)
            
            # K-means clustering clipped NDWI raters to 3 clusters 
            # (water, wet non-water, and dry non-water) (1 more cluster for the value of -0.5)
            ndwiA =np.copy(ndwiRescaled).astype(np.float32)
            x = ndwiA.ravel()
            km = KMeans(n_clusters=3, random_state=0)  #random_state=0, n_init=10
            km.fit(x.reshape(-1,1)) 
            z = km.cluster_centers_
            cluster_labels = km.predict(ndwiA.reshape(-1, 1))
            cluster_raster = cluster_labels.reshape(ndwiA.shape)
            max_center_index = np.argmax(np.sum(z, axis=1))
            z1 = z[max_center_index]
            second_max_center_index = np.argpartition(np.sum(z, axis=1), -2)[-2]
            z2 = z[second_max_center_index]
            # third_max_center_index = np.argpartition(np.sum(z, axis=1), -3)[-3]
            # z3 = z[third_max_center_index]
            max_cluster_mask1D = (cluster_labels == max_center_index)
            max_cluster_mask2D = max_cluster_mask1D.reshape(ndwiA.shape) 
            threshold = round((z1[0]+z2[0])/2,3)
            print("   K-Means clustering threshold = "+str(threshold))

            water_cluster = np.copy(max_cluster_mask2D).astype(np.float32)
            # Assess image quality
            zone_mask = np.copy(ZONE).astype(np.float32)
            zone_mask[np.isnan(zone_mask) ==1] = 0
            count_zm = np.zeros(no_zones)
            for i in range(0, no_zones):
                count_zm[i] = np.count_nonzero(zone_mask == i+1)
                
            cluster_zone = np.copy(zone_mask).astype(np.float32)
            cluster_zone[water_cluster == 0] = 0
            count_cl = np.zeros(no_zones)
            ratio = np.zeros(no_zones)
            N_10 = 0
            for i in range(0, no_zones):
                count_cl[i] = np.count_nonzero(cluster_zone == i+1)
                ratio[i] = count_cl[i]/(count_zm[i] + 1.0e-20)
                if ratio[i] >= 0.1:
                    N_10 += 1
            print("   Ratio of zone 50 = "+str(round(ratio[no_zones-1],3)))
            print("   No. of zones having >=10% water pixels = "+str(int(N_10)))
              
            # Improve image classification
            ratio_nm = ratio*100/(max(ratio) + 1.0e-20)
            x_axis = np.zeros(no_zones)
            for i in range(0, no_zones):
                x_axis[i] = i + 1
            xx = np.vstack((x_axis, ratio_nm)).T
            kkm = KMeans(n_clusters=2, n_init=10).fit(xx) #random_state=0, n_init=10
            llb = kkm.labels_
            minx0 = no_zones
            minx1 = no_zones
            for i in range(0, no_zones):
                if llb[i] == 0:
                    if x_axis[i] < minx0:
                        minx0 = x_axis[i]
                elif llb[i] == 1:
                    if x_axis[i] < minx1:
                        minx1 = x_axis[i]                 
            s_index = max(minx0, minx1)
            if minx0 == s_index:
                water_id = 0
            elif minx1 == s_index:
                water_id = 1 
             
            # recall_zm = np.copy(zone_mask).astype(np.float32)
            add = np.copy(zone_mask).astype(np.float32)                
            add[np.where(add < s_index)] = 0
            improved = added_cluster = water_cluster + add
            improved[np.where(added_cluster > 1)] = 1
            
            freq_image = np.copy(zone_mask).astype(np.uint8)
            img_image = np.copy(improved).astype(np.uint8)
            window_size = 10
            stride = int(window_size/2)
            for y in range(0, freq_image.shape[0] - window_size + 1, stride):
                for x in range(0, freq_image.shape[1] - window_size + 1, stride):
                    window_freq = freq_image[y:y+window_size, x:x+window_size]
                    window_binary = img_image[y:y+window_size, x:x+window_size]
                    if np.count_nonzero(window_binary)>0:
                        mean_freq_water = np.mean(window_freq[window_binary == 1])
                        window_img = np.copy(window_binary).astype(np.uint8) 
                        window_img[window_freq > mean_freq_water] = 1 
                        img_image[y:y+window_size, x:x+window_size] = window_img

            bf_area = np.count_nonzero(water_cluster == 1)*0.0009
            af_area = np.count_nonzero(img_image == 1)*0.0009 
            res_max_area = np.count_nonzero(res_area == 1)*0.0009
            print("   Water surface area:")
            print("      Before improvement: "+str(round(bf_area,3))+" km2")
            print("      After improvement: "+str(round(af_area,3))+" km2")       
            if af_area > res_max_area:
                fn_area = bf_area
                qual = 0
                print("      Image cannot be improved")
            if bf_area == 0:
                fn_area = bf_area
                qual = 0
                print("      Image cannot be improved")
            else:
                if threshold < -1:    #-1
                    fn_area = bf_area
                    qual = 0
                    print("      Image cannot be improved")
                else:
                    if ratio[49] == 0:
                        fn_area = bf_area
                        qual = 0
                        print("      Image cannot be improved")
                    else:
                        if N_10 == 0:
                            fn_area = bf_area
                            qual = 0
                            print("      Image cannot be improved")
                        else:
                            fn_area = af_area
                            qual = 1
            print("      Final area: "+str(round(fn_area,3))+" km2")
            print("   ") 
            # Adding to thr list
            Cloud_percentage= 0
            results = np.append(results, [[filename[8:10], filename[16:26], round(Cloud_percentage,3),
                                              int(qual), round(bf_area,3),
                                              round(af_area,3), round(fn_area,3)]], axis=0)
            slno += 1
            continue
        
        except:
            continue

    
    # ==========================================EXPORT RESULTS AS A CSV FILE
    print("Exporting results as a csv file ...")
    os.chdir(Outputs_directory)
    with open('WSA.csv',"w", newline='') as my_csv:
        csvWriter = csv.writer(my_csv)
        csvWriter.writerows(results)
    print("  ")
    print("Done")    
    print(res_name)   
   
#=================================== Update Landsat-based E-A database with DEM-based E-A-S relationship
    os.chdir(Outputs_directory) 
    curve = pd.read_csv('Curve.csv')
    landsat_wsa = pd.read_csv('WSA.csv')
    ImageList = pd.read_csv('Image_List.csv')
    landsat_wsa.Cloud_percentage = ImageList.Cloud_percentage
    
    # Function calling to get best-fit degree of polinomial
    from scipy.interpolate import interp1d
    area = curve.iloc[:, 1]    # area 
    elevation = curve.iloc[:, 0]    # elevation (ydata)
    storage = curve.iloc[:, 2]
    interpolation_function_elevation = interp1d(area, elevation, kind='linear', fill_value='extrapolate')
    interpolation_function_storage = interp1d(area, storage, kind='linear', fill_value='extrapolate')
    
    Wsa= landsat_wsa
    Wsa["dem_value_m"] = None
    Wsa["Tot_res_volume_mcm"] = None
    for i in range(0,len(Wsa)):
        interpolated_elevation_value = interpolation_function_elevation(Wsa.Final_area[i])
        interpolated_storage_value = interpolation_function_storage(Wsa.Final_area[i])
        Wsa.dem_value_m[i] = interpolated_elevation_value
        Wsa.Tot_res_volume_mcm[i] = interpolated_storage_value
    
    #delete_id = Wsa[(Wsa['Quality'] == 0) | (Wsa['dem_value_m'] > 350) | (Wsa['dem_value_m'] < 10)].index  #min_wl# (Wsa['Quality'] == 0) | 
    #Wsa = Wsa.drop(delete_id)
    # ========================================== EXPORT RESULTS AS A CSV FILE
    Wsa.to_csv('WSA_updated.csv', index=False)
    print("Done")    
    print(res_name)
    
    # =================== Finally  moving all .png/jpg files in a seperate folder for better organisation   
    import shutil
    pictures_folder = "JPG_files"
    os.makedirs(pictures_folder, exist_ok=True)
    files = os.listdir()
    for file in files:
        if file.lower().endswith(".png"):
            file_path = os.path.join(os.getcwd(), file)
            shutil.move(file_path, os.path.join(pictures_folder, file))

#============EXTRA
