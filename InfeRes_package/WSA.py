# IMPORTING LIBRARY
import os
import csv
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from osgeo import gdal_array


def wsa(res_name, res_directory):
    # IMPROVE NDWI-BASED LANDSAT IMAGE CLASSIFICATION
    results = [["Landsat", "Type", "Date", "Threshold", "R_50", "N_10", "S_zone", 
                "Quality", "Bf_area", "Af_area", "Fn_area"]]
    drtr = res_directory +  "/Outputs"     
    os.chdir(res_directory + "/LandsatData_Clip")
    #os.chdir('./Landsat_'+str(LS))
    directory = os.getcwd()
    filtered_files = [file for file in os.listdir(directory) if "NDWI" in file]
    slno =0
    no_zones = 50
    for filename in filtered_files:
        try:
            if filename.startswith("Clipped_"):
                slno = slno+1
                print(slno)        
                print(filename)
                ndwi = gdal_array.LoadFile(filename).astype(np.float32)
                ndwi = np.nan_to_num(ndwi, nan = -0.5)
                
                # Clip NDWI rasters by the expanded mask 
                exp_mask = gdal_array.LoadFile(drtr+"\Expanded_Mask.tif").astype(np.float32)
                clip_ndwi = ndwi
                clip_ndwi[np.where(exp_mask == 0)] = -0.5
                
                # K-means clustering clipped NDWI raters to 3 clusters 
                # (water, wet non-water, and dry non-water) (1 more cluster for the value of -0.5)
                rows = len(clip_ndwi)
                columns = len(clip_ndwi[0])
                x = clip_ndwi.ravel()
                km = KMeans(n_clusters=4, n_init=10)
                km.fit(x.reshape(-1,1)) 
                z = km.cluster_centers_
                z1 = max(z)
                z2 = -1
                z3 = -1
                for i in range(0, 4):
                    if z[i] < z1 and z[i] > z2:
                        z2 = z[i]
                for i in range(0, 4):
                    if z[i] < z2 and z[i] > z3:
                        z3 = z[i]        
                threshold = round(float((z1+z2)/2),3)
                print("   K-Means clustering threshold = "+str(threshold))
                # plt.figure(figsize=[30,15])
                # plt.hist(x, bins=200, range=[-0.49, 0.5], color='c')
                # plt.axvline(z[0], color='navy', linestyle='dashed', linewidth=2)
                # plt.axvline(z[1], color='navy', linestyle='dashed', linewidth=2)
                # plt.axvline(z[2], color='navy', linestyle='dashed', linewidth=2)
                # plt.axvline(z[3], color='navy', linestyle='dashed', linewidth=2)
                # plt.axvline((z1+z2)/2, color='red', linestyle='dashed', linewidth=2)
                # plt.axvline((z2+z3)/2, color='red', linestyle='dashed', linewidth=2)
                # plt.title(filename, fontsize=30)
                # plt.xlabel('NDWI', fontsize=30)
                # plt.xticks(fontsize=30)
                # plt.yticks(fontsize=30)
                # plt.show()
                labels = np.reshape(km.labels_,(-1,columns)) 
                water_label = 0
                for i in range(0, 4):
                    if z[i] == max(z):
                        water_label = i 
                water_cluster = labels - labels
                water_cluster[np.where(labels == water_label)] = 1
                
                # Assess image quality
                zone_mask = gdal_array.LoadFile(drtr+"\Zone_Mask.tif").astype(np.float32)
                count_zm = np.zeros(no_zones)
                for i in range(0, no_zones):
                    count_zm[i] = np.count_nonzero(zone_mask == i+1)
                cluster_zone = zone_mask
                cluster_zone[np.where(water_cluster == 0)] = 0
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
                kkm = KMeans(n_clusters=2).fit(xx)
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
                # print("   Additional water pixels start from zone "+str(int(s_index)))    
                # colors = ['navy' if x==water_id else 'lightblue' for x in llb]
                # plt.figure(figsize=[30,15])
                # plt.bar(x_axis, ratio, color=colors)
                # plt.ylim(top=1)
                # plt.axvline(x=s_index,color='red',linestyle='--')
                # plt.title(filename, fontsize=30)
                # plt.xlabel('Zone', fontsize=30)
                # plt.ylabel('Ratio', fontsize=30)
                # plt.xticks(fontsize=30)
                # plt.yticks(fontsize=30)
                # plt.show()
                 
                recall_zm = gdal_array.LoadFile(drtr+"\Zone_Mask.tif").astype(np.float32)
                add = recall_zm
                add[np.where(recall_zm < s_index)] = 0
                improved = added_cluster = water_cluster + add
                improved[np.where(added_cluster > 1)] = 1
                bf_area = np.count_nonzero(water_cluster == 1)*0.0009 
                af_area = np.count_nonzero(improved == 1)*0.0009        
                print("   Water surface area:")
                print("      Before improvement: "+str(round(bf_area,3))+" km2")
                print("      After improvement: "+str(round(af_area,3))+" km2")       
                if bf_area == 0:
                    fn_area = bf_area
                    qual = 0
                    print("      Image cannot be improved")
                else:
                    if threshold < -0.5:
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
                results = np.append(results, [[str(filename[8]), str(filename[10:12]),filename[18:28],
                                               round(threshold,3), round(ratio[49],3), 
                                               int(N_10), int(s_index), int(qual), 
                                               round(bf_area,3), round(af_area,3), 
                                               round(fn_area,3)]], axis=0)
                continue
            else:
                continue
        except:
            continue
    
    
    # ==========================================EXPORT RESULTS AS A CSV FILE
    print("Exporting results as a csv file ...")
    os.chdir(res_directory +  "/Outputs")
    with open("WSA.csv","w", newline='') as my_csv:
        csvWriter = csv.writer(my_csv)
        csvWriter.writerows(results)
    print("  ")
    print("Done")    
    print("  ")   
   