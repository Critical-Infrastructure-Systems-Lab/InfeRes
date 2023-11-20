#==============================================================================
# Run this code if your reserovoir contains two or more landsat tiles.
# Run this only after succesfully processing all the individual parts (landsat tiles) of the reservoir using "InfeRes.py"
# Asign input (i.e. res_name) as "ReservoirNAME".
# Where, "ReservoirNAME" is the name of the multi-tile reservoir.
# The code will automatically search for all other parts of the reservoir and compute the combined storage
# Please save different parts of the reservoir in folder 'ReservoirNAME_tile1', 'ReservoirNAME_tile2'...and so on.
#==============================================================================

import csv
import os 
os.chdir("H:/My Drive/NUSproject/ReservoirExtraction/")
import pandas as pd
import numpy as np

if __name__ == "__main__":

    #====================================>> USER INPUT PARAMETERS 
    parent_directory = "H:/My Drive/NUSproject/ReservoirExtraction/Reservoirs/"
    os.chdir(parent_directory)
    multi_tile_res = 'Xayabouri'
    max_wl = 285
    
    print("Two tiles reservoir")
    directory = os.getcwd()
    filtered_files = [file for file in os.listdir(directory) if multi_tile_res in file]
    data_range = pd.DataFrame()
    n=0
    for filename in filtered_files:
        os.chdir(parent_directory)
        os.chdir(filename + '/Outputs')
        curve = pd.read_csv('Curve_' + filename + '.csv')
        first_column = curve.iloc[:, 0]
        data_range.loc[n, 'Min'] = int(first_column.min())
        data_range.loc[n, 'Max'] = int(first_column.max())
        n += 1    
    filename = None
    
    curve_data = [["Level (m)", "Area (skm)", "Storage (mcm)"]]
    min_val= int(min(data_range.iloc[:, 0]))
    max_val= int(max(data_range.iloc[:, 1]))
    data = pd.DataFrame({'RefElev': range(min_val, max_val+1)})
    n=0
    for filename in filtered_files:
        n += 1
        os.chdir(parent_directory)
        os.chdir(filename + '/Outputs')
        curve = pd.read_csv('Curve_' + filename + '.csv')
        val = curve.iloc[:, 0] 
        data["Level"+str(n)] = 0
        data["Area"+str(n)] = 0
        data["Storage"+str(n)] = 0
        i=0
        for i in range(0,len(curve)):
            #print(i)
            index = np.where(data.loc[:,"RefElev"] == val.iloc[i])[0]
            if (len(index)>0 and n==1):
               data.iloc[index,n] = curve.iloc[i,0]
               data.iloc[index,n+1] = curve.iloc[i,1]
               data.iloc[index,n+2] = curve.iloc[i,2] 
               
            if (len(index)>0 and n==2):
               data.iloc[index,n+2] = curve.iloc[i,0]
               data.iloc[index,n+3] = curve.iloc[i,1]
               data.iloc[index,n+4] = curve.iloc[i,2] 
            
            if (len(index)>0 and n==3):
               data.iloc[index,n+4] = curve.iloc[i,0]
               data.iloc[index,n+5] = curve.iloc[i,1]
               data.iloc[index,n+6] = curve.iloc[i,2] 
               
    if len(filtered_files)==2:
        Lev = data['RefElev']
        Ar = data['Area1'] + data['Area2'] 
        Vol = data['Storage1'] + data['Storage2']
            
    if len(filtered_files)==3:
        Lev = data['RefElev']
        Ar = data['Area1'] + data['Area2'] + data['Area3']
        Vol = data['Storage1'] + data['Storage2'] + data['Storage3']
              
    curve_data = pd.concat([Lev , Ar , Vol], axis=1, keys=["Level (m)", "Area (skm)", "Storage (mcm)"])    
    curve_data['Diff'] = curve_data['Storage (mcm)'] - curve_data['Storage (mcm)'].shift(1) 
    pos = curve_data.index[curve_data['Diff'] <= 0].tolist()       
    curve_data_new = curve_data.iloc[:pos[0]]
    curve_data_new = curve_data_new.drop('Diff', axis=1)
    curve_data_new_list = [["Level (m)", "Area (skm)", "Storage (mcm)"]]
    # Initialize curve_data_new_list with the header as the first row
    curve_data_new_list = [list(curve_data_new.columns)]    
    # Convert DataFrame rows to a list and append to curve_data_new_list
    curve_data_new_list += curve_data_new.values.tolist()
    
    os.chdir(parent_directory)
    os.chdir(filtered_files[0] + '/Outputs')
    # saving output as a csv file
    with open("Curve_"+ multi_tile_res +".csv","w", newline='') as my_csv:
          csvWriter = csv.writer(my_csv)
          csvWriter.writerows(curve_data_new_list)
        
    del data, Lev, Ar, Vol, curve_data, curve_data_new
    
    #========== ESTIMATING TOTAL RESERVOIR AREA USING PREVIOUSLY GENERATED CURVE FOR THE BIGGER TILE
    
    curve = pd.read_csv('Curve_' + multi_tile_res + '.csv')
    landsat_wsa = pd.read_csv('WSA_updated_' + multi_tile_res + '_tile1.csv')
    dead_wl = landsat_wsa['dem_value_m'][0]       
    
    Wsa= landsat_wsa
    Wsa["Fn_area"] = None
    Wsa["Tot_res_volume_mcm"] = None
    for i in range(0,len(Wsa)):    
        index = np.where(curve.iloc[:, 0] == Wsa.dem_value_m[i])
        area = curve.iloc[index[0], 1]
        volume = curve.iloc[index[0], 2]
        Wsa.Fn_area[i] = area.values[0]
        Wsa.Tot_res_volume_mcm[i] = volume.values[0]
    
    delete_id = Wsa[(Wsa['Quality'] == 0) | (Wsa['dem_value_m'] < dead_wl) | (Wsa['dem_value_m'] > max_wl+20)].index
    Wsa = Wsa.drop(delete_id)
    # ========================================== EXPORT RESULTS AS A CSV FILE
    Wsa.to_csv('WSA_updated_' + multi_tile_res + '.csv', index=False)
    print("Done")    
    print("  ")
    
   

        
        
        
        
        
        
        
        
        
        
        
        