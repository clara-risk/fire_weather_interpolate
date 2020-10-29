#coding: utf-8
print('Checkpoint1 passed') 
#import
import geopandas as gpd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import warnings
import os,sys,time
import math,statistics
import json 
import pandas as pd 

warnings.filterwarnings("ignore") #Runtime warning suppress, this suppresses the /0 warning

import get_data as GD
import idw as idw
import idew as idew
import tps as tps
import rf as rf

import cluster_3d as c3d
print('Checkpoint2 passed: imports complete') 
#Locations of the input data we will need

dirname =  #Insert the directory name 
file_path_daily = os.path.join(dirname, 'datasets/weather/daily_feather/')
file_path_hourly = os.path.join(dirname, 'datasets/weather/hourly_feather/')
shapefile = os.path.join(dirname, 'datasets/study_area/QC_ON_albers_dissolve.shp')

file_path_elev = os.path.join(dirname,'datasets/lookup_files/elev_csv.csv')
idx_list = GD.get_col_num_list(file_path_elev,'elev')

with open(dirname+'datasets/json/daily_lookup_file_TEMP.json', 'r') as fp:
   date_dictionary = json.load(fp) #Get the lookup file for the stations with data on certain months/years

with open(dirname+'datasets/json/daily_lat_lon_TEMP.json', 'r') as fp:
   daily_dictionary = json.load(fp) #Get the latitude and longitude for the stations

with open(dirname+'datasets/json/hourly_lat_lon_TEMP.json', 'r') as fp:
   hourly_dictionary = json.load(fp) #Get the latitude and longitude for the stations

file_path_elev = os.path.join(dirname,'datasets/lookup_files/elev_csv.csv')
idx_list = GD.get_col_num_list(file_path_elev,'elev')
print('Checkpoint3 passed: data loaded into program') 
#Get example elev array
temperature = GD.get_noon_temp('1956-07-01 13:00',file_path_hourly)
idw1_grid, maxmin, elev_array = idew.IDEW(hourly_dictionary,temperature,'2018-07-01 13:00','temp',shapefile,False,file_path_elev,idx_list,1) #Example grid for blocking
   
print('Checkpoint4 passed: elevation array complete') 
 
#Creating the list of test dates 
years = [] 
for x in range(1956,2019):
   years.append(str(x))
overall_dates = []   

for year in years: 
   overall_dates.append((year)+'-07-01 13:00')

print('Checkpoint5 passed: size array complete') 
#Starting the procedure....
block_list = [9,16,25] 
variables = ['temp','rh','wind','pcp'] 
for var in variables: 
   for blockingNumber in block_list: 
      error_dict = {}

      for input_date in sorted(overall_dates):
         print('Processing %s'%(input_date))
         start = time.time()
         if var == 'temp': 
            temperature = GD.get_noon_temp(input_date,file_path_hourly)

         if var == 'rh':
            temperature = GD.get_relative_humidity(input_date,file_path_hourly)

         if var == 'wind':
            temperature = GD.get_wind_speed(input_date,file_path_hourly)

         if var == 'pcp': 
            temperature = GD.get_pcp(input_date[0:10],file_path_daily,date_dictionary)

         end = time.time()
         time_elapsed = (end-start)
         print('Completed getting dictionary, it took %s seconds..'%(time_elapsed)) 
         #Run the xval procedure for the clusters using 30 repetitions
         start = time.time() 

         num_stations_w = int(len(temperature.keys())) 
         phi_input = int(num_stations_w)-(math.sqrt(2*num_stations_w))
         if var != 'pcp':
            #Can use 'cluster' or 'block' here 
            #cluster_size,MAE = tps.spatial_kfold_tps(idw1_grid,hourly_dictionary,temperature,shapefile,phi_input,file_path_elev,elev_array,idx_list,blockingNumber,'cluster',False)
            cluster_size,MAE = idew.spatial_kfold_IDEW(idw1_grid,hourly_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,2,blockingNumber,'block')
            #cluster_size,MAE = rf.spatial_kfold_rf(idw1_grid,hourly_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,blockingNumber,'block',False)
            #cluster_size,MAE = idw.spatial_kfold_idw(idw1_grid,hourly_dictionary,temperature,shapefile,2,file_path_elev,idx_list,blockingNumber,'block',False)

         if var == 'pcp':
            cluster_size,MAE = idew.spatial_kfold_IDEW(idw1_grid,daily_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,2,blockingNumber,'block')
            #cluster_size,MAE = rf.spatial_kfold_rf(idw1_grid,daily_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,blockingNumber,'block',False)
            #cluster_size,MAE = idw.spatial_kfold_idw(idw1_grid,daily_dictionary,temperature,shapefile,2,file_path_elev,idx_list,blockingNumber,'block',False)
            #cluster_size,MAE = tps.spatial_kfold_tps(idw1_grid,daily_dictionary,temperature,shapefile,phi_input,file_path_elev,elev_array,idx_list,blockingNumber,'cluster',False)


         #Uncomment for GPR specifically
##         if var != 'pcp' and var != 'temp':
##            cluster_size,MAE = gpr.spatial_kfold_gpr(idw1_grid,hourly_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,0.3,blockingNumber,'cluster')
##         if var == 'temp':
##            cluster_size,MAE = gpr.spatial_kfold_gpr(idw1_grid,hourly_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,0.1,blockingNumber,'cluster')
##         if var == 'pcp':
##            cluster_size,MAE = gpr.spatial_kfold_gpr(idw1_grid,daily_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,0.3,blockingNumber,'cluster')
##         
         print(cluster_size)

         print(MAE)
         error_dict[input_date] = [MAE,cluster_size]
         end = time.time()
         time_elapsed = (end-start)
         print('Completed cluster operation, it took %s seconds..'%(time_elapsed)) 
        
      df = pd.DataFrame(error_dict)
      df = df.transpose()
      df.iloc[:,0] = df.iloc[:,0].astype(str).str.strip('[|]')
      df.iloc[:,1]= df.iloc[:,1].astype(str).str.strip('[|]')
      file_out = '' #Where you want to save the output csv

      df.to_csv(file_out+'IDEW2_spatialBlock_'+var+str(blockingNumber)+'.csv', header=None, sep=',', mode='a')
    
