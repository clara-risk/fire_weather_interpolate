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
import Eval as Eval
import gpr as gpr
import cluster_3d as c3d
print('Checkpoint2 passed: imports complete') 
#Locations of the input data we will need

dirname = '' #Insert the directory name 
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
print('Checkpoint3 passed: data loaded from server') 
#Get example elev array
temperature = GD.get_noon_temp('1956-07-01 13:00',file_path_hourly)
idw1_grid, maxmin, elev_array = idew.IDEW(hourly_dictionary,temperature,'2018-07-01 13:00','temp',shapefile,False,file_path_elev,idx_list,1)
   
print('Checkpoint4 passed: elevation array complete') 
 
#Creating the list of test dates 
years = [] 
for x in range(1987,1988):
   years.append(str(x))
overall_dates = []   

for year in years: 
   overall_dates.append((year)+'-07-01 13:00')

print('Checkpoint5 passed: size array complete') 
#Starting the procedure....
variables = ['pcp','temp','rh','wind'] 
for var in variables: 
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
         if var != 'pcp': #and var != 'temp': 

            #absolute_error_dictionary = tps.cross_validate_tps(hourly_dictionary,temperature,shapefile,phi_input,False)
            #absolute_error_dictionary = gpr.cross_validate_gpr(hourly_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,0.3)
            #absolute_error_dictionary = rf.cross_validate_rf(hourly_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,False)
            #absolute_error_dictionary = idw.cross_validate_IDW(hourly_dictionary,rainfall,shapefile,2,False)
            absolute_error_dictionary = idew.cross_validate_IDEW(hourly_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,2)
            
            MAE, MAE_max = Eval.get_MAE(absolute_error_dictionary)
   
         #if var == 'temp':
            #absolute_error_dictionary = gpr.cross_validate_gpr(hourly_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,0.1)
            #MAE, MAE_max = Eval.get_MAE(absolute_error_dictionary)
         if var == 'pcp':

            #absolute_error_dictionary = tps.cross_validate_tps(daily_dictionary,temperature,shapefile,phi_input,False)
            #absolute_error_dictionary = idw.cross_validate_IDW(daily_dictionary,rainfall,shapefile,2,False)
            #absolute_error_dictionary = rf.cross_validate_rf(daily_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,False)
            #absolute_error_dictionary = gpr.cross_validate_gpr(daily_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,0.3)
            absolute_error_dictionary = idew.cross_validate_IDEW(daily_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,2)
            MAE, MAE_max = Eval.get_MAE(absolute_error_dictionary)
 
         cluster_size = 'LOOCV'
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

      df.to_csv(file_out+'IDEW2_loocv_xval_'+var+'.csv', header=None, sep=',', mode='a')
    