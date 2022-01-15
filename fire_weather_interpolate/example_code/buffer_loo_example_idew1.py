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
import gpr as gpr

import cluster_3d as c3d
print('Checkpoint2 passed: imports complete') 
#Locations of the input data we will need

dirname = 'C:/Users/clara/Documents/cross-validation/' #Insert the directory name 
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
#print(temperature)
idw1_grid, maxmin, elev_array = idew.IDEW(hourly_dictionary,temperature,'2018-07-01 13:00','temp',shapefile,False,\
                                          file_path_elev,idx_list,1,False,res=10000)
print(elev_array)
   
print('Checkpoint4 passed: elevation array complete') 
 
#Creating the list of test dates 
years = [] 
for x in range(1956,2020): #Insert years here 
   years.append(str(x))
overall_dates = []   

for year in years: 
   overall_dates.append((year)+'-07-01 13:00') #July 1 each year


print('Checkpoint5 passed: size array complete') 
#Starting the procedure....
variables = ['temp','rh','wind','pcp'] #,'pcp','temp','rh','wind']
buffer_sizes = {'temp': 500, 'rh': 100, 'wind': 20, 'pcp': 20}
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
      if var != 'pcp':
         #if var == 'temp': 
            #MAE = gpr.shuffle_split_gpr(hourly_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,0.1,10)
         #kernels = ['316**2 * Matern(length_scale=[5e+05, 5e+05, 6.01e+03], nu=0.5)']
         #kernels = ['307**2 * Matern(length_scale=[9.51e+04, 9.58e+04, 3.8e+05], nu=0.5)']
         #kernels = ['316**2 * Matern(length_scale=[5e+05, 5e+05, 4.67e+05], nu=0.5)'] 
         #else:
         #MAE = gpr.shuffle_split_gpr(hourly_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,\
                                     #kernels,10)
         
         #MAE = tps.shuffle_split_tps(hourly_dictionary,temperature,shapefile,phi_input,10)
         #MAE = idew.shuffle_split_IDEW(hourly_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,2,10)
         #MAE = idw.shuffle_split(hourly_dictionary,temperature,shapefile,2,10,False)
         #MAE = idw.buffer_LOO(hourly_dictionary,temperature,shapefile,1,buffer_sizes[0],False)
         MAE = idew.buffer_LOO_IDEW(hourly_dictionary,temperature,shapefile,\
                                          file_path_elev,elev_array,idx_list,1,buffer_sizes[var])

         #MAE = rf.shuffle_split_rf(hourly_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,10)
      if var == 'pcp':
         MAE = idew.buffer_LOO_IDEW(daily_dictionary,temperature,shapefile,\
                                          file_path_elev,idx_list,1,buffer_sizes[var])
         #kernels = ['316**2 * Matern(length_scale=[5e+05, 5e+05, 4.67e+05], nu=0.5)']
         #MAE = gpr.shuffle_split_gpr(daily_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,\
                                     #kernels,10)
         #MAE = tps.shuffle_split_tps(daily_dictionary,temperature,shapefile,phi_input,10)
         #MAE = idew.shuffle_split_IDEW(daily_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,2,10)
         #MAE = idw.shuffle_split(daily_dictionary,temperature,shapefile,2,10,False)
         #MAE = rf.shuffle_split_rf(daily_dictionary,temperature,shapefile,file_path_elev,elev_array,idx_list,10)
      average_list = []
      for key, val in MAE.items():
         average_list.append(val)
      average = np.nanmean(np.array(average_list))
      print(average)
      #print(cluster_size)

      #print(MAE)
      error_dict[input_date] = [MAE,average]
      end = time.time()
      time_elapsed = (end-start)
      print('Completed operation, it took %s seconds..'%(time_elapsed))

   df = pd.DataFrame(error_dict)
   df = df.transpose()
   df.iloc[:,0] = df.iloc[:,0].astype(str).str.strip('[|]')
   #df.iloc[:,1]= df.iloc[:,1].astype(str).str.strip('[|]')
   file_out = 'C:/Users/clara/Documents/cross-validation/' #Where you want to save the output csv

   df.to_csv(file_out+'IDEW1_buffer_'+var+'.csv', header=None, sep=',')
