#coding: utf-8

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
import ok as ok 
import cluster_3d as c3d

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


#Get example elev array
temperature = GD.get_noon_temp('2018-07-01 13:00',file_path_hourly)
idw_grid, maxmin, elev_array = idew.IDEW(hourly_dictionary,temperature,'2018-07-01 13:00','temp',shapefile,False,file_path_elev,idx_list,1)
   
#Creating the list of test dates 
years = [] 
for x in range(1956,2019):
   years.append(str(x))
overall_dates = []   

for year in years: 
   overall_dates.append((year)+'-07-01 13:00')

#Get the selected semivariogram type from the lookup file
file_name1 = dirname+'datasets/best_semivariogram.csv' 
df = pd.read_csv(file_name1)
models = list(df['WS']) 

#Starting the procedure....
error_dict = {}
count = 0 
for input_date in sorted(overall_dates):
   
   print('Processing %s'%(input_date))
   start = time.time() 
   wind_speed = GD.get_wind_speed(input_date,file_path_hourly)
   end = time.time()
   time_elapsed = (end-start)
   print('Completed getting dictionary, it took %s seconds..'%(time_elapsed)) 
   #Run the xval procedure for the clusters using 30 repetitions
   start = time.time() 

   model = models[count]
   print(model) 
   

   MAE = ok.shuffle_split_OK(hourly_dictionary,temperature,shapefile,model,10)
   
   print(MAE) 
   
   error_dict[input_date] = [MAE]
   end = time.time()
   time_elapsed = (end-start)
   print('Completed cluster operation, it took %s seconds..'%(time_elapsed))
   count +=1
  
df = pd.DataFrame(error_dict)
df = df.transpose()
df.iloc[:,0] = df.iloc[:,0].astype(str).str.strip('[|]')
#df.iloc[:,1]= df.iloc[:,1].astype(str).str.strip('[|]')
file_out = os.path.join(dirname, 'datasets/') #Where you want to save the output csv

df.to_csv(file_out+'OK_montecarloCluster_xval_ws.csv', header=None, sep=',', mode='a')
  
#coding: utf-8

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
import ok as ok 
import cluster_3d as c3d

#Locations of the input data we will need

dirname = 'C:/Users/Clara/Documents/code_to_run/' #Insert the directory name 
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


#Get example elev array
temperature = GD.get_noon_temp('2018-07-01 13:00',file_path_hourly)
idw_grid, maxmin, elev_array = idew.IDEW(hourly_dictionary,temperature,'2018-07-01 13:00','temp',shapefile,False,file_path_elev,idx_list,1)
   
#Creating the list of test dates 
years = [] 
for x in range(1956,2019):
   years.append(str(x))
overall_dates = []   

for year in years: 
   overall_dates.append((year)+'-07-01 13:00')

file_name1 = dirname+'datasets/best_semivariogram.csv' 
df = pd.read_csv(file_name1)
models = list(df['WS'])
#Starting the procedure....
error_dict = {}
count = 0 
for input_date in sorted(overall_dates):
   
   print('Processing %s'%(input_date))
   start = time.time() 
   temperature = GD.get_wind_speed(input_date,file_path_hourly)
   end = time.time()
   time_elapsed = (end-start)
   print('Completed getting dictionary, it took %s seconds..'%(time_elapsed)) 
   #Run the xval procedure for the clusters using 30 repetitions
   start = time.time() 

   model = models[count]
   print(model) 
   

   MAE = ok.shuffle_split_OK(hourly_dictionary,temperature,shapefile,model,10)
   
   print(MAE) 
   
   error_dict[input_date] = [MAE]
   end = time.time()
   time_elapsed = (end-start)
   print('Completed cluster operation, it took %s seconds..'%(time_elapsed))
   count +=1
  
df = pd.DataFrame(error_dict)
df = df.transpose()
df.iloc[:,0] = df.iloc[:,0].astype(str).str.strip('[|]')

file_out = '' #Where you want to save the output csv

df.to_csv(file_out+'OK_montecarlo_xval_ws.csv', header=None, sep=',', mode='a')
  
