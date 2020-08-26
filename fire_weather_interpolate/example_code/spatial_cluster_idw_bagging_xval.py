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
 
#Creating the list of test dates 
years = [] 
for x in range(1956,2019):
   years.append(str(x))
overall_dates = []   

for year in years: 
   overall_dates.append((year)+'-07-01 13:00')

#Make the example idw grid for referencing the size 
xtemp_dict = GD.get_noon_temp('2018-07-01 13:00',file_path_hourly)
idw1_grid,maxmin = idw.IDW(hourly_dictionary,xtemp_dict,'2018-07-01 13:00','Temperature (⁰C)',shapefile,False,1)

#Starting the procedure....
error_dict = {}

for input_date in sorted(overall_dates):
   start = time.time() 
   print('Processing %s'%(input_date))
   
   temperature = GD.get_noon_temp(input_date,file_path_hourly)
   #Run the xval procedure for the clusters using 30 repetitions 
   cluster_size, MAE = idw.select_block_size_IDW(30,'clusters',hourly_dictionary,temperature,idw1_grid,shapefile)
   
   error_dict[input_date] = [MAE,cluster_size]
   end = time.time()
   time_elapsed = (end-start)
   print('Completed, it took %s seconds..'%(time_elapsed)) 
  
df = pd.DataFrame(error_dict)
df = df.transpose()
df.iloc[:,0] = df.iloc[:,0].astype(str).str.strip('[|]')
df.iloc[:,1]= df.iloc[:,1].astype(str).str.strip('[|]')
file_out = '' #Where you want to save the output csv

df.to_csv(file_out+'IDW1_baggingCluster_xval_temperature.csv', header=None, sep=',', mode='a')
  
