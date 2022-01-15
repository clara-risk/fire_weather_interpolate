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
import mogp as mogp

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
kernels={'temp':['316**2 * Matern(length_scale=[5e+05, 5e+05, 6.01e+03], nu=0.5)']\
                            ,'rh':['307**2 * Matern(length_scale=[5e+05, 6.62e+04, 1.07e+04], nu=0.5)'],\
                            'pcp':['316**2 * Matern(length_scale=[5e+05, 5e+05, 4.67e+05], nu=0.5)'],\
                            'wind':['316**2 * Matern(length_scale=[5e+05, 6.62e+04, 1.07e+04], nu=0.5)']}
for var in variables:

   error_dict = {}

   for input_date in sorted(overall_dates):
      print('Processing %s'%(input_date))
      start = time.time()
      temperature = GD.get_noon_temp(input_date,file_path_hourly)
      rh = GD.get_relative_humidity(input_date,file_path_hourly)
      wind = GD.get_wind_speed(input_date,file_path_hourly)

      mogp.MOGP_interpolator(hourly_dictionary,temperature,rh,wind,input_date,'Temperature',shapefile,False,\
                                          file_path_elev,idx_list,False,res=50*1000)

