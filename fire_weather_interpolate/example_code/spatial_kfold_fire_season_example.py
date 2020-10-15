#coding: utf-8

"""
Summary
-------
References
----------
"""
    
#import
import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import os, sys, json
import time
from datetime import datetime, timedelta, date
import gc

import get_data as GD
import idw as idw
import idew as idew
import tps as tps
import fwi as fwi
import Eval as Eval
import rf as rf
import math


if __name__ == "__main__":
    dirname = '' #directory name

    file_path_daily = os.path.join(dirname, 'datasets/weather/daily_feather/')
    file_path_hourlyf = os.path.join(dirname, 'datasets/weather/hourly_feather/')
    #file_path_hourly = 'C:/Users/clara/OneDrive/Documents/Thesis/fwi-interpolate/datasets/weather/hourly_csv/'
    file_path_daily = os.path.join(dirname, 'datasets/weather/all_daily/all_daily/') #path to daily csv files from Environment & Climate Change Canada, needed for faster processing
    shapefile = os.path.join(dirname, 'datasets/study_area/QC_ON_albers_dissolve.shp')

    file_path_elev = os.path.join(dirname,'datasets/lookup_files/elev_csv.csv')
    idx_list = GD.get_col_num_list(file_path_elev,'elev')

    file_path_slope = os.path.join(dirname,'datasets/lookup_files/slope_csv.csv')
    idx_slope = GD.get_col_num_list(file_path_slope,'slope')

    file_path_drainage = os.path.join(dirname,'datasets/lookup_files/drainage_csv.csv')
    idx_drainage = GD.get_col_num_list(file_path_drainage,'drainage')

    with open(dirname+'datasets/json/daily_lookup_file_TEMP.json', 'r') as fp:
        date_dictionary = json.load(fp) #Get the lookup file for the stations with data on certain months/years

    with open(dirname+'datasets/json/daily_lat_lon_TEMP.json', 'r') as fp:
        daily_dictionary = json.load(fp) #Get the latitude and longitude for the stations

    with open(dirname+'datasets/json/hourly_lat_lon_TEMP.json', 'r') as fp:
        hourly_dictionary = json.load(fp) #Get the latitude and longitude for the stations

    #days_dict, latlon_station = fwi.end_date_calendar_csv(file_path_daily,'2018')
    #daysurface, maxmin= idw.IDW(latlon_station,days_dict,'2018','# Days Since March 1',shapefile,True,4) #Interpolate the start date
                

    #Get example elev array
    temperature = GD.get_noon_temp('1956-07-01 13:00',file_path_hourlyf)
    idw1_grid, maxmin, elev_array = idew.IDEW(hourly_dictionary,temperature,'2018-07-01 13:00','temp',shapefile,False,file_path_elev,idx_list,1)
   

    varlist = ['start','end'] #'start'
    interpolator_list = ['IDW4'] #'IDW2','IDW3','IDW4','RF'
    for var in varlist:
        for interpolator in interpolator_list:
            error_dict = {}
            for year in range(1954,2020):
                start = time.time() 
                year = str(year) #Year of interest, right now we do not have an overwinter procedure so each year is run separately 
                #input_date = year+'-09-15 13:00' #FWI metrics are calculated at 13h00
                #print('Starting program, loading the json dictionaries...')
                print(year)

                #print('Completed... Getting the start & end dates') 
                start = time.time()
                if var == 'start':
                    
                    days_dict, latlon_station = fwi.start_date_calendar_csv(file_path_daily,year) #Get two things: start date for each station and the lat lon of the station
                elif var == 'end': 
                    days_dict, latlon_station = fwi.end_date_calendar_csv(file_path_daily,year)
                else:
                    print('That is not a correct variable!')

                if interpolator == 'IDW2': 
                    num_stations = int(len(days_dict.keys()))
                    cluster_num = round(num_stations/10)
                    cluster_size,MAE = idw.spatial_kfold_idw(idw1_grid,latlon_station,days_dict,shapefile,2,file_path_elev,idx_list,cluster_num,'cluster')
                    


                elif interpolator == 'IDW3':

                    num_stations = int(len(days_dict.keys()))
                    cluster_num = round(num_stations/10)
                    cluster_size,MAE = idw.spatial_kfold_idw(idw1_grid,latlon_station,days_dict,shapefile,3,file_path_elev,idx_list,cluster_num,'cluster')

                elif interpolator == 'IDW4':

                    num_stations = int(len(days_dict.keys()))
                    cluster_num = round(num_stations/10)
                    cluster_size,MAE = idw.spatial_kfold_idw(idw1_grid,latlon_station,days_dict,shapefile,4,file_path_elev,idx_list,cluster_num,'cluster')
                    
                elif interpolator == 'RF':

                    num_stations = int(len(days_dict.keys()))
                    cluster_num = round(num_stations/10)
                    cluster_size,MAE = rf.spatial_kfold_rf(idw1_grid,latlon_station,days_dict,shapefile,file_path_elev,elev_array,idx_list,cluster_num,'cluster')

                elif interpolator == 'TPSS':
                    num_stations = int(len(days_dict.keys()))
                    phi_input = int(num_stations)-(math.sqrt(2*num_stations))
                    cluster_num = round(num_stations/10)
                    cluster_size,MAE = tps.spatial_kfold_tps(idw1_grid,latlon_station,days_dict,shapefile,phi_input,file_path_elev,elev_array,idx_list,cluster_num,'cluster')


                print(MAE)
                error_dict[year] = [MAE]
                end = time.time()
                time_elapsed = (end-start)/60
                print('Completed LOOCV operation, it took %s minutes..'%(time_elapsed)) 
            
            df = pd.DataFrame(error_dict)
            df = df.transpose()
            df.iloc[:,0] = df.iloc[:,0].astype(str).str.strip('[|]')
            #df.iloc[:,1]= df.iloc[:,1].astype(str).str.strip('[|]')
            file_out = '' #Where you want to save the output csv

            df.to_csv(file_out+interpolator+'_'+var+'_fire_season_spatial_bootstrap.csv', header=None, sep=',', mode='a')

