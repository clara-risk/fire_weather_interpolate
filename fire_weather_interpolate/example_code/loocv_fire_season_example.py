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
import math, statistics


if __name__ == "__main__":

    dirname = '' #Insert data directory path 
    file_path_daily = os.path.join(dirname, 'datasets/weather/daily_feather/')
    file_path_hourlyf = os.path.join(dirname, 'datasets/weather/hourly_feather/')
    file_path_hourly = 'C:/Users/clara/OneDrive/Documents/Thesis/fwi-interpolate/datasets/weather/hourly_csv/'
    file_path_daily = 'C:/Users/clara/OneDrive/Documents/Thesis/summer2020/weather_engine/all_daily/'
    shapefile = os.path.join(dirname, 'datasets/study_area/QC_ON_albers_dissolve.shp')

    file_path_elev = os.path.join(dirname,'datasets/lookup_files/elev_csv.csv')
    idx_list = GD.get_col_num_list(file_path_elev,'elev')

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
   

    varlist = ['start','end']
    interpolator_list = ['IDW2','IDW3','IDW4','RF','TPSS'] 
    for var in varlist:
        for interpolator in interpolator_list:
            error_dict = {}
            for year in range(1954,2020):
                start = time.time() 
                year = str(year) #Year of interest, right now we do not have an overwinter procedure so each year is run separately 

                print(year)

                start = time.time()
                if var == 'start':
                    
                    days_dict, latlon_station = fwi.start_date_calendar_csv(file_path_daily,year) #Get two things: start date for each station and the lat lon of the station
                    if int(year) >= 1994: #When many hourly stations are added
                        hourly_dictH, latlon_stationH = fwi.start_date_add_hourly(file_path_hourlyf,year)
                        if hourly_dictH is not None: #Sometimes there are no stations with unbroken records
                            days_dict = GD.combine_stations(days_dict,hourly_dictH)
                            latlon_station = GD.combine_stations(latlon_station,latlon_stationH)                                   
                
                elif var == 'end': 
                    days_dict, latlon_station = fwi.end_date_calendar_csv(file_path_daily,year)
                    if int(year) >= 1994: #When many hourly stations are added
                        hourly_dictH, latlon_stationH = fwi.end_date_add_hourly(file_path_hourlyf,year)
                        if hourly_dictH is not None: #Sometimes there are no stations with unbroken records
                            days_dict = GD.combine_stations(days_dict,hourly_dictH)
                            latlon_station = GD.combine_stations(latlon_station,latlon_stationH) 
                else:
                    print('That is not a correct variable!')

                if interpolator == 'IDW2': 
                
                    #daysurface, maxmin= idw.IDW(latlon_station,days_dict,year,'# Days Since March 1',shapefile,False,2) #Interpolate the start date
                    #endsurface, maxmin = idw.IDW(latlon_station,end_dict,year,'# Days Since Oct 1',shapefile,True,2) #Interpolate the end date
                    absolute_error_dictionary = idw.cross_validate_IDW(latlon_station,days_dict,shapefile,2,False)
                    MAE, MAE_max = Eval.get_MAE(absolute_error_dictionary)
                    #Get the standard deviation 
                    error_at_station = absolute_error_dictionary.values() 
                    stdev_stations = statistics.stdev(error_at_station)
                    

                elif interpolator == 'IDW3':
                    absolute_error_dictionary = idw.cross_validate_IDW(latlon_station,days_dict,shapefile,3,False)
                    MAE, MAE_max = Eval.get_MAE(absolute_error_dictionary)
                    error_at_station = absolute_error_dictionary.values() 
                    stdev_stations = statistics.stdev(error_at_station)

                elif interpolator == 'IDW4':
                    absolute_error_dictionary = idw.cross_validate_IDW(latlon_station,days_dict,shapefile,4,False)
                    MAE, MAE_max = Eval.get_MAE(absolute_error_dictionary)
                    error_at_station = absolute_error_dictionary.values() 
                    stdev_stations = statistics.stdev(error_at_station)

                elif interpolator == 'RF':
                    absolute_error_dictionary = rf.cross_validate_rf(latlon_station,days_dict,shapefile,file_path_elev,elev_array,idx_list,False)
                    MAE, MAE_max = Eval.get_MAE(absolute_error_dictionary)
                    error_at_station = absolute_error_dictionary.values() 
                    stdev_stations = statistics.stdev(error_at_station)

                elif interpolator == 'TPSS':
                    num_stations = int(len(days_dict.keys()))
                    phi_input = int(num_stations)-(math.sqrt(2*num_stations))
                    absolute_error_dictionary = tps.cross_validate_tps(latlon_station,days_dict,shapefile,phi_input,False)
                    MAE, MAE_max = Eval.get_MAE(absolute_error_dictionary)
                    error_at_station = absolute_error_dictionary.values() 
                    stdev_stations = statistics.stdev(error_at_station)
                else: 
                    print('That is not a valid interpolator')
            
                
                print('MAE:'+str(MAE)) 
                print('STDEV:'+str(stdev_stations)) 
                error_dict[year] = [MAE, stdev_stations]
                end = time.time()
                time_elapsed = (end-start)/60
                print('Completed LOOCV operation, it took %s minutes..'%(time_elapsed)) 
            
            df = pd.DataFrame(error_dict)
            df = df.transpose()
            df.iloc[:,0] = df.iloc[:,0].astype(str).str.strip('[|]')
            df.iloc[:,1]= df.iloc[:,1].astype(str).str.strip('[|]')
            file_out = '' #Where you want to save the output csv

            df.to_csv(file_out+interpolator+'_'+var+'_fire_season.csv', header=None, sep=',', mode='a')
        


        
