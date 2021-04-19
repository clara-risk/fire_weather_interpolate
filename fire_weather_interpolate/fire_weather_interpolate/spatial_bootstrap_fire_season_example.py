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
    dirname = '' #File path to where the data is stored 

    file_path_daily = os.path.join(dirname, 'datasets/weather/daily_feather/')
    file_path_hourlyf = os.path.join(dirname, 'datasets/weather/hourly_feather/')
    file_path_daily = os.path.join(dirname, 'datasets/weather/all_daily/all_daily/')
    shapefile = os.path.join(dirname, 'datasets/study_area/QC_ON_albers_dissolve.shp')

    file_path_elev = os.path.join(dirname,'datasets/lookup_files/elev_csv.csv')
    idx_list = GD.get_col_num_list(file_path_elev,'elev')

    file_path_slope = os.path.join(dirname,'datasets/lookup_files/slope_csv.csv')
    idx_slope = GD.get_col_num_list(file_path_slope,'slope')

    file_path_drainage = os.path.join(dirname,'datasets/lookup_files/drainage_csv.csv')
    idx_drainage = GD.get_col_num_list(file_path_drainage,'drainage')

    #Get example elev array
    temperature = GD.get_noon_temp('1956-07-01 13:00',file_path_hourlyf)
    idw1_grid, maxmin, elev_array = idew.IDEW(hourly_dictionary,temperature,'2018-07-01 13:00','temp',shapefile,False,file_path_elev,idx_list,1)
   

    varlist = ['start','end'] #'start'
    interpolator_list = ['RF'] #'IDW2','IDW3','IDW4','RF','TPSS'
    for var in varlist:
        for interpolator in interpolator_list:
            error_dict = {}
            for year in range(1956,2019):
                start = time.time() 
                year = str(year) #Year of interest, right now we do not have an overwinter procedure so each year is run separately 
                print(year)
                start = time.time()
                if var == 'start':
                    
                    days_dict, latlon_station = fwi.start_date_calendar_csv(file_path_daily,year) #Get two things: start date for each station and the lat lon of the station
                elif var == 'end': 
                    days_dict, latlon_station = fwi.end_date_calendar_csv(file_path_daily,year)
                else:
                    print('That is not a correct variable!')

                if interpolator == 'IDW2': 
                
                    num_stations = int(len(days_dict.keys()))
                    print('Number of stations:'+str(num_stations))
                    cluster_num1 = round(num_stations/5)

                    cluster_num2 = round(num_stations/10)

                    cluster_num3 = round(num_stations/15)

                    cluster_size,MAE = idw.select_block_size_IDW(10,'clusters',latlon_station,days_dict,idw1_grid,shapefile,file_path_elev,idx_list,2,\
                                                             cluster_num1,cluster_num2,cluster_num3)


                elif interpolator == 'IDW3':
                    num_stations = int(len(days_dict.keys()))
                    cluster_num1 = round(num_stations/5)

                    cluster_num2 = round(num_stations/10)

                    cluster_num3 = round(num_stations/15)

                    cluster_size,MAE = idw.select_block_size_IDW(10,'clusters',latlon_station,days_dict,idw1_grid,shapefile,file_path_elev,idx_list,3,\
                                                             cluster_num1,cluster_num2,cluster_num3)

                elif interpolator == 'IDW4':

                    num_stations = int(len(days_dict.keys()))
                    cluster_num1 = round(num_stations/5)

                    cluster_num2 = round(num_stations/10)

                    cluster_num3 = round(num_stations/15)

                    cluster_size,MAE = idw.select_block_size_IDW(10,'clusters',latlon_station,days_dict,idw1_grid,shapefile,file_path_elev,idx_list,4,\
                                                             cluster_num1,cluster_num2,cluster_num3)
                    
                elif interpolator == 'RF':
                    num_stations = int(len(days_dict.keys()))
                    print('Number of stations:'+str(num_stations))
                    cluster_num1 = round(num_stations/5)

                    cluster_num2 = round(num_stations/10)

                    cluster_num3 = round(num_stations/15)
                    cluster_size,MAE = rf.select_block_size_rf(10,'clusters',latlon_station,days_dict,idw1_grid,shapefile,file_path_elev,idx_list,cluster_num1,cluster_num2,cluster_num3)
                elif interpolator == 'TPSS':
                    num_stations = int(len(days_dict.keys()))
                    phi_input = int(num_stations)-(math.sqrt(2*num_stations))

                    print('Number of stations:'+str(num_stations))
                    cluster_num1 = round(num_stations/5)

                    cluster_num2 = round(num_stations/10)

                    cluster_num3 = round(num_stations/15)
                    cluster_size,MAE = tps.select_block_size_tps(10,'clusters',latlon_station,days_dict,idw1_grid,shapefile,file_path_elev,idx_list,\
                                                                 phi_input,cluster_num1,cluster_num2,cluster_num3)

                print(MAE)
                error_dict[year] = [MAE]
                end = time.time()
                time_elapsed = (end-start)/60
                print('Completed spatial bootstrap operation, it took %s minutes..'%(time_elapsed)) 
            
            df = pd.DataFrame(error_dict)
            df = df.transpose()
            df.iloc[:,0] = df.iloc[:,0].astype(str).str.strip('[|]')
            #df.iloc[:,1]= df.iloc[:,1].astype(str).str.strip('[|]')
            file_out = '' #Where you want to save the output csv

            df.to_csv(file_out+interpolator+'_'+var+'_fire_season_spatial_bootstrap.csv', header=None, sep=',', mode='a')
        

