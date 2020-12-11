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

    dirname = '' #insert dirname here


    file_path_daily = os.path.join(dirname, 'datasets/weather/daily_feather/')
    file_path_hourlyf = os.path.join(dirname, 'datasets/weather/hourly_feather/')
    #file_path_hourly = 'C:/Users/clara/OneDrive/Documents/Thesis/fwi-interpolate/datasets/weather/hourly_csv/'
    file_path_daily = os.path.join(dirname, 'datasets/weather/all_daily/all_daily/')
    shapefile = os.path.join(dirname, 'datasets/study_area/QC_ON_albers_dissolve.shp')
    boreal_shapefile = os.path.join(dirname, 'datasets/study_area/boreal_forest_SP.shp')

    file_path_elev = os.path.join(dirname,'datasets/lookup_files/elev_csv_200km.csv')
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
    interpolator_list = ['IDW3'] #,'IDW4','RF','TPSS','GPR'
    ecozones = ['boreal1_ecozone61','boreal2_easternC5','hudson','taiga_shield'] 
    for var in varlist:
        for interpolator in interpolator_list:
            error_dict = {}
            for year in range(1922,2020):
                start = time.time() 
                year = str(year) #Year of interest, right now we do not have an overwinter procedure so each year is run separately 

                print(year)

                start = time.time()
                if var == 'start':
                    
                    days_dict, latlon_station = fwi.start_date_calendar_csv(file_path_daily,year) #Get two things: start date for each station and the lat lon of the station
                    if int(year) >= 2020: #When many hourly stations are added
                        hourly_dictH, latlon_stationH = fwi.start_date_add_hourly(file_path_hourlyf,year)
                        if hourly_dictH is not None: #Sometimes there are no stations with unbroken records
                            days_dict = GD.combine_stations(days_dict,hourly_dictH)
                            latlon_station = GD.combine_stations(latlon_station,latlon_stationH)                                   
                
                elif var == 'end': 
                    days_dict, latlon_station = fwi.end_date_calendar_csv(file_path_daily,year)
                    if int(year) >= 2020: #When many hourly stations are added
                        hourly_dictH, latlon_stationH = fwi.end_date_add_hourly(file_path_hourlyf,year)
                        if hourly_dictH is not None: #Sometimes there are no stations with unbroken records
                            days_dict = GD.combine_stations(days_dict,hourly_dictH)
                            latlon_station = GD.combine_stations(latlon_station,latlon_stationH) 
                else:
                    print('That is not a correct variable!')

                if interpolator == 'IDW2':
                    grd_size,maxmin= idw.IDW(latlon_station,days_dict,year,'End Date (# Days since September 1)',shapefile,False,2,True)
                
                    try:
                        inBoreal = GD.is_station_in_boreal(latlon_station,days_dict,boreal_shapefile)
                        filtered_dict = {k: v for k, v in days_dict.items() if k in inBoreal}
                        num_stations = len(filtered_dict.keys()) #Number clusters= # stations / 3, /5, /10
                        cluster_num1 = int(round(num_stations/3)) 
                        cluster_num2 = int(round(num_stations/5))
                        cluster_num3 = int(round(num_stations/10)) 
                        cluster_num,MAE,stdev_stations = idw.select_block_size_IDW(10,'clusters',latlon_station,days_dict,grd_size,shapefile,\
                                                                                       file_path_elev,idx_list,2,cluster_num1,cluster_num2,\
                                                                                       cluster_num3,True,boreal_shapefile)
                    except ZeroDivisionError:
                        print('There are no stations in the boreal zone!')
                        MAE = 'NA'
                        stdev_stations = 'NA'
                        
                    ecozone_values = [] 
                    for zone in ecozones:
                        cwd = os.getcwd()
                        ecozone_shapefile = cwd+'/ecozone_shp/'+zone+'.shp'
                        boolean_map = GD.get_intersect_boolean_array(ecozone_shapefile,shapefile,False,True)
                        surface, maxmin= idw.IDW(latlon_station,days_dict,year,'# Days',shapefile,False,2,True,True)
                        AvVal = GD.get_average_in_ecozone(boolean_map,surface)
                        ecozone_values.append(AvVal)
                        
                elif interpolator == 'IDW3':
                    grd_size,maxmin= idw.IDW(latlon_station,days_dict,year,'End Date (# Days since September 1)',shapefile,False,3,True)
                
                    try:
                        inBoreal = GD.is_station_in_boreal(latlon_station,days_dict,boreal_shapefile)
                        filtered_dict = {k: v for k, v in days_dict.items() if k in inBoreal}
                        num_stations = len(filtered_dict.keys()) #Number clusters= # stations / 3, /5, /10
                        cluster_num1 = int(round(num_stations/3)) 
                        cluster_num2 = int(round(num_stations/5))
                        cluster_num3 = int(round(num_stations/10)) 
                        cluster_num,MAE,stdev_stations = idw.select_block_size_IDW(10,'clusters',latlon_station,days_dict,grd_size,shapefile,\
                                                                                       file_path_elev,idx_list,2,cluster_num1,cluster_num2,\
                                                                                       cluster_num3,True,boreal_shapefile)
                    except ZeroDivisionError:
                        print('There are no stations in the boreal zone!')
                        MAE = 'NA'
                        stdev_stations = 'NA'
                        
                    ecozone_values = [] 
                    for zone in ecozones:
                        cwd = os.getcwd()
                        ecozone_shapefile = cwd+'/ecozone_shp/'+zone+'.shp'
                        boolean_map = GD.get_intersect_boolean_array(ecozone_shapefile,shapefile,False,True)
                        surface, maxmin= idw.IDW(latlon_station,days_dict,year,'# Days',shapefile,False,2,True)
                        AvVal = GD.get_average_in_ecozone(boolean_map,surface)
                        ecozone_values.append(AvVal)

                elif interpolator == 'IDW4':
                    grd_size,maxmin= idw.IDW(latlon_station,days_dict,year,'End Date (# Days since September 1)',shapefile,False,4,True)
                
                    try:
                        inBoreal = GD.is_station_in_boreal(latlon_station,days_dict,boreal_shapefile)
                        filtered_dict = {k: v for k, v in days_dict.items() if k in inBoreal}
                        num_stations = len(filtered_dict.keys()) #Number clusters= # stations / 3, /5, /10
                        cluster_num1 = int(round(num_stations/3)) 
                        cluster_num2 = int(round(num_stations/5))
                        cluster_num3 = int(round(num_stations/10)) 
                        cluster_num,MAE,stdev_stations = idw.select_block_size_IDW(10,'clusters',latlon_station,days_dict,grd_size,shapefile,\
                                                                                       file_path_elev,idx_list,4,cluster_num1,cluster_num2,\
                                                                                       cluster_num3,True,boreal_shapefile)
                    except ZeroDivisionError:
                        print('There are no stations in the boreal zone!')
                        MAE = 'NA'
                        stdev_stations = 'NA'
                        
                    ecozone_values = [] 
                    for zone in ecozones:
                        cwd = os.getcwd()
                        ecozone_shapefile = cwd+'/ecozone_shp/'+zone+'.shp'
                        boolean_map = GD.get_intersect_boolean_array(ecozone_shapefile,shapefile,False,True)
                        surface, maxmin= idw.IDW(latlon_station,days_dict,year,'# Days',shapefile,False,4,True)
                        AvVal = GD.get_average_in_ecozone(boolean_map,surface)
                        ecozone_values.append(AvVal)

                elif interpolator == 'RF':
                    grd_size,maxmin= rf.random_forest_interpolator(latlon_station,days_dict,year,'# Days',shapefile,False,file_path_elev,idx_list,True)
                
                    try:
                        inBoreal = GD.is_station_in_boreal(latlon_station,days_dict,boreal_shapefile)
                        filtered_dict = {k: v for k, v in days_dict.items() if k in inBoreal}
                        num_stations = len(filtered_dict.keys()) #Number clusters= # stations / 3, /5, /10
                        cluster_num1 = int(round(num_stations/3)) 
                        cluster_num2 = int(round(num_stations/5))
                        cluster_num3 = int(round(num_stations/10)) 
                        cluster_num,MAE,stdev_stations = rf.select_block_size_rf(10,'clusters',latlon_station,days_dict,grd_size,shapefile,\
                                                                                       file_path_elev,idx_list,cluster_num1,cluster_num2,\
                                                                                       cluster_num3,True,boreal_shapefile)
                    except ZeroDivisionError:
                        print('There are no stations in the boreal zone!')
                        MAE = 'NA'
                        stdev_stations = 'NA'

                        
                    ecozone_values = [] 
                    for zone in ecozones:
                        cwd = os.getcwd()
                        ecozone_shapefile = cwd+'/ecozone_shp/'+zone+'.shp'
                        boolean_map = GD.get_intersect_boolean_array(ecozone_shapefile,shapefile,False,True)
                        surface, maxmin = rf.random_forest_interpolator(latlon_station,days_dict,year,'# Days',shapefile,False,file_path_elev,idx_list,True)
                        AvVal = GD.get_average_in_ecozone(boolean_map,surface)
                        ecozone_values.append(AvVal)

                elif interpolator == 'TPSS':
                    grd_size,maxmin= tps.TPS(latlon_station,days_dict,year,'# Days',shapefile,False,None,True,True)
                
                    try:
                        inBoreal = GD.is_station_in_boreal(latlon_station,days_dict,boreal_shapefile)
                        filtered_dict = {k: v for k, v in days_dict.items() if k in inBoreal}
                        num_stations = len(filtered_dict.keys()) #Number clusters= # stations / 3, /5, /10
                        cluster_num1 = int(round(num_stations/3)) 
                        cluster_num2 = int(round(num_stations/5))
                        cluster_num3 = int(round(num_stations/10)) 
                        cluster_num,MAE,stdev_stations = tps.select_block_size_tps(10,'clusters',latlon_station,days_dict,grd_size,shapefile,\
                                                                                       file_path_elev,idx_list,cluster_num1,cluster_num2,\
                                                                                       cluster_num3,True,boreal_shapefile,True)
                    except ZeroDivisionError:
                        print('There are no stations in the boreal zone!')
                        MAE = 'NA'
                        stdev_stations = 'NA'
                        
                    ecozone_values = []
                    for zone in ecozones:
                        cwd = os.getcwd()
                        ecozone_shapefile = cwd+'/ecozone_shp/'+zone+'.shp'
                        boolean_map = GD.get_intersect_boolean_array(ecozone_shapefile,shapefile,False,True)
                        surface, maxmin = tps.TPS(latlon_station,days_dict,year,'# Days',shapefile,False,None,True,True)
                        AvVal = GD.get_average_in_ecozone(boolean_map,surface)
                        ecozone_values.append(AvVal)


                elif interpolator == 'GPR':
                    grd_size,maxmin= gpr.GPR_interpolator(latlon_station,days_dict,year,'# Days',shapefile,file_path_elev,idx_list,0.1,True)
                
                    try:
                        inBoreal = GD.is_station_in_boreal(latlon_station,days_dict,boreal_shapefile)
                        filtered_dict = {k: v for k, v in days_dict.items() if k in inBoreal}
                        num_stations = len(filtered_dict.keys()) #Number clusters= # stations / 3, /5, /10
                        cluster_num1 = int(round(num_stations/3)) 
                        cluster_num2 = int(round(num_stations/5))
                        cluster_num3 = int(round(num_stations/10)) 
                        cluster_num,MAE,stdev_stations = gpr.select_block_size_gpr(10,'clusters',latlon_station,days_dict,grd_size,shapefile,\
                                                                                       file_path_elev,idx_list,cluster_num1,cluster_num2,\
                                                                                       cluster_num3,True,boreal_shapefile,0.1)
                    except ZeroDivisionError:
                        print('There are no stations in the boreal zone!')
                        MAE = 'NA'
                        stdev_stations = 'NA'
                        
                    ecozone_values = []
                    for zone in ecozones:
                        cwd = os.getcwd()
                        ecozone_shapefile = cwd+'/ecozone_shp/'+zone+'.shp'
                        boolean_map = GD.get_intersect_boolean_array(ecozone_shapefile,shapefile,False,True)
                        surface, maxmin = gpr.GPR_interpolator(latlon_station,days_dict,year,'# Days',shapefile,file_path_elev,idx_list,0.1,True)
                        AvVal = GD.get_average_in_ecozone(boolean_map,surface)
                        ecozone_values.append(AvVal)
                else: 
                    print('That is not a valid interpolator')
            
                
                print('MAE:'+str(MAE)) 
                print('STDEV:'+str(stdev_stations))
                print('AVERAGE VALS IN ECOZONES - BOREAL WEST - BOREAL EAST - HUDSON - TAIGA:'+str(ecozone_values))
                error_dict[year] = [MAE, stdev_stations,ecozone_values[0],ecozone_values[1],ecozone_values[2],ecozone_values[3]]
                end = time.time()
                time_elapsed = (end-start)/60
                print('Completed SSS crossvalidation operation, it took %s minutes..'%(time_elapsed)) 
            
            df = pd.DataFrame(error_dict)
            df = df.transpose()
            df.iloc[:,0] = df.iloc[:,0].astype(str).str.strip('[|]')
            df.iloc[:,1]= df.iloc[:,1].astype(str).str.strip('[|]')
            df.iloc[:,2]= df.iloc[:,2].astype(str).str.strip('[|]')
            df.iloc[:,3]= df.iloc[:,3].astype(str).str.strip('[|]')
            df.iloc[:,4]= df.iloc[:,4].astype(str).str.strip('[|]')
            df.iloc[:,5]= df.iloc[:,5].astype(str).str.strip('[|]')
            file_out = '' #Where you want to save the output csv

            df.to_csv(file_out+interpolator+'_'+var+'_fire_season_averages_added_dec9.csv', header=None, sep=',', mode='a')
