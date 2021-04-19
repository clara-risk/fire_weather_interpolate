#coding: utf-8
"""
Summary
-------
Code to make a figure showing the resulting dc code maps from all the interpolation methods. 
"""
    
#import
import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import os, sys, json
import time 
from descartes import PolygonPatch
import math

import get_data as GD
import idw as idw
import idew as idew
import ok as ok
import tps as tps
import fwi as fwi
import Eval as Eval
import rf as rf 

    
if __name__ == "__main__":
    
    dirname = '' #Here you can insert the path to the directory you are working in 

    file_path_daily = os.path.join(dirname, 'datasets/weather/daily_feather/')
    file_path_hourlyf = os.path.join(dirname, 'datasets/weather/hourly_feather/')
    shapefile = os.path.join(dirname, 'datasets/study_area/QC_ON_albers_dissolve.shp')
    lookup_file = os.path.join(dirname,'datasets/lookup_files/fire_dates.csv')

    with open(dirname+'datasets/json/daily_lookup_file_TEMP.json', 'r') as fp:
        date_dictionary = json.load(fp) #Get the lookup file for the stations with data on certain months/years

    with open(dirname+'datasets/json/daily_lat_lon_TEMP.json', 'r') as fp:
        latlon_dictionary = json.load(fp) #Get the latitude and longitude for the stations

    input_date = '2018-07-02 13:00'
    year = '2018'
    temp_dict = GD.get_noon_temp(input_date,file_path_hourlyf)
    file_path_hourlyf = os.path.join(dirname, 'datasets/weather/hourly_feather/')

    with open(dirname+'datasets/json/hourly_lat_lon_TEMP.json', 'r') as fp:
        latlon_dict = json.load(fp) #Get the latitude and longitude for the stations

    idw1_grid,maxmin = idw.IDW(latlon_dict,temp_dict,input_date,'Temperature (⁰C)',shapefile,False,1) #Run this to get maxmin, which is the extent of the map
    
    dirname2 = '' #Directory to where the json flat files are stored 
    dc_list_pathIDW1 = os.path.join(dirname2,'json_from_jupyter/dc_list/IDW-1/')
    with open(dc_list_pathIDW1+year+'_DC_IDW-1.json', 'r') as fp:
        dc_listIDW1 = json.load(fp)
    dc_listIDW1 = [np.array(x) for x in dc_listIDW1]

    daysinceMarch = fwi.get_date_index(input_date[0:4],input_date,3)
    idw1_grid = dc_listIDW1[daysinceMarch]

    dc_list_pathIDW2 = os.path.join(dirname2,'json_from_jupyter/dc_list/IDW-2/')
    with open(dc_list_pathIDW2+year+'_DC_IDW-2.json', 'r') as fp:
        dc_listIDW2 = json.load(fp)
    dc_listIDW2 = [np.array(x) for x in dc_listIDW2]

    daysinceMarch = fwi.get_date_index(input_date[0:4],input_date,3)
    idw2_grid = dc_listIDW2[daysinceMarch]

    dc_list_pathIDEW1 = os.path.join(dirname2,'json_from_jupyter/dc_list/IDEW-1/')
    with open(dc_list_pathIDEW1+year+'_DC_IDEW-1.json', 'r') as fp:
        dc_listIDEW1 = json.load(fp)
    dc_listIDEW1 = [np.array(x) for x in dc_listIDEW1]

    daysinceMarch = fwi.get_date_index(input_date[0:4],input_date,3)
    idew1_grid = dc_listIDEW1[daysinceMarch]

    dc_list_pathIDEW2 = os.path.join(dirname2,'json_from_jupyter/dc_list/IDEW-2/')
    with open(dc_list_pathIDEW2+year+'_DC_IDEW-2.json', 'r') as fp:
        dc_listIDEW2 = json.load(fp)
    dc_listIDEW2 = [np.array(x) for x in dc_listIDEW2]

    daysinceMarch = fwi.get_date_index(input_date[0:4],input_date,3)
    idew2_grid = dc_listIDEW2[daysinceMarch]

    dc_list_pathTPSS = os.path.join(dirname2,'json_from_jupyter/dc_list/TPSS/')
    with open(dc_list_pathTPSS+year+'_DC_TPSS.json', 'r') as fp:
        dc_listTPSS = json.load(fp)
    dc_listTPSS = [np.array(x) for x in dc_listTPSS]

    daysinceMarch = fwi.get_date_index(input_date[0:4],input_date,3)
    tpss_grid = dc_listTPSS[daysinceMarch]

    dc_list_pathRF = os.path.join(dirname2,'json_from_jupyter/dc_list/RF/')
    with open(dc_list_pathRF+year+'_DC_RF.json', 'r') as fp:
        dc_listRF = json.load(fp)
    dc_listRF = [np.array(x) for x in dc_listRF]

    daysinceMarch = fwi.get_date_index(input_date[0:4],input_date,3)
    rf_grid = dc_listRF[daysinceMarch]

    dc_list_pathOK = os.path.join(dirname2,'json_from_jupyter/dc_list/OK/')
    with open(dc_list_pathOK+year+'_DC_OK.json', 'r') as fp:
        dc_listOK = json.load(fp)
    dc_listOK = [np.array(x) for x in dc_listOK]

    daysinceMarch = fwi.get_date_index(input_date[0:4],input_date,3)
    ok_grid = dc_listOK[daysinceMarch]
    
    Eval.plot(shapefile,maxmin,idw1_grid,idw2_grid,idew1_grid,idew2_grid,tpss_grid,rf_grid,ok_grid,input_date,'Drought Code')
