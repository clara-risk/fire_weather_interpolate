import fiona #must be here 
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

def plot_MI(paths):

    variables = ['ws','temp','rh','pcp'] 

    dfs = []

    for fp in paths:

        df = pd.read_csv(fp, delimiter = ",")
        dfs.append(df)

    years = []
    values = []
    count = 0 
    for df in dfs:
        dates = list(df['Date'])
        years = [float(x[0:4]) for x in dates]
        if count == 0:
            years.append(dates)
        
        df['Moran_I'].loc[df['p'] > 0.01] = np.nan
        df['Moran_I'].loc[df['Pos_Autocorrelation'] == 'No'] = np.nan
        values.append(list(df['Moran_I']))

        mean_val = np.nanmean(np.array(list(df['Moran_I'])))
        print(variables[count])
        print(mean_val)
        count+=1

if __name__ == "__main__":

    plot_MI(['20km_ws.txt','20km.txt','20km_rh.txt'])
            
    dirname = 'C:/Users/clara/Documents/cross-validation/'

    file_path_daily = os.path.join(dirname, 'datasets/weather/daily_feather/')
    file_path_se_dates = os.path.join(dirname, 'datasets/weather/all_daily/all_daily/')
    #file_path_se_dates  = 'C:/Users/clara/OneDrive/Documents/Thesis/summer2020/weather_engine/all_daily/'
    file_path_hourly = os.path.join(dirname, 'datasets/weather/hourly_feather/')
    shapefile = os.path.join(dirname, 'datasets/study_area/QC_ON_albers_dissolve.shp')
    #shapefile = os.path.join(dirname, 'datasets/study_area/justON_proj.shp')
    boreal_shapefile = os.path.join(dirname, 'datasets/study_area/boreal_forest_SP.shp')


    file_path_elev = os.path.join(dirname,'datasets/lookup_files/elev_csv.csv')
    idx_list = GD.get_col_num_list(file_path_elev,'elev')

    save = 'C:/Users/clara/Documents/fire_season/fwi_surfaces/'

    with open(dirname+'datasets/json/daily_lookup_file_TEMP.json', 'r') as fp:
        date_dictionary = json.load(fp) #Get the lookup file for the stations with data on certain months/years

    with open(dirname+'datasets/json/daily_lat_lon_TEMP.json', 'r') as fp:
        daily_dictionary = json.load(fp) #Get the latitude and longitude for the stations

    with open(dirname+'datasets/json/hourly_lat_lon_TEMP.json', 'r') as fp:
        hourly_dictionary = json.load(fp) #Get the latitude and longitude for the stations
 
    years = [] 
    for x in range(1956,2020):
        years.append(str(x))
    overall_dates = []
    
    for year in years: 
       overall_dates.append((year)+'-07-01 13:00')


    mi_list = []
    p_list =[]
    date = []
    num_pairs = []
    ref_list = []
    pos_autocorrelation = []

    for input_date in overall_dates:
         print(input_date)
         gc.collect()
         #Get the dictionary
         
         temp = GD.get_noon_temp(str(input_date)[0:10]+' 13:00',file_path_hourly)
         #print(temp) 
         rh = GD.get_relative_humidity(str(input_date)[0:10]+' 13:00',file_path_hourly)
         wind = GD.get_wind_speed(str(input_date)[0:10]+' 13:00',file_path_hourly)
         pcp = GD.get_pcp(str(input_date)[0:10],file_path_daily,date_dictionary)

         import pysal
         #print(hourly_dictionary)

         points_for_weights = []
         temp_list = [] 
        
         for key,val in pcp.items():
             #coord = hourly_dictionary[key]
             if key in list(daily_dictionary.keys()):
                 coord = daily_dictionary[key]
                 lon = float(coord[1])
                 lat = float(coord[0])
                 Plat, Plon = pyproj.Proj('esri:102001')(lon, lat)
                 points_for_weights.append((Plon,Plat,))
                 temp_list.append(val)

        
         #print(points_for_weights)
             
        
         lag1 = 0 * 1000
         w1 = pysal.weights.DistanceBand.from_array(points_for_weights,lag1)

         lag2 = 20 * 1000
         w2 = pysal.weights.DistanceBand.from_array(points_for_weights,lag2)

        
         filter_out = []
         filter_in = []
         filter_temp = [] 
         for i in range(0,len(points_for_weights)):
             if len(w1[i]) > 0:
                filter_out.append(points_for_weights[i])
                
             else:
                filter_in.append(points_for_weights[i])
                filter_temp.append(temp_list[i])
                
         w = pysal.weights.DistanceBand.from_array(filter_in,lag2)
         #w = pysal.Kernel(points_for_weights,bandwidth = 15.0)
         
         #print('Points inside spatial lag:' +str(len(w[0])))
         #print(str(len(w[1])))

         #calculate n (total number of pairs)
         
         count = 0
         for i in range(0,len(filter_in)):
             num_p = len(w[i])
             count += num_p

        
        
         num_pairs.append(count)
         print('Num pairs in class: '+str(count))
         print('Morans I:') 

         mi = pysal.Moran(filter_temp, w, two_tailed=False)
         print("%.3f"%mi.I)

         print('p:')
         print("%.5f"%mi.p_norm)

         #Calculate ref value
         ref = -1/(count-1)

         ref_list.append(ref)
         print('Ref: '+str(ref))

         if float(mi.I) > ref and float(mi.I < 1):
             pos_autocorrelation.append('Yes')
             print('Yes')
         else:
             pos_autocorrelation.append('No')
             print('No') 

         mi_list.append(mi.I)
         p_list.append(mi.p_norm)
         date.append(input_date)

    df = pd.DataFrame()
    df['Date'] = date

    df['Moran_I'] = mi_list
    df['p'] = p_list
    df['Ref'] = ref_list
    df['Pos_Autocorrelation'] = pos_autocorrelation
    df['Num_Pairs'] = num_pairs
    
    print(df)

    df.to_csv('20km_pcp.txt',sep=',') 
