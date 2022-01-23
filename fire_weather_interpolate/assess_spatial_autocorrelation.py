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

    dirname = 'C:/Users/clara/Documents/Earth_Space_Revisions/Moran/'

    variables = ['temp'] 

    dfs = []
    count = 0 
    for fp in paths:

        df = pd.read_csv(dirname+variables[0]+'/'+fp, delimiter = ",")
        dfs.append(df.dropna(how='any'))

    years = []
    values = []
    count = 0 
    for df in dfs:
        dates = list(df['Date'])
        years = [float(x[0:4]) for x in dates]
        if count == 0:
            years.append(dates)
        
##        df['Moran_I'].loc[df['p'] > 0.05] = np.nan
##        #df['Moran_I'].loc[df['Pos_Autocorrelation'] == 'No'] = np.nan
##        #df['Moran_I'].loc[df['Moran_I'] > 1] = 1
##        df['Moran_I'].loc[df['Num_Pairs'] < 30] = np.nan
##        values.append(list(df['Moran_I']))
##        print(list(df['Moran_I']))
##        mean_val = np.nanmean(np.array(list(df['Moran_I'])))
##        print(variables[count])
##        print(mean_val)
##        print(len(df['Moran_I'].dropna(how='any')))

        df = df.dropna(how='any') 
        df1 = df[df['Pos_Autocorrelation'] == 'No']
        df1['Pos_Autocorrelation'].loc[df['p'] < 0.05] = np.nan
        df1['Pos_Autocorrelation'].loc[df['Num_Pairs'] < 30] = np.nan
        df1.dropna(how='any')
        print('% neg') 
        print(len(df1)/len(df))
        df2 = df[df['Pos_Autocorrelation'] == 'Yes']
        df2['Pos_Autocorrelation'].loc[df['p'] >= 0.05] = np.nan
        df2['Pos_Autocorrelation'].loc[df['Num_Pairs'] < 30] = np.nan
        print('% pos')
        print(len(df2)/len(df))
        count+=1

def get_variogram(points,data,theor_model,num_lags,max_lag_150000):

    y = [i[0] for i in points]
    x = [j[0] for j in points]
    cval = data

    if max_lag_150000: 
    
        V = Variogram(list(zip(x, y)), cval, normalize=False, n_lags=num_lags, \
                      maxlag =  150000, model=theor_model,fit_method='trf')

    else: 

        #Variogram without max lag 

        V = Variogram(list(zip(x, y)), cval, normalize=False,n_lags=num_lags,\
                      model=theor_model,fit_method='trf')
    V.plot()
    print(V.describe())
    print(V.r)


#def get_heatmap():

    
    
if __name__ == "__main__":
    

    #plot_MI(['temp200p.txt'])

    #import time
    #time.sleep(60)
    
    from skgstat import Variogram

            
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
    for x in range(2000,2020):
        years.append(str(x))
    overall_dates = []
    
    for year in years: 
       overall_dates.append((year)+'-07-01 13:00')

    for x in [200,500,800,1000,1300,100000000000000000]: #10,20,50,100,500]: 
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
             
             #temp = GD.get_noon_temp(str(input_date)[0:10]+' 13:00',file_path_hourly)
             
             #print(temp) 
             #temp = GD.get_relative_humidity(str(input_date)[0:10]+' 13:00',file_path_hourly)
             temp = GD.get_wind_speed(str(input_date)[0:10]+' 13:00',file_path_hourly)
             #temp = GD.get_pcp(str(input_date)[0:10],file_path_daily,date_dictionary)

             import pysal
             #print(hourly_dictionary)

             points_for_weights = []
             temp_list = [] 
            
             for key,val in temp.items():
                 #coord = hourly_dictionary[key]
                 if key in list(hourly_dictionary.keys()): #daily
                     coord = hourly_dictionary[key]
                     lon = float(coord[1])
                     lat = float(coord[0])
                     Plat, Plon = pyproj.Proj('esri:102001')(lon, lat)
                     points_for_weights.append((Plon,Plat,))
                     temp_list.append(val)

            
             #get_variogram(points_for_weights,temp_list,'spherical',10,True)

             #import time
             #time.sleep(60)
             #print(points_for_weights)
                 
            
             lag1 = 0 * 1000
             w1 = pysal.weights.DistanceBand.from_array(points_for_weights,lag1)

             lag2 = x * 1000
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

            
             lag = x * 1000         
             #w = pysal.weights.DistanceBand.from_array(points_for_weights,lag,binary=True)
             w = pysal.weights.DistanceBand.from_array(filter_in,lag,binary=True)
             #w = pysal.Kernel(points_for_weights,bandwidth = 15.0)
             
             print('Points inside spatial lag:' +str(len(w[0])))
             #print(str(len(w[1])))

             #calculate n (total number of pairs)
             
             count = 0
##             for i in range(0,len(points_for_weights)):
##                 num_p = len(w[i])
##                 count += num_p
             for i in range(0,len(filter_in)):
                 num_p = len(w[i])
                 count += num_p
            
            
             num_pairs.append(count)
             print('Num pairs in class: '+str(count))
             print('Morans I:') 

             #mi = pysal.Moran(temp_list, w, two_tailed=False)
             try: 
                 mi = pysal.Moran(filter_temp, w, two_tailed=False)
                 print("%.3f"%mi.I)

                 print('p:')
                 print("%.5f"%mi.p_norm)

                 #Calculate ref value
                 ref2 =  mi.EI
                 ref = -1/(count-1)

                 ref_list.append(ref2)
                 print('Ref: '+str(ref))
                 print('Ref-2: '+str(ref2))

                 if float(mi.I) > ref2 and float(mi.p_norm) < 0.05: #and float(mi.I < 1):
                     pos_autocorrelation.append(1)
                     print('Yes')
                 else:
                     pos_autocorrelation.append(0)
                     print('No') 

                 mi_list.append(mi.I)
                 p_list.append(mi.p_norm)
                 date.append(input_date)
             except:
                 mi_list.append(np.nan)
                 p_list.append(np.nan)
                 date.append(np.nan)
                 ref_list.append(ref)
                 pos_autocorrelation.append('No')

        df = pd.DataFrame()
        df['Date'] = date

        df['Moran_I'] = mi_list
        df['p'] = p_list
        df['Ref'] = ref_list
        df['Pos_Autocorrelation'] = pos_autocorrelation
        
        print(df)
        out_dir = 'C:/Users/clara/Documents/Earth_Space_Revisions/Moran/wind/'
        df.to_csv(out_dir + str(x)+'wind_new_overall.csv') 

