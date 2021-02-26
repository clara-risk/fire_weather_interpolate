#coding: utf-8

"""
Summary
-------
Auto-selection protocol for spatial interpolation methods using shuffle-split cross-validation. 
References
----------
"""
import idw as idw
import idew as idew
import gpr as gpr
import tps as tps
import rf as rf

import fwi as fwi 
import get_data as GD
import Eval as Eval

import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import os, sys, json
import time
from datetime import datetime, timedelta, date
import gc


def run_comparison(var_name,input_date,interpolation_types,rep,loc_dictionary,cvar_dictionary,file_path_elev,elev_array,idx_list,phi_input=None,calc_phi=True,\
                   kernels={'temp':['316**2 * Matern(length_scale=[5e+05, 5e+05, 6.01e+03], nu=0.5)']\
                            ,'rh':['307**2 * Matern(length_scale=[5e+05, 6.62e+04, 1.07e+04], nu=0.5)'],\
                            'pcp':['316**2 * Matern(length_scale=[5e+05, 5e+05, 4.67e+05], nu=0.5)'],\
                            'wind':['316**2 * Matern(length_scale=[5e+05, 6.62e+04, 1.07e+04], nu=0.5)']}):
     '''Execute the shuffle-split cross-validation for the given interpolation types 
     Parameters
         interpolation_types (list of str): list of interpolation types to consider
     Returns 
         interpolation_best (str): returns the selected interpolation type name 
     '''
     MAE_dict = {} 
     for method in interpolation_types:
         if method not in ['IDW2','IDW3','IDW4','IDEW2','IDEW3','IDEW4','TPS','GPR','RF']:
             print('The method %s is not currently a supported interpolation type.'%(method))
             sys.exit() 
                
         else:
             if method == 'IDW2':
                 MAE = idw.shuffle_split(loc_dictionary,cvar_dictionary,shapefile,2,rep,False)
                 MAE_dict[method] = MAE
                 
             if method == 'IDW3':
                 MAE = idw.shuffle_split(loc_dictionary,cvar_dictionary,shapefile,3,rep,False)
                 MAE_dict[method] = MAE
                
             if method == 'IDW4':
                 MAE = idw.shuffle_split(loc_dictionary,cvar_dictionary,shapefile,4,rep,False)
                 MAE_dict[method] = MAE

             if method == 'IDEW2':
                 MAE = idew.shuffle_split_IDEW(loc_dictionary,cvar_dictionary,shapefile,file_path_elev,elev_array,idx_list,2,rep)
                 MAE_dict[method] = MAE

             if method == 'IDEW3':
                 MAE = idew.shuffle_split_IDEW(loc_dictionary,cvar_dictionary,shapefile,file_path_elev,elev_array,idx_list,3,rep)
                 MAE_dict[method] = MAE

             if method == 'IDEW4':
                 MAE = idew.shuffle_split_IDEW(loc_dictionary,cvar_dictionary,shapefile,file_path_elev,elev_array,idx_list,4,rep)
                 MAE_dict[method] = MAE

             if method == 'TPS':
                 MAE= tps.shuffle_split_tps(loc_dictionary,cvar_dictionary,shapefile,phi_input,10)
                 MAE_dict[method] = MAE

             if method == 'RF':
                 MAE = rf.shuffle_split_rf(loc_dictionary,cvar_dictionary,shapefile,file_path_elev,elev_array,idx_list,10)
                 MAE_dict[method] = MAE
                  
             if method == 'GPR':
                 MAE = gpr.shuffle_split_gpr(loc_dictionary,cvar_dictionary,shapefile,file_path_elev,elev_array,idx_list,kernels[var_name],10)
                 MAE_dict[method] = MAE

                     
     best_method = min(MAE_dict, key=MAE_dict.get)
     print('The best method for %s is: '%(best_method)) 
     if method == 'IDW2':
         choix_surf, maxmin = idw.IDW(loc_dictionary,cvar_dictionary,input_date,'Variable',shapefile,False,2,False) #Expand_area is not supported yet
         
     if method == 'IDW3':
         choix_surf, maxmin = idw.IDW(loc_dictionary,cvar_dictionary,input_date,'Variable',shapefile,False,3,False) #Expand_area is not supported yet
        
     if method == 'IDW4':
         choix_surf, maxmin = idw.IDW(loc_dictionary,cvar_dictionary,input_date,'Variable',shapefile,False,4,False) #Expand_area is not supported yet
         
     if method == 'IDEW2':
         choix_surf, maxmin, elev_array = idew.IDEW(loc_dictionary,cvar_dictionary,input_date,'Variable',shapefile,False,file_path_elev,idx_list,2) #Expand_area is not supported yet

     if method == 'IDEW3':
         choix_surf, maxmin, elev_array = idew.IDEW(loc_dictionary,cvar_dictionary,input_date,'Variable',shapefile,False,file_path_elev,idx_list,3)

     if method == 'IDEW4':
         choix_surf, maxmin, elev_array = idew.IDEW(loc_dictionary,cvar_dictionary,input_date,'Variable',shapefile,False,file_path_elev,idx_list,4)

     if method == 'TPS':
         choix_surf, maxmin = tps.TPS(loc_dictionary,cvar_dictionary,input_date,'Variable',shapefile,False,phi_input,False,calc_phi)
             
     if method == 'RF':
         choix_surf, maxmin = rf.random_forest_interpolator(loc_dictionary,cvar_dictionary,input_date,'Variable',shapefile,False,file_path_elev,idx_list,False)

          
     if method == 'GPR':
         choix_surf, maxmin = gpr.GPR_interpolator(loc_dictionary,cvar_dictionary,input_date,'Variable',shapefile,False,\
                     file_path_elev,idx_list,False,kernels[var_name],None,None,False,False)
        

     return best_method, choix_surf, maxmin

def execute_sequential_calc(file_path_hourly,file_path_daily,file_path_daily_csv,loc_dictionary_hourly, loc_dictionary_daily, date_dictionary,\
                            year,interpolation_types,rep,file_path_elev,idx_list,save_path,phi_input=None,calc_phi=True,\
                   kernels={'temp':['316**2 * Matern(length_scale=[5e+05, 5e+05, 6.01e+03], nu=0.5)']\
                            ,'rh':['307**2 * Matern(length_scale=[5e+05, 6.62e+04, 1.07e+04], nu=0.5)'],\
                            'pcp':['316**2 * Matern(length_scale=[5e+05, 5e+05, 4.67e+05], nu=0.5)'],\
                            'wind':['316**2 * Matern(length_scale=[5e+05, 6.62e+04, 1.07e+04], nu=0.5)']}):
     '''Execute the DC, DMC, FFMC seq calculations  
     Parameters
         interpolation_types (list of str): list of interpolation types to consider
     Returns 
         interpolation_best (str): returns the selected interpolation type name 
     '''
     #Fire season start and end dates
     start = time.time() 
     start_dict, latlon_station = fwi.start_date_calendar_csv(file_path_daily_csv,year) #Get two things: start date for each station and the lat lon of the station
     end_dict, latlon_station = fwi.end_date_calendar_csv(file_path_daily_csv,year,'oct') #start searching from Oct 1
        
        
     daysurface, maxmin= idw.IDW(latlon_station,start_dict,year,'# Days Since March 1',shapefile,False,3,False) #Interpolate the start date, IDW3
     endsurface, maxmin= idw.IDW(latlon_station,end_dict,year,'# Days Since Oct 1',shapefile,False,3,False) #Interpolate the end date

     end_dc_vals = np.zeros(endsurface.shape) #For now, no overwinter procedure
     end = time.time()
     time_elapsed = (end-start)/60
     print('Finished getting season start & end dates, it took %s minutes'%(time_elapsed/60))

        #Initialize the input elev_array (which is stable)
     placeholder_surf, maxmin, elev_array = idew.IDEW(loc_dictionary_hourly,end_dict,'placeholder','Variable',shapefile,False,\
                                                          file_path_elev,idx_list,2)
     
     #Get the dates in the fire season, overall, the surfaces will take care of masking
     sdate = pd.to_datetime(year+'-03-01').date() #Get the start date to start (if its too early everything will be masked out so can put any day before april)
     edate = pd.to_datetime(year+'-12-31').date() #End date, for right now it's Dec 31
     dates = list(pd.date_range(sdate,edate-timedelta(days=1),freq='d')) #Get the dates for all the potential days in the season 
     dc_list = []
     count = 0 
     for input_date in dates:
         print(input_date)
         gc.collect()
         #Get the dictionary
         start = time.time() 
         temp = GD.get_noon_temp(str(input_date)[:-3],file_path_hourly)
         rh = GD.get_relative_humidity(str(input_date)[:-3],file_path_hourly)
         wind = GD.get_wind_speed(str(input_date)[:-3],file_path_hourly)
         pcp = GD.get_pcp(str(input_date)[0:10],file_path_daily,date_dictionary)

         end = time.time()
         time_elapsed = end-start
         print('Finished getting weather dictionaries, it took %s seconds'%(time_elapsed))

         start = time.time() 

    
         best_interp_temp,choice_surf_temp,maxmin = run_comparison('temp',input_date,interpolation_types,rep,loc_dictionary_hourly,temp,file_path_elev,elev_array,idx_list)

         best_interp_rh,choice_surf_rh,maxmin = run_comparison('rh',input_date,interpolation_types,rep,loc_dictionary_hourly,rh,file_path_elev,elev_array,idx_list)
         best_interp_wind,choice_surf_wind,maxmin  = run_comparison('wind',input_date,interpolation_types,rep,loc_dictionary_hourly,wind,file_path_elev,elev_array,idx_list)
         best_interp_pcp,choice_surf_pcp,maxmin  = run_comparison('pcp',input_date,interpolation_types,rep,loc_dictionary_daily,pcp,file_path_elev,elev_array,idx_list)

         end = time.time()
         time_elapsed = end-start
         print('Finished getting best methods & surfaces, it took %s seconds'%(time_elapsed))

         #Get date index information
         year = str(input_date)[0:4]
         index = dates.index(input_date)
         dat = str(input_date)
         day_index= get_date_index(year,dat,3)
         eDay_index = get_date_index(year,dat,10)

         start = time.time() 

         mask1 = fwi.make_start_date_mask(day_index,daysurface)
         if eDay_index < 0:
             endMask = np.ones(endsurface.shape) #in the case that the index is before Oct 1
         else:
             endMask = fwi.make_end_date_mask(eDay_index,endsurface)
            
         
            
         if count > 0:  
            dc_array = dc_list[count-1] #the last one added will be yesterday's val, but there's a lag bc none was added when count was0, so just use count-1
            index = count-1
            dc = fwi.DC(input_date,choice_surf_pcp,choice_surf_rh,choice_surf_temp,choice_surf_wind,maxmin,\
                    dc_array,index,False,shapefile,mask1,endMask,None,False)
            dc_list.append(dc)
            
         else:
             rain_shape = rain_grid.shape
             dc_initialize = np.zeros(rain_shape)+15 #merge with the other overwinter array once it's calculated
             dc_yesterday1 = dc_initialize*mask1
             dc_list.append(dc_yesterday1) #placeholder
         end = time.time()
         time_elapsed = end-start
         print('Finished getting DC for date in stream, it took %s seconds'%(time_elapsed))

         count += 1

     with open(save_path+year+'_DC_auto_select.json', 'w') as fp:
        json.dump(dc_list, fp)

     fwi.plot_july(dc_list,maxmin,year,'DC',shapefile)
    
     return dc_list 
        
            
#Let's test
if __name__ == "__main__":

    dirname = 'C:/Users/clara/OneDrive/Documents/fire_weather_interpolate-master/fire_weather_interpolate-master/fire_weather_interpolate/'  #Insert the directory name (where the code is) 
    file_path_daily = os.path.join(dirname, 'datasets/weather/daily_feather/')
    #file_path_se_dates = 'C:/Users/clara/OneDrive/Documents/november/all_daily/'
    file_path_se_dates  = 'C:/Users/clara/OneDrive/Documents/Thesis/summer2020/weather_engine/all_daily/'
    file_path_hourly = os.path.join(dirname, 'datasets/weather/hourly_feather/')
    shapefile = os.path.join(dirname, 'datasets/study_area/QC_ON_albers_dissolve.shp')

    file_path_elev = os.path.join(dirname,'datasets/lookup_files/elev_csv.csv')
    idx_list = GD.get_col_num_list(file_path_elev,'elev')

    save = 'C:/Users/clara/OneDrive/Documents/february/auto_test/'

    with open(dirname+'datasets/json/daily_lookup_file_TEMP.json', 'r') as fp:
        date_dictionary = json.load(fp) #Get the lookup file for the stations with data on certain months/years

    with open(dirname+'datasets/json/daily_lat_lon_TEMP.json', 'r') as fp:
        daily_dictionary = json.load(fp) #Get the latitude and longitude for the stations

    with open(dirname+'datasets/json/hourly_lat_lon_TEMP.json', 'r') as fp:
        hourly_dictionary = json.load(fp) #Get the latitude and longitude for the stations
 


    execute_sequential_calc(file_path_hourly,file_path_daily,file_path_se_dates,hourly_dictionary, daily_dictionary, date_dictionary,\
                            str(1987),['IDW2','IDW3'],10,file_path_elev,idx_list,save,phi_input=None,calc_phi=True,\
                   kernels={'temp':['316**2 * Matern(length_scale=[5e+05, 5e+05, 6.01e+03], nu=0.5)']\
                            ,'rh':['307**2 * Matern(length_scale=[5e+05, 6.62e+04, 1.07e+04], nu=0.5)'],\
                            'pcp':['316**2 * Matern(length_scale=[5e+05, 5e+05, 4.67e+05], nu=0.5)'],\
                            'wind':['316**2 * Matern(length_scale=[5e+05, 6.62e+04, 1.07e+04], nu=0.5)']})
