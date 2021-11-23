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
from osgeo import ogr, gdal,osr

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
     for method in interpolation_types[var_name]:
         if method not in ['IDW2','IDW3','IDW4','IDEW2','IDEW3','IDEW4','TPS','GPR','RF']:
             print('The method %s is not currently a supported interpolation type.'%(method))
             sys.exit() 
                
         else:
             if method == 'IDW2':
                 MAE = idw.shuffle_split(loc_dictionary,cvar_dictionary,shapefile,2,rep,False,res=10000)
                 MAE_dict[method] = MAE
                 
             if method == 'IDW3':
                 MAE = idw.shuffle_split(loc_dictionary,cvar_dictionary,shapefile,3,rep,False,res=10000)
                 MAE_dict[method] = MAE
                
             if method == 'IDW4':
                 MAE = idw.shuffle_split(loc_dictionary,cvar_dictionary,shapefile,4,rep,False,res=10000)
                 MAE_dict[method] = MAE

             if method == 'IDEW2':
                 MAE = idew.shuffle_split_IDEW(loc_dictionary,cvar_dictionary,shapefile,file_path_elev,elev_array,idx_list,2,rep,res=10000)
                 MAE_dict[method] = MAE

             if method == 'IDEW3':
                 MAE = idew.shuffle_split_IDEW(loc_dictionary,cvar_dictionary,shapefile,file_path_elev,elev_array,idx_list,3,rep,res=10000)
                 MAE_dict[method] = MAE

             if method == 'IDEW4':
                 MAE = idew.shuffle_split_IDEW(loc_dictionary,cvar_dictionary,shapefile,file_path_elev,elev_array,idx_list,4,rep,res=10000)
                 MAE_dict[method] = MAE

             if method == 'TPS':
                 MAE= tps.shuffle_split_tps(loc_dictionary,cvar_dictionary,shapefile,10,res=10000)
                 MAE_dict[method] = MAE

             if method == 'RF':
                 MAE = rf.shuffle_split_rf(loc_dictionary,cvar_dictionary,shapefile,file_path_elev,elev_array,idx_list,10,res=10000)
                 MAE_dict[method] = MAE
                  
             if method == 'GPR':
                 MAE = gpr.shuffle_split_gpr(loc_dictionary,cvar_dictionary,shapefile,file_path_elev,elev_array,idx_list,kernels[var_name],10,res=10000)
                 MAE_dict[method] = MAE

                     
     best_method = min(MAE_dict, key=MAE_dict.get)
     print('The best method for %s is: %s'%(var_name,best_method)) 
     if method == 'IDW2':
         choix_surf, maxmin = idw.IDW(loc_dictionary,cvar_dictionary,input_date,'Variable',shapefile,False,2,False,res=10000) #Expand_area is not supported yet
         
     if method == 'IDW3':
         choix_surf, maxmin = idw.IDW(loc_dictionary,cvar_dictionary,input_date,'Variable',shapefile,False,3,False,res=10000) #Expand_area is not supported yet
        
     if method == 'IDW4':
         choix_surf, maxmin = idw.IDW(loc_dictionary,cvar_dictionary,input_date,'Variable',shapefile,False,4,False,res=10000) #Expand_area is not supported yet
         
     if method == 'IDEW2':
         choix_surf, maxmin, elev_array = idew.IDEW(loc_dictionary,cvar_dictionary,input_date,'Variable',shapefile,False,file_path_elev,idx_list,2,False,res=10000) #Expand_area is not supported yet

     if method == 'IDEW3':
         choix_surf, maxmin, elev_array = idew.IDEW(loc_dictionary,cvar_dictionary,input_date,'Variable',shapefile,False,file_path_elev,idx_list,3,False,res=10000)

     if method == 'IDEW4':
         choix_surf, maxmin, elev_array = idew.IDEW(loc_dictionary,cvar_dictionary,input_date,'Variable',shapefile,False,file_path_elev,idx_list,4,False,res=10000)

     if method == 'TPS':
         choix_surf, maxmin = tps.TPS(loc_dictionary,cvar_dictionary,input_date,'Variable',shapefile,False,phi_input,False,calc_phi,res=10000)
             
     if method == 'RF':
         choix_surf, maxmin = rf.random_forest_interpolator(loc_dictionary,cvar_dictionary,input_date,'Variable',shapefile,False,\
                                                            file_path_elev,idx_list,False,res=10000)

          
     if method == 'GPR':
         choix_surf, maxmin = gpr.GPR_interpolator(loc_dictionary,cvar_dictionary,input_date,'Variable',shapefile,False,\
                     file_path_elev,idx_list,False,kernels[var_name],0,False,False,res=10000)
         
        

     return best_method, choix_surf, maxmin

def execute_sequential_calc(file_path_hourly,file_path_daily,file_path_daily_csv,loc_dictionary_hourly, loc_dictionary_daily, date_dictionary,\
                            year,interpolation_types,rep,file_path_elev,idx_list,save_path,shapefile,shapefile2,phi_input=None,calc_phi=True,\
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
        #Initialize the input elev_array (which is stable)
     start_dict, latlon_station = fwi.start_date_calendar_csv(file_path_daily_csv,year) #Get two things: start date for each station and the lat lon of the station
     end_dict, latlon_station = fwi.end_date_calendar_csv(file_path_daily_csv,year,'oct') #start searching from Oct 1
     #Initialize the input elev_array (which is stable)
     placeholder_surf, maxmin, elev_array = idew.IDEW(loc_dictionary_hourly,end_dict,'placeholder','Variable',shapefile,False,\
                                                          file_path_elev,idx_list,2,True,res=10000)        
        
     #daysurface, maxmin= idw.IDW(latlon_station,start_dict,year,'# Days Since March 1',shapefile,False,3,False,res=10000) #Interpolate the start date, IDW3
     #endsurface, maxmin= idw.IDW(latlon_station,end_dict,year,'# Days Since Oct 1',shapefile,False,3,False,res=10000) #Interpolate the end date
     daysurface, maxmin = rf.random_forest_interpolator(latlon_station,start_dict,year,'# Days Since Oct 1',shapefile,False,file_path_elev,idx_list,False,res=10000)
     endsurface, maxmin = rf.random_forest_interpolator(latlon_station,end_dict,year,'# Days Since Oct 1',shapefile,False,file_path_elev,idx_list,False,res=10000) #Interpolate the end date
     
     end_dc_vals = np.zeros(endsurface.shape) #For now, no overwinter procedure
     end = time.time()
     time_elapsed = (end-start)/60
     print('Finished getting season start & end dates, it took %s minutes'%(time_elapsed/60))

        #Initialize the input elev_array (which is stable)
     #placeholder_surf, maxmin, elev_array = idew.IDEW(loc_dictionary_hourly,end_dict,'placeholder','Variable',shapefile,False,\
                                                          #file_path_elev,idx_list,2,True,res=28000)
     
     #Get the dates in the fire season, overall, the surfaces will take care of masking
     sdate = pd.to_datetime(year+'-03-01').date() #Get the start date to start (if its too early everything will be masked out so can put any day before april)
     edate = pd.to_datetime(year+'-12-31').date() #End date, for right now it's Dec 31
     #dates = list(pd.date_range(sdate,edate-timedelta(days=1),freq='d')) #Get the dates for all the potential days in the season
     dates = list(pd.date_range(sdate,edate,freq='d'))
     dc_list = []
     dmc_list = []
     dmcalt_list = [] 
     ffmc_list = []
     isi_list = []
     bui_list = []
     fwi_list = []
     temp_list = []
     rh_list = []
     wind_list = []
     pcp_list = [] 
     count = 0 
     for input_date in dates:
         print(input_date)
         gc.collect()
         #Get the dictionary
         start = time.time() 
         temp = GD.get_noon_temp(str(input_date)[0:10]+' 13:00',file_path_hourly)
         rh = GD.get_relative_humidity(str(input_date)[0:10]+' 13:00',file_path_hourly)
         wind = GD.get_wind_speed(str(input_date)[0:10]+' 13:00',file_path_hourly)
         pcp = GD.get_pcp(str(input_date)[0:10],file_path_daily,date_dictionary)

         end = time.time()
         time_elapsed = end-start
         print('Finished getting weather dictionaries, it took %s seconds'%(time_elapsed))

         start = time.time() 

         
         best_interp_temp,choice_surf_temp,maxmin = run_comparison('temp',input_date,interpolation_types,rep,loc_dictionary_hourly,temp,file_path_elev,elev_array,idx_list)
         #best_interp_temp,choice_surf_temp,maxmin = get_surf('temp',input_date,interpolation_types,rep,loc_dictionary_hourly,temp,file_path_elev,elev_array,idx_list)
         temp_list.append(choice_surf_temp)
         print(choice_surf_temp.shape)

         best_interp_rh,choice_surf_rh,maxmin = run_comparison('rh',input_date,interpolation_types,rep,loc_dictionary_hourly,rh,file_path_elev,elev_array,idx_list)

         #best_interp_rh,choice_surf_rh,maxmin = get_surf('rh',input_date,interpolation_types,rep,loc_dictionary_hourly,rh,file_path_elev,elev_array,idx_list)
         choice_surf_rh[choice_surf_rh > 100] = 100 
         rh_list.append(choice_surf_rh)

         best_interp_wind,choice_surf_wind,maxmin  = run_comparison('wind',input_date,interpolation_types,rep,loc_dictionary_hourly,wind,file_path_elev,elev_array,idx_list)

         #best_interp_wind,choice_surf_wind,maxmin  = get_surf('wind',input_date,interpolation_types,rep,loc_dictionary_hourly,wind,file_path_elev,elev_array,idx_list)
         choice_surf_wind[choice_surf_wind < 0] = 0 
         wind_list.append(choice_surf_wind)

         best_interp_pcp,choice_surf_pcp,maxmin  = run_comparison('pcp',input_date,interpolation_types,rep,loc_dictionary_daily,pcp,file_path_elev,elev_array,idx_list)

         #best_interp_pcp,choice_surf_pcp,maxmin  = get_surf('pcp',input_date,interpolation_types,rep,loc_dictionary_daily,pcp,file_path_elev,elev_array,idx_list)
         choice_surf_pcp[choice_surf_pcp < 0] = 0 #mask out negative values (convert negatives to 0)
         pcp_list.append(choice_surf_pcp)

         end = time.time()
         time_elapsed = end-start
         print('Finished getting best methods & surfaces, it took %s seconds'%(time_elapsed))

         #Get date index information
         year = str(input_date)[0:4]
         index = dates.index(input_date)
         dat = str(input_date)
         day_index= fwi.get_date_index(year,dat,3)
         eDay_index = fwi.get_date_index(year,dat,10)

         start = time.time() 

         mask1 = fwi.make_start_date_mask(day_index,daysurface)
         if eDay_index < 0:
             endMask = np.ones(endsurface.shape) #in the case that the index is before Oct 1
         else:
             endMask = fwi.make_end_date_mask(eDay_index,endsurface)
            
         
            
         if count > 0:  
            dc_array = dc_list[count-1] #the last one added will be yesterday's val, but there's a lag bc none was added when count was0, so just use count-1
            print(dc_array.shape)
            dmc_array = dmc_list[count-1]
            dmcalt_array = dmcalt_list[count-1]
            ffmc_array = ffmc_list[count-1]
            index = count-1
            dc = fwi.DC(input_date,choice_surf_pcp,choice_surf_rh,choice_surf_temp,choice_surf_wind,maxmin,\
                    dc_array,index,False,shapefile,mask1,endMask,None,False)
            dmc = fwi.DMC(input_date,choice_surf_pcp,choice_surf_rh,choice_surf_temp,choice_surf_wind,maxmin,\
                    dmc_array,index,False,shapefile,mask1,endMask)
            dmc_alt = fwi.DMC_alt(input_date,choice_surf_pcp,choice_surf_rh,choice_surf_temp,choice_surf_wind,maxmin,\
                    dmcalt_array,index,False,shapefile,mask1,endMask)
            ffmc = fwi.FFMC(input_date,choice_surf_pcp,choice_surf_rh,choice_surf_temp,choice_surf_wind,maxmin,\
                    ffmc_array,index,False,shapefile,mask1,endMask)

            isi = fwi.ISI(ffmc,choice_surf_wind,maxmin,\
                    False,shapefile,mask1,endMask)
            bui = fwi.BUI(dmc,dc,maxmin,\
                    False,shapefile,mask1,endMask)
            fwi_val = fwi.FWI(isi,bui,maxmin,\
                    False,shapefile,mask1,endMask)
            
            dc_list.append(dc)
            dmc_list.append(dmc)
            dmcalt_list.append(dmc_alt)
            ffmc_list.append(ffmc)
            isi_list.append(isi)
            bui_list.append(bui)
            fwi_list.append(fwi_val)
            
         else:
             rain_shape = choice_surf_pcp.shape
             dc_initialize = np.zeros(rain_shape)+15 #merge with the other overwinter array once it's calculated
             dc_yesterday1 = dc_initialize*mask1
             dc_list.append(dc_yesterday1) #placeholder

             rain_shape = choice_surf_pcp.shape
             dmc_initialize = np.zeros(rain_shape)+6 #merge with the other overwinter array once it's calculated
             dmc_yesterday1 = dmc_initialize*mask1
             dmc_list.append(dmc_yesterday1) #placeholder

             rain_shape = choice_surf_pcp.shape
             dmcalt_initialize = np.zeros(rain_shape)+6 #merge with the other overwinter array once it's calculated
             dmcalt_yesterday1 = dmcalt_initialize*mask1
             dmcalt_list.append(dmcalt_yesterday1) #placeholder

             rain_shape = choice_surf_pcp.shape
             ffmc_initialize = np.zeros(rain_shape)+85 #merge with the other overwinter array once it's calculated
             ffmc_yesterday1 = ffmc_initialize*mask1
             ffmc_list.append(ffmc_yesterday1) #placeholder
           
             isi_list.append(np.zeros(rain_shape))
             bui_list.append(np.zeros(rain_shape))
             fwi_list.append(np.zeros(rain_shape))
         if count > 0:
             read_data(dc_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/DC/')
             read_data(dmc_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/DMC/')
             read_data(dmcalt_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/DMC_alt/') 
             read_data(ffmc_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/FFMC/')
             read_data(isi_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/ISI/')
             read_data(bui_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/BUI/')
             read_data(fwi_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/FWI/')
             read_data(wind_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/wind/')
             read_data(temp_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/temp/')
             read_data(rh_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/rh/')
             read_data(pcp_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/pcp/')
             
             
         end = time.time()
         time_elapsed = end-start
         print('Finished getting DC for date in stream, it took %s seconds'%(time_elapsed))

         count += 1

     #prep to serialize
##     dc_list = [x.tolist() for x in dc_list]
##
##     dmc_list = [x.tolist() for x in dmc_list]
##
##     ffmc_list = [x.tolist() for x in ffmc_list]
##
##     isi_list = [x.tolist() for x in isi_list]
##
##     bui_list = [x.tolist() for x in bui_list]
##
##     fwi_list = [x.tolist() for x in fwi_list]
##
##     with open(save_path+year+'_DC_auto_select.json', 'w') as fp:
##        json.dump(dc_list, fp)
##
##     with open(save_path+year+'_DC_auto_select.json', 'r') as fp:
##        dc_list = json.load(fp)
##
##     with open(save_path+year+'_DMC_auto_select.json', 'w') as fp:
##        json.dump(dmc_list, fp)
##
##     with open(save_path+year+'_FFMC_auto_select.json', 'w') as fp:
##        json.dump(ffmc_list, fp)
##
##     with open(save_path+year+'_ISI_auto_select.json', 'w') as fp:
##        json.dump(isi_list, fp)
##
##     with open(save_path+year+'_BUI_auto_select.json', 'w') as fp:
##        json.dump(bui_list, fp)
##
##     with open(save_path+year+'_FWI_auto_select.json', 'w') as fp:
##        json.dump(fwi_list, fp)
##
##     dc_list = [np.array(x) for x in dc_list] #convert to np array for plotting 
##
##     fwi.plot_june(dc_list,maxmin,year,'DC',shapefile,shapefile2)
    
     return dc_list

def restart_calc(file_path_hourly,file_path_daily,file_path_daily_csv,loc_dictionary_hourly, loc_dictionary_daily, date_dictionary,\
                            year,interpolation_types,rep,file_path_elev,idx_list,save_path,shapefile,shapefile2,phi_input=None,calc_phi=True,\
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
     placeholder_surf, maxmin, elev_array = idew.IDEW(loc_dictionary_hourly,end_dict,'placeholder','Variable',shapefile,False,\
                                                          file_path_elev,idx_list,2,True,res=10000)     
     
##     wind_export = pd.DataFrame(loc_dictionary_hourly.items(),columns = ['name','latlon'])
##     print(wind_export['latlon'].astype(str).str[2:7])
##     wind_export['lat'] = wind_export['latlon'].astype(str).str[2:7]
##     wind_export['lon'] = wind_export['latlon'].astype(str).str[11:17]
##     wind_export.to_csv('C:/Users/clara/Documents/fire_season/new/stations_loc.csv',index=False)
        
     #daysurface, maxmin= idw.IDW(latlon_station,start_dict,year,'# Days Since March 1',shapefile,False,3,False,res=10000) #Interpolate the start date, IDW3
     #endsurface, maxmin= idw.IDW(latlon_station,end_dict,year,'# Days Since Oct 1',shapefile,False,3,False,res=10000) #Interpolate the end date
     daysurface, maxmin = rf.random_forest_interpolator(latlon_station,start_dict,year,'# Days Since Oct 1',shapefile,False,file_path_elev,idx_list,False,res=10000)
     endsurface, maxmin = rf.random_forest_interpolator(latlon_station,end_dict,year,'# Days Since Oct 1',shapefile,False,file_path_elev,idx_list,False,res=10000) #Interpolate the end date
     

     end_dc_vals = np.zeros(endsurface.shape) #For now, no overwinter procedure
     end = time.time()
     time_elapsed = (end-start)/60
     print('Finished getting season start & end dates, it took %s minutes'%(time_elapsed/60))
     #Get the dates in the fire season, overall, the surfaces will take care of masking
     sdate = pd.to_datetime(year+'-03-01').date() #Get the start date to start (if its too early everything will be masked out so can put any day before april)
     edate = pd.to_datetime(year+'-12-31').date() #End date, for right now it's Dec 31
     #dates = list(pd.date_range(sdate,edate-timedelta(days=1),freq='d')) #Get the dates for all the potential days in the season
     dates = list(pd.date_range(sdate,edate,freq='d'))
     dates_existing = [x[0:10] for x in os.listdir(save_path+str(year)+'/DC/')]
     
     dates_updated = [] 
     for dat1 in dates:
          if str(dat1)[0:10] not in dates_existing:
               dates_updated.append(dat1)
             
     
     dc_list = []
     dmc_list = []
     dmcalt_list = [] 
     ffmc_list = []
     isi_list = []
     bui_list = []
     fwi_list = []
     temp_list = []
     rh_list = []
     wind_list = []
     pcp_list = [] 
     count = 0 
     for input_date in dates_updated[1:]: #Skip first date, since this is March 1, which is the initiation date
         print(input_date)
         gc.collect()
         #Get the dictionary
         start = time.time() 
         temp = GD.get_noon_temp(str(input_date)[0:10]+' 13:00',file_path_hourly)
         rh = GD.get_relative_humidity(str(input_date)[0:10]+' 13:00',file_path_hourly)
         wind = GD.get_wind_speed(str(input_date)[0:10]+' 13:00',file_path_hourly)
         #wind_export = pd.DataFrame(wind.items())
         #wind_export.to_csv('C:/Users/clara/Documents/fire_season/new/june15_stations.csv',index=False)
         pcp = GD.get_pcp(str(input_date)[0:10],file_path_daily,date_dictionary)

         end = time.time()
         time_elapsed = end-start
         print('Finished getting weather dictionaries, it took %s seconds'%(time_elapsed))

         start = time.time() 

         
         best_interp_temp,choice_surf_temp,maxmin = run_comparison('temp',input_date,interpolation_types,rep,loc_dictionary_hourly,temp,file_path_elev,elev_array,idx_list)
         #best_interp_temp,choice_surf_temp,maxmin = get_surf('temp',input_date,interpolation_types,rep,loc_dictionary_hourly,temp,file_path_elev,elev_array,idx_list)
         temp_list.append(choice_surf_temp)
         print(choice_surf_temp.shape)

         best_interp_rh,choice_surf_rh,maxmin = run_comparison('rh',input_date,interpolation_types,rep,loc_dictionary_hourly,rh,file_path_elev,elev_array,idx_list)

         #best_interp_rh,choice_surf_rh,maxmin = get_surf('rh',input_date,interpolation_types,rep,loc_dictionary_hourly,rh,file_path_elev,elev_array,idx_list)
         choice_surf_rh[choice_surf_rh > 100] = 100 
         rh_list.append(choice_surf_rh)

         best_interp_wind,choice_surf_wind,maxmin  = run_comparison('wind',input_date,interpolation_types,rep,loc_dictionary_hourly,wind,file_path_elev,elev_array,idx_list)

         #best_interp_wind,choice_surf_wind,maxmin  = get_surf('wind',input_date,interpolation_types,rep,loc_dictionary_hourly,wind,file_path_elev,elev_array,idx_list)
         choice_surf_wind[choice_surf_wind < 0] = 0 
         wind_list.append(choice_surf_wind)

         best_interp_pcp,choice_surf_pcp,maxmin  = run_comparison('pcp',input_date,interpolation_types,rep,loc_dictionary_daily,pcp,file_path_elev,elev_array,idx_list)

         #best_interp_pcp,choice_surf_pcp,maxmin  = get_surf('pcp',input_date,interpolation_types,rep,loc_dictionary_daily,pcp,file_path_elev,elev_array,idx_list)
         choice_surf_pcp[choice_surf_pcp < 0] = 0 #mask out negative values (convert negatives to 0)
         pcp_list.append(choice_surf_pcp)

         end = time.time()
         time_elapsed = end-start
         print('Finished getting best methods & surfaces, it took %s seconds'%(time_elapsed))

         #Get date index information
         year = str(input_date)[0:4]
         index = dates.index(input_date)
         dat = str(input_date)
         day_index= fwi.get_date_index(year,dat,3)
         eDay_index = fwi.get_date_index(year,dat,10)

         start = time.time() 

         mask1 = fwi.make_start_date_mask(day_index,daysurface)
         if eDay_index < 0:
             endMask = np.ones(endsurface.shape) #in the case that the index is before Oct 1
         else:
             endMask = fwi.make_end_date_mask(eDay_index,endsurface)
            
         
            
         if count > 0:  
            dc_array = dc_list[count-1] #the last one added will be yesterday's val, but there's a lag bc none was added when count was0, so just use count-1
            dmc_array = dmc_list[count-1]
            ffmc_array = ffmc_list[count-1]
            index = count-1
            dc = fwi.DC(input_date,choice_surf_pcp,choice_surf_rh,choice_surf_temp,choice_surf_wind,maxmin,\
                    dc_array,index,False,shapefile,mask1,endMask,None,False)
            dmc = fwi.DMC(input_date,choice_surf_pcp,choice_surf_rh,choice_surf_temp,choice_surf_wind,maxmin,\
                    dmc_array,index,False,shapefile,mask1,endMask)
            dmc_alt = fwi.DMC_alt(input_date,choice_surf_pcp,choice_surf_rh,choice_surf_temp,choice_surf_wind,maxmin,\
                    dmcalt_array,index,False,shapefile,mask1,endMask)
            ffmc = fwi.FFMC(input_date,choice_surf_pcp,choice_surf_rh,choice_surf_temp,choice_surf_wind,maxmin,\
                    ffmc_array,index,False,shapefile,mask1,endMask)

            isi = fwi.ISI(ffmc,choice_surf_wind,maxmin,\
                    False,shapefile,mask1,endMask)
            bui = fwi.BUI(dmc,dc,maxmin,\
                    False,shapefile,mask1,endMask)
            fwi_val = fwi.FWI(isi,bui,maxmin,\
                    False,shapefile,mask1,endMask)
            
            dc_list.append(dc)
            dmc_list.append(dmc)
            dmcalt_list.append(dmc_alt)
            ffmc_list.append(ffmc)
            isi_list.append(isi)
            bui_list.append(bui)
            fwi_list.append(fwi_val)
            
         else:
             # Read in yesterdays dates from raster
             format_dat = datetime.strptime(dat, '%Y-%m-%d %H:%M:%S')
             yesterday = format_dat - timedelta(1)
             src_dc = gdal.Open(save_path+str(year)+'/DC/'+str(yesterday)[0:10]+'.tif')
             cols = src_dc.RasterXSize
             rows = src_dc.RasterYSize
             dc_initialize = src_dc.ReadAsArray(0, 0, cols, rows)
             dc_yesterday1 = dc_initialize*mask1
             dc_list.append(dc_yesterday1) #placeholder
             print('DC collected') 

             src_dmc = gdal.Open(save_path+str(year)+'/DMC/'+str(yesterday)[0:10]+'.tif')              
             dmc_initialize = src_dmc.ReadAsArray(0, 0, cols, rows)
             dmc_yesterday1 = dmc_initialize*mask1
             dmc_list.append(dmc_yesterday1) #placeholder
             print('DMC collected')

             src_dmc_a = gdal.Open(save_path+str(year)+'/DMC_alt/'+str(yesterday)[0:10]+'.tif')              
             dmc_initialize_a = src_dmc_a.ReadAsArray(0, 0, cols, rows)
             dmc_yesterday1_a = dmc_initialize_a*mask1
             dmcalt_list.append(dmc_yesterday1_a) #placeholder
             print('DMC_alt collected') 

             src_ffmc = gdal.Open(save_path+str(year)+'/FFMC/'+str(yesterday)[0:10]+'.tif')           
             ffmc_initialize = src_ffmc.ReadAsArray(0, 0, cols, rows)
             ffmc_yesterday1 = ffmc_initialize*mask1
             ffmc_list.append(ffmc_yesterday1) #placeholder
             print('FFMC collected')
             
             src_isi = gdal.Open(save_path+str(year)+'/ISI/'+str(yesterday)[0:10]+'.tif')               
             isi_initialize = src_isi.ReadAsArray(0, 0, cols, rows)
             isi_list.append(isi_initialize)
             
             src_bui = gdal.Open(save_path+str(year)+'/BUI/'+str(yesterday)[0:10]+'.tif')              
             bui_initialize = src_bui.ReadAsArray(0, 0, cols, rows)
             bui_list.append(bui_initialize)
             
             src_fwi = gdal.Open(save_path+str(year)+'/FWI/'+str(yesterday)[0:10]+'.tif')           
             fwi_initialize = src_fwi.ReadAsArray(0, 0, cols, rows)
             fwi_list.append(fwi_initialize)

             
         #if count > 0:
         print('Writing to drive.....') 
         read_data(dc_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/DC/')
         read_data(dmc_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/DMC/')
         read_data(dmcalt_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/DMC_alt/') 
         read_data(ffmc_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/FFMC/')
         read_data(isi_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/ISI/')
         read_data(bui_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/BUI/')
         read_data(fwi_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/FWI/')
         read_data(wind_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/wind/')
         read_data(temp_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/temp/')
         read_data(rh_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/rh/')
         read_data(pcp_list[-1],10000,shapefile,str(dat)[0:10],save_path+year+'/pcp/')
             
         end = time.time()
         time_elapsed = end-start
         print('Finished getting DC for date in stream, it took %s seconds'%(time_elapsed))

         count += 1
    
     return dc_list
     

from osgeo import ogr, gdal,osr
def read_data(nd_array,res,shp,day,save_path):

     # Read in the data in pandas

     array = np.array(nd_array)
     print(np.nanmax(array))

     # Create the raster

     na_map = gpd.read_file(shp)
     bounds = na_map.bounds
     xmax = bounds['maxx']
     xmin = bounds['minx']
     ymax = bounds['maxy']
     ymin = bounds['miny']

     pixelHeight = res
     pixelWidth = res

     num_col = int((xmax - xmin) / pixelHeight) + 1
     num_row = int((ymax - ymin) / pixelWidth) + 1


     print('Finished making raster parameters!') 

     #new_array = array.reshape((num_row,num_col))
     new_array = array.reshape((num_row,num_col))[::-1]

     driver = gdal.GetDriverByName('GTiff')
     srs = osr.SpatialReference()
     epsg_code = 'esri:102001'
     srs.SetFromUserInput(epsg_code)

     print('Making output file')

     output_file = driver.Create(save_path+str(day)+'.tif', num_col, num_row, 1, gdal.GDT_Float32) 
     output_file.SetGeoTransform((xmin, res, 0, ymax, 0, -res)) # Height and then width. 
     output_file.SetProjection(str(srs))
     output_file.GetRasterBand(1).WriteArray(new_array, 0, 0)

     output_file = None

def resample(output_res):
    '''Resample the rasters to a 10 by 10 grid 

    Parameters
    ----------
    output_res : int
        output resolution to resample to, 10000 for 10 km 
    path : int
        the path identifior of the image you want to obtain information about 
    row : int
        the row identifior of the image you want to obtain information about
    '''

    files_available = [name for name in os.listdir('outputs/')
                       if os.path.isfile(os.path.join('outputs/', name))
                       and os.path.join('outputs/', name).endswith('.tif')]

    for file_name in files_available:
        print(file_name) 
    
        out_path = 'outputs/resample_'+str(file_name)

        xres=output_res
        yres=output_res
        resample_algorithm = 'cubicspline'
        options = gdal.WarpOptions(xRes=xres, yRes=yres, resampleAlg=resample_algorithm)
        data_resampler = gdal.Warp(out_path, os.path.join('outputs/', file_name),options=options)
        data_resampler = None
        
            
#Let's test
if __name__ == "__main__":

    #dirname = 'C:/Users/clara/OneDrive/Documents/fire_weather_interpolate-master/fire_weather_interpolate-master/fire_weather_interpolate/'  #Insert the directory name (where the code is)
    #laptop
     
    dirname = 'C:/Users/clara/Documents/cross-validation/'

    file_path_daily = os.path.join(dirname, 'datasets/weather/daily_feather/')
    file_path_se_dates = os.path.join(dirname, 'datasets/weather/all_daily/all_daily/')
    #file_path_se_dates  = 'C:/Users/clara/OneDrive/Documents/Thesis/summer2020/weather_engine/all_daily/'
    file_path_hourly = os.path.join(dirname, 'datasets/weather/hourly_feather/')
    #shapefile = os.path.join(dirname, 'datasets/study_area/QC_ON_albers_dissolve.shp')
    shapefile = os.path.join(dirname, 'datasets/study_area/justON_proj.shp')
    boreal_shapefile = os.path.join(dirname, 'datasets/study_area/boreal_forest_SP.shp')


    file_path_elev = os.path.join(dirname,'datasets/lookup_files/elev_csv.csv')
    idx_list = GD.get_col_num_list(file_path_elev,'elev')

    save = 'C:/Users/clara/Documents/fire_season/1996_2013/'

    with open(dirname+'datasets/json/daily_lookup_file_TEMP.json', 'r') as fp:
        date_dictionary = json.load(fp) #Get the lookup file for the stations with data on certain months/years

    with open(dirname+'datasets/json/daily_lat_lon_TEMP.json', 'r') as fp:
        daily_dictionary = json.load(fp) #Get the latitude and longitude for the stations

    with open(dirname+'datasets/json/hourly_lat_lon_TEMP.json', 'r') as fp:
        hourly_dictionary = json.load(fp) #Get the latitude and longitude for the stations
 
    for year in range(2018,2019):

    #Uncomment if running for first time 

##        execute_sequential_calc(file_path_hourly,file_path_daily,file_path_se_dates,hourly_dictionary, daily_dictionary, date_dictionary,\
##                            str(year),{'temp':['TPS','GPR'],'rh':['IDW2','RF'],'wind':['IDW2','RF'],'pcp':['IDW4','GPR']},10,file_path_elev,\
##                                idx_list,save,shapefile,boreal_shapefile,phi_input=None,calc_phi=True,\
##                   kernels={'temp':['316**2 * Matern(length_scale=[5e+05, 5e+05, 6.01e+03], nu=0.5)']\
##                            ,'rh':['307**2 * Matern(length_scale=[5e+05, 6.62e+04, 1.07e+04], nu=0.5)'],\
##                            'pcp':['316**2 * Matern(length_scale=[5e+05, 5e+05, 4.67e+05], nu=0.5)'],\
##                            'wind':['316**2 * Matern(length_scale=[5e+05, 6.62e+04, 1.07e+04], nu=0.5)']})

    #Uncomment if running after program was halted 

        spath = 'C:/Users/clara/Documents/fire_season/new/'
        restart_calc(file_path_hourly,file_path_daily,file_path_se_dates,hourly_dictionary, daily_dictionary, date_dictionary,\
                            str(year),{'temp':['TPS','GPR'],'rh':['IDW2','RF'],'wind':['IDW2','RF'],'pcp':['IDW4','GPR']},10,file_path_elev,\
                                idx_list,spath,shapefile,boreal_shapefile,phi_input=None,calc_phi=True,\
                   kernels={'temp':['316**2 * Matern(length_scale=[5e+05, 5e+05, 6.01e+03], nu=0.5)']\
                            ,'rh':['307**2 * Matern(length_scale=[5e+05, 6.62e+04, 1.07e+04], nu=0.5)'],\
                            'pcp':['316**2 * Matern(length_scale=[5e+05, 5e+05, 4.67e+05], nu=0.5)'],\
                            'wind':['316**2 * Matern(length_scale=[5e+05, 6.62e+04, 1.07e+04], nu=0.5)']})


             
        


             
        
