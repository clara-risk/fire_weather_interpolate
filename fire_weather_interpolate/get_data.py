#coding: utf-8

"""
Summary
-------
The functions in this file are used for extracting data from the weather data files 
downloaded from Environment & Climate Change Canada and terrain lookup files (ex,
slope, drainage). 

References 
----------
get_b relies on information from Lawson & Armitage (2008)

Lawson, B., & Armitage, O. (2008). Weather Guide for the Canadian Forest Fire Danger Rating System. Natural Resources Canada.

"""

#import
from shapely.geometry import Point
import geopandas as gpd
import os, sys
from datetime import datetime, timedelta, date
import numpy as np
import json, feather
import pandas as pd
from scipy.spatial.distance import cdist
import pyproj
from shapely.ops import unary_union
import matplotlib.pyplot as plt

import fwi as fwi
import idw as idw
import tps as tps
import rf as rf
import gpr as gpr


def get_col_num_list(file_path,col_name): 
    '''Get the column in the look up file that corresponds to the correct data

    Parameters
    ----------
    file_path : string
        path to the lookup csv file, includes the file name
    col_name : string
        the column name of the data you want to use in the lookup file, ex. "ELEV"
    Returns
    ----------
    list
        - list containing all columns in the lookup table that are called col_name
    '''
    col_nums = [] #Create empty list for the relevent column numbers.
    header = []
    count = 0
    with open(file_path) as lookup_info:
        for line in lookup_info:
            row = line.rstrip('\n').split(',')
            if count == 0: 
                header.append(row[0:]) #Access the header 
            count += 1
    
    #What is the index of col_name in the header? 
    idx_list = [i for i, x in enumerate(header[0]) if col_name in x.lower()] 

    lookup_info.close() 

    return idx_list
    
def finding_data_frm_lookup(projected_coordinates_list,file_path,idx_list):
    '''Get the dictionary of values for each input point from the lookup file

    Parameters
    ----------
    projected_coordinates_list : list
        list of coordinates (list of tuples) that you want to find data in the lookup file for, must be in lon, lat format 
    file_path : string
        file path to the lookup file, includes the file name and ending
    idx_list : list
        output of get_col_name_list, we will just take the first entry
        
    Returns
    ----------
    dictionary
       - a dictionary of values for each coordinate (taken from the closest point in the lookup file to the input coordinate)
    '''
    distance_dictionary = {} 
    L_dictionary = {}
    min_distance_L = {}
    temp = []
    with open(file_path) as L_information:
        next(L_information) # Skip header. 
        for line in L_information:
            line2 = line.rstrip('\n').split(',')
            #Store information in Python dictionary so we can access it efficiently. 
            L_dictionary[tuple([float(line2[1]),float(line2[2])])] = float(line2[idx_list[0]])
            temp.append([float(line2[1]),float(line2[2])]) #Temp lat lon list 

    L_information.close() 

    L_locarray = np.array(temp) #Make an array of the lat lons. 

    loc_dict = {} 

    for coord in projected_coordinates_list: # projected_coordinates_list --> list of tuples.
        ts_lat = coord[1] # Obtain the lat/lon. 
        ts_lon =  coord[0]
        
        L = [] # Create a place to store the elevation.

        reshaped_coord= np.array([[ts_lon,ts_lat]]) #Input will be in lon, lat format. 


        hypot = cdist(L_locarray,reshaped_coord) #Get distance between input coords and coords in file.


        where_distance_small =  np.argmin(hypot) #Get the smallest distance index. 

        
        distance_dictionary[coord] = L_locarray[where_distance_small] #Access the lat/lon at the index.
        #We don't care if there's more than one closest coordinate, same dist in all directions. 

        convert_to_lst = L_locarray[where_distance_small] 
        convert_to_tuple = tuple(convert_to_lst) #Convert to tuple to look up value in dictionary. 
        
        min_distance_L[coord] = L_dictionary[convert_to_tuple] #Get value out of dictionary. 
        
    
    return min_distance_L
  
def is_station_in_boreal(loc_dict,days_dict,boreal_shapefile):
    '''For the fire season code, we are only interested in stations within the boreal zone,
    so we need to get a list of stations that are located there.
    
    Parameters
    ----------
    loc_dict : dictionary
        locations of the stations
    days_dict : dictionary
        stations we are interested in, output of get_start_date_calendar_csv/end_date
    boreal_shapefile : string
        file path leading to the .shp file that delineates the boreal forest 
    Returns
    ----------
    dictionary
        - dictionary of whether it is in area, organized boreal_dict[station_name] = True 
    '''
    boreal_dict = {}
    boreal_zone = gpd.read_file(boreal_shapefile)
    borealDF = gpd.GeoDataFrame(boreal_zone)
    borealDF_union = borealDF.geometry.unary_union #It is a multipolygon

    for location in loc_dict.keys():
        #We need to project the latlon b/c the shapefile is projected
        latitude = float(loc_dict[location][0])
        longitude = float(loc_dict[location][1])

        source_proj = pyproj.Proj(proj='latlong', datum = 'NAD83')
        xProj, yProj = pyproj.Proj('esri:102001')(longitude,latitude)        

        
        station_loc = Point((xProj,yProj))
        pointDF = pd.DataFrame([station_loc])
        gdf = gpd.GeoDataFrame(pointDF, geometry=[station_loc])
    
        if (boreal_zone.geometry.contains(gdf.geometry)).any():
            boreal_dict[location] = True
        else:
            pass
    return boreal_dict 
    
def calc_season_duration(start_surface,end_surface,year): 
    '''For fire season, once we have created the continuous surfaces, we need to calculate for 
    each pixel what is the estimated duration (for the year).
    
    Parameters
    ----------
    start_surface : ndarray
        the array of the start dates generated from a function such as IDW()
    end_surface : ndarray
        the array of the end dates *these need to be the same size and should automatically be 
    year : int
        the year that we are calculating the duration for, mostly needed to account for a leap year
             
    Returns
    ----------
    ndarray
        - an array of the study area for the estimated fire season in the array
    '''
    #Convert to days since January 1
    dJ = date(int(year), 1, 1)
    d0 = date(int(year), 3, 1) #March 1, Year 
    d1 = date(int(year), 9, 1) #Sep 1, Year
    d_between = int((d0-dJ).days) 
    d2_between = int((d1-dJ).days) 
    Sdate_array = start_surface + d_between
    Edate_array = end_surface + d2_between
    #Now we have an array of dates, we can end date-start date
    season_duration = Edate_array - Sdate_array
    #We have created a new array 
    return season_duration
    
def combine_stations(dictionary_daily,dictionary_hourly):
    '''Combine dictionaries with station names as keys. For example, hourly stations
    & daily stations for the fire season calculations. Daily dictionary takes precedence,
    meaning the overlapping key will be dropped from the hourly one 
    
    Parameters
    ----------
    dictionary_daily : dictionary
        dictionary of daily stations and the weather values there
    dictionary_hourly : dictionary
        dictionary of hourly stations and the weather values there

    Returns
    ----------
    dictionary
        - dictionary contining information from both the hourly and daily stations 
    '''
    #Which dictionary takes priority? For now, the daily dictionary will take priority over
    #the hourly dictionary, which means that an overlapping key will be dropped from the
    #hourly one

    for key in dictionary_hourly:
        if key not in dictionary_daily.keys():
            dictionary_daily[key] = dictionary_hourly[key]

    return dictionary_daily 

def stack_and_average(year1,year2,file_path_daily,file_path_hourly,shapefile,file_path_elev,idx_list,method):
    '''Get the fire season duration for every year in between the two input years
    and average them. Output the average array.

    Parameters
    ----------

    year1 : int
        first year taken into account
    year2 : int
        last year taken into account
    file_path_daily : string
        path to the daily weather csv files from Environment & Climate Change Canada
    file_path_hourly : string
        path to the hourly feather files
    shapefile : string
        path to the study area shapefile
    file_path_elev : string
        path to the elevation lookup file
    idx_list : list
        column index of elevation information in the lookup file
    method : string
        type of interpolation to use to create the yearly arrays, one of: 'IDW2', 'IDW3', 'IDW4', 'TPSS', 'RF'

    Returns
    ----------
    ndarray
        - average of each pixel of all the years considered in array format 
    '''
    list_of_arrays = [] 
    for year in range(int(year1),int(year2)+1):
        print('Processing...'+str(year))
        days_dict, latlon_station = fwi.start_date_calendar_csv(file_path_daily,str(year))
        end_dict, latlon_station2 = fwi.end_date_calendar_csv(file_path_daily,str(year),'oct')
        if year >= 2020: 
            hourly_dict, latlon_stationH = fwi.start_date_add_hourly(file_path_hourly, str(year))
            hourly_end, latlon_stationE = fwi.end_date_add_hourly(file_path_hourly, str(year))

            days_dict = combine_stations(days_dict,hourly_dict)
            latlon_station = combine_stations(latlon_station,latlon_stationH)

            end_dict = combine_stations(end_dict,hourly_end)
            latlon_station2 = combine_stations(latlon_station2,latlon_stationE)

        if method == 'IDW2': 

            start_surface,maxmin = idw.IDW(latlon_station,days_dict,str(year),'Start',shapefile, False, 2,True)
            end_surface,maxmin = idw.IDW(latlon_station2,end_dict,str(year),'End',shapefile, False, 2,True)

        elif method == 'IDW3':
            
            start_surface,maxmin = idw.IDW(latlon_station,days_dict,str(year),'Start',shapefile, False, 3,True)
            end_surface,maxmin = idw.IDW(latlon_station2,end_dict,str(year),'End',shapefile, False, 3,True)

        elif method == 'IDW4':

            start_surface,maxmin = idw.IDW(latlon_station,days_dict,str(year),'Start',shapefile, False, 4,True)
            end_surface,maxmin = idw.IDW(latlon_station2,end_dict,str(year),'End',shapefile, False, 4,True)

        elif method == 'TPSS':
            num_stationsS = int(len(days_dict.keys()))
            phi_inputS = int(num_stations)-(math.sqrt(2*num_stations))
            num_stationsE = int(len(end_dict.keys()))
            phi_inputE = int(num_stations)-(math.sqrt(2*num_stations))
            start_surface,maxmin = tps.TPS(latlon_station,days_dict,str(year),'Start',shapefile,False,phi_inputS,True,True)
            end_surface,maxmin = tps.TPS(latlon_station2,end_dict,str(year),'End',shapefile,False,phi_inputE,True,True)

        elif method == 'RF':
            start_surface,maxmin = rf.random_forest_interpolator(latlon_station,days_dict,str(year),'Start',shapefile,False,file_path_elev,idx_list,True)
            end_surface,maxmin = rf.random_forest_interpolator(latlon_station2,end_dict,str(year),'End',shapefile,False,file_path_elev,idx_list,True)

        else:
            print('Either that method does not exist or there is no support for it. You can use IDW2-4, TPSS, or RF') 
            
        dur_array = calc_season_duration(start_surface,end_surface,year)
        list_of_arrays.append(dur_array)
    voxels = np.dstack(list_of_arrays) #stack arrays based on depth
    averaged_voxels = np.array([[np.mean(x) for x in group] for group in voxels])
    return averaged_voxels 

def get_b(latlon_dict,file_path_slope,idx_slope,file_path_drainage,idx_drainage,shapefile):
    '''Create a permanent lookup file for b for study area for future processing to be used in overwinter DC procedure 

    Parameters
    ----------
    latlon_dict : dictionary
        dictionary of latitude and longitudes for the hourly stations
    file_path_slope : string
        path to the slope file, includes file name
    idx_slope : list
        index of the slope variable in the header of the slope lookup file 
    file_path_drainage : string
        path to the drainage file, includes file name
    idx_drainage : list
        index of the drainage variable in the header of the drainage lookup file 
    shapefile : string
        path to the shapefile of the study area (.shp format)
    '''

    lat = [] #Initialize empty lists to store data 
    lon = []
    for station_name in latlon_dict.keys(): #Loop through the list of stations 
        loc = latlon_dict[station_name]
        latitude = loc[0]
        longitude = loc[1]
        lat.append(float(latitude))
        lon.append(float(longitude))

    y = np.array(lat) #Convert to a numpy array for faster processing speed 
    x = np.array(lon)

    na_map = gpd.read_file(shapefile)
    bounds = na_map.bounds #Get the bounding box of the shapefile 
    xmax = bounds['maxx']
    xmin= bounds['minx']
    ymax = bounds['maxy']
    ymin = bounds['miny']
    pixelHeight = 10000 #We want a 10 by 10 pixel, or as close as we can get 
    pixelWidth = 10000
            
    num_col = int((xmax - xmin) / pixelHeight) #Calculate the number of rows cols to fill the bounding box at that resolution 
    num_row = int((ymax - ymin) / pixelWidth)


    #We need to project to a projected system before making distance matrix
    source_proj = pyproj.Proj(proj='latlong', datum = 'NAD83') #We dont know but assume NAD83
    xProj, yProj = pyproj.Proj('esri:102001')(x,y) #Convert to Canada Albers Equal Area 

    yProj_extent=np.append(yProj,[bounds['maxy'],bounds['miny']]) #Add the bounding box coords to the dataset so we can extrapolate the interpolation to cover whole area
    xProj_extent=np.append(xProj,[bounds['maxx'],bounds['minx']])

    Yi = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row) #Get the value for lat lon in each cell we just made 
    Xi = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col)

    Xi,Yi = np.meshgrid(Xi,Yi) #Make a rectangular grid (because eventually we will map the values)
    Xi,Yi = Xi.flatten(), Yi.flatten() #Then we flatten the arrays for easier processing 
    #X and then Y for a reason 
    concat = np.array((Xi.flatten(), Yi.flatten())).T #Preparing the coordinates to send to the function that will get the elevation grid 
    send_to_list = concat.tolist()
    send_to_tuple = [tuple(x) for x in send_to_list] #The elevation function takes a tuple 

    #in cython dictionaries maintain insertion order 
    Xi1_grd=[]
    Yi1_grd=[]
    slope_grd = []
    drainage_grd = [] 
    slope_grd_dict = finding_data_frm_lookup(send_to_tuple,file_path_slope,idx_slope) #Get the elevations from the lookup file 
    drainage_grd_dict = finding_data_frm_lookup(send_to_tuple,file_path_drainage,idx_drainage)
    for keys in slope_grd_dict.keys(): #The keys are each lat lon pair 
        x= keys[0]
        y = keys[1]
        Xi1_grd.append(x)
        Yi1_grd.append(y)
        slope_grd.append(slope_grd_dict[keys])
        drainage_grd.append(drainage_grd_dict[keys])

    #combine the arrays
    slope_array = np.array(slope_grd)
    drainage_array = np.array(drainage_grd)
    
    #return the b array to be passed to the other function
    b_array = np.empty(slope_array.shape)
    b_array[drainage_array == 3] = 0.5
    b_array[drainage_array == 2] = 0.75
    b_array[drainage_array == 0] = 0.75
    b_array[drainage_array == 1] = 0.9
    b_array[slope_array > 0.5] = 0.5

    b_list = list(b_array)

    with open('b_list.json', 'w') as fp: #write to hard drive for faster processing later 
        json.dump(b_list, fp)

def get_wind_speed(input_date,file_path): 
    '''Create a dictionary for wind speed data on the input date

    Parameters
    ----------
    input_date : string
        input date for the date of interest, in the format: YYYY-MM-DD HH:MM
    file_path : string
        path to the feather files containing the hourly data from Environment & Climate Change Canada 
    Returns
    ----------
    dictionary
        - a dictionary of wind speed values for all the active & non-null stations on the input date 
    '''
    
def get_wind_speed(input_date,file_path): 
    '''Create a dictionary for wind speed data on the input date

    Parameters
    ----------
    input_date : string
        input date for the date of interest, in the format: YYYY-MM-DD HH:MM
    file_path : string
        path to the feather files containing the hourly data from Environment & Climate Change Canada 
    Returns
    ----------
    dictionary
        - a dictionary of wind speed values for all the active & non-null stations on the input date 
    '''
    
    ws_dictionary = {}

    search_date = datetime.strptime(input_date, '%Y-%m-%d %H:%M') # Get the datetime object for input date

    for station_name in os.listdir(file_path):
        for file_name in os.listdir(file_path+station_name+'/'):
            if input_date[5:7] == file_name[29:31]: #This is a trick to speed up the code, only look at files which have the month/day in the name
                if input_date[0:4]== file_name[32:36]:
                    file = file_path+station_name+'/'+file_name

                    df = feather.read_dataframe(file)
                    try: 
                        if pd.notnull(df.loc[df['Date/Time'] == input_date, 'Wind Spd (km/h)'].item()):

                            #Put the value into the dictionary. 
                            
                            if float(df.loc[df['Date/Time'] == input_date, 'Wind Spd (km/h)'].item()) >= 315: 
                                print('The wind speed for %s corresponds to the most severe class of Tornado for the Enhanced Fujita Scale - Canada'%(station_name))

                            elif float(df.loc[df['Date/Time'] == input_date, 'Wind Spd (km/h)'].item()) < 0: 
                                print('The wind speed for %s is less than 0'%(station_name))
                            else:
                                ws_dictionary[station_name] = df.loc[df['Date/Time'] == input_date, 'Wind Spd (km/h)'].item()

                        else: 
                            pass
                    except ValueError:
                        pass 

    return ws_dictionary


def get_noon_temp(input_date,file_path):
    '''Create a dictionary for noon temp data on the input date

    Parameters
    ----------
    input_date : string
        input date for the date of interest, in the format: YYYY-MM-DD HH:MM
    file_path : string
        path to the feather files containing the hourly data from Environment & Climate Change Canada 
    Returns
    ----------
    dictionary
        - a dictionary of temperature values for all the active & non-null stations on the input date 
    '''
    
    temp_dictionary = {}
    
    search_date = datetime.strptime(input_date, '%Y-%m-%d %H:%M')


    for station_name in os.listdir(file_path):
        for file_name in os.listdir(file_path+station_name+'/'):
            if input_date[5:7] == file_name[29:31]:
                if input_date[0:4]== file_name[32:36]:
                    file = file_path+station_name+'/'+file_name

                    df = feather.read_dataframe(file)
                    

                    try: 
                        if pd.notnull(df.loc[df['Date/Time'] == input_date, 'Temp (Â°C)'].item()):


                            temp_dictionary[station_name] = df.loc[df['Date/Time'] == input_date, 'Temp (Â°C)'].item()
                            
                            if float(df.loc[df['Date/Time'] == input_date, 'Temp (Â°C)'].item()) > 42.2 or \
                            float(df.loc[df['Date/Time'] == input_date, 'Temp (Â°C)'].item()) < -58.3: 
                                print('The temperature for %s is either greater than the record high temperature recorded in Ontario \
                                or QuÃ©bec or lower than the record lowest temperature'%(station_name))

                        else: 
                            pass
                    except ValueError:
                        pass 

    return temp_dictionary
    
    
def get_relative_humidity(input_date,file_path):
    '''Create a dictionary for rh% data on the input date

    Parameters
    ----------
    input_date : string
        input date for the date of interest, in the format: YYYY-MM-DD HH:MM
    file_path : string
        path to the feather files containing the hourly data from Environment & Climate Change Canada 
    Returns
    ----------
    dictionary
        - a dictionary of relative humidity values for all the active & non-null stations on the input date 
    '''

    RH_dictionary = {}

    search_date = datetime.strptime(input_date, '%Y-%m-%d %H:%M')


    for station_name in os.listdir(file_path):
        for file_name in os.listdir(file_path+station_name+'/'):
            if input_date[5:7] == file_name[29:31]:
                if input_date[0:4]== file_name[32:36]:
                    file = file_path+station_name+'/'+file_name

                    df = feather.read_dataframe(file)
                    try: 
                        if pd.notnull(df.loc[df['Date/Time'] == input_date, 'Rel Hum (%)'].item()):
                            
                            if float(df.loc[df['Date/Time'] == input_date, 'Rel Hum (%)'].item()) > 100: 
                                print('The relative humidity for %s is greater than 100%'%(station_name))
                            else:
                                RH_dictionary[station_name] = df.loc[df['Date/Time'] == input_date, 'Rel Hum (%)'].item()

                        else: 
                            pass
                    except ValueError:
                        pass 

    return RH_dictionary

def get_pcp_dictionary_by_year(file_path):
    '''Create a lookup file for the year_month that each daily station has data for faster 
    processing later --> this is an input to get_pcp
    
    Parameters
    ----------
    file_path : string
        file path to the daily csv files provided by Environment & Climate Change Canada, including the name of the file 
    '''
    
    date_dictionary = {}
    for station_name in os.listdir(file_path):
        
        yearList = []
        count = 0
        with open(file_path+station_name, encoding='latin1') as year_information:
            for row in year_information:
                information = row.rstrip('\n').split(',')
                information_stripped = [i.replace('"','') for i in information]
                if count==0:
                    
                    header= information_stripped

                    keyword = 'month' #There is also the flag which is why we include the (
                    idx_list = [i for i, x in enumerate(header) if keyword in x.lower()]
                    if len(idx_list) >1:
                        print('The program is confused because there is more than one field name that could \
                        contain the month. Please check on this.') #there could be no index if the file is empty, which sometimes happens 
                        sys.exit()

                    keyword2 = 'year'
                    idx_list2 = [i for i, x in enumerate(header) if keyword2 in x.lower()]
                    if len(idx_list2) > 1: # There should only be one field 
                        print('The program is confused because there is more than one field name that could \
                        contain the year. Please check on this.')
                        sys.exit()
                if count > 0:
                    if int(information_stripped[idx_list[0]]) >= 10: 
                        year_month = str(int(information_stripped[idx_list2[0]]))+'-'+str(int(information_stripped[idx_list[0]]))
                    else:
                        year_month = str(int(information_stripped[idx_list2[0]]))+'-0'+str(int(information_stripped[idx_list[0]]))
                    if year_month not in yearList: 
                        yearList.append(year_month)
                count+=1
        date_dictionary[station_name[:-4]] =yearList 

    with open('daily_lookup_file_TEMP.json', 'w') as fp:
        json.dump(date_dictionary, fp)
        
def get_daily_lat_lon(file_path):
    '''Get the latitude and longitude of the daily stations and store in a json file in the directory

    Parameters
    ----------
    file_path : string
        file path to the daily csv files provided by Environment & Climate Change Canada, including the name of the file 
    '''
    latlon_dictionary = {} 
    #for station_name in lat_lon_list: #This is the list of available stations on that day
    for station_name in os.listdir(file_path):
        latlon_list = []
        with open(file_path+station_name, encoding='latin1') as latlon_information:
            
            count=0
            for row in latlon_information:
                
                information = row.rstrip('\n').split(',')
                information_stripped = [i.replace('"','') for i in information]
                
                if count==0:
                    header= information_stripped #We will look for latitude and longitude keywords in the header and find the index
                    keyword = 'lon'
                    idx_list = [i for i, x in enumerate(header) if keyword in x.lower()]
                    keyword2 = 'lat'
                    idx_list2 = [i for i, x in enumerate(header) if keyword2 in x.lower()]
                    if len(idx_list) > 1: # There should only be one field 
                        print('The program is confused because there is more than one field name that could \
                        contain the longitude. Please check on this.')
                        sys.exit()
                    if len(idx_list2) > 1: # There should only be one field 
                        print('The program is confused because there is more than one field name that could \
                        contain the latitude. Please check on this.')
                        sys.exit()
                        
                if count == 1:
                    if float(information_stripped[idx_list2[0]]) != 0 or float(information_stripped[idx_list[0]]) != 0: #sometimes lat and lon is 0, need to exclude

                        latlon_list.append((information_stripped[idx_list2[0]],information_stripped[idx_list[0]]))
                        break
                    else:
                        pass 
                    
                count+=1
                
        if len(set(latlon_list)) > 1:
            print('For %s there is more than one location in the list! You can only have one record per station so please check the data.'%(station_name))
        elif len(set(latlon_list)) == 0:
            print('A valid lat lon for that station was not found in the file.') 
        else:
            try: 
                latlon_dictionary[station_name[:-4]]=latlon_list[0]

            except:
                print('There is a problem with the files for %s and the location has not been recorded. Please check.'%(station_name))
    with open('daily_lat_lon_TEMP.json', 'w') as fp:
        json.dump(latlon_dictionary, fp)
        
def get_pcp(input_date,file_path,date_dictionary):
    '''Get a dictionary of the precipitation data from the feather files of the daily stations

    Parameters
    ----------
    input_date : string
        input date for the date of interest, in the format: YYYY-MM-DD HH:MM
    file_path : string
        path to the feather files containing the hourly data from Environment & Climate Change Canada
    date_dictionary : dictionary
        lookup file that has what day/month pairs each station is active on to speed up processing, loaded from the .json file 
    Returns
    ----------
    dictionary
        - dictionary containing rain amount for each station 
    '''


    rain_dictionary = {}

    yearMonth = input_date[0:7]

    for station_name in os.listdir(file_path):

        yearsMonths = date_dictionary[station_name[:-8]] #-8 bc we are now working with feather files

        if yearMonth in yearsMonths:

            file = file_path+station_name


            df = feather.read_dataframe(file)
            try: 
                if pd.notnull(df.loc[df['date'] == input_date, 'total_precip'].item()):


                    rain_dictionary[station_name[:-8]] = df.loc[df['date'] == input_date, 'total_precip'].item()
                    
                    if float(df.loc[df['date'] == input_date, 'total_precip'].item()) > 264: 
                        print('The amount of 24hr precipitation for %s exceeds the record recorded in Ontario or QuÃ©bec'%(station_name)) 

                else: 
                    pass
            except ValueError:
                pass #trace precip

    return rain_dictionary
    
def get_lat_lon(file_path):
    '''Get the latitude and longitude of the hourly stations and write to hard drive as a json file

    Parameters
    ----------
        file_path : string
            file path to the hourly csv files provided by Environment & Climate Change Canada, including the name of the file 
    '''
    latlon_dictionary = {} 

    for station_name in os.listdir(file_path):
        latlon_list = []
        files = os.listdir(file_path+station_name+'/')[0]
        with open(file_path+station_name+'/'+files, encoding='latin1') as latlon_information:
            print(station_name)
            count=0
            for row in latlon_information:
                
                information = row.rstrip('\n').split(',')
                information_stripped = [i.replace('"','') for i in information]
                
                if count==0:
                    header= information_stripped #We will look for latitude and longitude keywords in the header and find the index
                    keyword = 'longitude'
                    idx_list = [i for i, x in enumerate(header) if keyword in x.lower()]
                    keyword2 = 'latitude'
                    idx_list2 = [i for i, x in enumerate(header) if keyword2 in x.lower()]
                    if len(idx_list) > 1: # There should only be one field 
                        print('The program is confused because there is more than one field name that could \
                        contain the longitude. Please check on this.')
                        sys.exit()
                    if len(idx_list2) > 1: # There should only be one field 
                        print('The program is confused because there is more than one field name that could \
                        contain the latitude. Please check on this.')
                        sys.exit()
                        
                if count == 1:
                    latlon_list.append((information_stripped[idx_list2[0]],information_stripped[idx_list[0]]))

                    break 
                    
                count+=1
                
        if len(set(latlon_list)) > 1:
            print('For %s there is more than one location in the list! You can only have one record per station so please check the data.'%(station_name))
        else:
            try: 
                latlon_dictionary[station_name]=latlon_list[0]
            except:
                print('There is a problem with the files for %s and the location has not been recorded. Please check.'%(station_name))

    with open('hourly_lat_lon_TEMP.json', 'w') as fp:
        json.dump(latlon_dictionary, fp)
        
def convert_to_feather(file_path,out_path):
    '''Convert the Environment & Climate Change Canada csv files into feather files, to allow for faster processing

    Parameters
    ----------
    file_path : string
        file path to the csv files provided by Environment & Climate Change Canada, not including the name of the file 
    out_path : string
        where you want the new feather file to be written to in the computer, not including the new file name
    '''
    for station_name in os.listdir(file_path):
        file = file_path+station_name
        df = pd.read_csv(file, sep=',', engine='c', low_memory=False,encoding='latin1')
        feather.write_dataframe(df,out_path+station_name[:-4]+'.feather')


def get_intersect_boolean_array(ecozone_shapefile,shapefile,show,expand_area):
    '''Obtain a boolean array of where the ecozone is 0 = pixel NOT in ecozone, otherwise 1

    Parameters
    ----------
    ecozone_shapefile : string
        path to ecozone shapefile, including name
    shapefile : string
        path to shapefile 
    show : bool
        show a map if you want to check it has rasterized the shapefile correctly
    expand_area : bool
        whether to expand study area by 200km 
    Returns
    ----------
    ndarray
        - array 1/0 if pixel was inside ecozone
    '''
    study_map = gpd.read_file(shapefile)
    eco_map = gpd.read_file(ecozone_shapefile)

    #First, study area 

    bounds = study_map.bounds #Get the bounding box of the shapefile
    if expand_area:
        xmax = bounds['maxx']+200000 
        xmin= bounds['minx']-200000 
        ymax = bounds['maxy']+200000 
        ymin = bounds['miny']-200000
    else:
        xmax = bounds['maxx']
        xmin= bounds['minx']
        ymax = bounds['maxy']
        ymin = bounds['miny']    

    pixelHeight = 10000 #We want a 10 by 10 pixel, or as close as we can get 
    pixelWidth = 10000
            
    num_col = int((xmax - xmin) / pixelHeight) #Calculate the number of rows cols to fill the bounding box at that resolution 
    num_row = int((ymax - ymin) / pixelWidth)

    Yi = np.linspace(float(ymin),float(ymax),num_row+1) #Get the value for lat lon in each cell we just made 
    Xi = np.linspace(float(xmin),float(xmax),num_col+1) #+1 otherwise we get banding on the image

    Xi,Yi = np.meshgrid(Xi,Yi)
    concat = np.array((Xi.flatten(), Yi.flatten())).T #Because we are not using the lookup file, send in X,Y order 
    send_to_list = concat.tolist() #List of points inside the study area using the generated grid
    meshPoints = [Point(item) for item in send_to_list]
    study_df = pd.DataFrame(meshPoints) #our point dataframe
    study_gdf = gpd.GeoDataFrame(study_df, geometry=meshPoints)

    #Second, ecozone get points in the list we made that are inside the geodataframe
    pointList = [] 
    for location in meshPoints: 
        if (eco_map.geometry.contains(location)).any(): #if contained in any polygon in multipolygon
            pointList.append(location)
    
    #Make a grid of zeros in the right shape
    bool_initiate = np.zeros((num_row+1,num_col+1)) #Make consistent

    #Fill in the ones in the correct places
    for loc in pointList: 
        pair = list(loc.coords)
        coord_pair = (pair[0][0],pair[0][1],)#lat,lon
        x_orig = int((coord_pair[0] - float(xmin))/pixelHeight) #lon 
        y_orig = int((coord_pair[1] - float(ymin))/pixelWidth) #lat
        bool_initiate[y_orig][x_orig] = 1

    #Plot to make sure everything is ok  if new study area and you want to be sure
    if show: 
        fig, ax = plt.subplots(figsize= (15,15))
        crs = {'init': 'esri:102001'}

            
        plt.imshow(bool_initiate,extent=(float(xmin)-1,float(xmax)+1,float(ymax)+1,float(ymin)-1)) 
        study_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.2)
        plt.gca().invert_yaxis()        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        ax.ticklabel_format(useOffset=False, style='plain')

        plt.show()


    #return the array
    return bool_initiate 

    
def get_average_in_ecozone(boolean_ecozone,continuous_surface):
    '''This is a function to get the average of the array cells with centroids that are within an ecozone

    Parameters
    ----------
    boolean_ecozone : ndarray
        a boolean array delineating the raster cells belonging to the ecozone
    continuous_surface : ndarray
        an array of values for the study area
    Returns
    ----------
    float
        - the average if the array values that fall inside the ecozone
    '''

    continuous_surface[boolean_ecozone == 0] = -9999
    mean_value = np.mean(list(continuous_surface[continuous_surface > -9999].flatten())) #Mask out values not in ecozone

    return mean_value


def get_stations_plus_200km(latlon_dict,Cvar_dict,input_date,shapefile): 
    '''Dictionary for plotting of the stations included in the analysis

    Parameters
    ----------
    latlon_dict : dictionary
        the latitude and longitudes of the hourly stations
    Cvar_dict : dictionary
        dictionary of weather variable values for each station 
    input_date : string
        the date you want to interpolate for 
    shapefile : string
        path to the study area shapefile 
    Returns
    ----------
    dictionary
        - contains values only for stations inside study area plus 200km
    '''
    plotting_dictionary ={} #if expanding the study area, we need to return a dictionary of the stations used
    lat = []
    lon = []
    Cvar = []

    source_proj = pyproj.Proj(proj='latlong', datum = 'NAD83')
    na_map = gpd.read_file(shapefile)
    bounds = na_map.bounds

    xmax = bounds['maxx']+200000
    xmin= bounds['minx']-200000
    ymax = bounds['maxy']+200000
    ymin = bounds['miny']-200000

    for station_name in Cvar_dict.keys():

        if station_name in latlon_dict.keys():

            loc = latlon_dict[station_name]
            latitude = loc[0]
            longitude = loc[1]
            proj_coord = pyproj.Proj('esri:102001')(longitude,latitude) #Filter out stations outside of grid
            if (proj_coord[1] <= float(ymax[0]) and proj_coord[1] >= float(ymin[0]) and proj_coord[0] <= float(xmax[0]) and proj_coord[0] >= float(xmin[0])):
                 cvar_val = Cvar_dict[station_name]
                 lat.append(float(latitude))
                 lon.append(float(longitude))
                 Cvar.append(cvar_val)
                 plotting_dictionary[station_name] = cvar_val

    #Can return plotting dictionary if need be, add 3rd to return statement
    return plotting_dictionary
