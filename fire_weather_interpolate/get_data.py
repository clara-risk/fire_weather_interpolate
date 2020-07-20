#coding: latin1

"""
Summary
-------
The functions in this file are used for extracting data from the weather data files 
downloaded from Environment & Climate Change Canada and terrain lookup files (ex,
slope, drainage). 

References 
----------
get_b relies on information from Lawson & Armitage (2008) 

"""

#import
import geopandas as gpd
import os, sys
from datetime import datetime, timedelta, date
import numpy as np
import json, feather
import pandas as pd
from scipy.spatial.distance import cdist
import pyproj


def get_col_num_list(file_path,col_name): 
    '''Get the column in the look up file that corresponds to the correct data
    Parameters: 
        file_path (str): path to the lookup csv file, includes the file name
        col_name (str): the column name of the data you want to use in the lookup file, ex. "ELEV"
    Returns 
        idx_list (list): list containing all columns in the lookup table that are called col_name
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
    '''Get the dictionary of values for each input point frm lookup file
    Parameters
        projected_coordinates_list (list of tuples): list of coordinates that you want to find data
        in the lookup file for, must be in lon, lat format 
        file_path (str): file path to the lookup file, includes the file name and ending
        idx_list (list): output of get_col_name_list, we will just take the first entry
    Returns 
        min_distance_L (dict): a dictionary of values for each coordinate (taken from the closest 
        point in the lookup file to the input coordinate)
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
    
    
def get_b(latlon_dict,file_path_slope,idx_slope,file_path_drainage,idx_drainage,shapefile):
    '''Create a permanent lookup file for b for study area for future processing
    This is used in overwinter dc procedure 
    Parameters
        latlon_dict (dictionary, to be loaded from json file): dictionary of latitude and longitudes 
        for the hourly stations
        file_path_slope (str): path to the slope file, includes file name
        idx_slope (list): index of the slope variable in the header of the slope lookup file 
        file_path_drainage (str): path to the drainage file, includes file name
        idx_drainage (list): index of the drainage variable in the header of the drainage lookup file 
        shapefile (str): path to the shapefile of the study area (.shp format)
    Returns 
        The function will write a .json file to the hard drive, which can be loaded later. 
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
        input_date (str): input date for the date of interest, in the format: YYYY-MM-DD HH:MM
        file_path (str): path to the feather files containing the hourly data from Environment & 
        Climate Change Canada 
    Returns 
        ws_dictionary (dict): a dictionary of wind speed values for all the active & non-null stations
        on the input date 
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
                            ws_dictionary[station_name] = df.loc[df['Date/Time'] == input_date, 'Wind Spd (km/h)'].item()
                            
                            if float(df.loc[df['Date/Time'] == input_date, 'Wind Spd (km/h)'].item()) >= 315: 
                                print('The wind speed for %s corresponds to the most severe class of Tornado for the Enhanced Fujita Scale - Canada'%(station_name))

                        else: 
                            pass
                    except ValueError:
                        pass 

    return ws_dictionary

def get_noon_temp(input_date,file_path):
    '''Create a dictionary for noon temp data on the input date 
    Parameters
        input_date (str): input date for the date of interest, in the format: YYYY-MM-DD HH:MM
        file_path (str): path to the feather files containing the hourly data from Environment & 
        Climate Change Canada 
    Returns 
        temp_dictionary (dict): a dictionary of temperature values for all the active & non-null stations
        on the input date 
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
        input_date (str): input date for the date of interest, in the format: YYYY-MM-DD HH:MM
        file_path (str): path to the feather files containing the hourly data from Environment & 
        Climate Change Canada 
    Returns 
        temp_dictionary (dict): a dictionary of relative humidity values for all the active & 
        non-null stations on the input date 
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


                            RH_dictionary[station_name] = df.loc[df['Date/Time'] == input_date, 'Rel Hum (%)'].item()
                            
                            if float(df.loc[df['Date/Time'] == input_date, 'Rel Hum (%)'].item()) > 100: 
                                print('The relative humidity for %s is greater than 100%'%(station_name))

                        else: 
                            pass
                    except ValueError:
                        pass 

    return RH_dictionary

def get_pcp_dictionary_by_year(file_path):
    '''Create a lookup file for the year_month that each daily station has data for faster 
    processing later --> this is an input to get_pcp 
    Parameters
        file_path (str): file path to the daily csv files provided by Environment & Climate Change
        Canada, including the name of the file 
    Returns 
        The function writes a .json file to the hard drive that can be read back into the code before
        running get_pcp 
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
        file_path (str): file path to the daily csv files provided by Environment & Climate Change
        Canada, including the name of the file 
    Returns 
        The function writes a .json file to the hard drive that can be read back into the code before
        running get_pcp 
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
        input_date (str): input date for the day of interest, in the format YYYY:MM:DD 
        file_path (str): file path to the daily feather files 
        date_dictionary (dict, loaded from .json): lookup file that has what day/month pairs each 
        station is active on 
    Returns 
        rain_dictionary (dict): dictionary containing rain amount for each station 
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
        file_path (str): file path to the hourly csv files provided by Environment & Climate Change
        Canada, including the name of the file 
    Returns 
        The function writes a .json file to the hard drive that can be read back into the code before
        running get_pcp 
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
        
   
