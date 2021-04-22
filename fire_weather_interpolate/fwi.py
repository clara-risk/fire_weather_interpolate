#coding: utf-8

"""
Summary
-------

Code for calculating the FWI metrics and calculating fire season duration.

References
----------

Wang, X., Wotton, B. M., Cantin, A. S., Parisien, M. A., Anderson, K., Moore,
B., & Flannigan, M. D. (2017). cffdrs: an R package for the Canadian Forest
Fire Danger Rating System. Ecological Processes, 6(1). https://doi.org/10.1186/s13717-017-0070-z

Wotton, B. M. (2009). Interpreting and using outputs from the Canadian Forest
Fire Danger Rating System in research applications. Environmental and Ecological
Statistics, 16(2), 107–131. https://doi.org/10.1007/s10651-007-0084-2

Wotton, B. M., & Flannigan, M. D. (1993). Length of the fire season in a
changing climate. Forestry Chronicle, 69(2), 187–192. https://doi.org/10.5558/tfc69187-2

Code is partially translated from R package:
https://github.com/cran/cffdrs/tree/master/R

"""

# import
from shapely.geometry import Point
import geopandas as gpd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
from itertools import groupby
from datetime import datetime, timedelta, date
import pandas as pd
import math
import os
import sys
import gc
import feather
import csv
from calendar import isleap
from descartes import PolygonPatch

import get_data as GD
import idw as idw
import idew as idew
import gpr as gpr
import tps as tps
import rf as rf


def start_date_calendar_csv(file_path_daily, year):
    '''Returns a dictionary of where each station meets the start up criteria,
    plus a reference dictionary for the lat lon of the stations
    
    Parameters
    ----------
    file_path_daily : string
        path to the csv files containing the daily data from Environment & Climate Change Canada 
    year : string
        year to find the fire season start up date for 
              
    Returns
    ----------
    dictionary 
        - dictionary containing the start up date for each station (days since Mar 1)
    dictionary
        - the latitude and longitude of those stations 
    '''

    # Input: path to hourly data, string of the year, i.e. '1998'
    maxTempList_dict = {}  # Locations where we will store the data
    maxTemp_dictionary = {}
    date_dict = {}
    latlon_dictionary = {}

    # The dictionary will be keyed by the hourly (temperature) station names, which means all the names must be unique
    for station_name in os.listdir(file_path_daily):
        # We will need an empty dictionary to store the data due to data ordering issues
        Temp_subdict = {}
        temp_list = []  # Initialize an empty list to temporarily store data we will later send to a permanent dictionary
        count = 0
        # for csv in os.listdir(file_path_hourly+station_name+'/'): #Loop through the csv in the station folder
        # if year in csv: #Only open if it is the csv for the year of interest (this is contained in the csv name)
        # +'/'+csv
        # Open the file - for CAN data we use latin 1 due to à, é etc.
        with open(file_path_daily+station_name, encoding='latin1') as year_information:
            for row in year_information:  # Look at each row
                # Split each row into a list so we can loop through
                information = row.rstrip('\n').split(',')
                # Get rid of extra quotes in the header
                information_stripped = [
                    i.replace('"', '') for i in information]
                if count == 0:  # This is getting the first row

                    header = information_stripped

                    keyword = 'max_temp'  # Look for this keyword in the header
                    filter_out_keyword = 'flag'  # We don't want flag temperature, we want to skip over it
                    idx_list1 = [i for i, x in enumerate(header) if keyword in x.lower(
                    ) and filter_out_keyword not in x.lower()]  # Get the index of the temperature column

                    if len(idx_list1) > 1:  # There should only be one field
                        print('The program is confused because there is more than one field name that could \
                        contain the temp data. Please check on this.')
                        sys.exit()
                    keyword2 = 'date'  # Getting the index of the datetime object so we can later make sure we are using 13h00 value
                    idx_list2 = [i for i, x in enumerate(
                        header) if keyword2 in x.lower()]

                    if len(idx_list2) > 1:  # There should only be one field
                        print('The program is confused because there is more than one field name that could \
                        contain the date. Please check on this.')
                        sys.exit()

                    keyword3 = 'lat'  # Here we use the same methods to get the latitude and longitude
                    idx_list3 = [i for i, x in enumerate(
                        header) if keyword3 in x.lower()]
                    if len(idx_list3) > 1:  # There should only be one field
                        print('The program is confused because there is more than one field name that could \
                        contain the latitude. Please check on this.')
                        sys.exit()
                    keyword4 = 'lon'
                    idx_list4 = [i for i, x in enumerate(
                        header) if keyword4 in x.lower()]
                    if len(idx_list4) > 1:  # There should only be one field
                        print('The program is confused because there is more than one field name that could \
                        contain the latitude. Please check on this.')
                        sys.exit()

                if count > 0:  # Now we are looking at the rest of the file, after the header

                    if count == 1:  # Lat/lon will be all the same so only record it once
                        # If the file is corrupted (it usually looks like a bunch of random characters) we will get an error, so we need a try/except loop
                        try:
                            lat = float(information_stripped[idx_list3[0]])
                            lon = float(information_stripped[idx_list4[0]])
                            # Get the lat lon and send the tuple to the dictionary
                            latlon_dictionary[station_name[:-4]
                                              ] = tuple((lat, lon))
                        except:
                            print(
                                'Something is wrong with the lat/lon header names for %s!' % (station_name))
                            break

                        try:
                            # Make sure we have the right year
                            if information_stripped[idx_list2[0]][0:4] == year:
                                if information_stripped[idx_list2[0]][5:7] == '03' or information_stripped[idx_list2[0]][5:7] == '04' or \
                                   information_stripped[idx_list2[0]][5:7] == '05' or information_stripped[idx_list2[0]][5:7] == '06' or \
                                   information_stripped[idx_list2[0]][5:7] == '07' or information_stripped[idx_list2[0]][5:7] == '08':  # Make sure we are only considering months since March in case of heat wave in another month
                                    # if information_stripped[idx_list2[0]][11:13] == '13': #We are only interested in checking the 13h00 temperature
                                    Temp_subdict[information_stripped[idx_list2[0]]] = float(
                                        information_stripped[idx_list1[0]])
                                    # Get the 13h00 temperature, send to temp list
                                    temp_list.append(
                                        float(information_stripped[idx_list1[0]]))

                        except:  # In the case of a nodata value
                            Temp_subdict[information_stripped[idx_list2[0]]] = -9999
                            temp_list.append(-9999)

                    else:  # Proceed down the rows
                        try:

                            if information_stripped[idx_list2[0]][0:4] == year:
                                if information_stripped[idx_list2[0]][5:7] == '03' or information_stripped[idx_list2[0]][5:7] == '04' or information_stripped[idx_list2[0]][5:7] == '05'\
                                   or information_stripped[idx_list2[0]][5:7] == '06' or information_stripped[idx_list2[0]][5:7] == '07' or information_stripped[idx_list2[0]][5:7] == '08':
                                    # if information_stripped[idx_list2[0]][11:13] == '13':
                                    Temp_subdict[information_stripped[idx_list2[0]]] = float(
                                        information_stripped[idx_list1[0]])
                                    temp_list.append(
                                        float(information_stripped[idx_list1[0]]))

                        except:
                            Temp_subdict[information_stripped[idx_list2[0]]] = -9999
                            temp_list.append(-9999)

                count += 1

        maxTemp_dictionary[station_name[:-4]] = Temp_subdict
        # Store the information for each station in the permanent dictionary
        maxTempList_dict[station_name[:-4]] = temp_list

        vals = maxTempList_dict[station_name[:-4]]

        # if 'NA' not in vals and len(vals) == 184: #only consider the stations with unbroken records, num_days between March-August is 153

        varray = np.array(vals)
        where_g12 = np.array(varray >= 12)  # Where is the temperature >=12?

        # Put the booleans in groups, ex. [True, True], [False, False, False]
        groups = [list(j) for i, j in groupby(where_g12)]

        # Obtain a list of where the groups are three or longer which corresponds to at least 3 days >= 12
        length = [x for x in groups if len(x) >= 3 and x[0] == True]

        if len(length) > 0:

            index = groups.index(length[0])  # Get the index of the group
            group_len = [len(x) for x in groups]  # Get length of each group
            length_sofar = 0  # We need to get the number of days up to where the criteria is met
            # loop through each group until you get to the index and add the length of that group
            for i in range(0, index):
                length_sofar += group_len[i]

            # We need to filter out the stations with No Data before that point
            # So slice to the index
            vals_behind = varray[0:length_sofar]
            if -9999 not in vals_behind:
                try:

                    # Go three days ahead for the fourth day
                    Sdate = list(
                        sorted(maxTemp_dictionary[station_name[:-4]].keys()))[length_sofar+3]

                    d0 = date(int(year), 3, 1)  # March 1, Year
                    # Convert to days since march 1 so we can interpolate
                    d1 = date(int(Sdate[0:4]), int(
                        Sdate[5:7]), int(Sdate[8:10]))
                    delta = d1 - d0
                    day = int(delta.days)  # Convert to integer
                    # Store the integer in the dictionary
                    date_dict[station_name[:-4]] = day
                except:
                    print(
                        'There are no data values upstream! Need to take an alternative method.')
                    Sdate = list(
                        sorted(maxTemp_dictionary[station_name[:-4]].keys()))[length_sofar]
                    d0 = date(int(year), 3, 1)
                    d1 = date(int(Sdate[0:4]), int(Sdate[5:7]), int(
                        Sdate[8:10])) + timedelta(days=3)
                    delta = d1 - d0
                    day = int(delta.days)  # Convert to integer
                    date_dict[station_name[:-4]] = day

        else:
            #print('Station %s did not start up by September 1 or had NA values upstream of the start date.'%station_name[:-4])
            pass  # Do not include the station

            #print('The start date for %s for %s is %s'%(station_name[:-4],year,Sdate))

    # Return the dates for each station
    # print(date_dict)
    return date_dict, latlon_dictionary

def end_date_calendar_csv(file_path_daily, year, search_month):
    '''Returns a dictionary of where each station meets the end criteria see 
    Wotton & Flannigan 1993, plus a reference dictionary for the lat lon of the stations
    
    Parameters
    ----------
    file_path_daily : string
        path to the csv files containing the daily data from Environment & Climate Change Canada 
    year : string
        year to find the fire season start up date for
    search_month : string 
        the month (day 1) you want to start searching for the end date, enter 'sep' or 'oct'
        
    Returns
    ----------
    dictionary 
        - dictionary containing the start up date for each station (days since Mar 1)
    dictionary
        - the latitude and longitude of those stations 
    '''
    # Input: path to hourly data, string of the year, i.e. '1998'
    maxTempList_dict = {}  # Locations where we will store the data
    maxTemp_dictionary = {}
    date_dict = {}
    latlon_dictionary = {}

    # The dictionary will be keyed by the hourly (temperature) station names, which means all the names must be unique
    for station_name in os.listdir(file_path_daily):
        # We will need an empty dictionary to store the data due to data ordering issues
        Temp_subdict = {}
        temp_list = []  # Initialize an empty list to temporarily store data we will later send to a permanent dictionary
        count = 0
        # for csv in os.listdir(file_path_hourly+station_name+'/'): #Loop through the csv in the station folder
        # if year in csv: #Only open if it is the csv for the year of interest (this is contained in the csv name)

        # Open the file - for CAN data we use latin 1 due to à, é etc.
        with open(file_path_daily+station_name, encoding='latin1') as year_information:
            for row in year_information:  # Look at each row
                # Split each row into a list so we can loop through
                information = row.rstrip('\n').split(',')
                # Get rid of extra quotes in the header
                information_stripped = [
                    i.replace('"', '') for i in information]
                if count == 0:  # This is getting the first row

                    header = information_stripped

                    keyword = 'max_temp'  # Look for this keyword in the header
                    filter_out_keyword = 'flag'  # We don't want flag temperature, we want to skip over it
                    idx_list1 = [i for i, x in enumerate(header) if keyword in x.lower(
                    ) and filter_out_keyword not in x.lower()]  # Get the index of the temperature column

                    if len(idx_list1) > 1:  # There should only be one field
                        print('The program is confused because there is more than one field name that could \
                        contain the temp data. Please check on this.')
                        sys.exit()
                    keyword2 = 'date'  # Getting the index of the datetime object so we can later make sure we are using 13h00 value
                    idx_list2 = [i for i, x in enumerate(
                        header) if keyword2 in x.lower()]

                    if len(idx_list2) > 1:  # There should only be one field
                        print('The program is confused because there is more than one field name that could \
                        contain the date. Please check on this.')
                        sys.exit()

                    keyword3 = 'lat'  # Here we use the same methods to get the latitude and longitude
                    idx_list3 = [i for i, x in enumerate(
                        header) if keyword3 in x.lower()]
                    if len(idx_list3) > 1:  # There should only be one field
                        print('The program is confused because there is more than one field name that could \
                        contain the latitude. Please check on this.')
                        sys.exit()
                    keyword4 = 'lon'
                    idx_list4 = [i for i, x in enumerate(
                        header) if keyword4 in x.lower()]
                    if len(idx_list4) > 1:  # There should only be one field
                        print('The program is confused because there is more than one field name that could \
                        contain the latitude. Please check on this.')
                        sys.exit()

                if count > 0:  # Now we are looking at the rest of the file, after the header

                    if count == 1:  # Lat/lon will be all the same so only record it once
                        # If the file is corrupted (it usually looks like a bunch of random characters) we will get an error, so we need a try/except loop
                        try:
                            lat = float(information_stripped[idx_list3[0]])
                            lon = float(information_stripped[idx_list4[0]])
                            # Get the lat lon and send the tuple to the dictionary
                            latlon_dictionary[station_name[:-4]
                                              ] = tuple((lat, lon))
                        except:
                            print(
                                'Something is wrong with the lat/lon header names for %s!' % (station_name))
                            break

                        try:
                            # Make sure we have the right year
                            if information_stripped[idx_list2[0]][0:4] == year:
                                if search_month == 'sep':
                                    if information_stripped[idx_list2[0]][5:7] == '09' or information_stripped[idx_list2[0]][5:7] == '10' or \
                                       information_stripped[idx_list2[0]][5:7] == '11' or information_stripped[idx_list2[0]][5:7] == '12':  # Make sure we are only considering months after October
                                        # if information_stripped[idx_list2[0]][11:13] == '13': #We are only interested in checking the 13h00 temperature
                                        Temp_subdict[information_stripped[idx_list2[0]]] = float(
                                            information_stripped[idx_list1[0]])
                                        # Get the max temperature, send to temp list
                                        temp_list.append(
                                            float(information_stripped[idx_list1[0]]))
                                elif search_month == 'oct':
                                    if information_stripped[idx_list2[0]][5:7] == '10' or \
                                       information_stripped[idx_list2[0]][5:7] == '11' or information_stripped[idx_list2[0]][5:7] == '12':  # Make sure we are only considering months after October
                                        # if information_stripped[idx_list2[0]][11:13] == '13': #We are only interested in checking the 13h00 temperature
                                        Temp_subdict[information_stripped[idx_list2[0]]] = float(
                                            information_stripped[idx_list1[0]])
                                        # Get the max temperature, send to temp list
                                        temp_list.append(
                                            float(information_stripped[idx_list1[0]]))
                                else:
                                    print('That is not a valid search month!')
                                    break

                        except:  # In the case of a nodata value
                            Temp_subdict[information_stripped[idx_list2[0]]] = -9999
                            temp_list.append(-9999)

                    else:  # Proceed down the rows
                        try:

                            if information_stripped[idx_list2[0]][0:4] == year:
                                if search_month == 'sep':
                                    if information_stripped[idx_list2[0]][5:7] == '09' or information_stripped[idx_list2[0]][5:7] == '10' or \
                                       information_stripped[idx_list2[0]][5:7] == '11' or information_stripped[idx_list2[0]][5:7] == '12':  # Make sure we are only considering months after October
                                        # if information_stripped[idx_list2[0]][11:13] == '13': #We are only interested in checking the 13h00 temperature
                                        Temp_subdict[information_stripped[idx_list2[0]]] = float(
                                            information_stripped[idx_list1[0]])
                                        # Get the max temperature, send to temp list
                                        temp_list.append(
                                            float(information_stripped[idx_list1[0]]))
                                elif search_month == 'oct':
                                    if information_stripped[idx_list2[0]][5:7] == '10' or \
                                       information_stripped[idx_list2[0]][5:7] == '11' or information_stripped[idx_list2[0]][5:7] == '12':  # Make sure we are only considering months after October
                                        # if information_stripped[idx_list2[0]][11:13] == '13': #We are only interested in checking the 13h00 temperature
                                        Temp_subdict[information_stripped[idx_list2[0]]] = float(
                                            information_stripped[idx_list1[0]])
                                        # Get the max temperature, send to temp list
                                        temp_list.append(
                                            float(information_stripped[idx_list1[0]]))
                                else:
                                    print('That is not a valid search month!')
                                    break

                        except:
                            Temp_subdict[information_stripped[idx_list2[0]]] = -9999
                            temp_list.append(-9999)

                count += 1

        maxTemp_dictionary[station_name[:-4]] = Temp_subdict
        # Store the information for each station in the permanent dictionary
        maxTempList_dict[station_name[:-4]] = temp_list

        vals = maxTempList_dict[station_name[:-4]]

        # if 'NA' not in vals and len(vals) == 122: #only consider the stations with unbroken records, num_days between Oct1-Dec31 = 92, Sep1-Dec31=122

        varray = np.array(vals)
        where_g12 = np.array(varray < 5)  # Where is the temperature < 5?

        # Put the booleans in groups, ex. [True, True], [False, False, False]
        groups = [list(j) for i, j in groupby(where_g12)]

        # Obtain a list of where the groups are three or longer which corresponds to at least 3 days < 5
        length = [x for x in groups if len(x) >= 3 and x[0] == True]

        if len(length) > 0:
            index = groups.index(length[0])  # Get the index of the group
            group_len = [len(x) for x in groups]  # Get length of each group
            length_sofar = 0  # We need to get the number of days up to where the criteria is met
            # loop through each group until you get to the index and add the length of that group
            for i in range(0, index):
                length_sofar += group_len[i]

            # We need to filter out the stations with No Data before that point
            # So slice to the index
            vals_behind = varray[0:length_sofar]
            if -9999 not in vals_behind:

                try:

                    # Go three days ahead for the fourth day (end day)
                    Sdate = list(
                        sorted(maxTemp_dictionary[station_name[:-4]].keys()))[length_sofar+3]
                    if search_month == 'sep':
                        d0 = date(int(year), 9, 1)  # Sep 1, Year
                    elif search_month == 'oct':
                        d0 = date(int(year), 10, 1)  # Oct 1, Year
                    else:
                        print('That is not a valid search month!')
                        break
                    # Convert to days since Oct 1 so we can interpolate
                    d1 = date(int(Sdate[0:4]), int(
                        Sdate[5:7]), int(Sdate[8:10]))
                    delta = d1 - d0
                    day = int(delta.days)  # Convert to integer
                    # Store the integer in the dictionary
                    date_dict[station_name[:-4]] = day
                except:
                    print(
                        'There are no data values upstream! Need to take an alternative method.')
                    Sdate = list(
                        sorted(maxTemp_dictionary[station_name[:-4]].keys()))[length_sofar]
                    if search_month == 'sep':
                        d0 = date(int(year), 9, 1)  # Sep 1, Year
                    elif search_month == 'oct':
                        d0 = date(int(year), 10, 1)  # Oct 1, Year
                    else:
                        print('That is not a valid search month!')
                        break
                    d1 = date(int(Sdate[0:4]), int(Sdate[5:7]), int(
                        Sdate[8:10])) + timedelta(days=3)
                    delta = d1 - d0
                    day = int(delta.days)  # Convert to integer
                    date_dict[station_name[:-4]] = day

        else:
            #print('Station %s did not end by December 31.'%station_name[:-4])
            pass  # Do not include the station

        #print('The end date for %s for %s is %s'%(station_name,year,Sdate))

    # Return the dates for each station
    # print(date_dict)
    return date_dict, latlon_dictionary


def calc_season_change(earlier_array, later_array):
    '''Calculate the change between seasons so we can evaluate how much the season has changed over time.

    Parameters
    ----------
    earlier_array : ndarray
        array of fire season duration values for the earlier year, ex 1922
    later_array : ndarray
        array of fire season duration values for the later year, ex 2019
        
    Returns
    ----------
    ndarray 
        - array containing the difference for each pixel
    '''
    change_array = earlier_array-later_array
    return change_array


def calc_duration_in_ecozone(file_path_daily, file_path_elev, idx_list, shapefile, list_of_ecozone_names, year1,
                             year2, method, expand_area):
    '''Calculation the yearly duration between years 1-2 and output to dictionary for graphing

    Parameters
    ----------
    file_path_daily : string
        path to the daily files 
    file_path_elev : string
        path to the elevation lookup file
    idx_list : list
        the index of the elevation data column in the lookup file
    shapefile : string
        path to the shapefile of the study area
    list_of_ecozone_names : list
        list of ecozone names you want to calculate duration for, ex ['taiga','hudson'],
        must correspond to the names of the shapefiles in the folder ecozone_shp in the working directory 
    year1 : int
        start year
    year2 : int
        end year 
    method : string
        spatial model, one of: IDW2,IDW3,IDW4,TPSS,RF
    expand_area : bool
        whether or not to expand area by 200km
        
    Returns
    ----------
    dictionary
        - dictionary keyed by year then ecozone that contains a list of durations from year1-year2 (year2 inclusive)
    '''
    duration_dict = {}
    for year in range(int(year1), int(year2)+1):
        print('Processing...'+str(year))
        days_dict, latlon_station = start_date_calendar_csv(
            file_path_daily, str(year))
        end_dict, latlon_station2 = end_date_calendar_csv(
            file_path_daily, str(year))

        if method == 'IDW2':

            start_surface, maxmin = idw.IDW(latlon_station, days_dict, str(
                year), 'Start', shapefile, False, 2, expand_area)
            end_surface, maxmin = idw.IDW(latlon_station2, end_dict, str(
                year), 'End', shapefile, False, 2, expand_area)

        elif method == 'IDW3':

            start_surface, maxmin = idw.IDW(latlon_station, days_dict, str(
                year), 'Start', shapefile, False, 3, expand_area)
            end_surface, maxmin = idw.IDW(latlon_station2, end_dict, str(
                year), 'End', shapefile, False, 3, expand_area)

        elif method == 'IDW4':

            start_surface, maxmin = idw.IDW(latlon_station, days_dict, str(
                year), 'Start', shapefile, False, 4, expand_area)
            end_surface, maxmin = idw.IDW(latlon_station2, end_dict, str(
                year), 'End', shapefile, False, 4, expand_area)

        elif method == 'TPSS':
            num_stationsS = int(len(days_dict.keys()))
            phi_inputS = None
            num_stationsE = int(len(end_dict.keys()))
            #phi_inputE = int(num_stationsE)-(math.sqrt(2*num_stationsS))
            phi_inputE = None
            start_surface, maxmin = tps.TPS(latlon_station, days_dict, str(
                year), 'Start', shapefile, False, phi_inputS, expand_area, True)
            end_surface, maxmin = tps.TPS(latlon_station2, end_dict, str(
                year), 'End', shapefile, False, phi_inputE, expand_area, True)

        elif method == 'RF':
            start_surface, maxmin = rf.random_forest_interpolator(latlon_station, days_dict, str(
                year), 'Start', shapefile, False, file_path_elev, idx_list, expand_area)
            end_surface, maxmin = rf.random_forest_interpolator(latlon_station2, end_dict, str(
                year), 'End', shapefile, False, file_path_elev, idx_list, expand_area)

        elif method == 'GPR':
            start_surface, maxmin = gpr.GPR_interpolator(latlon_station, days_dict, str(
                year), 'Start', shapefile, False, file_path_elev, idx_list, 0.1, expand_area)
            end_surface, maxmin = gpr.GPR_interpolator(latlon_station2, end_dict, str(
                year), 'End', shapefile, False, file_path_elev, idx_list, 0.1, expand_area)

        else:
            print('Either that method does not exist or there is no support for it. You can use IDW2-4, TPSS, or RF')

        ecozones = list_of_ecozone_names
        yearly_dict = {}
        cwd = os.getcwd()  # get the working directory
        for zone in ecozones:
            print(zone)
            # For this to work, the shapefiles MUST be in this location
            ecozone_shapefile = cwd+'/ecozone_shp/'+zone+'.shp'
            boolean_map = GD.get_intersect_boolean_array(
                ecozone_shapefile, shapefile, False, expand_area)
            dur_matrix = GD.calc_season_duration(
                start_surface, end_surface, year)
            AvVal = GD.get_average_in_ecozone(boolean_map, dur_matrix)
            yearly_dict[zone] = AvVal
            print(AvVal)

        duration_dict[year] = yearly_dict

    return duration_dict


def get_date_index(year, input_date, month):
    '''Get the number of days for the date of interest from the first of the month of interest
    Example, convert to days since March 1
    
    Parameters
    ----------
    year : string
        year of interest 
    input_date : string
        input date of interest
    month : int
        the month from when you want to calculate the date (ex, 04 for March)
        
    Returns
    ----------
    int 
        - days since 1st of the month of interest 
    '''
    d0 = date(int(year), month, 1)
    input_date = str(input_date)
    # convert to days since march 1/oct 1 so we can interpolate
    d1 = date(int(input_date[0:4]), int(
        input_date[5:7]), int(input_date[8:10]))
    delta = d1 - d0
    day = int(delta.days)
    return day


def make_start_date_mask(day_index, day_interpolated_surface):
    '''Turn the interpolated surface of start dates into a numpy array

    Parameters
    ----------
    day_index : int
        index of the day of interest since Mar 1
    day_interpolated_surface : ndarray
        the interpolated surface of the start dates across the study area
        
    Returns
    ----------
    ndarray
        - a mask array of the start date, either activated (1) or inactivated (np.nan)
    '''
    shape = day_interpolated_surface.shape
    new = np.ones(shape)
    # If the day in the interpolated surface is before the index, it is activated
    new[day_interpolated_surface <= day_index] = 1
    # If it is the opposite it will be masked out, so assign it to np.nan (no data)
    new[day_interpolated_surface > day_index] = np.nan
    return new


def make_end_date_mask(day_index, day_interpolated_surface):
    '''Turn the interpolated surface of end dates into a numpy array

    Parameters
    ----------
    day_index : int
        index of the day of interest since Oct 1 (or potentially Sep 1) 
    day_interpolated_surface : ndarray
        the interpolated surface of the start dates across the study area
        
    Returns
    ----------
    ndarray
        - a mask array of the end date, either activated (1) or inactivated (np.nan)
    '''
    shape = day_interpolated_surface.shape
    new = np.ones(shape)
    # If the day in the interpolated surface is before the index, its closed
    new[day_interpolated_surface <= day_index] = np.nan
    # If it is the opposite it will be left open, so assign it to 1
    new[day_interpolated_surface > day_index] = 1
    return new


def get_overwinter_pcp(overwinter_dates, file_path_daily, start_surface, end_surface, maxmin, shapefile,
                       show, date_dictionary, latlon_dictionary, file_path_elev, idx_list, json):
    '''Get the total amount of overwinter pcp for the purpose of knowing where to use the 
    overwinter DC procedure
    
    Parameters
    ----------
    overwinter_dates : list
        list of dates that are in the winter (i.e. station shut down to start up), you can just input generally
        Oct 1-June 1 and stations that are still active will be masked out
    file_path_daily : string
        file path to the daily feather files containing the precipitation data
    start_surface : ndarray
        array containing the interpolated start-up date for each cell 
    end_surface : ndarray
        array containing the interpolated end date for each cell, from the year before 
    maxmin : list
        bounds of the study area 
    shapefile : string
        path to the study area shapefile
    show : bool
        whether to print a map 
    date_dictionary : dictionary
        lookup file that has what day/month pairs each station contains data for (loaded from .json file) 
    latlon_dictionary : dictionary
        lat lons of the daily stations (loaded from .json file) 
    file_path_elev : string
        file path to the elevation lookup file 
    idx_list : list
        the index of the elevation data column in the lookup file 
    json : bool
        if True, convert the array to a flat list so it can be written as a .json file to the hard drive
    Returns
    ----------
    ndarray 
        - array of interpolated overwinter precipitation for the study area
    ndarray
        - array indicating where the overwinter DC procedure is needed
    '''
    pcp_list = []

    # dates is from Oct 1 year before to start day current year (up to Apr 1)
    for o_dat in overwinter_dates:
        year = str(o_dat)[0:4]
        # we need to take index while its still a timestamp before convert to str
        index = overwinter_dates.index(o_dat)
        o_dat = str(o_dat)
        day_index = get_date_index(year, o_dat, 3)
        eDay_index = get_date_index(year, o_dat, 10)
        if int(str(o_dat)[5:7]) >= 10:

            invertEnd = np.ones(end_surface.shape)
            endMask = make_end_date_mask(eDay_index, end_surface)
            # 0 in place of np.nan, because we use sum function later
            invertEnd[endMask == 1] = 0
            invertEnd[np.where(np.isnan(endMask))] = 1

            endMask = invertEnd
            # No stations will start up until the next spring
            mask = np.ones(start_surface.shape)
        # All stations are in summer
        elif int(str(o_dat)[5:7]) < 10 and int(str(o_dat)[5:7]) >= 7:

            endMask = np.zeros(end_surface.shape)
            mask = np.zeros(start_surface.shape)
        else:

            # by this time (Jan 1) all stations are in winter
            endMask = np.ones(end_surface.shape)
            invertMask = np.ones(start_surface.shape)
            mask = make_start_date_mask(day_index, start_surface)
            # If the station has started, stop counting it
            invertMask[mask == 1] = 0
            # If the station is still closed, keep counting
            invertMask[np.where(np.isnan(mask))] = 1
            mask = invertMask

        rainfall = GD.get_pcp(
            str(o_dat)[0:10], file_path_daily, date_dictionary)
        rain_grid, maxmin = idw.IDW(
            latlon_dictionary, rainfall, o_dat, 'Precipitation', shapefile, False, 1)

        masked = rain_grid * mask * endMask

        masked[np.where(np.isnan(masked))] = 0  # sum can't handle np.nan

        pcp_list.append(masked)

    # when we sum, we need to treat nan as 0, otherwise any nan in the list will cause the whole val to be nan
    pcp_overwinter = sum(pcp_list)

    overwinter_reqd = np.ones(pcp_overwinter.shape)
    overwinter_reqd[pcp_overwinter > 200] = np.nan  # not Required
    overwinter_reqd[pcp_overwinter <= 200] = 1  # required

    if show:
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        fig, ax = plt.subplots(figsize=(15, 15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)

        plt.imshow(overwinter_reqd, extent=(min_xProj_extent-1,
                   max_xProj_extent+1, max_yProj_extent-1, min_yProj_extent+1))
        na_map.plot(ax=ax, color='white', edgecolor='k',
                    linewidth=2, zorder=10, alpha=0.1)

        plt.gca().invert_yaxis()

        title = 'Areas Requiring Overwinter DC Procedure for %s' % (year)
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        plt.show()

    if json:
        # json cannot handle numpy arrays
        pcp_overwinter = list(pcp_overwinter.flatten())
        overwinter_reqd = list(overwinter_reqd.flatten())

    return pcp_overwinter, overwinter_reqd  # This is the array, This is a mask


def get_last_dc_before_shutdown(dc_list, endsurface, overwinter_reqd, show, maxmin):
    '''Get an array of the last dc vals before station shutdown for winter for the study area

    Parameters
    ----------

    dc_list : list
        a list of the interpolated surfaces for the drought code for each day in the fire season (from auto_select module, execute_sequential_calc) 
    endsurface : ndarray
        array for end dates for the year before the year of interest 
    overwinter_reqd : ndarray
        where the overwinter procedure is required 
    show : bool
        whether you want to plot the map 
    maxmin : list
        bounds of the study area
        
    Returns
    ----------
    ndarray
        array of dc values before shutdown for cells requiring overwinter procedure 
    '''

    # flatten the arrays for easier processing - avoid 3d array
    flatten_dc = list(map(lambda x: x.flatten(), dc_list))
    # Create an array from the list that we can index
    stackDC = np.stack(flatten_dc, axis=-1)

    # add 214 days (mar-aug31 to convert to days since March 1) to get index in the stack... based on make_end_date_mask() -1 for day before
    days_since_mar1 = endsurface.flatten().astype(int)+214-1

    # Index each cell in the array by the end date to get the last val
    last_DC_val_before_shutdown = stackDC[np.arange(
        len(stackDC)), days_since_mar1]
    last_DC_val_before_shutdown_masked = last_DC_val_before_shutdown * \
        overwinter_reqd.flatten()  # Mask the areas that don't require the overwinter procedure
    last_DC_val_before_shutdown_masked_reshape = last_DC_val_before_shutdown_masked.reshape(
        endsurface.shape)

    if show:
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        fig, ax = plt.subplots(figsize=(15, 15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)

        plt.imshow(last_DC_val_before_shutdown_masked_reshape, extent=(
            min_xProj_extent-1, max_xProj_extent+1, max_yProj_extent-1, min_yProj_extent+1))
        na_map.plot(ax=ax, color='white', edgecolor='k',
                    linewidth=2, zorder=10, alpha=0.1)

        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label('DC')

        title = 'Last DC'
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        plt.show()

    return last_DC_val_before_shutdown_masked_reshape


def dc_overwinter_procedure(last_DC_val_before_shutdown_masked_reshape, overwinter_pcp, b_list):
    '''Apply the overwinter procedure, see Wang et al. (2017) for more details

    Parameters
    ----------
    last_DC_val_before_shutdown_masked_reshape : ndarray
        output of get_last_dc_before_shutdown, array containing the last dc value before the station shut
        down interpolated across the study area 
    overwinter_pcp : ndarray
        overwinter precipitation amount in each cell 
    b_list : list
        list containing information for the b parameter, can be formatted into an array (loaded from .json) 
    Returns
    ----------
    ndarray 
    - the new values for the start-up DC procedure, to be used in areas identified as needing the overwinter procedure 
    '''
    a = 1.0
    b_array = np.array(b_list).reshape(overwinter_pcp.shape)
    Qf = 800 * np.exp(-last_DC_val_before_shutdown_masked_reshape / 400)
    Qs = a * Qf + b_array * (3.94 * overwinter_pcp)
    DCs = 400 * np.log(800 / Qs)  # this is natural logarithm

    DCs[DCs < 15] = 15
    return DCs

def DC(input_date, rain_grid, rh_grid, temp_grid, wind_grid, maxmin, dc_yesterday, index, show, shapefile, mask, endMask,
       last_DC_val_before_shutdown, overwinter):
    '''Calculate the DC.

    Parameters
    ----------
    input_date : string
        input date of interest
    rain_grid : ndarray
        interpolated surface for rainfall on the date of interest
    temp_grid : ndarray
        interpolated surface for temperature on the date of interest
    wind_grid : ndarray
        interpolated surface for wind on the date of interest
    maxmin : list
        bounds of the study area 
    dc_yesterday : ndarray
        array of DC values for yesterday 
    index : int
        index of the date since Mar 1
    show : bool
        whether you want to show the map 
    shapefile : string
        path to the study area shapefile
    mask : ndarray
        mask for the start dates 
    endMask : ndarray
        mask for the end dates
    last_DC_val_before_shutdown : ndarray
        array for last dc values before cell shut down, if no areas required the procedure, you can input an empty
        array of the correct size (if not using overwinter, input the empty array)
    overwinter : bool
        whether or not to implement the overwinter procedure
        
    Returns
    ----------
    ndarray 
        - array of dc values on the date on interest for the study area
    '''

    yesterday_index = index-1
    if yesterday_index == -1:
        if overwinter:
            rain_shape = rain_grid.shape
            dc_initialize = np.zeros(rain_shape)
            dc_initialize[np.isnan(last_DC_val_before_shutdown)] = 15
            dc_initialize[~np.isnan(last_DC_val_before_shutdown)] = last_DC_val_before_shutdown[~np.isnan(
                last_DC_val_before_shutdown)]
            dc_yesterday1 = dc_initialize*mask
        else:
            rain_shape = rain_grid.shape
            dc_initialize = np.zeros(rain_shape)+15
            dc_yesterday1 = dc_initialize
            dc_yesterday1 = dc_yesterday1*mask  # mask out areas that haven't started
    else:
        if overwinter:
            dc_yesterday1 = dc_yesterday
            dc_yesterday1[np.where(np.isnan(dc_yesterday1) & ~np.isnan(mask) & ~np.isnan(last_DC_val_before_shutdown))] = last_DC_val_before_shutdown[np.where(
                np.isnan(dc_yesterday1) & ~np.isnan(mask) & ~np.isnan(last_DC_val_before_shutdown))]
            dc_yesterday1[np.where(np.isnan(dc_yesterday1) & ~np.isnan(
                mask) & np.isnan(last_DC_val_before_shutdown))] = 15
        else:
            dc_yesterday1 = dc_yesterday
            # set started pixels since yesterday to 15
            dc_yesterday1[np.where(
                np.isnan(dc_yesterday1) & ~np.isnan(mask))] = 15

    input_date = str(input_date)
    month = int(input_date[6])
    # Get day length factor

    f101 = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5, 2.4, 0.4, -1.6, -1.6]

    # Put constraint on low end of temp
    temp_grid[temp_grid < -2.8] = -2.8

    # E22 potential evapT

    #pe = (0.36*(temp_grid+2.8)+f101[month])/2

    # Checked code from 419 spreadsheet
    pe = (0.36*(temp_grid+2.8)+f101[month-1])

    # Make empty dc array
    new_shape = dc_yesterday1.shape
    dc = np.zeros(new_shape)

    # starting rain

    netRain = 0.83*rain_grid-1.27

    # eq 19
    smi = 800*np.exp(-1*dc_yesterday1/400)

    # eq 21
    # dr0 = dc_yesterday1 -400*np.log(1+3.937*netRain/smi) #log is the natural logarithm

    # eq21 from 419
    dr_ini = np.array(smi+3.937*netRain)
    dr0 = np.array(400*np.log(800/dr_ini))

    dr0[dr0 < 0] = 0
    dr0[rain_grid <= 2.8] = dc_yesterday1[rain_grid <= 2.8]

    #dc1 = dr0 + pe

    # from 419

    dc1 = np.array(dr0 + 0.5*pe)
    dc1[dc1 < 0] = 0

    dc1 = dc1 * mask * endMask
    if show == True:
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        fig, ax = plt.subplots(figsize=(15, 15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)
        circ = PolygonPatch(na_map['geometry'][0], visible=False)
        ax.add_patch(circ)
        plt.imshow(dc1, extent=(xProj_extent.min(), xProj_extent.max(), yProj_extent.max(), yProj_extent.min()), clip_path=circ,
                   clip_on=True)

        # plt.imshow(dc1,extent=(min_xProj_extent-1,max_xProj_extent+1,max_yProj_extent-1,min_yProj_extent+1))
        na_map.plot(ax=ax, color='white', edgecolor='k',
                    linewidth=2, zorder=10, alpha=0.1)

        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label('DC')

        title = 'DC for %s' % (input_date)
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        plt.show()

    return dc1


def DMC(input_date, rain_grid, rh_grid, temp_grid, wind_grid, maxmin, dmc_yesterday, index, show, shapefile,
        mask, endMask):
    '''Calculate the DMC.

    Parameters
    ----------
    input_date : string
        input date of interest
    rain_grid : ndarray
        interpolated surface for rainfall on the date of interest
    temp_grid : ndarray
        interpolated surface for temperature on the date of interest
    wind_grid : ndarray
        interpolated surface for wind on the date of interest
    maxmin : list
        bounds of the study area 
    dc_yesterday : ndarray
        array of DC values for yesterday 
    index : int
        index of the date since Mar 1
    show : bool
        whether you want to show the map 
    shapefile : string
        path to the study area shapefile
    mask : ndarray
        mask for the start dates 
    endMask : ndarray
        mask for the end dates
    Returns
    ----------
    ndarray
        - array of dmc values on the date on interest for the study area
    '''
    yesterday_index = index-1

    if yesterday_index == -1:
        rain_shape = rain_grid.shape
        dmc_initialize = np.zeros(rain_shape)+6
        dmc_yesterday1 = dmc_initialize*mask
    else:
        dmc_yesterday1 = dmc_yesterday
        dmc_yesterday1[np.where(np.isnan(dmc_yesterday1)
                                & ~np.isnan(mask))] = 6

    #dmc_yesterday = dmc_yesterday1.flatten()
    input_date = str(input_date)
    month = int(input_date[6])
    # Get day length factor

    ell01 = [6.5, 7.5, 9, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8, 7, 6]

    # Put constraint on low end of temp
    temp_grid[temp_grid < -1.1] = -1.1

    # Log drying rate
    rk = 1.894*(temp_grid+1.1)*(100-rh_grid)*ell01[month-1]*1.0E-6

    # Make empty dmc array
    new_shape = dmc_yesterday1.shape
    dmc = np.zeros(new_shape)

    # starting rain

    netRain = 0.92*rain_grid-1.27

    wmi = 20 + (np.exp(5.6348-(dmc_yesterday1/43.43)))

    # if else depending on yesterday dmc, eq.13
    b = np.zeros(new_shape)

    b[dmc_yesterday1 <= 33] = 100 / \
        (0.5+0.3*dmc_yesterday1[dmc_yesterday1 <= 33])
    b[(dmc_yesterday1 > 33) & (dmc_yesterday1 < 65)] = 14-1.3 * \
        np.log(dmc_yesterday1[(dmc_yesterday1 > 33) &
               (dmc_yesterday1 < 65)])  # np.log is ln
    b[dmc_yesterday1 >= 65] = 6.5 * \
        np.log(dmc_yesterday1[dmc_yesterday1 >= 65])-17.2

    # eq 14, modified in R package
    wmr = wmi + 1000 * netRain/(48.77 + b * netRain)

    pr0 = np.array(244.72-(43.43 * (np.log(wmr-20)))

    pr0[pr0 < 0] = 0

    rk_pr0 =pr0 + (100*rk)

    rk_ydmc = dmc_yesterday1 + (100*rk) #we want to add rk because that's the drying rate
                   
    dmc[netRain > 1.5] = rk_pr0[netRain > 1.5]
    dmc[netRain <= 1.5] = rk_ydmc[netRain <= 1.5]

    dmc[dmc < 0] = 0

    dmc = dmc * mask * endMask  # mask out areas that haven't been activated

    if show == True:
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        fig, ax = plt.subplots(figsize=(15, 15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)

        circ = PolygonPatch(na_map['geometry'][0], visible=False)
        ax.add_patch(circ)
        plt.imshow(dmc, extent=(min_xProj_extent-1, max_xProj_extent+1, max_yProj_extent-1, min_yProj_extent+1), clip_path=circ,
                   clip_on=True)

        # plt.imshow(dmc,extent=(min_xProj_extent-1,max_xProj_extent+1,max_yProj_extent-1,min_yProj_extent+1))
        na_map.plot(ax=ax, color='white', edgecolor='k',
                    linewidth=2, zorder=10, alpha=0.1)

        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label('DMC')

        title = 'DMC for %s' % (input_date)
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        plt.show()

    return dmc


def FFMC(input_date, rain_grid, rh_grid, temp_grid, wind_grid, maxmin, ffmc_yesterday, index, show, shapefile,
         mask, endMask):
    '''Calculate the FFMC. 
    Parameters
    ----------
    input_date : string
        input date of interest
    rain_grid : ndarray
        interpolated surface for rainfall on the date of interest
    temp_grid : ndarray
        interpolated surface for temperature on the date of interest
    wind_grid : ndarray
        interpolated surface for wind on the date of interest
    maxmin : list
        bounds of the study area 
    dc_yesterday : ndarray
        array of DC values for yesterday 
    index : int
        index of the date since Mar 1
    show : bool
        whether you want to show the map 
    shapefile : string
        path to the study area shapefile
    mask : ndarray
        mask for the start dates 
    endMask : ndarray
        mask for the end dates
    Returns
    ----------
    ndarray
        - array of ffmc values on the date on interest for the study area
    '''
    yesterday_index = index-1

    if yesterday_index == -1:
        rain_shape = rain_grid.shape
        ffmc_initialize = np.zeros(rain_shape)+85
        ffmc_yesterday1 = ffmc_initialize*mask  # mask out areas that haven't started
    else:
        ffmc_yesterday1 = ffmc_yesterday
        # set started pixels since yesterday to 85
        ffmc_yesterday1[np.where(
            np.isnan(ffmc_yesterday1) & ~np.isnan(mask))] = 85

    wmo = 147.2*(101-ffmc_yesterday)/(59.5+ffmc_yesterday)

    rain_grid[rain_grid > 0.5] = rain_grid[rain_grid > 0.5] - 0.5

    wmo[wmo >= 150] = wmo[wmo >= 150]+0.0015*(wmo[wmo >= 150]-150) *\
        (wmo[wmo >= 150] - 150)*np.sqrt(rain_grid[wmo >= 150]) + 42.5\
        * rain_grid[wmo >= 150]*np.exp(-100/(251-wmo[wmo >= 150])) *\
        (1-np.exp(-6.93/rain_grid[wmo >= 150]))

    wmo[wmo < 150] = wmo[wmo < 150]+42.5*rain_grid[wmo < 150]*np.exp(-100/(251-wmo[wmo < 150]))\
        * (1-np.exp(-6.93/rain_grid[wmo < 150]))

    wmo[rain_grid < 0.5] = 147.2 * \
        (101-ffmc_yesterday[rain_grid < 0.5]) / \
        (59.5+ffmc_yesterday[rain_grid < 0.5])

    wmo[wmo > 250] = 250

    ed = 0.942*np.power(rh_grid, 0.679)+(11*np.exp((rh_grid-100)/10))+0.18*(21.1-temp_grid)\
        * (1-1/np.exp(rh_grid*0.115))

    ew = 0.618*np.power(rh_grid, 0.753)+(10*np.exp((rh_grid-100)/10))+0.18*(21.1-temp_grid) *\
        (1-1/np.exp(rh_grid*0.115))

    shape = rain_grid.shape
    z = np.zeros(shape)
    z[np.where((wmo < ed) & (wmo < ew))] = 0.424*(1-np.power((rh_grid[np.where((wmo < ed) & (wmo < ew))]/100), 1.7))\
        + 0.0694*np.sqrt(wind_grid[np.where((wmo < ed) & (wmo < ew))]) *\
        (1-np.power((rh_grid[np.where((wmo < ed) & (wmo < ew))]/100), 8))

    z[np.where((wmo >= ed) & (wmo >= ew))] = 0

    x = z*0.581*np.exp(0.0365*temp_grid)

    shape = rain_grid.shape
    wm = np.zeros(shape)

    wm[np.where((wmo < ed) & (wmo < ew))] = ew[np.where((wmo < ed) & (wmo < ew))] -\
        (ew[np.where((wmo < ed) & (wmo < ew))] -
         wmo[np.where((wmo < ed) & (wmo < ew))])/(np.power(10, x[np.where((wmo < ed) & (wmo < ew))]))

    wm[np.where((wmo >= ed) & (wmo >= ew))
       ] = wmo[np.where((wmo >= ed) & (wmo >= ew))]

    z[wmo > ed] = 0.424*(1-np.power((rh_grid[wmo > ed]/100), 1.7))+0.0694\
                       * np.sqrt(wind_grid[wmo > ed])*(1-np.power((rh_grid[wmo > ed]/100), 8))

    x = z*0.581*np.exp(0.0365 * temp_grid)
    wm[wmo > ed] = ed[wmo > ed] + \
        (wmo[wmo > ed] - ed[wmo > ed])/(np.power(10, x[wmo > ed]))

    ffmc1 = (59.5*(250-wm))/(147.2+wm)

    ffmc1[ffmc1 > 101] = 101

    ffmc1[ffmc1 < 0] = 0

    ffmc1 = ffmc1*mask * endMask

    if show:
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        fig, ax = plt.subplots(figsize=(15, 15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)

        circ = PolygonPatch(na_map['geometry'][0], visible=False)
        ax.add_patch(circ)
        plt.imshow(ffmc1, extent=(min_xProj_extent-1, max_xProj_extent+1, max_yProj_extent-1, min_yProj_extent+1), clip_path=circ,
                   clip_on=True)

        # plt.imshow(ffmc1,extent=(min_xProj_extent-1,max_xProj_extent+1,max_yProj_extent-1,min_yProj_extent+1))
        na_map.plot(ax=ax, color='white', edgecolor='k',
                    linewidth=2, zorder=10, alpha=0.1)

        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label('FFMC')

        title = 'FFMC for %s' % (input_date)
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        plt.show()

    return ffmc1


def BUI(dmc, dc, maxmin, show, shapefile, mask, endMask):  # BUI can be calculated on the fly
    ''' Calculate BUI
    Parameters
        dmc (np_array): the dmc array for the date of interest
        dc (np_array): the dc array for the date of interest
        maxmin (list): bounds of the study area 
        show (bool): whether or not to display the map 
        shapefile (str): path to the study area shapefile 
        mask (np_array): mask for the start up date 
        endMask (np_array): mask for the shut down date 
    Returns 
        bui1 (np_array): array containing BUI values for the study area 
    '''
    shape = dmc.shape
    bui1 = np.zeros(shape)

    bui1[np.where((dmc == 0) & (dc == 0))] = 0
    bui1[np.where((dmc > 0) & (dc > 0))] = 0.8 * dc[np.where((dmc > 0) & (dc > 0))] *\
        dmc[np.where((dmc > 0) & (dc > 0))]/(dmc[np.where((dmc > 0) & (dc > 0))]
                                             + 0.4 * dc[np.where((dmc > 0) & (dc > 0))])
    p = np.zeros(shape)
    p[dmc == 0] = 0
    p[dmc > 0] = (dmc[dmc > 0] - bui1[dmc > 0])/dmc[dmc > 0]

    cc = 0.92 + (np.power((0.0114 * dmc), 1.7))

    bui0 = dmc - cc * p

    bui0[bui0 < 0] = 0

    bui1[bui1 < dmc] = bui0[bui1 < dmc]

    bui1 = bui1*mask * endMask

    if show:
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        fig, ax = plt.subplots(figsize=(15, 15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)

        plt.imshow(bui1, extent=(min_xProj_extent-1,
                   max_xProj_extent+1, max_yProj_extent-1, min_yProj_extent+1))
        na_map.plot(ax=ax, color='white', edgecolor='k',
                    linewidth=2, zorder=10, alpha=0.1)

        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label('BUI')

        title = 'BUI for %s' % (input_date)
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        plt.show()

    return bui1


def ISI(ffmc, wind_grid, maxmin, show, shapefile, mask, endMask):
    ''' Calculate ISI
    Parameters
        ffmc (np_array): ffmc array for the date of interest
        wind_grid (np_array): wind speed interpolated array for the date of interest
        maxmin (list): bounds of the shapefile
        show (bool): whether or not to display the map 
        shapefile (str): path to the study area shapefile
        mask (np_array): mask for the start up date
        endMask (np_array): mask for the shutdown date
    Returns 
        isi (np_array): the calculated array for ISI for the study area 
    '''

    fm = 147.2 * (101 - ffmc)/(59.5 + ffmc)

    fW = np.exp(0.05039 * wind_grid)

    fF = 91.9 * np.exp((-0.1386) * fm) * (1 + (fm**5.31) / 49300000)

    isi = 0.208 * fW * fF

    isi = isi*mask * endMask
    if show:
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        fig, ax = plt.subplots(figsize=(15, 15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)

        plt.imshow(isi, extent=(min_xProj_extent-1, max_xProj_extent +
                   1, max_yProj_extent-1, min_yProj_extent+1))
        na_map.plot(ax=ax, color='white', edgecolor='k',
                    linewidth=2, zorder=10, alpha=0.1)

        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label('ISI')

        title = 'ISI for %s' % (input_date)
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        plt.show()

    return isi


def FWI(isi, bui, maxmin, show, shapefile, mask, endMask):
    ''' Calculate FWI
    Parameters
        isi (np_array): calculated isi surface for the date of interest 
        bui (np_array): calculated bui surface for the date of interest 
        maxmin (list): bounds of the study area
        show (bool): whether or not to show the map 
        shapefile (str): path to the shapefile 
        mask (np_array): start up mask 
        endMask (np_array): shut down mask 
    Returns 
        fwi (np_array): calculated FWI surface for the study area 
    '''

    shape = isi.shape
    bb = np.zeros(shape)

    bb[bui > 80] = 0.1 * isi[bui > 80] * \
        (1000/(25 + 108.64/np.exp(0.023 * bui[bui > 80])))
    bb[bui <= 80] = 0.1 * isi[bui <= 80] * \
        (0.626 * np.power(bui[bui <= 80], 0.809) + 2)

    fwi = np.zeros(shape)
    fwi[bb <= 1] = bb[bb <= 1]
    # natural logarithm
    fwi[bb > 1] = np.exp(2.72 * ((0.434 * np.log(bb[bb > 1]))**0.647))

    fwi = fwi * mask * endMask

    if show:
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        fig, ax = plt.subplots(figsize=(15, 15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)

        plt.imshow(fwi, extent=(min_xProj_extent-1, max_xProj_extent +
                   1, max_yProj_extent-1, min_yProj_extent+1))
        na_map.plot(ax=ax, color='white', edgecolor='k',
                    linewidth=2, zorder=10, alpha=0.1)

        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label('FWI')

        title = 'FWI for %s' % (input_date)
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        plt.show()

    return fwi


def plot_july(fwi_list, maxmin, year, var, shapefile, shapefile2):
    ''' Visualize all values for July. **DO NOT HAVE TO CHANGE INDEX IF LEAP YEAR** WHY? B/C WE ARE COUNTING FRM MAR1
    Parameters
        fwi_list (list): list of fwi metric arrays for a certain measure (i.e. dmc)
        maxmin (list): bounds of study area
        year (str): year of interest
        var (str): variable name of interest (i.e. "Duff Moisture Code")
        shapefile (str): path to study area shapefile
        shapefile2 (str): path to masking shapefile 
    Returns
        plots a figure with fwi metric map for each day in month of July 
    '''

    fig = plt.figure()
    COUNT = 0
    # The range refers to the start and end indexes of where July is in the list
    for index in range(121, 152):
        ax = fig.add_subplot(4, 8, COUNT+1)
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        max_list = []

        for arr in fwi_list:

            max_list.append(np.amax(arr))

        maxval = max(max_list)

        crs = {'init': 'esri:102001'}
        plt.rcParams["font.family"] = "Calibri"  # "Times New Roman"
        plt.rcParams.update({'font.size': 16})
        # plt.rcParams['image.cmap']='RdYlBu_r'
        # plt.rcParams['image.cmap']='Spectral_r'
        plt.rcParams['image.cmap'] = 'RdYlGn_r'

        na_map = gpd.read_file(shapefile)
        bor_map = gpd.read_file(shapefile2)

        title = str(COUNT+1)
        circ = PolygonPatch(bor_map['geometry'][0], visible=False)
        ax.add_patch(circ)

        im = ax.imshow(fwi_list[index], extent=(min_xProj_extent-1, max_xProj_extent+1, max_yProj_extent-1,
                       min_yProj_extent+1), vmin=0, vmax=maxval, clip_path=circ, clip_on=True, origin='upper')
        na_map.plot(ax=ax, facecolor="none", edgecolor='k', linewidth=1)

        ax.tick_params(axis='both', which='both', bottom=False, top=False,
                       labelbottom=False, right=False, left=False, labelleft=False)
        ax.ticklabel_format(useOffset=False, style='plain')

        ax.set_title(title)
        ax.invert_yaxis()

        COUNT += 1

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar1 = fig.colorbar(im, orientation="vertical", cax=cbar_ax, pad=0.2)
    cbar1.set_label(var)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    fig.text(0.5, 0.04, 'Longitude', ha='center')
    fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical')
    # title = '%s for June %s'%(var,year) #No title for now
    #fig.suptitle(title, fontsize=14)
    plt.show()


def plot_june(fwi_list, maxmin, year, var, shapefile, shapefile2):
    ''' Visualize all values for June. **DO NOT HAVE TO CHANGE INDEX IF LEAP YEAR** WHY? B/C WE ARE COUNTING FRM MAR1
    Parameters
        fwi_list (list): list of fwi metric arrays for a certain measure (i.e. dmc)
        maxmin (list): bounds of study area
        year (str): year of interest
        var (str): variable name of interest (i.e. "Duff Moisture Code")
        shapefile (str): path to study area shapefile
        shapefile2 (str): path to masking shapefile 
    Returns
        plots a figure with fwi metric map for each day in month of June 
    '''

    fig = plt.figure()
    COUNT = 0
    # The range refers to the start and end indexes of where June is in the list
    for index in range(91, 121):
        ax = fig.add_subplot(4, 8, COUNT+1)
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        max_list = []

        for arr in fwi_list:

            if np.isfinite(np.amax(arr.flatten())):

                max_list.append(np.amax(arr.flatten()))

        maxval = np.amax(max_list)
        print(maxval)

        crs = {'init': 'esri:102001'}
        plt.rcParams["font.family"] = "Calibri"  # "Times New Roman"
        plt.rcParams.update({'font.size': 16})
        # plt.rcParams['image.cmap']='RdYlBu_r'
        # plt.rcParams['image.cmap']='Spectral_r'
        plt.rcParams['image.cmap'] = 'RdYlGn_r'

        na_map = gpd.read_file(shapefile)
        bor_map = gpd.read_file(shapefile2)

        title = str(COUNT+1)
        circ = PolygonPatch(bor_map['geometry'][0], visible=False)
        ax.add_patch(circ)

        im = ax.imshow(fwi_list[index], extent=(min_xProj_extent-1, max_xProj_extent+1, max_yProj_extent-1,
                       min_yProj_extent+1), vmin=0, vmax=maxval, clip_path=circ, clip_on=True, origin='upper')
        na_map.plot(ax=ax, facecolor="none", edgecolor='k', linewidth=1)

        ax.tick_params(axis='both', which='both', bottom=False, top=False,
                       labelbottom=False, right=False, left=False, labelleft=False)
        ax.ticklabel_format(useOffset=False, style='plain')

        ax.set_title(title)
        ax.invert_yaxis()

        COUNT += 1

    fig.subplots_adjust(right=0.85)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar1 = fig.colorbar(im, orientation="vertical",
                         cax=cbar_ax, pad=0.2, aspect=10)
    cbar1.set_label(var)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    #fig.text(0.5, 0.04, 'Longitude', ha='center')
    #fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical')
    # title = '%s for June %s'%(var,year) #No title for now
    #fig.suptitle(title, fontsize=14)
    plt.show()


def plot_all(fwi_list, maxmin, year, var, shapefile, shapefile2):
    ''' Visualize all values for all in list. **DO NOT HAVE TO CHANGE INDEX IF LEAP YEAR** WHY? B/C WE ARE COUNTING FRM MAR1
    Parameters
        fwi_list (list): list of fwi metric arrays for a certain measure (i.e. dmc)
        maxmin (list): bounds of study area
        year (str): year of interest
        var (str): variable name of interest (i.e. "Duff Moisture Code")
        shapefile (str): path to study area shapefile
        shapefile2 (str): path to masking shapefile 
    Returns
        plots a figure with fwi metric map for each day 
    '''

    fig = plt.figure()
    COUNT = 0
    # The range refers to the start and end indexes of where June is in the list
    for index in range(0, len(fwi_list)):
        ax = fig.add_subplot(4, 8, COUNT+1)
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        max_list = []

        for arr in fwi_list:

            max_list.append(np.amax(arr))

        maxval = max(max_list)

        crs = {'init': 'esri:102001'}
        plt.rcParams["font.family"] = "Calibri"  # "Times New Roman"
        plt.rcParams.update({'font.size': 16})
        # plt.rcParams['image.cmap']='RdYlBu_r'
        # plt.rcParams['image.cmap']='Spectral_r'
        plt.rcParams['image.cmap'] = 'RdYlGn_r'

        na_map = gpd.read_file(shapefile)
        bor_map = gpd.read_file(shapefile2)

        title = str(COUNT+1)
        circ = PolygonPatch(bor_map['geometry'][0], visible=False)
        ax.add_patch(circ)

        im = ax.imshow(fwi_list[index], extent=(min_xProj_extent-1, max_xProj_extent+1, max_yProj_extent-1,
                       min_yProj_extent+1), vmin=0, vmax=maxval, clip_path=circ, clip_on=True, origin='upper')
        na_map.plot(ax=ax, facecolor="none", edgecolor='k', linewidth=1)

        ax.tick_params(axis='both', which='both', bottom=False, top=False,
                       labelbottom=False, right=False, left=False, labelleft=False)
        ax.ticklabel_format(useOffset=False, style='plain')

        ax.set_title(title)
        ax.invert_yaxis()

        COUNT += 1

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar1 = fig.colorbar(im, orientation="vertical", cax=cbar_ax, pad=0.2)
    cbar1.set_label(var)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    fig.text(0.5, 0.04, 'Longitude', ha='center')
    fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical')
    # title = '%s for June %s'%(var,year) #No title for now
    #fig.suptitle(title, fontsize=14)
    plt.show()


def extract_fire_season_frm_NFDB(file_path, year1, year2, ecozone_path, out_path, search_date_end, search_date_start):
    '''Get the first and last lightning-caused ignitions from the database in ecozone
    Parameters
        file_path (str): path to ignition lookup file
        year1 (int): start year
        year2 (int): end year
        ecozone_path (str): path to the ecozone shapefile
        out_path (str): where to save the results file
        search_date_end (str): 'oct' or 'sep', when to start looking for the last date
        search_date_start (str): 'feb' or 'mar', when to start looking for the last date,
        january start dates considered unrealistic...
    Returns
        first_date (str): first lightning caused ignition in ecozone
        last_date (str): last lightning caused ignition in ecozone
    '''
    first_fire = []
    last_fire = []
    year_list = []
    for year in range(year1, year2+1):
        print('Processing..........'+str(year))
        fire_locs = []
        lookup_dict = {}
        data = pd.read_csv(file_path)
        df2 = data.loc[data['YEAR'] == year]
        #df2 = df2.loc[df2['CAUSE'] == 'L']
        df2 = df2.loc[(df2['SRC_AGENCY'] == 'ON') |
                      (df2['SRC_AGENCY'] == 'QC')]
        fire_locs = list(zip(df2['LATITUDE'], df2['LONGITUDE']))
        initiate_dict = list(
            zip(df2['FIRE_ID'], df2['LATITUDE'], df2['LONGITUDE'], df2['REP_DATE']))
        lookup_dict = {i[0]: [i[1], i[2], i[3]] for i in initiate_dict}

        # How many fires in the year??
        print('Number of fires in year: '+str(len(initiate_dict)))

        proj_dict = {}
        # Project the latitude and longitudes
        for k, v in lookup_dict.items():
            lat = v[0]
            lon = v[1]
            x, y = pyproj.Proj('esri:102001')(lon, lat)
            # Make sure v2 is not before Mar 1

            try:
                if search_date_start == 'mar':

                    d0 = date(int(str(v[2])[0:4]), 3, 1)  # Revert to Mar 1
                    d_End = date(int(str(v[2])[0:4]), 12, 1)  # Dec 1
                    d1 = date(int(str(v[2])[0:4]), int(
                        v[2][5:7]), int(v[2][8:10]))
                    if d0 <= d1 and d_End > d1:  # Exclude fires occurring before Mar 1 and after Dec 1
                        proj_dict[k] = [x, y, v[2]]
                elif search_date_start == 'feb':  # Feb 1
                    d0 = date(int(str(v[2])[0:4]), 2, 1)
                    d_End = date(int(str(v[2])[0:4]), 12, 1)  # Dec 1
                    d1 = date(int(str(v[2])[0:4]), int(
                        v[2][5:7]), int(v[2][8:10]))
                    if d0 <= d1 and d_End > d1:  # Exclude fires occurring before Feb 1 and after Dec 1
                        proj_dict[k] = [x, y, v[2]]
                else:
                    print('That is not a valid search date!')

            except:
                print('Skipping nan value!')

        # check if leap
        is_leap = isleap(int(year))
        if not is_leap:
            num_days_to_feb = 31
            num_days_to_march = 31+28
            num_days_to_sep = 31+28+31+30+31+30+31+31
            num_days_to_oct = 31+28+31+30+31+30+31+31+30
            num_days_to_dec = 31+28+31+30+31+30+31+31+30+31+30
        else:
            num_days_to_feb = 31
            num_days_to_march = 31+29
            num_days_to_sep = 31+29+31+30+31+30+31+31
            num_days_to_oct = 31+29+31+30+31+30+31+31+30
            num_days_to_dec = 31+29+31+30+31+30+31+31+30+31+30

        # Get fires inside the ecozone
        eco_zone = gpd.read_file(ecozone_path)
        ecoDF = gpd.GeoDataFrame(eco_zone)
        ecoDF_union = ecoDF.geometry.unary_union

        updating_list_first = []
        updating_list_last = []
        num_fires_in_zone = 0
        for k, v in proj_dict.items():

            latitude = float(v[1])
            longitude = float(v[0])

            fire_loc = Point((latitude, longitude))
            pointDF = pd.DataFrame([fire_loc])
            gdf = gpd.GeoDataFrame(pointDF, geometry=[fire_loc])
            if (eco_zone.geometry.contains(gdf.geometry)).any():

                # filter out nan
                if len(updating_list_first) > 0 and len(str(v[2])) == 19:
                    num_fires_in_zone += 1

                    if updating_list_first[0] > v[2]:
                        # Get days since January 1

                        updating_list_first[0] = v[2]
                        #print('Overwrite first!')
                        # print(v[2])
                elif len(updating_list_first) == 0:
                    num_fires_in_zone += 1

                    d1 = date(int(v[2][0:4]), int(v[2][5:7]), int(v[2][8:10]))
                    # Calculate from Jan 1
                    d0 = date(int(v[2][0:4]), 1, 1)
                    delta_check = d1 - d0

                    if search_date_end == 'sep':

                        # only if it is before sep 1
                        if delta_check < timedelta(days=num_days_to_sep):
                            updating_list_first.append(v[2])

                    elif search_date_end == 'oct':
                        # only if it is before sep 1
                        if delta_check < timedelta(days=num_days_to_oct):
                            updating_list_first.append(v[2])
                    else:
                        print('That is not a valid search date!')

                else:
                    print('Date is nodata')

                if len(updating_list_last) > 0 and len(str(v[2])) == 19:
                    if updating_list_last[0] < v[2]:

                        updating_list_last[0] = v[2]
                       #print('Overwrite last!')
                       # print(v[2])
                elif len(updating_list_last) == 0:

                    d0 = date(int(str(v[2])[0:4]), 1, 1)
                    d1 = date(int(str(v[2])[0:4]), int(
                        v[2][5:7]), int(v[2][8:10]))
                    if d0 < d1:  # Exclude jan 1
                        updating_list_last.append(v[2])
                else:
                    print('Date is nodata')

        print(num_fires_in_zone)

        if len(updating_list_first) > 0:
            if search_date_start == 'mar':
                # Jan 1 --> Mar 1 Dec 28
                d0 = date(int(updating_list_first[0][0:4]), 3, 1)
            elif search_date_start == 'feb':
                # Jan 1 --> Mar 1 Dec 28
                d0 = date(int(updating_list_first[0][0:4]), 2, 1)
            else:
                print('That is not a valid search date!')
            d1 = date(int(updating_list_first[0][0:4]), int(
                updating_list_first[0][5:7]), int(updating_list_first[0][8:10]))

            delta = d1 - d0

            # Calculate from Jan 1

            d0_jan1 = date(int(updating_list_first[0][0:4]), 1, 1)
            delta_print_to_file = d1 - d0_jan1

            if 'd' not in str(delta_print_to_file)[0:3]:
                if delta >= timedelta(days=0):
                    first_fire.append(str(delta_print_to_file)[0:3])
                    #print('First fire: '+str(delta)[0:3])
                else:
                    first_fire.append(-9999)
            else:
                if delta >= timedelta(days=0):
                    first_fire.append(str(delta_print_to_file)[0:1])
                    #print('First fire: '+str(delta)[0:1])
                else:
                    first_fire.append(-9999)

        else:
            first_fire.append(-9999)
        if len(updating_list_last) > 0:

            if search_date_end == 'sep':

                # Sep 1- revert
                d0 = date(int(updating_list_last[0][0:4]), 9, 1)

            elif search_date_end == 'oct':
                # Sep 1- revert
                d0 = date(int(updating_list_last[0][0:4]), 10, 1)

            else:
                print('That is not a valid search date!')

            d1 = date(int(updating_list_last[0][0:4]), int(
                updating_list_last[0][5:7]), int(updating_list_last[0][8:10]))
            delta = d1 - d0

            # Calculate from Jan 1

            d0_jan1 = date(int(updating_list_last[0][0:4]), 1, 1)
            delta_print_to_file = d1 - d0_jan1

            if 'd' not in str(delta_print_to_file)[0:3]:
                if delta >= timedelta(days=0):
                    last_fire.append(str(delta_print_to_file)[0:3])
                    #print('Last fire: '+str(delta)[0:3])
                else:
                    last_fire.append(-9999)
            else:
                if delta >= timedelta(days=0):
                    last_fire.append(str(delta_print_to_file)[0:1])
                    #print('Last fire: '+str(delta)[0:1])
                else:
                    last_fire.append(-9999)

        else:
            last_fire.append(-9999)

        print('There are '+str(num_fires_in_zone) + ' fires in the zone')
        if num_fires_in_zone <= 5:  # Not enough fires
            first_fire[-1] = -9999
            last_fire[-1] = -9999

        print(first_fire[-1])
        print(last_fire[-1])

        year_list.append(year)
        if int(last_fire[-1]) < 150 and int(last_fire[-1]) != -9999:
            print('Error 1!')
            print(last_fire[-1])
        if int(first_fire[-1]) < 31 and int(first_fire[-1]) != -9999:
            print('Error 2!')
            print(first_fire[-1])

    print(year_list)
    if len(year_list) != len(first_fire) or len(year_list) != len(last_fire):
        print('Error! A year is missing a value!')
    rows = zip(year_list, first_fire, last_fire)
    # Print to a results file
    with open(out_path, "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        for row in rows:
            writer.writerow(row)


def extract_fire_season_frm_fire_archive_report(file_path, year1, year2, ecozone_path, out_path, search_date_end, search_date_start):
    '''Get the first and last lightning-caused ignitions from the extra dataset 
    Parameters
        file_path (str): path to ignition lookup file
        year1 (int): start year
        year2 (int): end year
        ecozone_path (str): path to the ecozone shapefile
        out_path (str): where to save the results file
        search_date_end (str): 'oct' or 'sep', when to start looking for the last date
        search_date_start (str): 'feb' or 'mar', when to start looking for the last date,
        january start dates considered unrealistic...
    Returns
        first_date (str): first lightning caused ignition in ecozone
        last_date (str): last lightning caused ignition in ecozone
        writes output to csv file 
    '''
    first_fire = []
    last_fire = []
    year_list = []
    for year in range(year1, year2+1):
        print('Processing..........'+str(year))
        fire_locs = []
        lookup_dict = {}
        data = pd.read_csv(file_path)
        df2 = data.loc[data['FIRE_YEAR'] == year]
        # df2 = df.loc[df['GENERAL_CAUSE'] == 'LTG'] #All fires
        fire_locs = list(zip(df2['LATITUDE'], df2['LONGITUDE']))
        initiate_dict = list(zip(
            df2['UNIQUE_ID'], df2['LATITUDE'], df2['LONGITUDE'], df2['C_START_DATE_DayofYear']))
        lookup_dict = {i[0]: [i[1], i[2], i[3]] for i in initiate_dict}

        proj_dict = {}
        # Project the latitude and longitudes
        for k, v in lookup_dict.items():
            lat = v[0]
            lon = v[1]
            x, y = pyproj.Proj('esri:102001')(lon, lat)
            proj_dict[k] = [x, y, v[2]]

        # check if leap
        is_leap = isleap(int(year))
        if is_leap:
            num_days_to_feb = 31
            num_days_to_march = 31+28
            num_days_to_sep = 31+28+31+30+31+30+31+31
            num_days_to_oct = 31+28+31+30+31+30+31+31+30
        else:
            num_days_to_feb = 31
            num_days_to_march = 31+29
            num_days_to_sep = 31+29+31+30+31+30+31+31
            num_days_to_oct = 31+29+31+30+31+30+31+31+30

        # Get fires inside the ecozone
        eco_zone = gpd.read_file(ecozone_path)
        ecoDF = gpd.GeoDataFrame(eco_zone)
        ecoDF_union = ecoDF.geometry.unary_union

        updating_list_first = []
        updating_list_last = []
        num_fires_in_zone = 0
        for k, v in proj_dict.items():

            latitude = float(v[1])
            longitude = float(v[0])

            fire_loc = Point((latitude, longitude))
            pointDF = pd.DataFrame([fire_loc])
            gdf = gpd.GeoDataFrame(pointDF, geometry=[fire_loc])
            if (eco_zone.geometry.contains(gdf.geometry)).any():
                num_fires_in_zone += 1
                if search_date_start == 'mar':
                    if v[2] >= num_days_to_march:  # 1 is Jan 1, we exclude
                        if len(updating_list_first) > 0 and updating_list_first[0] > v[2]:
                            updating_list_first[0] = v[2]
                        elif len(updating_list_first) == 0:
                            updating_list_first.append(v[2])
                        else:
                            print('...')
                elif search_date_start == 'feb':
                    if v[2] >= num_days_to_feb:  # 1 is Jan 1, we exclude
                        if len(updating_list_first) > 0 and updating_list_first[0] > v[2]:
                            updating_list_first[0] = v[2]
                        elif len(updating_list_first) == 0:
                            updating_list_first.append(v[2])
                        else:
                            print('...')

                else:
                    print('That is not a valid report date!')

                # End date
                if search_date_end == 'sep':
                    if v[2] >= num_days_to_sep:  # 1 is Jan 1, we exclude
                        if len(updating_list_last) > 0 and updating_list_last[0] < v[2]:
                            updating_list_last[0] = v[2]
                        elif len(updating_list_last) == 0:
                            updating_list_last.append(v[2])
                        else:
                            print('...')
                elif search_date_end == 'oct':
                    if v[2] >= num_days_to_oct:  # 1 is Jan 1, we exclude
                        if len(updating_list_last) > 0 and updating_list_last[0] < v[2]:
                            updating_list_last[0] = v[2]
                        elif len(updating_list_last) == 0:
                            updating_list_last.append(v[2])
                        else:
                            print('...')
                else:
                    print('That is not a valid report date!')

            # if len(updating_list_first) > 0:
                #print('First fire: '+str(updating_list_first[0]))
            # if len(updating_list_last) > 0:
                #print('Last fire: '+str(updating_list_last[0]))
        year_list.append(year)
        try:
            first_fire.append(updating_list_first[0])
        except:
            first_fire.append(-9999)
        try:
            last_fire.append(updating_list_last[0])
        except:
            last_fire.append(-9999)

        if num_fires_in_zone <= 5:  # Not enough fires
            first_fire[-1] = -9999
            last_fire[-1] = -9999
        print(num_fires_in_zone)
        print(first_fire[-1])
        print(last_fire[-1])

    rows = zip(year_list, first_fire, last_fire)
    # Print to a results file
    with open(out_path, "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        for row in rows:
            writer.writerow(row)


def select_and_output_earliest_year(file_path1, file_path2, year1, year2, out_path):
    '''Get the first and last lightning-caused ignitions from the two sources using the csv files
    (we are basically combining them) 
    Parameters
        file_path1 (str): path to the output csv file from national fire database
        file_path2 (str): path to the output csv file from the extra dataset
        year1 (int): start year
        year2 (int): end year
        out_path (str): where to save the results file 
    Returns
        first_date (str): first lightning caused ignition in ecozone
        last_date (str): last lightning caused ignition in ecozone
        writes output to csv file
    '''
    # Get the pandas dataframes
    first_fire = []
    last_fire = []
    year_list = []
    for year in range(year1, year2+1):
        year_list.append(year)
        print('Processing..........'+str(year))
        fire_locs = []
        lookup_dict = {}
        data = pd.read_csv(file_path1)
        df = data.loc[data['YEAR'] == year]
        initiate_dict = list(zip(df['YEAR'], df['START'], df['END']))
        lookup_dict = {i[0]: [i[1], i[2]] for i in initiate_dict}
        data2 = pd.read_csv(file_path2)
        df2 = data2.loc[data2['YEAR'] == year]
        initiate_dict2 = list(zip(df2['YEAR'], df2['START'], df2['END']))
        lookup_dict2 = {i[0]: [i[1], i[2]] for i in initiate_dict2}

        # which column value is smaller for the start date?
        # which column is larger for the end date?

        if lookup_dict2[year][0] == -9999 and lookup_dict[year][0] == -9999:  # if both are NaN
            first_fire.append(-9999)
        else:
            if (lookup_dict2[year][0] <= lookup_dict[year][0]) and lookup_dict2[year][0] != -9999:
                first_fire.append(lookup_dict2[year][0])  # the earlier one

            elif lookup_dict[year][0] == -9999:
                first_fire.append(lookup_dict2[year][0])
            elif lookup_dict2[year][0] == -9999:
                first_fire.append(lookup_dict[year][0])
            else:
                first_fire.append(lookup_dict[year][0])

        if lookup_dict2[year][1] == -9999 and lookup_dict[year][1] == -9999:
            last_fire.append(-9999)
        else:
            if (lookup_dict2[year][1] >= lookup_dict[year][1]) and lookup_dict[year][1] != -9999:
                last_fire.append(lookup_dict2[year][1])

            elif lookup_dict[year][1] == -9999:
                last_fire.append(lookup_dict2[year][1])
            elif lookup_dict2[year][1] == -9999:
                last_fire.append(lookup_dict[year][1])
            else:
                last_fire.append(lookup_dict[year][1])

    # write the merged file to a new file
    rows = zip(year_list, first_fire, last_fire)
    # Print to a results file
    with open(out_path, "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        for row in rows:
            writer.writerow(row)
