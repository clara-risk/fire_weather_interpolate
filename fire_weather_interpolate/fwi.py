#coding: utf-8

"""
Summary
-------
Code for calculating the FWI metrics.

References
----------
Wang, X., Wotton, B. M., Cantin, A. S., Parisien, M. A., Anderson, K., Moore,
B., & Flannigan, M. D. (2017). cffdrs: an R package for the Canadian Forest
Fire Danger Rating System. Ecological Processes, 6(1). https://doi.org/10.1186/s13717-017-0070-z

Wotton, B. M. (2009). Interpreting and using outputs from the Canadian Forest
Fire Danger Rating System in research applications. Environmental and Ecological
Statistics, 16(2), 107–131. https://doi.org/10.1007/s10651-007-0084-2

Code is translated from R package:
https://github.com/cran/cffdrs/tree/master/R

"""
    
#import
import geopandas as gpd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
from itertools import groupby
from datetime import datetime, timedelta, date
import pandas as pd
import math
import os, sys
import gc
import feather 

import get_data as GD
import idw as idw
import idew as idew
import ok as ok
import tps as tps
import rf as rf


#functions 
#Using the feather fires, which are copied faster to another computer, but the code is slower than
#if we use the csv files 
def start_date_calendar(file_path_daily,year):
    '''Returns a dictionary of where each station meets the start up criteria, plus a reference dictionary for the lat lon of the stations
    Parameters
        file_path (str): path to the feather files containing the hourly data from Environment & 
        Climate Change Canada 
        year (str): year we want to find the fire season start up date for 
    Returns 
        date_dict (dict): dictionary containing the start up date for each station (days since Mar 1)
        latlon_dictionary (dict): the latitude and longitude of those stations 
    '''

    #Input: path to hourly data, string of the year, i.e. '1998' 
    maxTempList_dict = {} #Locations where we will store the data
    maxTemp_dictionary = {}
    date_dict = {}
    latlon_dictionary = {}

    for station_name in os.listdir(file_path_hourly): #The dictionary will be keyed by the hourly (temperature) station names, which means all the names must be unique
        Temp_subdict = {} #We will need an empty dictionary to store the data due to data ordering issues 
        temp_list = [] #Initialize an empty list to temporarily store data we will later send to a permanent dictionary 
        for csv in os.listdir(file_path_hourly+station_name+'/'): #Loop through the csv in the station folder
            if year in csv: #Only open if it is the csv for the year of interest (this is contained in the csv name)
                file = file_path_hourly+station_name+'/'+csv  #Open the file - for CAN data we use latin 1 due to à, é etc.
                df = feather.read_dataframe(file)

                count = 0 

                for index, row in df.iterrows():
                    if count == 0:
                        
                        try: 
                        
                            latlon_dictionary[station_name] = (row['Latitude (y)'], row['ï»¿"Longitude (x)"']) #unicode characters at beginning, not sure why 
                            
                        except KeyError: 
                            latlon_dictionary[station_name] = (row['Latitude (y)'], row[0]) #The start unicode problem changes based on the computer... lon should always be in place 0 anyways
                    if str(row['Year']) == year:

                        if str(row['Month']) == '3' or str(row['Month']) == '4' or str(row['Month']) == '5' or \
                           str(row['Month']) == '6' or str(row['Month']) == '7':
                            
                            if str(row['Time']) == '13:00':

                                if pd.notnull(row['Temp (Â°C)']):

                                    Temp_subdict[str(row['Date/Time'])] = float(row['Temp (Â°C)'])
                                    temp_list.append(float(row['Temp (Â°C)'])) #Get the 13h00 temperature, send to temp list
                                else:
                                    Temp_subdict[str(row['Date/Time'])] = 'NA'
                                    temp_list.append('NA')
                    count +=1 

        maxTemp_dictionary[station_name] = Temp_subdict
        maxTempList_dict[station_name] = temp_list #Store the information for each station in the permanent dictionary 

        vals = maxTempList_dict[station_name]

        if 'NA' not in vals and len(vals) == 153: #only consider the stations with unbroken records, num_days between March-July is 153

            varray = np.array(vals)
            where_g12 = np.array(varray >= 12) #Where is the temperature >=12? 


            groups = [list(j) for i, j in groupby(where_g12)] #Put the booleans in groups, ex. [True, True], [False, False, False] 

            length = [x for x in groups if len(x) >= 3 and x[0] == True] #Obtain a list of where the groups are three or longer which corresponds to at least 3 days >= 12


            if len(length) > 0: 
                index = groups.index(length[0]) #Get the index of the group
                group_len = [len(x) for x in groups] #Get length of each group
                length_sofar = 0 #We need to get the number of days up to where the criteria is met 
                for i in range(0,index): #loop through each group until you get to the index and add the length of that group 
                    length_sofar += group_len[i]

                Sdate = list(sorted(maxTemp_dictionary[station_name].keys()))[length_sofar+2] #Go two days ahead for the third day 

                d0 = date(int(year), 3, 1) #March 1, Year 
                d1 = date(int(Sdate[0:4]), int(Sdate[5:7]), int(Sdate[8:10])) #Convert to days since march 1 so we can interpolate
                delta = d1 - d0
                day = int(delta.days) #Convert to integer 
                date_dict[station_name] = day #Store the integer in the dictionary

            else:
                print('Station %s did not start up by August 1.'%station_name) 
                pass #Do not include the station - no start up by August 1 is pretty unrealistic I think... (?) 



            #print('The start date for %s for %s is %s'%(station_name,year,Sdate))

    #Return the dates for each station

    return date_dict, latlon_dictionary

def start_date_calendar_csv(file_path_daily,year):
    '''Returns a dictionary of where each station meets the start up criteria, plus a reference dictionary for the lat lon of the stations
    Parameters
        file_path (str): path to the csv files containing the hourly data from Environment & 
        Climate Change Canada 
        year (str): year we want to find the fire season start up date for 
    Returns 
        date_dict (dict): dictionary containing the start up date for each station (days since Mar 1)
        latlon_dictionary (dict): the latitude and longitude of those stations 
    '''

    #Input: path to hourly data, string of the year, i.e. '1998' 
    maxTempList_dict = {} #Locations where we will store the data
    maxTemp_dictionary = {}
    date_dict = {}
    latlon_dictionary = {}

    for station_name in os.listdir(file_path_hourly): #The dictionary will be keyed by the hourly (temperature) station names, which means all the names must be unique
        Temp_subdict = {} #We will need an empty dictionary to store the data due to data ordering issues 
        temp_list = [] #Initialize an empty list to temporarily store data we will later send to a permanent dictionary 
        count=0
        #for csv in os.listdir(file_path_hourly+station_name+'/'): #Loop through the csv in the station folder 
            #if year in csv: #Only open if it is the csv for the year of interest (this is contained in the csv name)
                #+'/'+csv
        with open(file_path_hourly+station_name, encoding='latin1') as year_information: #Open the file - for CAN data we use latin 1 due to à, é etc. 
            for row in year_information: #Look at each row 
                information = row.rstrip('\n').split(',') #Split each row into a list so we can loop through 
                information_stripped = [i.replace('"','') for i in information] #Get rid of extra quotes in the header
                if count==0: #This is getting the first row 
                    
                    header= information_stripped


                    keyword = 'max_temp' #Look for this keyword in the header 
                    filter_out_keyword = 'flag' #We don't want flag temperature, we want to skip over it 
                    idx_list1 = [i for i, x in enumerate(header) if keyword in x.lower() and filter_out_keyword not in x.lower()] #Get the index of the temperature column

                    if len(idx_list1) > 1: # There should only be one field 
                        print('The program is confused because there is more than one field name that could \
                        contain the temp data. Please check on this.')
                        sys.exit()
                    keyword2 = 'date' #Getting the index of the datetime object so we can later make sure we are using 13h00 value 
                    idx_list2 = [i for i, x in enumerate(header) if keyword2 in x.lower()]

                    if len(idx_list2) > 1: # There should only be one field 
                        print('The program is confused because there is more than one field name that could \
                        contain the date. Please check on this.')
                        sys.exit()

                    keyword3 = 'lat' #Here we use the same methods to get the latitude and longitude 
                    idx_list3 = [i for i, x in enumerate(header) if keyword3 in x.lower()]
                    if len(idx_list3) > 1: # There should only be one field 
                        print('The program is confused because there is more than one field name that could \
                        contain the latitude. Please check on this.')
                        sys.exit()
                    keyword4 = 'lon'
                    idx_list4 = [i for i, x in enumerate(header) if keyword4 in x.lower()]
                    if len(idx_list4) > 1: # There should only be one field 
                        print('The program is confused because there is more than one field name that could \
                        contain the latitude. Please check on this.')
                        sys.exit()
                        
                if count > 0: #Now we are looking at the rest of the file, after the header 

                    if count == 1: #Lat/lon will be all the same so only record it once
                        try: #If the file is corrupted (it usually looks like a bunch of random characters) we will get an error, so we need a try/except loop
                            lat =float(information_stripped[idx_list3[0]])
                            lon =float(information_stripped[idx_list4[0]])
                            latlon_dictionary[station_name[:-4]] = tuple((lat,lon)) #Get the lat lon and send the tuple to the dictionary 
                        except:
                            print('Something is wrong with the lat/lon header names for %s!'%(station_name))
                            break 

  
                        try:
                            if information_stripped[idx_list2[0]][0:4] == year: #Make sure we have the right year 
                                if information_stripped[idx_list2[0]][5:7] == '03' or information_stripped[idx_list2[0]][5:7] == '04' or \
                                   information_stripped[idx_list2[0]][5:7] == '05' or information_stripped[idx_list2[0]][5:7] == '06' or \
                                   information_stripped[idx_list2[0]][5:7] == '07' or information_stripped[idx_list2[0]][5:7] == '08': #Make sure we are only considering months since March in case of heat wave in another month
                                    #if information_stripped[idx_list2[0]][11:13] == '13': #We are only interested in checking the 13h00 temperature
                                    Temp_subdict[information_stripped[idx_list2[0]]] = float(information_stripped[idx_list1[0]])
                                    temp_list.append(float(information_stripped[idx_list1[0]])) #Get the 13h00 temperature, send to temp list
                                    


                        except: #In the case of a nodata value
                            Temp_subdict[information_stripped[idx_list2[0]]] = 'NA'
                            temp_list.append('NA')
                            

                    else: #Proceed down the rows 
                        try:

                            if information_stripped[idx_list2[0]][0:4] == year: 
                                if information_stripped[idx_list2[0]][5:7] == '03' or information_stripped[idx_list2[0]][5:7] == '04' or information_stripped[idx_list2[0]][5:7] == '05'\
                                   or information_stripped[idx_list2[0]][5:7] == '06' or information_stripped[idx_list2[0]][5:7] == '07' or information_stripped[idx_list2[0]][5:7] == '08':
                                    #if information_stripped[idx_list2[0]][11:13] == '13':
                                    Temp_subdict[information_stripped[idx_list2[0]]] = float(information_stripped[idx_list1[0]])
                                    temp_list.append(float(information_stripped[idx_list1[0]]))



                        except:
                            Temp_subdict[information_stripped[idx_list2[0]]] = 'NA'
                            temp_list.append('NA')

                count+=1   

        maxTemp_dictionary[station_name[:-4]] = Temp_subdict
        maxTempList_dict[station_name[:-4]] = temp_list #Store the information for each station in the permanent dictionary 

        vals = maxTempList_dict[station_name[:-4]]

        if 'NA' not in vals and len(vals) == 184: #only consider the stations with unbroken records, num_days between March-August is 153

            varray = np.array(vals)
            where_g12 = np.array(varray >= 12) #Where is the temperature >=12? 


            groups = [list(j) for i, j in groupby(where_g12)] #Put the booleans in groups, ex. [True, True], [False, False, False] 

            length = [x for x in groups if len(x) >= 3 and x[0] == True] #Obtain a list of where the groups are three or longer which corresponds to at least 3 days >= 12


            if len(length) > 0: 

                index = groups.index(length[0]) #Get the index of the group
                group_len = [len(x) for x in groups] #Get length of each group
                length_sofar = 0 #We need to get the number of days up to where the criteria is met 
                for i in range(0,index): #loop through each group until you get to the index and add the length of that group 
                    length_sofar += group_len[i]

                Sdate = list(sorted(maxTemp_dictionary[station_name[:-4]].keys()))[length_sofar+3] #Go three days ahead for the fourth day 

                d0 = date(int(year), 3, 1) #March 1, Year 
                d1 = date(int(Sdate[0:4]), int(Sdate[5:7]), int(Sdate[8:10])) #Convert to days since march 1 so we can interpolate
                delta = d1 - d0
                day = int(delta.days) #Convert to integer 
                date_dict[station_name[:-4]] = day #Store the integer in the dictionary
                
            else:
                print('Station %s did not start up by September 1.'%station_name[:-4]) 
                pass #Do not include the station 


            #print('The start date for %s for %s is %s'%(station_name[:-4],year,Sdate))

    #Return the dates for each station
    #print(date_dict)
    return date_dict, latlon_dictionary 

def start_date_add_hourly(file_path_hourly, year):
    ''' There are not enough daily stations for accurate calculation of the fire season duration in years after about 1996(?) so we need to add in some of the hourly ones. 
    This is computationally more intensive so should only be used if necessary (ie not for years that already have enough daily stations). 
    Parameters
        file_path_hourly (str): path to the hourly feather files
        year (str): str of the year of interest
    Returns
        date_dict (dict): start up date dictionary 
        latlon_dictionary (dict): dictionary of the station locations
    '''
    maxTempList_dict = {} #Locations where we will store the data
    maxTemp_dictionary = {}
    date_dictH = {}
    latlon_dictionary = {}
    
    for station_name in os.listdir(file_path_hourly): #The dictionary will be keyed by the hourly (temperature) station names, which means all the names must be unique

        Temp_subdict = {} #We will need an empty dictionary to store the data due to data ordering issues 
        temp_list = [] #Initialize an empty list to temporarily store data we will later send to a permanent dictionary 
        for csv in os.listdir(file_path_hourly+station_name+'/'): #Loop through the csv in the station folder
            if csv[-16:-12] == year and (csv[-19:-17] == '03' or csv[-19:-17] == '04'or csv[-19:-17] == '05'\
               or csv[-19:-17] == '06' or csv[-19:-17] == '07' or csv[-19:-17] == '08'): #Only open if it is the csv for the year of interest (this is contained in the csv name)
                file = file_path_hourly+station_name+'/'+csv  #Open the file - for CAN data we use latin 1 due to à, é etc.
                df = feather.read_dataframe(file)
                unique_dates = set([x[0:10] for x in df['Date/Time'].unique()]) 
                count = 0
                for dat in sorted(unique_dates): #We need dictionary insertion order maintained
                    if dat in Temp_subdict.keys():
                        #print('Date is already in the sub-dictionary')
                        break
                    temp_24 = {} 
                
                    for index, row in df.iterrows():
                        if count == 0:
                            
                            try: 
                            
                                latlon_dictionary[station_name] = (row['Latitude (y)'], row['ï»¿"Longitude (x)"']) #unicode characters at beginning, not sure why 
                                
                            except KeyError: 
                                latlon_dictionary[station_name] = (row['Latitude (y)'], row[0]) #The start unicode problem changes based on the computer... lon should always be in place 0 anyways

                        if str(row['Year']) == year:

                            if str(row['Month']) == '3' or str(row['Month']) == '4' or str(row['Month']) == '5' or \
                               str(row['Month']) == '6' or str(row['Month']) == '7' or str(row['Month']) == '8':
                                #if str(row['Time']) == '13:00':
                                if str(row['Date/Time'])[0:10] == dat:
                                    if pd.notnull(row['Temp (Â°C)']):
                                        #print(float(row['Temp (Â°C)']))
                                        temp_24[str(row['Date/Time'])] = float(row['Temp (Â°C)'])
                                        #temp_list.append(float(row['Temp (Â°C)']))
                                    else:
                                        pass
                        else:
                             break
                            
                            
                    if len(temp_24.values()) == 24: #Make sure unbroken record
                    
                                
                        Temp_subdict[dat] = max(temp_24.values()) #Send max temp to dictionary 
                        temp_list.append(max(temp_24.values())) #Get the 13h00 temperature, send to temp list
                    else:
                        Temp_subdict[dat] = 'NA'
                        temp_list.append('NA')
                    count +=1 

        maxTemp_dictionary[station_name] = Temp_subdict
        maxTempList_dict[station_name] = temp_list #Store the information for each station in the permanent dictionary 

        vals = maxTempList_dict[station_name]

        if 'NA' not in vals and len(vals) == 184: #only consider the stations with unbroken records, num_days between March-July is 153
            

            varray = np.array(vals)
            where_g12 = np.array(varray >= 12) #Where is the temperature >=12? 


            groups = [list(j) for i, j in groupby(where_g12)] #Put the booleans in groups, ex. [True, True], [False, False, False]
            

            length = [x for x in groups if len(x) >= 3 and x[0] == True] #Obtain a list of where the groups are three or longer which corresponds to at least 3 days >= 12
            

            if len(length) > 0: 
                index = groups.index(length[0]) #Get the index of the group
                group_len = [len(x) for x in groups] #Get length of each group
                length_sofar = 0 #We need to get the number of days up to where the criteria is met 
                for i in range(0,index): #loop through each group until you get to the index and add the length of that group 
                    length_sofar += group_len[i]

                Sdate = list(sorted(maxTemp_dictionary[station_name].keys()))[length_sofar+3] #Go two days ahead for the third day 

                d0 = date(int(year), 3, 1) #March 1, Year 
                d1 = date(int(Sdate[0:4]), int(Sdate[5:7]), int(Sdate[8:10])) #Convert to days since march 1 so we can interpolate
                delta = d1 - d0
                day = int(delta.days) #Convert to integer
                if station_name in os.listdir(file_path_hourly): #avoid some hidden files 
                    date_dictH[station_name] = day #Store the integer in the dictionary
                    print(station_name)
                    print(day)

            else:
                print('Station %s did not start up by September 1.'%station_name) 
                pass #Do not include the station - no start up by August 1 is pretty unrealistic I think... (?) 
=
    if len(date_dictH.keys()) == 0:
        return None, None #Save overhead associated with creating an empty dictionary 
    else:
        return date_dictH, latlon_dictionary 

def end_date_calendar(file_path_daily,year):
    '''Returns a dictionary of where each station meets the start up criteria, 
    plus a reference dictionary for the lat lon of the stations
    Parameters
        file_path (str): path to the feather files containing the hourly data from Environment & 
        Climate Change Canada 
        year (str): year we want to find the fire season end date for 
    Returns 
        date_dict (dict): dictionary containing the end date for each station (days since Oct 1)
        latlon_dictionary (dict): the latitude and longitude of those stations 
    '''
    #Input: path to hourly data, string of the year, i.e. '1998'
    maxTempList_dict = {} #Locations where we will store the data
    maxTemp_dictionary = {}
    date_dict = {}
    latlon_dictionary = {}
    for station_name in os.listdir(file_path_hourly): #The dictionary will be keyed by the hourly (temperature) station names, which means all the names must be unique
        Temp_subdict = {} #We will need an empty dictionary to store the data due to data ordering issues 
        temp_list = [] #Initialize an empty list to temporarily store data we will later send to a permanent dictionary 
        for csv in os.listdir(file_path_hourly+station_name+'/'): #Loop through the csv in the station folder
            if year in csv: #Only open if it is the csv for the year of interest (this is contained in the csv name)
                file = file_path_hourly+station_name+'/'+csv  #Open the file - for CAN data we use latin 1 due to à, é etc.
                df = feather.read_dataframe(file)

                count = 0 

                for index, row in df.iterrows():
                    if count == 0:
                        
                        try: 
                        
                            latlon_dictionary[station_name] = (row['Latitude (y)'], row['ï»¿"Longitude (x)"']) #unicode characters at beginning, not sure why 
                        except KeyError: 
                            latlon_dictionary[station_name] = (row['Latitude (y)'], row[0]) #to allow the code to be moved computers 
                    if str(row['Year']) == year:

                        if str(row['Month']) == '10' or str(row['Month']) == '11' or str(row['Month']) == '12':

                            if str(row['Time']) == '13:00':

                                if pd.notnull(row['Temp (Â°C)']):
                                    Temp_subdict[row['Date/Time']] = float(row['Temp (Â°C)'])
                                    temp_list.append(float(row['Temp (Â°C)'])) #Get the 13h00 temperature, send to temp list
                                else:
                                    Temp_subdict[row['Date/Time']] = 'NA'
                                    temp_list.append('NA')
                    count +=1 

                    
        maxTemp_dictionary[station_name] = Temp_subdict
        maxTempList_dict[station_name] = temp_list #Store the information for each station in the permanent dictionary
        vals = maxTempList_dict[station_name]

        
        if 'NA' not in vals and len(vals) == 92: #only consider the stations with unbroken records, num_days between Oct1-Dec31 = 92

            varray = np.array(vals)
            where_g12 = np.array(varray < 5) #Where is the temperature < 5? 


            groups = [list(j) for i, j in groupby(where_g12)] #Put the booleans in groups, ex. [True, True], [False, False, False] 

            length = [x for x in groups if len(x) >= 3 and x[0] == True] #Obtain a list of where the groups are three or longer which corresponds to at least 3 days < 5

            
            index = groups.index(length[0]) #Get the index of the group
            group_len = [len(x) for x in groups] #Get length of each group
            length_sofar = 0 #We need to get the number of days up to where the criteria is met 
            for i in range(0,index): #loop through each group until you get to the index and add the length of that group 
                length_sofar += group_len[i]

            Sdate = list(sorted(maxTemp_dictionary[station_name].keys()))[length_sofar+2] #Go two days ahead for the third day 

            d0 = date(int(year), 10, 1) #Oct 1, Year 
            d1 = date(int(Sdate[0:4]), int(Sdate[5:7]), int(Sdate[8:10])) #Convert to days since Oct 1 so we can interpolate
            delta = d1 - d0
            day = int(delta.days) #Convert to integer 
            date_dict[station_name] = day #Store the integer in the dictionary 


            #print('The end date for %s for %s is %s'%(station_name,year,Sdate))

    #Return the dates for each station
    return date_dict, latlon_dictionary

def end_date_add_hourly(file_path_hourly, year):
    '''Returns a dictionary of where each station meets the end criteria, 
    plus a reference dictionary for the lat lon of the stations
    Parameters
        file_path (str): path to the feather files containing the hourly data from Environment & 
        Climate Change Canada 
        year (str): year we want to find the fire season end date for 
    Returns 
        date_dict (dict): dictionary containing the end date for each station (days since Oct 1)
        latlon_dictionary (dict): the latitude and longitude of those stations 
    '''
    maxTempList_dict = {} #Locations where we will store the data
    maxTemp_dictionary = {}
    date_dict = {}
    latlon_dictionary = {}

    for station_name in os.listdir(file_path_hourly): #The dictionary will be keyed by the hourly (temperature) station names, which means all the names must be unique
        print(station_name) 
        Temp_subdict = {} #We will need an empty dictionary to store the data due to data ordering issues 
        temp_list = [] #Initialize an empty list to temporarily store data we will later send to a permanent dictionary 
        for csv in os.listdir(file_path_hourly+station_name+'/'): #Loop through the csv in the station folder
            if csv[-16:-12] == year and (csv[-19:-17] == '09' or csv[-19:-17] == '10'or csv[-19:-17] == '11'\
               or csv[-19:-17] == '12'): #Only open if it is the csv for the year of interest (this is contained in the csv name)
                file = file_path_hourly+station_name+'/'+csv  #Open the file - for CAN data we use latin 1 due to à, é etc.
                df = feather.read_dataframe(file)
                unique_dates = set([x[0:10] for x in df['Date/Time'].unique()]) 
                count = 0
                for dat in sorted(unique_dates): #We need dictionary insertion order maintained
                    if dat in Temp_subdict.keys():
                        #print('Date is already in the sub-dictionary')
                        break
                    temp_24 = {} 
                
                    for index, row in df.iterrows():
                        if count == 0:
                            
                            try: 
                            
                                latlon_dictionary[station_name] = (row['Latitude (y)'], row['ï»¿"Longitude (x)"']) #unicode characters at beginning, not sure why 
                                
                            except KeyError: 
                                latlon_dictionary[station_name] = (row['Latitude (y)'], row[0]) #The start unicode problem changes based on the computer... lon should always be in place 0 anyways

                        if str(row['Year']) == year:

                            if str(row['Month']) == '9' or str(row['Month']) == '10' or str(row['Month']) == '11' or \
                               str(row['Month']) == '12':
                                #if str(row['Time']) == '13:00':
                                if str(row['Date/Time'])[0:10] == dat:
                                    if pd.notnull(row['Temp (Â°C)']):
                                        #print(float(row['Temp (Â°C)']))
                                        temp_24[str(row['Date/Time'])] = float(row['Temp (Â°C)'])
                                        #temp_list.append(float(row['Temp (Â°C)']))
                                    else:
                                        pass
                        else:
                             break
                            
                            
                    if len(temp_24.values()) == 24: #Make sure unbroken record
                    
                                
                        Temp_subdict[dat] = max(temp_24.values()) #Send max temp to dictionary 
                        temp_list.append(max(temp_24.values())) #Get the 13h00 temperature, send to temp list
                    else:
                        Temp_subdict[dat] = 'NA'
                        temp_list.append('NA')
                    count +=1 

        maxTemp_dictionary[station_name] = Temp_subdict
        maxTempList_dict[station_name] = temp_list #Store the information for each station in the permanent dictionary 

        vals = maxTempList_dict[station_name]

        if 'NA' not in vals and len(vals) == 122: #only consider the stations with unbroken records, num_days between Oct1-Dec31 = 92

            varray = np.array(vals)
            where_g12 = np.array(varray < 5) #Where is the temperature < 5? 


            groups = [list(j) for i, j in groupby(where_g12)] #Put the booleans in groups, ex. [True, True], [False, False, False] 

            length = [x for x in groups if len(x) >= 3 and x[0] == True] #Obtain a list of where the groups are three or longer which corresponds to at least 3 days < 5

            if len(length) > 0: 
                index = groups.index(length[0]) #Get the index of the group
                group_len = [len(x) for x in groups] #Get length of each group
                length_sofar = 0 #We need to get the number of days up to where the criteria is met 
                for i in range(0,index): #loop through each group until you get to the index and add the length of that group 
                    length_sofar += group_len[i]

                Sdate = list(sorted(maxTemp_dictionary[station_name].keys()))[length_sofar+2] #Go two days ahead for the third day 

                d0 = date(int(year), 10, 1) #Oct 1, Year 
                d1 = date(int(Sdate[0:4]), int(Sdate[5:7]), int(Sdate[8:10])) #Convert to days since Oct 1 so we can interpolate
                delta = d1 - d0
                day = int(delta.days) #Convert to integer 
                date_dict[station_name] = day #Store the integer in the dictionary

            else:
                print('Station %s did not end by December 31.'%station_name[:-4]) 
                pass #Do not include the station 

    if len(date_dict.keys()) == 0:
        return None, None #Save overhead associated with creating an empty dictionary 
    else:
        return date_dict, latlon_dictionary

def end_date_calendar_csv(file_path_daily,year):
    '''Returns a dictionary of where each station meets the end criteria see 
    Wotton & Flannigan 1993, plus a reference dictionary for the lat lon of the stations
    Parameters
        file_path (str): path to the csv files containing the hourly data from Environment & 
        Climate Change Canada 
        year (str): year we want to find the fire season end date for 
    Returns 
        date_dict (dict): dictionary containing the end date for each station (days since Oct 1)
        latlon_dictionary (dict): the latitude and longitude of those stations 
    '''
    #Input: path to hourly data, string of the year, i.e. '1998' 
    maxTempList_dict = {} #Locations where we will store the data
    maxTemp_dictionary = {}
    date_dict = {}
    latlon_dictionary = {}

    for station_name in os.listdir(file_path_hourly): #The dictionary will be keyed by the hourly (temperature) station names, which means all the names must be unique
        Temp_subdict = {} #We will need an empty dictionary to store the data due to data ordering issues 
        temp_list = [] #Initialize an empty list to temporarily store data we will later send to a permanent dictionary 
        count=0
        #for csv in os.listdir(file_path_hourly+station_name+'/'): #Loop through the csv in the station folder 
            #if year in csv: #Only open if it is the csv for the year of interest (this is contained in the csv name)

        with open(file_path_hourly+station_name, encoding='latin1') as year_information: #Open the file - for CAN data we use latin 1 due to à, é etc. 
            for row in year_information: #Look at each row 
                information = row.rstrip('\n').split(',') #Split each row into a list so we can loop through 
                information_stripped = [i.replace('"','') for i in information] #Get rid of extra quotes in the header
                if count==0: #This is getting the first row 
                    
                    header= information_stripped



                    keyword = 'max_temp' #Look for this keyword in the header 
                    filter_out_keyword = 'flag' #We don't want flag temperature, we want to skip over it 
                    idx_list1 = [i for i, x in enumerate(header) if keyword in x.lower() and filter_out_keyword not in x.lower()] #Get the index of the temperature column

                    if len(idx_list1) > 1: # There should only be one field 
                        print('The program is confused because there is more than one field name that could \
                        contain the temp data. Please check on this.')
                        sys.exit()
                    keyword2 = 'date' #Getting the index of the datetime object so we can later make sure we are using 13h00 value 
                    idx_list2 = [i for i, x in enumerate(header) if keyword2 in x.lower()]

                    if len(idx_list2) > 1: # There should only be one field 
                        print('The program is confused because there is more than one field name that could \
                        contain the date. Please check on this.')
                        sys.exit()

                    keyword3 = 'lat' #Here we use the same methods to get the latitude and longitude 
                    idx_list3 = [i for i, x in enumerate(header) if keyword3 in x.lower()]
                    if len(idx_list3) > 1: # There should only be one field 
                        print('The program is confused because there is more than one field name that could \
                        contain the latitude. Please check on this.')
                        sys.exit()
                    keyword4 = 'lon'
                    idx_list4 = [i for i, x in enumerate(header) if keyword4 in x.lower()]
                    if len(idx_list4) > 1: # There should only be one field 
                        print('The program is confused because there is more than one field name that could \
                        contain the latitude. Please check on this.')
                        sys.exit()
                                
                if count > 0: #Now we are looking at the rest of the file, after the header 

                    if count == 1: #Lat/lon will be all the same so only record it once
                        try: #If the file is corrupted (it usually looks like a bunch of random characters) we will get an error, so we need a try/except loop
                            lat =float(information_stripped[idx_list3[0]])
                            lon =float(information_stripped[idx_list4[0]])
                            latlon_dictionary[station_name[:-4]] = tuple((lat,lon)) #Get the lat lon and send the tuple to the dictionary 
                        except:
                            print('Something is wrong with the lat/lon header names for %s!'%(station_name))
                            break 

  
                        try:
                            if information_stripped[idx_list2[0]][0:4] == year: #Make sure we have the right year 
                                if information_stripped[idx_list2[0]][5:7] == '09' or information_stripped[idx_list2[0]][5:7] == '10' or \
                                   information_stripped[idx_list2[0]][5:7] == '11' or information_stripped[idx_list2[0]][5:7] == '12': #Make sure we are only considering months after October 
                                    #if information_stripped[idx_list2[0]][11:13] == '13': #We are only interested in checking the 13h00 temperature
                                    Temp_subdict[information_stripped[idx_list2[0]]] = float(information_stripped[idx_list1[0]])
                                    temp_list.append(float(information_stripped[idx_list1[0]])) #Get the 13h00 temperature, send to temp list
                                    


                        except: #In the case of a nodata value
                            Temp_subdict[information_stripped[idx_list2[0]]] = 'NA'
                            temp_list.append('NA')
                            

                    else: #Proceed down the rows 
                        try:

                            if information_stripped[idx_list2[0]][0:4] == year: 
                                if information_stripped[idx_list2[0]][5:7] == '09' or information_stripped[idx_list2[0]][5:7] == '10' or \
                                   information_stripped[idx_list2[0]][5:7] == '11' or information_stripped[idx_list2[0]][5:7] == '12':
                                    #if information_stripped[idx_list2[0]][11:13] == '13':
                                    Temp_subdict[information_stripped[idx_list2[0]]] = float(information_stripped[idx_list1[0]])
                                    temp_list.append(float(information_stripped[idx_list1[0]]))



                        except:
                            Temp_subdict[information_stripped[idx_list2[0]]] = 'NA'
                            temp_list.append('NA')

                count+=1   

        maxTemp_dictionary[station_name[:-4]] = Temp_subdict
        maxTempList_dict[station_name[:-4]] = temp_list #Store the information for each station in the permanent dictionary 

        vals = maxTempList_dict[station_name[:-4]]

        if 'NA' not in vals and len(vals) == 122: #only consider the stations with unbroken records, num_days between Oct1-Dec31 = 92, Sep1-Dec31=122

            varray = np.array(vals)
            where_g12 = np.array(varray < 5) #Where is the temperature < 5? 


            groups = [list(j) for i, j in groupby(where_g12)] #Put the booleans in groups, ex. [True, True], [False, False, False] 

            length = [x for x in groups if len(x) >= 3 and x[0] == True] #Obtain a list of where the groups are three or longer which corresponds to at least 3 days < 5

            
            if len(length) > 0: 
                index = groups.index(length[0]) #Get the index of the group
                group_len = [len(x) for x in groups] #Get length of each group
                length_sofar = 0 #We need to get the number of days up to where the criteria is met 
                for i in range(0,index): #loop through each group until you get to the index and add the length of that group 
                    length_sofar += group_len[i]

                Sdate = list(sorted(maxTemp_dictionary[station_name[:-4]].keys()))[length_sofar+3] #Go three days ahead for the fourth day (end day) 

                d0 = date(int(year), 9, 1) #Oct 1, Year 
                d1 = date(int(Sdate[0:4]), int(Sdate[5:7]), int(Sdate[8:10])) #Convert to days since Oct 1 so we can interpolate
                delta = d1 - d0
                day = int(delta.days) #Convert to integer 
                date_dict[station_name[:-4]] = day #Store the integer in the dictionary

            else:
                print('Station %s did not end by December 31.'%station_name[:-4]) 
                pass #Do not include the station 


            #print('The end date for %s for %s is %s'%(station_name,year,Sdate))

    #Return the dates for each station
    #print(date_dict)
    return date_dict, latlon_dictionary

def calc_season_change(earlier_array,later_array):
    '''Calculate the change between seasons so we can evaluate how much the season has changed over time.
    Parameters
        earlier_array (np_array): array of fire season duration values for the earlier year, ex 1919
        later_array (np_array): array of fire season duration values for the later year, ex 2019
    Returns
        change_array (np_array): array containing the difference for each pixel
    '''
    change_array = earlier_array-later_array
    return change_array 

def get_date_index(year,input_date,month):
    '''Get the number of days for the date of interest from the first of the month of interest
    Example, convert to days since March 1 
    Parameters
        year (str): year of interest 
        input_date (str): input date of interest
        month (str): the month from when you want to calculate the date (ex, Mar 1)
    Returns 
        day (int): days since 1st of the month of interest 
    '''
    d0 = date(int(year), month, 1)
    input_date = str(input_date)
    d1 = date(int(input_date[0:4]), int(input_date[5:7]), int(input_date[8:10])) #convert to days since march 1/oct 1 so we can interpolate
    delta = d1 - d0
    day = int(delta.days)
    return day

def make_start_date_mask(day_index,day_interpolated_surface):
    '''Turn the interpolated surface of start dates into a numpy array
    Parameters
        day_index (int): index of the day of interest since Mar 1
        day_interpolated_surface (np_array): the interpolated surface of the start dates across the 
        study area 
    Returns 
        new (np_array): a mask array of the start date, either activated (1) or inactivated (np.nan)
    '''
    shape = day_interpolated_surface.shape
    new = np.ones(shape)
    new[day_interpolated_surface <= day_index] = 1 #If the day in the interpolated surface is before the index, it is activated 
    new[day_interpolated_surface > day_index] = np.nan #If it is the opposite it will be masked out, so assign it to np.nan (no data) 
    return new

def make_end_date_mask(day_index,day_interpolated_surface):
    '''Turn the interpolated surface of end dates into a numpy array
    Parameters
        day_index (int): index of the day of interest since Oct 1
        day_interpolated_surface (np_array): the interpolated surface of the end dates across the 
        study area 
    Returns 
        new (np_array): a mask array of the end date, either activated (1) or inactivated (np.nan)
    '''
    shape = day_interpolated_surface.shape
    new = np.ones(shape)
    new[day_interpolated_surface <= day_index] = np.nan #If the day in the interpolated surface is before the index, its closed
    new[day_interpolated_surface > day_index] = 1 #If it is the opposite it will be left open, so assign it to 1
    return new

def get_overwinter_pcp(overwinter_dates, file_path_daily,start_surface,end_surface,maxmin, shapefile,
                       show,date_dictionary,latlon_dictionary, file_path_elev,idx_list, json):
    '''Get the total amount of overwinter pcp for the purpose of knowing where to use the 
    overwinter DC procedure 
    Parameters
        overwinter_dates (list): list of dates that are in the winter (i.e. station shut down to 
        start up), you can just input generally Oct 1-June 1 and stations that are still active 
        will be masked out 
        file_path_daily (str): file path to the daily feather files containing the precipitation data
        start_surface (np_array): array containing the interpolated start-up date for each cell 
        end_surface (np_array): array containing the interpolated end date for each cell, from the 
        year before 
        maxmin (list): bounds of the study area 
        shapefile (str): path to the study area shapefile 
        date_dictionary (dict, loaded from .json): lookup file that has what day/month pairs each 
        station contains data for 
        latlon_dictionary (dict, loaded from .json): lat lons of the daily stations
        file_path_elev (str): file path to the elevation lookup file 
        idx_list (list): the index of the elevation data column in the lookup file 
        json (bool): if True, convert the array to a flat list so it can be written as a .json file
        to the hard drive
    Returns 
        pcp_overwinter (np_array): array of interpolated overwinter precipitation for the study area 
        overwinter_reqd (np_array): array indicating where to do the overwinter DC procedure 
    '''
    pcp_list = [] 

    for o_dat in overwinter_dates: #dates is from Oct 1 year before to start day current year (up to Apr 1)
        year = str(o_dat)[0:4]
        index = overwinter_dates.index(o_dat) #we need to take index while its still a timestamp before convert to str
        o_dat = str(o_dat)
        day_index= get_date_index(year,o_dat,3)
        eDay_index = get_date_index(year,o_dat,10)
        if int(str(o_dat)[5:7]) >= 10:


            invertEnd = np.ones(end_surface.shape)
            endMask = make_end_date_mask(eDay_index,end_surface)
            invertEnd[endMask == 1] = 0 #0 in place of np.nan, because we use sum function later 
            invertEnd[np.where(np.isnan(endMask))] = 1

            endMask = invertEnd 
            mask = np.ones(start_surface.shape) #No stations will start up until the next spring
        elif int(str(o_dat)[5:7]) < 10 and int(str(o_dat)[5:7]) >= 7: #All stations are in summer

            endMask = np.zeros(end_surface.shape)
            mask = np.zeros(start_surface.shape)
        else:

            endMask = np.ones(end_surface.shape) #by this time (Jan 1) all stations are in winter
            invertMask = np.ones(start_surface.shape)
            mask = make_start_date_mask(day_index,start_surface)
            invertMask[mask == 1] = 0 #If the station has started, stop counting it 
            invertMask[np.where(np.isnan(mask))] = 1 #If the station is still closed, keep counting
            mask = invertMask

        rainfall = GD.get_pcp(str(o_dat)[0:10],file_path_daily,date_dictionary)
        rain_grid,maxmin = idw.IDW(latlon_dictionary,rainfall,o_dat,'Precipitation',shapefile,False,1)

        masked = rain_grid * mask * endMask

        masked[np.where(np.isnan(masked))] = 0 #sum can't handle np.nan

        pcp_list.append(masked)

        
    #when we sum, we need to treat nan as 0, otherwise any nan in the list will cause the whole val to be nan 
    pcp_overwinter = sum(pcp_list)


    overwinter_reqd = np.ones(pcp_overwinter.shape)
    overwinter_reqd[pcp_overwinter> 200] = np.nan #not Required
    overwinter_reqd[pcp_overwinter <= 200] = 1 #required 

    if show: 
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        fig, ax = plt.subplots(figsize= (15,15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)
        
      
        plt.imshow(overwinter_reqd,extent=(min_xProj_extent-1,max_xProj_extent+1,max_yProj_extent-1,min_yProj_extent+1)) 
        na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)
            
        plt.gca().invert_yaxis()
        
        title = 'Areas Requiring Overwinter DC Procedure for %s'%(year)
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        plt.show()

    if json:
        pcp_overwinter = list(pcp_overwinter.flatten()) #json cannot handle numpy arrays 
        overwinter_reqd = list(overwinter_reqd.flatten()) 

    return pcp_overwinter,overwinter_reqd #This is the array, This is a mask 

def dc_stack(dates,file_path_daily,file_path_hourly,var_name,shapefile,day_interpolated_surface,end_interpolated_surface,
             last_DC_val_before_shutdown,overwinter,file_path_elev,idx_list,date_dictionary,latlon_dict,latlon_dictionary,
             json,interpolation_method):
    '''Calc dc for each day in season. This is the only metric with overwinter procedure applied. 
    For notes see cffdrs R code.
    Parameters
        dates (list): list of all dates within the fire season, inactive stations will be masked out 
        so you can define it as Mar 1 - Dec 31
        file_path_daily (str): file path to the daily feather files
        file_path_hourly (str): file path to the hourly feather files
        var_name (str): name of the variable you are interpolating
        shapefile (str): path to the study area shapefile 
        day_interpolated_surface (np_array): array of start-up days (since Mar 1) for the study area
        end_interpolated_surface (np_array): array of end days (since Oct 1) for the study area
        last_DC_val_before_shutdown (np_array): the last DC value in each cell before that cell shut down
        for the winter (WITH THE OVERWINTER PROCEDURE APPLIED); only need to input this if overwinter 
        = True otherwise you can just input None 
        overwinter (bool): if True, the program overwinters the DC where it needs to be
        file_path_elev (str): file path to the elevation lookup file 
        idx_list (list): the index of the elevation data column in the lookup file 
        date_dictionary (dict, loaded from .json): lookup file that has what day/month pairs each 
        station contains data for 
        latlon_dict (dict, loaded from .json): lat lons of the hourly stations
        latlon_dictionary (dict, loaded from .json): lat lons of the daily stations
        json (bool): if True, convert the array to a flat list so it can be written as a .json file
        interpolation_method (str): the interpolation method to use to get the continuous DC surface, 
        there are eight options - IDW-1, IDW-2, IDEW-1, IDEW-2, TPS, TPSS, OK, RF
    Returns 
        dc_list (list of np_array): a list of the interpolated surfaces for the drought code for 
        each day in the fire season 
    '''

    print(interpolation_method)
    
    dc_list = [] 
    count = 0 
    for dat in dates:
        gc.collect() 
        year = str(dat)[0:4]
        index = dates.index(dat)
        dat = str(dat)
        day_index= get_date_index(year,dat,3)
        eDay_index = get_date_index(year,dat,10)

        mask1 = make_start_date_mask(day_index,day_interpolated_surface)
        if eDay_index < 0:
            endMask = np.ones(end_interpolated_surface.shape) #in the case that the index is before Oct 1
        else: 
            endMask = make_end_date_mask(eDay_index,end_interpolated_surface)

        hourly = str(dat)[0:10]+' 13:00'
        
        rainfall = GD.get_pcp(str(dat)[0:10],file_path_daily,date_dictionary)
        wind = GD.get_wind_speed(hourly,file_path_hourly) #Using the list, get the data for wind speed for those stations on the input date
        temp = GD.get_noon_temp(hourly,file_path_hourly) #Using the list, get the data for temperature for those stations on the input date
        rh =GD.get_relative_humidity(hourly,file_path_hourly) #Using the list, get the data for rh% for those stations on the input date

        #what type of interpolation are we using here?

        if interpolation_method == 'IDW-1': 

        
            rain_grid, maxmin = idw.IDW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,1)
            temp_grid, maxmin = idw.IDW(latlon_dict,temp,hourly,var_name,shapefile,False,1)
            rh_grid, maxmin = idw.IDW(latlon_dict,rh,hourly,var_name,shapefile,False,1)
            wind_grid, maxmin = idw.IDW(latlon_dict,wind,hourly,var_name,shapefile,False,1)

        if interpolation_method == 'IDW-2': 

        
            rain_grid, maxmin = idw.IDW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,2)
            temp_grid, maxmin = idw.IDW(latlon_dict,temp,hourly,var_name,shapefile,False,2)
            rh_grid, maxmin = idw.IDW(latlon_dict,rh,hourly,var_name,shapefile,False,2)
            wind_grid, maxmin = idw.IDW(latlon_dict,wind,hourly,var_name,shapefile,False,2)

        if interpolation_method == 'IDEW-1':

            rain_grid, maxmin, elev_array= idew.IDEW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,file_path_elev,idx_list,1)
            temp_grid, maxmin, elev_array= idew.IDEW(latlon_dict,temp,hourly,var_name,shapefile,False,file_path_elev,idx_list,1)
            rh_grid, maxmin, elev_array = idew.IDEW(latlon_dict,rh,hourly,var_name,shapefile,False,file_path_elev,idx_list,1)
            wind_grid, maxmin, elev_array= idew.IDEW(latlon_dict,wind,hourly,var_name,shapefile,False,file_path_elev,idx_list,1)
            
        if interpolation_method == 'IDEW-2':

            rain_grid, maxmin, elev_array = idew.IDEW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,file_path_elev,idx_list,2)
            temp_grid, maxmin, elev_array = idew.IDEW(latlon_dict,temp,hourly,var_name,shapefile,False,file_path_elev,idx_list,2)
            rh_grid, maxmin, elev_array = idew.IDEW(latlon_dict,rh,hourly,var_name,shapefile,False,file_path_elev,idx_list,2)
            wind_grid, maxmin, elev_array = idew.IDEW(latlon_dict,wind,hourly,var_name,shapefile,False,file_path_elev,idx_list,2)

        if interpolation_method == 'TPS':

            rain_grid, maxmin = tps.TPS(latlon_dictionary,rainfall,dat,var_name,shapefile,False,0)
            temp_grid, maxmin = tps.TPS(latlon_dict,temp,hourly,var_name,shapefile,False,0)
            rh_grid, maxmin = tps.TPS(latlon_dict,rh,hourly,var_name,shapefile,False,0)
            wind_grid, maxmin = tps.TPS(latlon_dict,wind,hourly,var_name,shapefile,False,0)

        if interpolation_method == 'TPSS':

            num_stations_R = len(rainfall.keys())
            num_stations_t = len(temp.keys())
            num_stations_rh = len(rh.keys())
            num_stations_w = len(wind.keys())
            
            smoothing_parameterR = int(num_stations_R)-(math.sqrt(2*num_stations_R))
            smoothing_parameterT = int(num_stations_t)-(math.sqrt(2*num_stations_t))
            smoothing_parameterRH = int(num_stations_rh)-(math.sqrt(2*num_stations_rh))
            smoothing_parameterW = int(num_stations_w)-(math.sqrt(2*num_stations_w))
            
            rain_grid, maxmin = tps.TPS(latlon_dictionary,rainfall,dat,var_name,shapefile,False,smoothing_parameterR)
            temp_grid, maxmin = tps.TPS(latlon_dict,temp,hourly,var_name,shapefile,False,smoothing_parameterT)
            rh_grid, maxmin = tps.TPS(latlon_dict,rh,hourly,var_name,shapefile,False,smoothing_parameterRH)
            wind_grid, maxmin = tps.TPS(latlon_dict,wind,hourly,var_name,shapefile,False,smoothing_parameterW)


        if interpolation_method == 'OK':

            models = ['exponential','gaussian','linear','spherical','power'] #The types of models we will test
            model_rain = ok.get_best_model(models,latlon_dictionary,rainfall,shapefile,1,10) #run the procedure once, leaving 10 stations out for crossval
            model_temp = ok.get_best_model(models,latlon_dict,temp,shapefile,1,10)
            model_rh = ok.get_best_model(models,latlon_dict,rh,shapefile,1,10)
            model_wind = ok.get_best_model(models,latlon_dict,wind,shapefile,1,10)
            try: 
                rain_grid, maxmin = ok.OKriging(latlon_dictionary,rainfall,dat,var_name,shapefile,model_rain,False)
            except:
                try: 
                    model_rain = 'linear'
                    rain_grid, maxmin = ok.OKriging(latlon_dictionary,rainfall,dat,var_name,shapefile,model_rain,False)
                except:
                    try: 
                        model_rain = 'exponential'
                        rain_grid, maxmin = ok.OKriging(latlon_dictionary,rainfall,dat,var_name,shapefile,model_rain,False)
                    except:
                        rain_grid_template, maxmin = idw.IDW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,1)
                        rain_grid = np.zeros(rain_grid_template.shape)
            try:
                temp_grid, maxmin = ok.OKriging(latlon_dict,temp,hourly,var_name,shapefile,model_temp,False)
            except:
                try: 
                    model_temp = 'linear'
                    temp_grid, maxmin = ok.OKriging(latlon_dict,temp,hourly,var_name,shapefile,model_temp,False)
                except: 
                    try: 
                        model_temp = 'exponential'
                        temp_grid, maxmin = ok.OKriging(latlon_dict,temp,hourly,var_name,shapefile,model_temp,False)
                    except:
                        rain_grid_template, maxmin = idw.IDW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,1)
                        temp_grid = np.zeros(rain_grid_template.shape)
            try: 
                rh_grid, maxmin = ok.OKriging(latlon_dict,rh,hourly,var_name,shapefile,model_rh,False)
            except:
                try: 
                    model_rh = 'linear'
                    rh_grid, maxmin = ok.OKriging(latlon_dict,rh,hourly,var_name,shapefile,model_rh,False)
                except: 
                    try: 
                        model_rh = 'exponential'
                        rh_grid, maxmin = ok.OKriging(latlon_dict,temp,hourly,var_name,shapefile,model_rh,False)
                    except:
                        rain_grid_template, maxmin = idw.IDW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,1)
                        rh_grid = np.zeros(rain_grid_template.shape)
                
            try: 
                wind_grid, maxmin = ok.OKriging(latlon_dict,wind,hourly,var_name,shapefile,model_wind,False)
            except:
                try: 
                    model_wind = 'linear'
                    wind_grid, maxmin = ok.OKriging(latlon_dict,wind,hourly,var_name,shapefile,model_wind,False)
                except: 
                    try: 
                        model_wind = 'exponential'
                        wind_grid, maxmin = ok.OKriging(latlon_dict,temp,hourly,var_name,shapefile,model_wind,False)
                    except:
                        rain_grid_template, maxmin = idw.IDW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,1)
                        wind_grid = np.zeros(rain_grid_template.shape) #If the procedure fails, just generate 0s 
            
        if interpolation_method == 'RF':

            rain_grid, maxmin, elev_array = rf.random_forest_interpolator(latlon_dictionary,rainfall,dat,var_name,shapefile,False,file_path_elev,idx_list)
            temp_grid, maxmin, elev_array = rf.random_forest_interpolator(latlon_dict,temp,hourly,var_name,shapefile,False,file_path_elev,idx_list)
            rh_grid, maxmin, elev_array = rf.random_forest_interpolator(latlon_dict,rh,hourly,var_name,shapefile,False,file_path_elev,idx_list)
            wind_grid, maxmin, elev_array = rf.random_forest_interpolator(latlon_dict,wind,hourly,var_name,shapefile,False,file_path_elev,idx_list)

        if (interpolation_method == 'RF' or interpolation_method == 'OK' or interpolation_method == 'TPSS' or interpolation_method == 'TPS' or interpolation_method == 'IDEW-2'\
           or interpolation_method == 'IDEW-1' or interpolation_method == 'IDW-2' or interpolation_method == 'IDW-1') != True:

            print('The entered interpolation method is not recognized')
            sys.exit() 
        
        if count > 0:  
            dc_array = dc_list[count-1] #the last one added will be yesterday's val, but there's a lag bc none was added when count was0, so just use count-1
            index = count-1
            dc = DC(dat,rain_grid,rh_grid,temp_grid,wind_grid,maxmin,dc_array,index,False,shapefile,mask1,endMask,last_DC_val_before_shutdown,overwinter)
            dc_list.append(dc)
        else: #Initialize procedure
            if overwinter:
                rain_shape = rain_grid.shape
                dc_initialize = np.zeros(rain_shape)
                dc_initialize[np.isnan(last_DC_val_before_shutdown)] = 15
                dc_initialize[~np.isnan(last_DC_val_before_shutdown)] = last_DC_val_before_shutdown[~np.isnan(last_DC_val_before_shutdown)]
                dc_list.append(dc_initialize*mask1)
            else: 
                rain_shape = rain_grid.shape
                dc_initialize = np.zeros(rain_shape)+15 #merge with the other overwinter array once it's calculated 
                dc_yesterday1 = dc_initialize*mask1
                dc_list.append(dc_yesterday1) #placeholder 
        count += 1

    if json:
        dc_list = [i.tolist() for i in dc_list]
        for arrayDC in dc_list: #Check for corruption before writing to json
            is_it_corrupted = all(i >= 2000 for i in arrayDC) #the DC value should never be that high 
            if is_it_corrupted:
                print('There is a problem with the calculations. The DC value is too large.') 

    return dc_list

def get_last_dc_before_shutdown(dc_list,endsurface,overwinter_reqd,show,maxmin):
    '''Get an array of the last dc vals before station shutdown for winter for the study area 
    Parameters
        dc_list (list of np_array): a list of the interpolated surfaces for the drought code for 
        each day in the fire season 
        endsurface (np_array): array for end dates for the year before the year of interest 
        overwinter_reqd (np_array): where the overwinter procedure is required 
        show (bool): whether you want to plot the map 
        maxmin (list): bounds of the study area 
    Returns 
        last_DC_val_before_shutdown_masked_reshape (np_array): array of dc values before shutdown for 
        cells requiring overwinter procedure 
    '''

    flatten_dc = list(map(lambda x:x.flatten(),dc_list)) #flatten the arrays for easier processing - avoid 3d array
    stackDC = np.stack(flatten_dc,axis=-1) #Create an array from the list that we can index

    days_since_mar1 = endsurface.flatten().astype(int)+214-1 #add 214 days (mar-aug31 to convert to days since March 1) to get index in the stack... based on make_end_date_mask() -1 for day before

    last_DC_val_before_shutdown = stackDC[np.arange(len(stackDC)), days_since_mar1] #Index each cell in the array by the end date to get the last val
    last_DC_val_before_shutdown_masked = last_DC_val_before_shutdown * overwinter_reqd.flatten() #Mask the areas that don't require the overwinter procedure
    last_DC_val_before_shutdown_masked_reshape = last_DC_val_before_shutdown_masked.reshape(endsurface.shape)

    if show: 
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        fig, ax = plt.subplots(figsize= (15,15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)
        
      
        plt.imshow(last_DC_val_before_shutdown_masked_reshape,extent=(min_xProj_extent-1,max_xProj_extent+1,max_yProj_extent-1,min_yProj_extent+1)) 
        na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)
            
        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label('DC') 
        
        title = 'Last DC'
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        plt.show()
        
    return last_DC_val_before_shutdown_masked_reshape

def dc_overwinter_procedure(last_DC_val_before_shutdown_masked_reshape,overwinter_pcp,b_list):
    '''Apply the overwinter procedure, see Wang et al. (2017) for more details 
    Parameters
        last_DC_val_before_shutdown_masked_reshape (np_array): output of get_last_dc_before_shutdown, array containing the last
        dc value before the station shut down interpolated across the study area 
        overwinter_pcp (np_array): overwinter precipitation amount in each cell 
        b_list (list, loaded from json): list containing information for the b parameter, can be formatted into an array 
    Returns 
        DCs (np_array): the new values for the start-up DC procedure, to be used in areas identified as needing the overwinter
        procedure 
    '''
    a = 1.0
    b_array = np.array(b_list).reshape(overwinter_pcp.shape)
    Qf = 800 * np.exp(-last_DC_val_before_shutdown_masked_reshape / 400)
    Qs = a * Qf + b_array * (3.94 * overwinter_pcp)
    DCs = 400 * np.log(800 / Qs) #this is natural logarithm 

    DCs[DCs < 15] = 15
    return DCs
      
def dmc_stack(dates,file_path_daily,file_path_hourly,var_name,shapefile,day_interpolated_surface,
end_interpolated_surface,file_path_elev,idx_list,date_dictionary,latlon_dict,latlon_dictionary,json,interpolation_method):
    '''Calc dmc for each day in fire season. For notes see cffdrs R code.
    Parameters
        dates (list): list of all dates within the fire season, inactive stations will be masked out 
        so you can define it as Mar 1 - Dec 31
        file_path_daily (str): file path to the daily feather files
        file_path_hourly (str): file path to the hourly feather files
        var_name (str): name of the variable you are interpolating
        shapefile (str): path to the study area shapefile 
        day_interpolated_surface (np_array): array of start-up days (since Mar 1) for the study area
        end_interpolated_surface (np_array): array of end days (since Oct 1) for the study area
        file_path_elev (str): file path to the elevation lookup file 
        idx_list (list): the index of the elevation data column in the lookup file 
        date_dictionary (dict, loaded from .json): lookup file that has what day/month pairs each 
        station contains data for 
        latlon_dict (dict, loaded from .json): lat lons of the hourly stations
        latlon_dictionary (dict, loaded from .json): lat lons of the daily stations
        json (bool): if True, convert the array to a flat list so it can be written as a .json file
        interpolation_method (str): the interpolation method to use to get the continuous DMC surface, 
        there are eight options - IDW-1, IDW-2, IDEW-1, IDEW-2, TPS, TPSS, OK, RF
    Returns 
        dmc_list (list of np_array): a list of the interpolated surfaces for the duff moisture code for 
        each day in the fire season 
    '''
    dmc_list = [] 
    count = 0 
    for dat in dates:
        index = dates.index(dat) #need to run BEFORE we convert to string 
        gc.collect() 
        year = str(dat)[0:4]
        dat = str(dat)
        day_index= get_date_index(year,dat,3)
        eDay_index = get_date_index(year,dat,10)

        mask = make_start_date_mask(day_index,day_interpolated_surface)
        if eDay_index < 0:
            endMask = np.ones(end_interpolated_surface.shape) #in the case that the index is before Oct 1
        else: 
            endMask = make_end_date_mask(eDay_index,end_interpolated_surface)
    
        hourly = str(dat)[0:10]+' 13:00'
        
        rainfall = GD.get_pcp(str(dat)[0:10],file_path_daily,date_dictionary)
        wind = GD.get_wind_speed(hourly,file_path_hourly) #Using the list, get the data for wind speed for those stations on the input date
        temp = GD.get_noon_temp(hourly,file_path_hourly) #Using the list, get the data for temperature for those stations on the input date
        rh = GD.get_relative_humidity(hourly,file_path_hourly) #Using the list, get the data for rh% for those stations on the input date
        
        #what type of interpolation are we using here?

        if interpolation_method == 'IDW-1': 

        
            rain_grid, maxmin = idw.IDW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,1)
            temp_grid, maxmin = idw.IDW(latlon_dict,temp,hourly,var_name,shapefile,False,1)
            rh_grid, maxmin = idw.IDW(latlon_dict,rh,hourly,var_name,shapefile,False,1)
            wind_grid, maxmin = idw.IDW(latlon_dict,wind,hourly,var_name,shapefile,False,1)

        if interpolation_method == 'IDW-2': 

        
            rain_grid, maxmin = idw.IDW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,2)
            temp_grid, maxmin = idw.IDW(latlon_dict,temp,hourly,var_name,shapefile,False,2)
            rh_grid, maxmin = idw.IDW(latlon_dict,rh,hourly,var_name,shapefile,False,2)
            wind_grid, maxmin = idw.IDW(latlon_dict,wind,hourly,var_name,shapefile,False,2)

        if interpolation_method == 'IDEW-1':

            rain_grid, maxmin, elev_array  = idew.IDEW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,file_path_elev,idx_list,1)
            temp_grid, maxmin, elev_array  = idew.IDEW(latlon_dict,temp,hourly,var_name,shapefile,False,file_path_elev,idx_list,1)
            rh_grid, maxmin, elev_array  = idew.IDEW(latlon_dict,rh,hourly,var_name,shapefile,False,file_path_elev,idx_list,1)
            wind_grid, maxmin, elev_array  = idew.IDEW(latlon_dict,wind,hourly,var_name,shapefile,False,file_path_elev,idx_list,1)
            
        if interpolation_method == 'IDEW-2':

            rain_grid, maxmin, elev_array  = idew.IDEW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,file_path_elev,idx_list,2)
            temp_grid, maxmin, elev_array  = idew.IDEW(latlon_dict,temp,hourly,var_name,shapefile,False,file_path_elev,idx_list,2)
            rh_grid, maxmin, elev_array  = idew.IDEW(latlon_dict,rh,hourly,var_name,shapefile,False,file_path_elev,idx_list,2)
            wind_grid, maxmin, elev_array  = idew.IDEW(latlon_dict,wind,hourly,var_name,shapefile,False,file_path_elev,idx_list,2)

        if interpolation_method == 'TPS':

            rain_grid, maxmin = tps.TPS(latlon_dictionary,rainfall,dat,var_name,shapefile,False,0)
            temp_grid, maxmin = tps.TPS(latlon_dict,temp,hourly,var_name,shapefile,False,0)
            rh_grid, maxmin = tps.TPS(latlon_dict,rh,hourly,var_name,shapefile,False,0)
            wind_grid, maxmin = tps.TPS(latlon_dict,wind,hourly,var_name,shapefile,False,0)

        if interpolation_method == 'TPSS':

            num_stations_R = len(rainfall.keys())
            num_stations_t = len(temp.keys())
            num_stations_rh = len(rh.keys())
            num_stations_w = len(wind.keys())
            
            smoothing_parameterR = int(num_stations_R)-(math.sqrt(2*num_stations_R))
            smoothing_parameterT = int(num_stations_t)-(math.sqrt(2*num_stations_t))
            smoothing_parameterRH = int(num_stations_rh)-(math.sqrt(2*num_stations_rh))
            smoothing_parameterW = int(num_stations_w)-(math.sqrt(2*num_stations_w))
            
            rain_grid, maxmin = tps.TPS(latlon_dictionary,rainfall,dat,var_name,shapefile,False,smoothing_parameterR)
            temp_grid, maxmin = tps.TPS(latlon_dict,temp,hourly,var_name,shapefile,False,smoothing_parameterT)
            rh_grid, maxmin = tps.TPS(latlon_dict,rh,hourly,var_name,shapefile,False,smoothing_parameterRH)
            wind_grid, maxmin = tps.TPS(latlon_dict,wind,hourly,var_name,shapefile,False,smoothing_parameterW)


        if interpolation_method == 'OK':

            models = ['exponential','gaussian','linear','spherical','power'] #The types of models we will test
            model_rain = ok.get_best_model(models,latlon_dictionary,rainfall,shapefile,1,10) #run the procedure once, leaving 10 stations out for crossval
            model_temp = ok.get_best_model(models,latlon_dict,temp,shapefile,1,10)
            model_rh = ok.get_best_model(models,latlon_dict,rh,shapefile,1,10)
            model_wind = ok.get_best_model(models,latlon_dict,wind,shapefile,1,10)
            try: 
                rain_grid, maxmin = ok.OKriging(latlon_dictionary,rainfall,dat,var_name,shapefile,model_rain,False)
            except:
                try: 
                    model_rain = 'linear'
                    rain_grid, maxmin = ok.OKriging(latlon_dictionary,rainfall,dat,var_name,shapefile,model_rain,False)
                except:
                    try: 
                        model_rain = 'exponential'
                        rain_grid, maxmin = ok.OKriging(latlon_dictionary,rainfall,dat,var_name,shapefile,model_rain,False)
                    except:
                        rain_grid_template, maxmin = idw.IDW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,1)
                        rain_grid = np.zeros(rain_grid_template.shape)
            try:
                temp_grid, maxmin = ok.OKriging(latlon_dict,temp,hourly,var_name,shapefile,model_temp,False)
            except:
                try: 
                    model_temp = 'linear'
                    temp_grid, maxmin = ok.OKriging(latlon_dict,temp,hourly,var_name,shapefile,model_temp,False)
                except: 
                    try: 
                        model_temp = 'exponential'
                        temp_grid, maxmin = ok.OKriging(latlon_dict,temp,hourly,var_name,shapefile,model_temp,False)
                    except:
                        rain_grid_template, maxmin = idw.IDW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,1)
                        temp_grid = np.zeros(rain_grid_template.shape)
            try: 
                rh_grid, maxmin = ok.OKriging(latlon_dict,rh,hourly,var_name,shapefile,model_rh,False)
            except:
                try: 
                    model_rh = 'linear'
                    rh_grid, maxmin = ok.OKriging(latlon_dict,rh,hourly,var_name,shapefile,model_rh,False)
                except: 
                    try: 
                        model_rh = 'exponential'
                        rh_grid, maxmin = ok.OKriging(latlon_dict,temp,hourly,var_name,shapefile,model_rh,False)
                    except:
                        rain_grid_template, maxmin = idw.IDW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,1)
                        rh_grid = np.zeros(rain_grid_template.shape)
                
            try: 
                wind_grid, maxmin = ok.OKriging(latlon_dict,wind,hourly,var_name,shapefile,model_wind,False)
            except:
                try: 
                    model_wind = 'linear'
                    wind_grid, maxmin = ok.OKriging(latlon_dict,wind,hourly,var_name,shapefile,model_wind,False)
                except: 
                    try: 
                        model_wind = 'exponential'
                        wind_grid, maxmin = ok.OKriging(latlon_dict,temp,hourly,var_name,shapefile,model_wind,False)
                    except:
                        rain_grid_template, maxmin = idw.IDW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,1)
                        wind_grid = np.zeros(rain_grid_template.shape) #If the procedure fails, just generate 0s 
            
        if interpolation_method == 'RF':

            rain_grid, maxmin, elev_array = rf.random_forest_interpolator(latlon_dictionary,rainfall,dat,var_name,shapefile,False,file_path_elev,idx_list)
            temp_grid, maxmin, elev_array = rf.random_forest_interpolator(latlon_dict,temp,hourly,var_name,shapefile,False,file_path_elev,idx_list)
            rh_grid, maxmin, elev_array = rf.random_forest_interpolator(latlon_dict,rh,hourly,var_name,shapefile,False,file_path_elev,idx_list)
            wind_grid, maxmin, elev_array = rf.random_forest_interpolator(latlon_dict,wind,hourly,var_name,shapefile,False,file_path_elev,idx_list)

        if (interpolation_method == 'RF' or interpolation_method == 'OK' or interpolation_method == 'TPSS' or interpolation_method == 'TPS' or interpolation_method == 'IDEW-2'\
           or interpolation_method == 'IDEW-1' or interpolation_method == 'IDW-2' or interpolation_method == 'IDW-1') != True:

            print('The entered interpolation method is not recognized')
            sys.exit()
            
        if count > 0:  
            dmc_array = dmc_list[count-1] #the last one added will be yesterday's val, but there's a lag bc none was added when count was0, so just use count-1
            index = count-1
            dmc = DMC(dat,rain_grid,rh_grid,temp_grid,wind_grid,maxmin,dmc_array,index,False,shapefile,mask,endMask)
            dmc_list.append(dmc)
        else:
            rain_shape = rain_grid.shape
            dmc_initialize = np.zeros(rain_shape)+6
            dmc_yesterday1 = dmc_initialize*mask
            dmc_list.append(dmc_yesterday1) #placeholder 
        count += 1

    if json:
        dmc_list = [i.tolist() for i in dmc_list]

    return dmc_list

def ffmc_stack(dates,file_path_daily,file_path_hourly,var_name,shapefile,day_interpolated_surface,
               end_interpolated_surface,file_path_elev,idx_list,date_dictionary,latlon_dict,latlon_dictionary,
               json,interpolation_method):
    '''Calc ffmc for each day in season. For notes see cffdrs R code.
    Parameters
        dates (list): list of all dates within the fire season, inactive stations will be masked out 
        so you can define it as Mar 1 - Dec 31
        file_path_daily (str): file path to the daily feather files
        file_path_hourly (str): file path to the hourly feather files
        var_name (str): name of the variable you are interpolating
        shapefile (str): path to the study area shapefile 
        day_interpolated_surface (np_array): array of start-up days (since Mar 1) for the study area
        end_interpolated_surface (np_array): array of end days (since Oct 1) for the study area
        file_path_elev (str): file path to the elevation lookup file 
        idx_list (list): the index of the elevation data column in the lookup file 
        date_dictionary (dict, loaded from .json): lookup file that has what day/month pairs each 
        station contains data for 
        latlon_dict (dict, loaded from .json): lat lons of the hourly stations
        latlon_dictionary (dict, loaded from .json): lat lons of the daily stations
        json (bool): if True, convert the array to a flat list so it can be written as a .json file
        interpolation_method (str): the interpolation method to use to get the continuous DMC surface, 
        there are eight options - IDW-1, IDW-2, IDEW-1, IDEW-2, TPS, TPSS, OK, RF
    Returns 
        ffmc_list (list of np_array): a list of the interpolated surfaces for the fine fuel moisture code for 
        each day in the fire season 
    '''
    ffmc_list = [] 
    count = 0 
    for dat in dates:
        index = dates.index(dat) #need to run BEFORE we convert to string 
        gc.collect() 
        year = str(dat)[0:4]
        dat = str(dat) #convert to str so that cython doesn't get confused 
        day_index= get_date_index(year,dat,3)
        eDay_index = get_date_index(year,dat,10)

        mask = make_start_date_mask(day_index,day_interpolated_surface)
        if eDay_index < 0:
            endMask = np.ones(end_interpolated_surface.shape) #in the case that the index is before Oct 1
        else: 
            endMask = make_end_date_mask(eDay_index,end_interpolated_surface)

        hourly = str(dat)[0:10]+' 13:00'
        rainfall = GD.get_pcp(str(dat)[0:10],file_path_daily,date_dictionary)
        wind = GD.get_wind_speed(hourly,file_path_hourly) #Using the list, get the data for wind speed for those stations on the input date
        temp = GD.get_noon_temp(hourly,file_path_hourly) #Using the list, get the data for temperature for those stations on the input date
        rh = GD.get_relative_humidity(hourly,file_path_hourly) #Using the list, get the data for rh% for those stations on the input date
        
        #what type of interpolation are we using here?

        if interpolation_method == 'IDW-1': 

        
            rain_grid, maxmin = idw.IDW(latlon_dictionary,rainfall,str(dat),var_name,shapefile,False,1)
            temp_grid, maxmin = idw.IDW(latlon_dict,temp,hourly,var_name,shapefile,False,1)
            rh_grid, maxmin = idw.IDW(latlon_dict,rh,hourly,var_name,shapefile,False,1)
            wind_grid, maxmin = idw.IDW(latlon_dict,wind,hourly,var_name,shapefile,False,1)

        if interpolation_method == 'IDW-2': 

        
            rain_grid, maxmin = idw.IDW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,2)
            temp_grid, maxmin = idw.IDW(latlon_dict,temp,hourly,var_name,shapefile,False,2)
            rh_grid, maxmin = idw.IDW(latlon_dict,rh,hourly,var_name,shapefile,False,2)
            wind_grid, maxmin = idw.IDW(latlon_dict,wind,hourly,var_name,shapefile,False,2)

        if interpolation_method == 'IDEW-1':

            rain_grid, maxmin, elev_array = idew.IDEW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,file_path_elev,idx_list,1)
            temp_grid, maxmin, elev_array = idew.IDEW(latlon_dict,temp,hourly,var_name,shapefile,False,file_path_elev,idx_list,1)
            rh_grid, maxmin, elev_array = idew.IDEW(latlon_dict,rh,hourly,var_name,shapefile,False,file_path_elev,idx_list,1)
            wind_grid, maxmin, elev_array = idew.IDEW(latlon_dict,wind,hourly,var_name,shapefile,False,file_path_elev,idx_list,1)
            
        if interpolation_method == 'IDEW-2':

            rain_grid, maxmin, elev_array = idew.IDEW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,file_path_elev,idx_list,2)
            temp_grid, maxmin, elev_array = idew.IDEW(latlon_dict,temp,hourly,var_name,shapefile,False,file_path_elev,idx_list,2)
            rh_grid, maxmin, elev_array = idew.IDEW(latlon_dict,rh,hourly,var_name,shapefile,False,file_path_elev,idx_list,2)
            wind_grid, maxmin, elev_array = idew.IDEW(latlon_dict,wind,hourly,var_name,shapefile,False,file_path_elev,idx_list,2)

        if interpolation_method == 'TPS':

            rain_grid, maxmin = tps.TPS(latlon_dictionary,rainfall,dat,var_name,shapefile,False,0)
            temp_grid, maxmin = tps.TPS(latlon_dict,temp,hourly,var_name,shapefile,False,0)
            rh_grid, maxmin = tps.TPS(latlon_dict,rh,hourly,var_name,shapefile,False,0)
            wind_grid, maxmin = tps.TPS(latlon_dict,wind,hourly,var_name,shapefile,False,0)

        if interpolation_method == 'TPSS':

            num_stations_R = len(rainfall.keys())
            num_stations_t = len(temp.keys())
            num_stations_rh = len(rh.keys())
            num_stations_w = len(wind.keys())
            
            smoothing_parameterR = int(num_stations_R)-(math.sqrt(2*num_stations_R))
            smoothing_parameterT = int(num_stations_t)-(math.sqrt(2*num_stations_t))
            smoothing_parameterRH = int(num_stations_rh)-(math.sqrt(2*num_stations_rh))
            smoothing_parameterW = int(num_stations_w)-(math.sqrt(2*num_stations_w))
            
            rain_grid, maxmin = tps.TPS(latlon_dictionary,rainfall,dat,var_name,shapefile,False,smoothing_parameterR)
            temp_grid, maxmin = tps.TPS(latlon_dict,temp,hourly,var_name,shapefile,False,smoothing_parameterT)
            rh_grid, maxmin = tps.TPS(latlon_dict,rh,hourly,var_name,shapefile,False,smoothing_parameterRH)
            wind_grid, maxmin = tps.TPS(latlon_dict,wind,hourly,var_name,shapefile,False,smoothing_parameterW)

        if interpolation_method == 'OK':

            models = ['exponential','gaussian','linear','spherical','power'] #The types of models we will test
            model_rain = ok.get_best_model(models,latlon_dictionary,rainfall,shapefile,1,10) #run the procedure once, leaving 10 stations out for crossval
            model_temp = ok.get_best_model(models,latlon_dict,temp,shapefile,1,10)
            model_rh = ok.get_best_model(models,latlon_dict,rh,shapefile,1,10)
            model_wind = ok.get_best_model(models,latlon_dict,wind,shapefile,1,10)
            try: 
                rain_grid, maxmin = ok.OKriging(latlon_dictionary,rainfall,dat,var_name,shapefile,model_rain,False)
            except:
                try: 
                    model_rain = 'linear'
                    rain_grid, maxmin = ok.OKriging(latlon_dictionary,rainfall,dat,var_name,shapefile,model_rain,False)
                except:
                    try: 
                        model_rain = 'exponential'
                        rain_grid, maxmin = ok.OKriging(latlon_dictionary,rainfall,dat,var_name,shapefile,model_rain,False)
                    except:
                        rain_grid_template, maxmin = idw.IDW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,1)
                        rain_grid = np.zeros(rain_grid_template.shape)
            try:
                temp_grid, maxmin = ok.OKriging(latlon_dict,temp,hourly,var_name,shapefile,model_temp,False)
            except:
                try: 
                    model_temp = 'linear'
                    temp_grid, maxmin = ok.OKriging(latlon_dict,temp,hourly,var_name,shapefile,model_temp,False)
                except: 
                    try: 
                        model_temp = 'exponential'
                        temp_grid, maxmin = ok.OKriging(latlon_dict,temp,hourly,var_name,shapefile,model_temp,False)
                    except:
                        rain_grid_template, maxmin = idw.IDW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,1)
                        temp_grid = np.zeros(rain_grid_template.shape)
            try: 
                rh_grid, maxmin = ok.OKriging(latlon_dict,rh,hourly,var_name,shapefile,model_rh,False)
            except:
                try: 
                    model_rh = 'linear'
                    rh_grid, maxmin = ok.OKriging(latlon_dict,rh,hourly,var_name,shapefile,model_rh,False)
                except: 
                    try: 
                        model_rh = 'exponential'
                        rh_grid, maxmin = ok.OKriging(latlon_dict,temp,hourly,var_name,shapefile,model_rh,False)
                    except:
                        rain_grid_template, maxmin = idw.IDW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,1)
                        rh_grid = np.zeros(rain_grid_template.shape)
                
            try: 
                wind_grid, maxmin = ok.OKriging(latlon_dict,wind,hourly,var_name,shapefile,model_wind,False)
            except:
                try: 
                    model_wind = 'linear'
                    wind_grid, maxmin = ok.OKriging(latlon_dict,wind,hourly,var_name,shapefile,model_wind,False)
                except: 
                    try: 
                        model_wind = 'exponential'
                        wind_grid, maxmin = ok.OKriging(latlon_dict,temp,hourly,var_name,shapefile,model_wind,False)
                    except:
                        rain_grid_template, maxmin = idw.IDW(latlon_dictionary,rainfall,dat,var_name,shapefile,False,1)
                        wind_grid = np.zeros(rain_grid_template.shape) #If the procedure fails, just generate 0s 
            
        if interpolation_method == 'RF':

            rain_grid, maxmin, elev_array = rf.random_forest_interpolator(latlon_dictionary,rainfall,dat,var_name,shapefile,False,file_path_elev,idx_list)
            temp_grid, maxmin, elev_array = rf.random_forest_interpolator(latlon_dict,temp,hourly,var_name,shapefile,False,file_path_elev,idx_list)
            rh_grid, maxmin, elev_array = rf.random_forest_interpolator(latlon_dict,rh,hourly,var_name,shapefile,False,file_path_elev,idx_list)
            wind_grid, maxmin, elev_array = rf.random_forest_interpolator(latlon_dict,wind,hourly,var_name,shapefile,False,file_path_elev,idx_list)

        if (interpolation_method == 'RF' or interpolation_method == 'OK' or interpolation_method == 'TPSS' or interpolation_method == 'TPS' or interpolation_method == 'IDEW-2'\
           or interpolation_method == 'IDEW-1' or interpolation_method == 'IDW-2' or interpolation_method == 'IDW-1') != True:

            print('The entered interpolation method is not recognized')
            sys.exit()
            
        if count > 0:  
            ffmc_array = ffmc_list[count-1] #the last one added will be yesterday's val, but there's a lag bc none was added when count was0, so just use count-1
            index = count-1
            ffmc = FFMC(dat,rain_grid,rh_grid,temp_grid,wind_grid,maxmin,ffmc_array,index,False,shapefile,mask,endMask)
            ffmc_list.append(ffmc)
        else:
            rain_shape = rain_grid.shape
            ffmc_initialize = np.zeros(rain_shape)+85
            ffmc_yesterday1 = ffmc_initialize*mask
            ffmc_list.append(ffmc_yesterday1) #placeholder
                
                
        count += 1

    if json:
        ffmc_list = [i.tolist() for i in ffmc_list]

    return ffmc_list

def DC(input_date,rain_grid,rh_grid,temp_grid,wind_grid,maxmin,dc_yesterday,index,show,shapefile,mask,endMask,
       last_DC_val_before_shutdown,overwinter):
    '''Calculate the DC. See cffdrs R code
    Parameters
        input_date (str): input date of interest
        rain_grid: interpolated surface for rainfall on the date of interest
        temp_grid: interpolated surface for temperature on the date of interest
        wind_grid: interpolated surface for wind on the date of interest
        maxmin: bounds of the study area 
        dc_yesterday: array of DC values for yesterday (from the dc stack list/if this function
        is being used inside dc_stack it is calculated then) 
        index (int): index of the date since Mar 1
        show (bool): whether you want to show the map 
        shapefile (str): path to the study area shapefile
        mask (np_array): mask for the start dates 
        endMask (np_array): mask for the end days 
        last_DC_val_before_shutdown (np_array): array for last dc values before cell shut down, if no areas 
        required the procedure, you can input an empty array of the correct size (if not using overwinter, input
        the empty array)
        overwinter (bool): whether or not to implement the overwinter procedure 
    Returns 
        dc1 (np_array): array of dc values on the date on interest for the study area
    '''
    
    yesterday_index = index-1
    if yesterday_index == -1:
        if overwinter:
            rain_shape = rain_grid.shape
            dc_initialize = np.zeros(rain_shape)
            dc_initialize[np.isnan(last_DC_val_before_shutdown)] = 15
            dc_initialize[~np.isnan(last_DC_val_before_shutdown)] = last_DC_val_before_shutdown[~np.isnan(last_DC_val_before_shutdown)]
            dc_yesterday1 = dc_initialize*mask 
        else: 
            rain_shape = rain_grid.shape
            dc_initialize = np.zeros(rain_shape)+15
            dc_yesterday1 = dc_initialize
            dc_yesterday1 = dc_yesterday1*mask #mask out areas that haven't started
    else:
        if overwinter:
            dc_yesterday1 = dc_yesterday
            dc_yesterday1[np.where(np.isnan(dc_yesterday1) & ~np.isnan(mask) & ~np.isnan(last_DC_val_before_shutdown))] = last_DC_val_before_shutdown[np.where(np.isnan(dc_yesterday1) & ~np.isnan(mask) & ~np.isnan(last_DC_val_before_shutdown))]
            dc_yesterday1[np.where(np.isnan(dc_yesterday1) & ~np.isnan(mask) & np.isnan(last_DC_val_before_shutdown))] = 15
        else: 
            dc_yesterday1 = dc_yesterday
            dc_yesterday1[np.where(np.isnan(dc_yesterday1) & ~np.isnan(mask))] = 15 #set started pixels since yesterday to 15

    input_date = str(input_date)
    month = int(input_date[6])
    #Get day length factor

    f101 = [-1.6,-1.6,-1.6,0.9,3.8,5.8,6.4,5,2.4,0.4,-1.6,-1.6]

    #Put constraint on low end of temp
    temp_grid[temp_grid < -2.8] = -2.8

    #E22 potential evapT

    pe = (0.36*(temp_grid+2.8)+f101[month])/2

    #Make empty dmc array
    new_shape = dc_yesterday1.shape
    dc = np.zeros(new_shape)

    #starting rain

    netRain = 0.83*rain_grid-1.27

    #eq 19
    smi = 800*np.exp(-1*dc_yesterday1/400)

    #eq 21
    dr0 = dc_yesterday1 -400*np.log(1+3.937*netRain/smi) #log is the natural logarithm 
    dr0[dr0<0] = 0
    dr0[rain_grid <= 2.8] = dc_yesterday1[rain_grid <= 2.8]

    dc1 = dr0 + pe
    dc1[dc1 < 0] = 0
        
    dc1 = dc1 * mask * endMask
    if show == True: 
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        fig, ax = plt.subplots(figsize= (15,15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)
        circ = PolygonPatch(na_map['geometry'][0],visible=False)
        ax.add_patch(circ) 
        plt.imshow(dc1,extent=(xProj_extent.min(),xProj_extent.max(),yProj_extent.max(),yProj_extent.min()),clip_path=circ, 
                   clip_on=True) 
      
        #plt.imshow(dc1,extent=(min_xProj_extent-1,max_xProj_extent+1,max_yProj_extent-1,min_yProj_extent+1)) 
        na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)
            
        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label('DC') 
        
        title = 'DC for %s'%(input_date) 
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        plt.show()

    return dc1

def DMC(input_date,rain_grid,rh_grid,temp_grid,wind_grid,maxmin,dmc_yesterday,index,show,shapefile,
        mask,endMask):
    '''Calculate the DMC. See cffdrs R code
    Parameters
        input_date (str): input date of interest
        rain_grid: interpolated surface for rainfall on the date of interest
        rh_grid: interpolated surface for relative humidity on the date of interest
        temp_grid: interpolated surface for temperature on the date of interest
        wind_grid: interpolated surface for wind on the date of interest
        maxmin: bounds of the study area 
        dmc_yesterday: array of DMC values for yesterday (from the dmc stack list/if this function
        is being used inside dmc_stack it is calculated then) 
        index (int): index of the date since Mar 1
        show (bool): whether you want to show the map 
        shapefile (str): path to the study area shapefile
        mask (np_array): mask for the start dates 
        endMask (np_array): mask for the end days 
    Returns 
        dmc (np_array): array of dmc values on the date on interest for the study area
    '''
    yesterday_index = index-1

    if yesterday_index == -1:
        rain_shape = rain_grid.shape
        dmc_initialize = np.zeros(rain_shape)+6
        dmc_yesterday1 = dmc_initialize*mask
    else: 
        dmc_yesterday1 = dmc_yesterday
        dmc_yesterday1[np.where(np.isnan(dmc_yesterday1) & ~np.isnan(mask))] = 6

    #dmc_yesterday = dmc_yesterday1.flatten()
    input_date = str(input_date)
    month = int(input_date[6])
    #Get day length factor

    ell01 = [6.5, 7.5, 9, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8, 7, 6]

    #Put constraint on low end of temp
    temp_grid[temp_grid < -1.1] = -1.1

    #Log drying rate
    rk = 1.84*(temp_grid+1.1)*(100-rh_grid)*ell01[month]*1.0E-4

    #Make empty dmc array
    new_shape = dmc_yesterday1.shape
    dmc = np.zeros(new_shape)

    #starting rain

    netRain = 0.92*rain_grid-1.27

    #initial moisture content, modified same as cffdrs package
    wmi = 20 + 280/np.exp(0.023*dmc_yesterday1)

    #if else depending on yesterday dmc, eq.13
    b = np.zeros(new_shape)


    b[dmc_yesterday1 <= 33] = 100/(0.5+0.3*dmc_yesterday1[dmc_yesterday1 <= 33])
    b[(dmc_yesterday1 > 33) & (dmc_yesterday1 < 65)] = 14-1.3*np.log(dmc_yesterday1[(dmc_yesterday1 > 33) & (dmc_yesterday1 < 65)]) # np.log is ln
    b[dmc_yesterday1 >= 65] = 6.5*np.log(dmc_yesterday1[dmc_yesterday1 >= 65])-17.2
        

    #eq 14, modified in R package
    wmr = wmi + 1000 * netRain/(48.77 + b * netRain)

    #eq 15 modified to be same as cffdrs package
    
    pr0 = 43.43 * (5.6348 - np.log(wmr-20)) #natural logarithm

    pr0[pr0 <0] = 0
    

    rk_pr0 =pr0 + rk
    rk_ydmc = dmc_yesterday1 + rk #we want to add rk because that's the drying rate 
    dmc[netRain > 1.5] = rk_pr0[netRain > 1.5]
    dmc[netRain <= 1.5] = rk_ydmc[netRain <= 1.5]



    dmc[dmc < 0]=0

    dmc = dmc * mask * endMask # mask out areas that haven't been activated



    if show == True: 
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        fig, ax = plt.subplots(figsize= (15,15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)
        
        circ = PolygonPatch(na_map['geometry'][0],visible=False)
        ax.add_patch(circ) 
        plt.imshow(dmc,extent=(min_xProj_extent-1,max_xProj_extent+1,max_yProj_extent-1,min_yProj_extent+1),clip_path=circ, 
                   clip_on=True) 
        
      
        #plt.imshow(dmc,extent=(min_xProj_extent-1,max_xProj_extent+1,max_yProj_extent-1,min_yProj_extent+1)) 
        na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)
            
        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label('DMC') 
        
        title = 'DMC for %s'%(input_date) 
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        plt.show()

    return dmc 


def FFMC(input_date,rain_grid,rh_grid,temp_grid,wind_grid,maxmin,ffmc_yesterday,index,show,shapefile,
         mask,endMask):
    '''Calculate the FFMC. See cffdrs R code
    Parameters
        input_date (str): input date of interest
        rain_grid: interpolated surface for rainfall on the date of interest
        rh_grid: interpolated surface for relative humidity on the date of interest
        temp_grid: interpolated surface for temperature on the date of interest
        wind_grid: interpolated surface for wind on the date of interest
        maxmin: bounds of the study area 
        ffmc_yesterday: array of FFMC values for yesterday (from the ffmc stack list/if this function
        is being used inside ffmc_stack it is calculated then) 
        index (int): index of the date since Mar 1
        show (bool): whether you want to show the map 
        shapefile (str): path to the study area shapefile
        mask (np_array): mask for the start dates 
        endMask (np_array): mask for the end days 
    Returns 
        ffmc1 (np_array): array of ffmc values on the date on interest for the study area
    '''
    yesterday_index = index-1

    if yesterday_index == -1:
        rain_shape = rain_grid.shape
        ffmc_initialize = np.zeros(rain_shape)+85
        ffmc_yesterday1 = ffmc_initialize*mask #mask out areas that haven't started
    else: 
        ffmc_yesterday1 = ffmc_yesterday
        ffmc_yesterday1[np.where(np.isnan(ffmc_yesterday1) & ~np.isnan(mask))] = 85 #set started pixels since yesterday to 85


    wmo = 147.2*(101-ffmc_yesterday)/(59.5+ffmc_yesterday)

    rain_grid[rain_grid > 0.5] = rain_grid[rain_grid > 0.5] - 0.5

    wmo[wmo>=150]=wmo[wmo >= 150]+0.0015*(wmo[wmo >= 150]-150)*\
                     (wmo[wmo >= 150]- 150)*np.sqrt(rain_grid[wmo >= 150]) + 42.5\
                     *rain_grid[wmo >= 150]*np.exp(-100/(251-wmo[wmo >= 150]))*\
                     (1-np.exp(-6.93/rain_grid[wmo >= 150]))
    
    wmo[wmo<150]=wmo[wmo<150]+42.5*rain_grid[wmo<150]*np.exp(-100/(251-wmo[wmo<150]))\
                  *(1-np.exp(-6.93/rain_grid[wmo<150]))

    wmo[rain_grid < 0.5] = 147.2*(101-ffmc_yesterday[rain_grid < 0.5])/(59.5+ffmc_yesterday[rain_grid < 0.5])

    wmo[wmo>250] = 250

    ed=0.942*np.power(rh_grid,0.679)+(11*np.exp((rh_grid-100)/10))+0.18*(21.1-temp_grid)\
        *(1-1/np.exp(rh_grid*0.115))
    
    ew=0.618*np.power(rh_grid,0.753)+(10*np.exp((rh_grid-100)/10))+0.18*(21.1-temp_grid)*\
        (1-1/np.exp(rh_grid*0.115))

    shape = rain_grid.shape 
    z = np.zeros(shape)
    z[np.where((wmo<ed) & (wmo<ew))]=0.424*(1-np.power((rh_grid[np.where((wmo<ed) & (wmo<ew))]/100),1.7))\
                          +0.0694*np.sqrt(wind_grid[np.where((wmo<ed) & (wmo<ew))])*\
                  (1-np.power((rh_grid[np.where((wmo<ed) & (wmo<ew))]/100),8))
    

    z[np.where((wmo>=ed) & (wmo>=ew))] = 0

    x=z*0.581*np.exp(0.0365*temp_grid)

    shape = rain_grid.shape 
    wm = np.zeros(shape)    

    wm[np.where((wmo<ed) & (wmo<ew))]= ew[np.where((wmo<ed) & (wmo<ew))]-\
                           (ew[np.where((wmo<ed) & (wmo<ew))]-\
                            wmo[np.where((wmo<ed) & (wmo<ew))])/(np.power(10,x[np.where((wmo<ed) & (wmo<ew))]))

    wm[np.where((wmo>=ed) & (wmo>=ew))] = wmo[np.where((wmo>=ed) & (wmo>=ew))]

    z[wmo>ed] = 0.424*(1-np.power((rh_grid[wmo>ed]/100),1.7))+0.0694\
                       *np.sqrt(wind_grid[wmo>ed])*(1-np.power((rh_grid[wmo>ed]/100),8))

    x=z*0.581*np.exp(0.0365 * temp_grid)
    wm[wmo>ed] = ed[wmo>ed] + (wmo[wmo>ed] - ed[wmo>ed])/(np.power(10,x[wmo>ed]))

    ffmc1 = (59.5*(250-wm))/(147.2+wm)
    
    ffmc1[ffmc1 > 101] = 101

    ffmc1[ffmc1 < 0] = 0

    ffmc1 = ffmc1*mask* endMask

    if show: 
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        fig, ax = plt.subplots(figsize= (15,15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)
        
        circ = PolygonPatch(na_map['geometry'][0],visible=False)
        ax.add_patch(circ) 
        plt.imshow(ffmc1,extent=(min_xProj_extent-1,max_xProj_extent+1,max_yProj_extent-1,min_yProj_extent+1),clip_path=circ, 
                   clip_on=True) 
        
        #plt.imshow(ffmc1,extent=(min_xProj_extent-1,max_xProj_extent+1,max_yProj_extent-1,min_yProj_extent+1)) 
        na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)
            
        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label('FFMC') 
        
        title = 'FFMC for %s'%(input_date) 
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        plt.show()

    return ffmc1

def BUI(dmc,dc,maxmin,show,shapefile,mask,endMask): #BUI can be calculated on the fly
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
    bui1[np.where((dmc > 0) & (dc > 0))] = 0.8 * dc[np.where((dmc > 0) & (dc > 0))]*\
                                          dmc[np.where((dmc > 0) & (dc > 0))]/(dmc[np.where((dmc > 0) & (dc > 0))]\
                                          + 0.4 * dc[np.where((dmc > 0) & (dc > 0))])
    p = np.zeros(shape)
    p[dmc == 0] = 0
    p[dmc > 0] = (dmc[dmc > 0] - bui1[dmc > 0])/dmc[dmc > 0]

    cc = 0.92 + (np.power((0.0114 * dmc),1.7))
    
    bui0 = dmc - cc * p

    bui0[bui0 < 0] = 0

    bui1[bui1 < dmc] = bui0[bui1 < dmc]

    bui1 = bui1*mask* endMask

    if show: 
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        fig, ax = plt.subplots(figsize= (15,15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)
        
      
        plt.imshow(bui1,extent=(min_xProj_extent-1,max_xProj_extent+1,max_yProj_extent-1,min_yProj_extent+1)) 
        na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)
            
        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label('BUI') 
        
        title = 'BUI for %s'%(input_date) 
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        plt.show()

    return bui1

def ISI(ffmc,wind_grid,maxmin,show,shapefile,mask,endMask):
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
    
    isi = isi*mask* endMask
    if show: 
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        fig, ax = plt.subplots(figsize= (15,15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)
        
      
        plt.imshow(isi,extent=(min_xProj_extent-1,max_xProj_extent+1,max_yProj_extent-1,min_yProj_extent+1)) 
        na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)
            
        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label('ISI') 
        
        title = 'ISI for %s'%(input_date) 
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        plt.show()
        
    return isi

def FWI(isi,bui,maxmin,show,shapefile,mask,endMask):
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

    bb[bui > 80] = 0.1 * isi[bui > 80] * (1000/(25 + 108.64/np.exp(0.023 * bui[bui > 80])))
    bb[bui <= 80] =  0.1 * isi[bui <= 80] * (0.626 * np.power(bui[bui <= 80],0.809) + 2)

    fwi = np.zeros(shape)
    fwi[bb <= 1] = bb[bb <= 1]
    fwi[bb > 1] = np.exp(2.72 * ((0.434 * np.log(bb[bb > 1]))**0.647)) #natural logarithm 

    fwi = fwi * mask * endMask

    if show: 
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        fig, ax = plt.subplots(figsize= (15,15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)
        
      
        plt.imshow(fwi,extent=(min_xProj_extent-1,max_xProj_extent+1,max_yProj_extent-1,min_yProj_extent+1)) 
        na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)
            
        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label('FWI') 
        
        title = 'FWI for %s'%(input_date) 
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        plt.show()
 
    return fwi

def plot_july(fwi_list,maxmin,year,var,shapefile):
    ''' Visualize all values for July. **DO NOT HAVE TO CHANGE INDEX IF LEAP YEAR** WHY? B/C WE ARE COUNTING FRM MAR1
    Parameters
        fwi_list (list): list of fwi metric arrays for a certain measure (i.e. dmc)
        maxmin (list): bounds of study area
        year (str): year of interest
        var (str): variable name of interest (i.e. "Duff Moisture Code")
        shapefile (str): path to study area shapefile
    Returns
        plots a figure with fwi metric map for each day in month of July 
    '''

    fig = plt.figure()
    COUNT = 0
    for index in range(121,152): #The range refers to the start and end indexes of where July is in the list 
        ax = fig.add_subplot(4,8,COUNT+1)
        min_yProj_extent = maxmin[0]
        max_yProj_extent = maxmin[1]
        max_xProj_extent = maxmin[2]
        min_xProj_extent = maxmin[3]

        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)
        
        na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)

        
        title = str(COUNT+1)

        im = ax.imshow(fwi_list[index],extent=(min_xProj_extent-1,max_xProj_extent+1,max_yProj_extent-1,min_yProj_extent+1)) 
        ax.set_title(title)
        ax.invert_yaxis()

        COUNT+=1



    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar1 = fig.colorbar(im, orientation="vertical", cax=cbar_ax, pad=0.2)
    cbar1.set_label(var)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    fig.text(0.5, 0.04, 'Longitude', ha='center')
    fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical')
    title = '%s for July %s'%(var,year)
    fig.suptitle(title, fontsize=14)
    plt.show()

