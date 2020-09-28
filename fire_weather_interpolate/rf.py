#coding: utf-8

"""
Summary
-------
Spatial interpolation functions for random forest interpolation
using the scikit-learn package. 

"""
    
#import
import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj

import warnings
warnings.filterwarnings("ignore") #Runtime warning suppress, this suppresses the /0 warning

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

import get_data as GD
import cluster_3d as c3d
import make_blocks as mbk
import Eval as Eval
import statistics 

def random_forest_interpolator(latlon_dict,Cvar_dict,input_date,var_name,shapefile,show,file_path_elev,idx_list): 
    lat = []
    lon = []
    Cvar = []
    for station_name in Cvar_dict.keys():
        if station_name in latlon_dict.keys():
            
            loc = latlon_dict[station_name]
            latitude = loc[0]
            longitude = loc[1]
            cvar_val = Cvar_dict[station_name]
            lat.append(float(latitude))
            lon.append(float(longitude))
            Cvar.append(cvar_val)
    y = np.array(lat)
    x = np.array(lon)
    z = np.array(Cvar) 

    na_map = gpd.read_file(shapefile)
    bounds = na_map.bounds
    xmax = bounds['maxx']
    xmin= bounds['minx']
    ymax = bounds['maxy']
    ymin = bounds['miny']
    pixelHeight = 10000 
    pixelWidth = 10000
            
    num_col = int((xmax - xmin) / pixelHeight)
    num_row = int((ymax - ymin) / pixelWidth)


    #We need to project to a projected system before making distance matrix
    source_proj = pyproj.Proj(proj='latlong', datum = 'NAD83') 
    xProj, yProj = pyproj.Proj('esri:102001')(x,y)
    
    df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'var': z})

    yProj_extent=np.append(yProj,[bounds['maxy'],bounds['miny']])
    xProj_extent=np.append(xProj,[bounds['maxx'],bounds['minx']])

    Yi = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row)
    Xi = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col)

    Xi,Yi = np.meshgrid(Xi,Yi)
    Xi,Yi = Xi.flatten(), Yi.flatten()
    
    
    maxmin = [np.min(yProj_extent),np.max(yProj_extent),np.max(xProj_extent),np.min(xProj_extent)]
    
    
    #Elevation 
    concat = np.array((Xi.flatten(), Yi.flatten())).T #Preparing the coordinates to send to the function that will get the elevation grid 
    send_to_list = concat.tolist()
    send_to_tuple = [tuple(x) for x in send_to_list] #The elevation function takes a tuple 


    Xi1_grd=[]
    Yi1_grd=[]
    elev_grd = []
    elev_grd_dict = GD.finding_data_frm_lookup(send_to_tuple,file_path_elev,idx_list) #Get the elevations from the lookup file 

    for keys in elev_grd_dict.keys(): #The keys are each lat lon pair 
        x= keys[0]
        y = keys[1]
        Xi1_grd.append(x)
        Yi1_grd.append(y)
        elev_grd.append(elev_grd_dict[keys]) #Append the elevation data to the empty list 

    elev_array = np.array(elev_grd) #make an elevation array

    

    elev_dict= GD.finding_data_frm_lookup(zip(xProj, yProj),file_path_elev,idx_list) #Get the elevations for the stations 

    xProj_input=[]
    yProj_input=[]
    e_input = []


    for keys in zip(xProj,yProj): #Repeat process for just the stations not the whole grid 
        x= keys[0]
        y = keys[1]
        xProj_input.append(x)
        yProj_input.append(y)
        e_input.append(elev_dict[keys])

    source_elev = np.array(e_input)
    
    Xi1_grd = np.array(Xi1_grd)
    Yi1_grd = np.array(Yi1_grd)
    
    df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'elevS':source_elev, 'var': z})
    
    df_testX = pd.DataFrame({'Xi': Xi1_grd, 'Yi': Yi1_grd, 'elev': elev_array})
    
    reg = RandomForestRegressor(n_estimators = 100, max_features='sqrt',random_state=1)   
    
    
    y = np.array(df_trainX['var']).reshape(-1,1)
    X_train = np.array(df_trainX[['xProj','yProj','elevS']])
    X_test = np.array(df_testX[['Xi','Yi','elev']])
    
    reg.fit(X_train, y)
    
    Zi = reg.predict(X_test)
    
    rf_grid = Zi.reshape(num_row,num_col)

    if show:
        fig, ax = plt.subplots(figsize= (15,15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)
        
      
        plt.imshow(rf_grid,extent=(xProj_extent.min()-1,xProj_extent.max()+1,yProj_extent.max()-1,yProj_extent.min()+1)) 
        na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)
            
        plt.scatter(xProj,yProj,c=z,edgecolors='k')

        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label(var_name) 
        
        title = 'RF Interpolation for %s on %s'%(var_name,input_date)
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude') 

        plt.show()


    return rf_grid, maxmin    

def cross_validate_rf(latlon_dict,Cvar_dict,shapefile,file_path_elev,elev_array,idx_list):
    '''Leave-one-out cross-validation procedure for IDEW
    Parameters
        latlon_dict (dict): the latitude and longitudes of the hourly or daily stations, loaded from the 
        .json file
        Cvar_dict (dict): dictionary of weather variable values for each station 
        shapefile (str): path to the study area shapefile 
        file_path_elev (str): file path to the elevation lookup file 
        elev_array (np_array): the elevation array for the study area 
        idx_list (list): the index of the elevation data column in the lookup file 
        d (int): the weighting function for IDW interpolation 
    Returns 
        absolute_error_dictionary (dict): a dictionary of the absolute error at each station when it
        was left out 
    '''
    x_origin_list = []
    y_origin_list = [] 

    absolute_error_dictionary = {} #for plotting
    station_name_list = []
    projected_lat_lon = {}

    for station_name in Cvar_dict.keys():
        if station_name in latlon_dict.keys():
            station_name_list.append(station_name)

            loc = latlon_dict[station_name]
            latitude = loc[0]
            longitude = loc[1]
            Plat, Plon = pyproj.Proj('esri:102001')(longitude,latitude)
            Plat = float(Plat)
            Plon = float(Plon)
            projected_lat_lon[station_name] = [Plat,Plon]



    for station_name_hold_back in station_name_list:

        lat = []
        lon = []
        Cvar = []
        for station_name in sorted(Cvar_dict.keys()):
            if station_name in latlon_dict.keys():
                if station_name != station_name_hold_back:
                    loc = latlon_dict[station_name]
                    latitude = loc[0]
                    longitude = loc[1]
                    cvar_val = Cvar_dict[station_name]
                    lat.append(float(latitude))
                    lon.append(float(longitude))
                    Cvar.append(cvar_val)
                else:

                    pass
                
        y = np.array(lat)
        x = np.array(lon)
        z = np.array(Cvar) 

        na_map = gpd.read_file(shapefile)
        bounds = na_map.bounds
        xmax = bounds['maxx']
        xmin= bounds['minx']
        ymax = bounds['maxy']
        ymin = bounds['miny']
        pixelHeight = 10000 
        pixelWidth = 10000
            
        num_col = int((xmax - xmin) / pixelHeight)
        num_row = int((ymax - ymin) / pixelWidth)


        #We need to project to a projected system before making distance matrix
        source_proj = pyproj.Proj(proj='latlong', datum = 'NAD83') 
        xProj, yProj = pyproj.Proj('esri:102001')(x,y)
    
        df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'var': z})

        yProj_extent=np.append(yProj,[bounds['maxy'],bounds['miny']])
        xProj_extent=np.append(xProj,[bounds['maxx'],bounds['minx']])

        Yi = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row)
        Xi = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col)

        Xi,Yi = np.meshgrid(Xi,Yi)
        Xi,Yi = Xi.flatten(), Yi.flatten()
    
    
        maxmin = [np.min(yProj_extent),np.max(yProj_extent),np.max(xProj_extent),np.min(xProj_extent)]
    
    
        #Elevation 
        concat = np.array((Xi.flatten(), Yi.flatten())).T #Preparing the coordinates to send to the function that will get the elevation grid 
        send_to_list = concat.tolist()
        send_to_tuple = [tuple(x) for x in send_to_list] #The elevation function takes a tuple 


        Xi1_grd=[]
        Yi1_grd=[]
        elev_grd = []
        elev_grd_dict = GD.finding_data_frm_lookup(send_to_tuple,file_path_elev,idx_list) #Get the elevations from the lookup file 

        for keys in elev_grd_dict.keys(): #The keys are each lat lon pair 
            x= keys[0]
            y = keys[1]
            Xi1_grd.append(x)
            Yi1_grd.append(y)
            elev_grd.append(elev_grd_dict[keys]) #Append the elevation data to the empty list 

        elev_array = np.array(elev_grd) #make an elevation array

    

        elev_dict= GD.finding_data_frm_lookup(zip(xProj, yProj),file_path_elev,idx_list) #Get the elevations for the stations 

        xProj_input=[]
        yProj_input=[]
        e_input = []


        for keys in zip(xProj,yProj): #Repeat process for just the stations not the whole grid 
            x= keys[0]
            y = keys[1]
            xProj_input.append(x)
            yProj_input.append(y)
            e_input.append(elev_dict[keys])

        source_elev = np.array(e_input)
    
        Xi1_grd = np.array(Xi1_grd)
        Yi1_grd = np.array(Yi1_grd)
    
        df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'elevS':source_elev, 'var': z})
    
        df_testX = pd.DataFrame({'Xi': Xi1_grd, 'Yi': Yi1_grd, 'elev': elev_array})
    
    
        reg = RandomForestRegressor(n_estimators = 100, max_features='sqrt',random_state=1)     
    
    
        y = np.array(df_trainX['var']).reshape(-1,1)
        X_train = np.array(df_trainX[['xProj','yProj','elevS']])
        X_test = np.array(df_testX[['Xi','Yi','elev']])
    
        reg.fit(X_train, y)
    
        Zi = reg.predict(X_test)
    
        rf_grid = Zi.reshape(num_row,num_col)

        #Calc the RMSE, MAE at the pixel loc
        #Delete at a certain point
        coord_pair = projected_lat_lon[station_name_hold_back]

        x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
        y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)

        interpolated_val = rf_grid[y_orig][x_orig] 

        original_val = Cvar_dict[station_name_hold_back]
        absolute_error = abs(interpolated_val-original_val)
        absolute_error_dictionary[station_name_hold_back] = absolute_error


    return absolute_error_dictionary

def shuffle_split_rf(latlon_dict,Cvar_dict,shapefile,file_path_elev,elev_array,idx_list,rep):
    '''Shuffle split cross-validation procedure for rf
    Parameters
        latlon_dict (dict): the latitude and longitudes of the hourly or daily stations, loaded from the 
        .json file
        Cvar_dict (dict): dictionary of weather variable values for each station 
        shapefile (str): path to the study area shapefile 
        file_path_elev (str): file path to the elevation lookup file 
        elev_array (np_array): the elevation array for the study area 
        idx_list (list): the index of the elevation data column in the lookup file 
        rep (int): number of repetitions to run 
    Returns 
        overall_error (float): average MAE value of all the reps 
    '''
    count = 1
    error_dictionary = {}
    while count <= rep:
        x_origin_list = []
        y_origin_list = [] 

        absolute_error_dictionary = {} #for plotting
        station_name_list = []
        projected_lat_lon = {}

        for station_name in Cvar_dict.keys():
            if station_name in latlon_dict.keys():
                station_name_list.append(station_name)

                loc = latlon_dict[station_name]
                latitude = loc[0]
                longitude = loc[1]
                Plat, Plon = pyproj.Proj('esri:102001')(longitude,latitude)
                Plat = float(Plat)
                Plon = float(Plon)
                projected_lat_lon[station_name] = [Plat,Plon]

        #Split the stations in two
        stations_input = [] #we can't just use Cvar_dict.keys() because some stations do not have valid lat/lon
        for station_code in Cvar_dict.keys():
            if station_code in latlon_dict.keys():
                stations_input.append(station_code)
          #Split the stations in two
        stations = np.array(stations_input)
        splits = ShuffleSplit(n_splits=1, train_size=.5) #Won't be exactly 50/50 if uneven num stations

        for train_index, test_index in splits.split(stations):

               train_stations = stations[train_index] 
               #print(train_stations)
               test_stations = stations[test_index]
               #print(test_stations)

          #They can't overlap

        for val in train_stations:
            if val in test_stations:
                print('Error, the train and test sets overlap!')
                sys.exit()
                    
        lat = []
        lon = []
        Cvar = []
        for station_name in sorted(Cvar_dict.keys()):
            if station_name in latlon_dict.keys():
                if station_name not in test_stations:
                    loc = latlon_dict[station_name]
                    latitude = loc[0]
                    longitude = loc[1]
                    cvar_val = Cvar_dict[station_name]
                    lat.append(float(latitude))
                    lon.append(float(longitude))
                    Cvar.append(cvar_val)
                else:

                    pass
                
        y = np.array(lat)
        x = np.array(lon)
        z = np.array(Cvar) 

        na_map = gpd.read_file(shapefile)
        bounds = na_map.bounds
        xmax = bounds['maxx']
        xmin= bounds['minx']
        ymax = bounds['maxy']
        ymin = bounds['miny']
        pixelHeight = 10000 
        pixelWidth = 10000
            
        num_col = int((xmax - xmin) / pixelHeight)
        num_row = int((ymax - ymin) / pixelWidth)


        #We need to project to a projected system before making distance matrix
        source_proj = pyproj.Proj(proj='latlong', datum = 'NAD83') 
        xProj, yProj = pyproj.Proj('esri:102001')(x,y)
    
        df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'var': z})

        yProj_extent=np.append(yProj,[bounds['maxy'],bounds['miny']])
        xProj_extent=np.append(xProj,[bounds['maxx'],bounds['minx']])

        Yi = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row)
        Xi = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col)

        Xi,Yi = np.meshgrid(Xi,Yi)
        Xi,Yi = Xi.flatten(), Yi.flatten()
    
    
        maxmin = [np.min(yProj_extent),np.max(yProj_extent),np.max(xProj_extent),np.min(xProj_extent)]
    
    
        #Elevation 
        concat = np.array((Xi.flatten(), Yi.flatten())).T #Preparing the coordinates to send to the function that will get the elevation grid 
        send_to_list = concat.tolist()
        send_to_tuple = [tuple(x) for x in send_to_list] #The elevation function takes a tuple 


        Xi1_grd=[]
        Yi1_grd=[]
        elev_grd = []
        elev_grd_dict = GD.finding_data_frm_lookup(send_to_tuple,file_path_elev,idx_list) #Get the elevations from the lookup file 

        for keys in elev_grd_dict.keys(): #The keys are each lat lon pair 
            x= keys[0]
            y = keys[1]
            Xi1_grd.append(x)
            Yi1_grd.append(y)
            elev_grd.append(elev_grd_dict[keys]) #Append the elevation data to the empty list 

        elev_array = np.array(elev_grd) #make an elevation array

    

        elev_dict= GD.finding_data_frm_lookup(zip(xProj, yProj),file_path_elev,idx_list) #Get the elevations for the stations 

        xProj_input=[]
        yProj_input=[]
        e_input = []


        for keys in zip(xProj,yProj): #Repeat process for just the stations not the whole grid 
            x= keys[0]
            y = keys[1]
            xProj_input.append(x)
            yProj_input.append(y)
            e_input.append(elev_dict[keys])

        source_elev = np.array(e_input)
    
        Xi1_grd = np.array(Xi1_grd)
        Yi1_grd = np.array(Yi1_grd)
    
        df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'elevS':source_elev, 'var': z})
    
        df_testX = pd.DataFrame({'Xi': Xi1_grd, 'Yi': Yi1_grd, 'elev': elev_array})
    
    
        reg = RandomForestRegressor(n_estimators = 100, max_features='sqrt',random_state=1)     
    
    
        y = np.array(df_trainX['var']).reshape(-1,1)
        X_train = np.array(df_trainX[['xProj','yProj','elevS']])
        X_test = np.array(df_testX[['Xi','Yi','elev']])
    
        reg.fit(X_train, y)
    
        Zi = reg.predict(X_test)
    
        rf_grid = Zi.reshape(num_row,num_col)

        #Calc the RMSE, MAE at the pixel loc
        #Delete at a certain point
        for statLoc in test_stations: 
            coord_pair = projected_lat_lon[statLoc]

            x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
            y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
            x_origin_list.append(x_orig)
            y_origin_list.append(y_orig)

            interpolated_val = rf_grid[y_orig][x_orig] 

            original_val = Cvar_dict[statLoc]
            absolute_error = abs(interpolated_val-original_val)
            absolute_error_dictionary[statLoc] = absolute_error
        error_dictionary[count]= sum(absolute_error_dictionary.values())/len(absolute_error_dictionary.values()) #average of all the withheld stations
        count+=1

    overall_error = sum(error_dictionary.values())/rep


    return overall_error
        

def spatial_kfold_rf(idw_example_grid,loc_dict,Cvar_dict,shapefile,file_path_elev,elev_array,idx_list,clusterNum,blocking_type):
    '''Spatially blocked k-folds cross-validation procedure for rf
    Parameters
        loc_dict (dict): the latitude and longitudes of the hourly or daily stations, loaded from the 
        .json file
        Cvar_dict (dict): dictionary of weather variable values for each station 
        shapefile (str): path to the study area shapefile 
        file_path_elev (str): file path to the elevation lookup file 
        elev_array (np_array): the elevation array for the study area 
        idx_list (list): the index of the elevation data column in the lookup file 
        clusterNum (int): number of clusters to form
    Returns 
        overall_error (float): average MAE value of all the reps 
    '''
    groups_complete = [] #If not using replacement, keep a record of what we have done 
    error_dictionary = {} 

    x_origin_list = []
    y_origin_list = [] 

    absolute_error_dictionary = {} 
    projected_lat_lon = {}

    #Selecting blocknum
    if blocking_type == 'cluster':
        cluster = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,clusterNum,file_path_elev,idx_list,False,False,False)
    elif blocking_type == 'block':
        np_array_blocks = mbk.make_block(idw_example_grid,clusterNum) #Get the numpy array that delineates the blocks
        cluster = mbk.sorting_stations(np_array_blocks,shapefile,loc_dict,Cvar_dict) #Now get the dictionary
    else:
        print('That is not a valid blocking method')
        sys.exit()
        
    for group in cluster.values():
        if group not in groups_complete:
            station_list = [k for k,v in cluster.items() if v == group]
            groups_complete.append(group)

    for station_name in Cvar_dict.keys():
        if station_name in loc_dict.keys():

            loc = loc_dict[station_name]
            latitude = loc[0]
            longitude = loc[1]
            Plat, Plon = pyproj.Proj('esri:102001')(longitude,latitude)
            Plat = float(Plat)
            Plon = float(Plon)
            projected_lat_lon[station_name] = [Plat,Plon]

                
    lat = []
    lon = []
    Cvar = []
    for station_name in sorted(Cvar_dict.keys()):
        if station_name in loc_dict.keys():
            if station_name not in station_list:
                loc = loc_dict[station_name]
                latitude = loc[0]
                longitude = loc[1]
                cvar_val = Cvar_dict[station_name]
                lat.append(float(latitude))
                lon.append(float(longitude))
                Cvar.append(cvar_val)
            else:

                pass
            
    y = np.array(lat)
    x = np.array(lon)
    z = np.array(Cvar) 

    na_map = gpd.read_file(shapefile)
    bounds = na_map.bounds
    xmax = bounds['maxx']
    xmin= bounds['minx']
    ymax = bounds['maxy']
    ymin = bounds['miny']
    pixelHeight = 10000 
    pixelWidth = 10000
        
    num_col = int((xmax - xmin) / pixelHeight)
    num_row = int((ymax - ymin) / pixelWidth)


    #We need to project to a projected system before making distance matrix
    source_proj = pyproj.Proj(proj='latlong', datum = 'NAD83') 
    xProj, yProj = pyproj.Proj('esri:102001')(x,y)

    df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'var': z})

    yProj_extent=np.append(yProj,[bounds['maxy'],bounds['miny']])
    xProj_extent=np.append(xProj,[bounds['maxx'],bounds['minx']])

    Yi = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row)
    Xi = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col)

    Xi,Yi = np.meshgrid(Xi,Yi)
    Xi,Yi = Xi.flatten(), Yi.flatten()


    maxmin = [np.min(yProj_extent),np.max(yProj_extent),np.max(xProj_extent),np.min(xProj_extent)]


    #Elevation 
    concat = np.array((Xi.flatten(), Yi.flatten())).T #Preparing the coordinates to send to the function that will get the elevation grid 
    send_to_list = concat.tolist()
    send_to_tuple = [tuple(x) for x in send_to_list] #The elevation function takes a tuple 


    Xi1_grd=[]
    Yi1_grd=[]
    elev_grd = []
    elev_grd_dict = GD.finding_data_frm_lookup(send_to_tuple,file_path_elev,idx_list) #Get the elevations from the lookup file 

    for keys in elev_grd_dict.keys(): #The keys are each lat lon pair 
        x= keys[0]
        y = keys[1]
        Xi1_grd.append(x)
        Yi1_grd.append(y)
        elev_grd.append(elev_grd_dict[keys]) #Append the elevation data to the empty list 

    elev_array = np.array(elev_grd) #make an elevation array



    elev_dict= GD.finding_data_frm_lookup(zip(xProj, yProj),file_path_elev,idx_list) #Get the elevations for the stations 

    xProj_input=[]
    yProj_input=[]
    e_input = []


    for keys in zip(xProj,yProj): #Repeat process for just the stations not the whole grid 
        x= keys[0]
        y = keys[1]
        xProj_input.append(x)
        yProj_input.append(y)
        e_input.append(elev_dict[keys])

    source_elev = np.array(e_input)

    Xi1_grd = np.array(Xi1_grd)
    Yi1_grd = np.array(Yi1_grd)

    df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'elevS':source_elev, 'var': z})

    df_testX = pd.DataFrame({'Xi': Xi1_grd, 'Yi': Yi1_grd, 'elev': elev_array})


    reg = RandomForestRegressor(n_estimators = 100, max_features='sqrt',random_state=1)     


    y = np.array(df_trainX['var']).reshape(-1,1)
    X_train = np.array(df_trainX[['xProj','yProj','elevS']])
    X_test = np.array(df_testX[['Xi','Yi','elev']])

    reg.fit(X_train, y)

    Zi = reg.predict(X_test)

    rf_grid = Zi.reshape(num_row,num_col)

    #Calc the RMSE, MAE at the pixel loc
    #Delete at a certain point
    for statLoc in station_list: 
        coord_pair = projected_lat_lon[statLoc]

        x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
        y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)

        interpolated_val = rf_grid[y_orig][x_orig] 

        original_val = Cvar_dict[statLoc]
        absolute_error = abs(interpolated_val-original_val)
        absolute_error_dictionary[statLoc] = absolute_error
        
    MAE= sum(absolute_error_dictionary.values())/len(absolute_error_dictionary.values()) #average of all the withheld stations
     
    return clusterNum,MAE

def select_block_size_rf(nruns,group_type,loc_dict,Cvar_dict,idw_example_grid,shapefile,file_path_elev,idx_list):
     '''Evaluate the standard deviation of MAE values based on consective runs of the cross-valiation, 
     in order to select the block/cluster size
     Parameters
         nruns (int): number of repetitions 
         group_type (str): whether using 'clusters' or 'blocks'
         loc_dict (dict): the latitude and longitudes of the daily/hourly stations, 
         loaded from the .json file
         Cvar_dict (dict): dictionary of weather variable values for each station 
         idw_example_grid (numpy array): used for reference of study area grid size
         shapefile (str): path to the study area shapefile 
         file_path_elev (str): path to the elevation lookup file
         idx_list (int): position of the elevation column in the lookup file
     Returns 
         lowest_stdev,ave_MAE (int,float): block/cluster number w/ lowest stdev, associated
         ave_MAE of all the runs 
     '''
     
     #Get group dictionaries

     if group_type == 'blocks': 

          folds25 = mbk.make_block(idw_example_grid,25)
          dictionaryGroups25 = mbk.sorting_stations(folds25,shapefile,Cvar_dict)
          folds16 = mbk.make_block(idw_example_grid,16)
          dictionaryGroups16 = mbk.sorting_stations(folds16,shapefile,Cvar_dict)
          folds9 = mbk.make_block(idw_example_grid,9)
          dictionaryGroups9 = mbk.sorting_stations(folds9,shapefile,Cvar_dict)

     elif group_type == 'clusters':

          dictionaryGroups25 = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,25,file_path_elev,idx_list,False,False,False)
          dictionaryGroups16 = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,16,file_path_elev,idx_list,False,False,False)
          dictionaryGroups9 = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,9,file_path_elev,idx_list,False,False,False)

     else:
          print('Thats not a valid group type')
          sys.exit() 
               
     block25_error = []
     block16_error = []
     block9_error = []
     if nruns <= 1:
          print('That is not enough runs to calculate the standard deviation!')
          sys.exit() 
     
     for n in range(0,nruns):

          block25 = spatial_groups_rf(idw_example_grid,loc_dict,Cvar_dict,shapefile,25,5,True,False,dictionaryGroups25,file_path_elev,idx_list)
          block25_error.append(block25) 

          block16 = spatial_groups_rf(idw_example_grid,loc_dict,Cvar_dict,shapefile,16,8,True,False,dictionaryGroups16,file_path_elev,idx_list)
          block16_error.append(block16)
          
          block9 = spatial_groups_rf(idw_example_grid,loc_dict,Cvar_dict,shapefile,9,14,True,False,dictionaryGroups9,file_path_elev,idx_list)
          block9_error.append(block9)

     stdev25 = statistics.stdev(block25_error) 
     stdev16 = statistics.stdev(block16_error)
     stdev9 = statistics.stdev(block9_error)

     list_stdev = [stdev25,stdev16,stdev9]
     list_block_name = [25,16,9]
     list_error = [block25_error,block16_error,block9_error]
     index_min = list_stdev.index(min(list_stdev))
     lowest_stdev = list_block_name[index_min]

     ave_MAE = sum(list_error[index_min])/len(list_error[index_min]) 

     print(lowest_stdev)
     print(ave_MAE) 
     return lowest_stdev,ave_MAE
    
def spatial_groups_rf(idw_example_grid,loc_dict,Cvar_dict,shapefile,blocknum,nfolds,replacement,dictionary_Groups,file_path_elev,idx_list):
     '''Spatially blocked bagging cross-validation procedure for IDW 
     Parameters
         idw_example_grid (numpy array): the example idw grid to base the size of the group array off of 
         loc_dict (dict): the latitude and longitudes of the hourly stations, loaded from the 
         .json file
         Cvar_dict (dict): dictionary of weather variable values for each station 
         shapefile (str): path to the study area shapefile 
         d (int): the weighting function for IDW interpolation
         nfolds (int): # number of folds. For 10-fold we use 10, etc. 
     Returns 
         error_dictionary (dict): a dictionary of the absolute error at each fold when it
         was left out 
     '''
     station_list_used = [] #If not using replacement, keep a record of what we have done 
     count = 1
     error_dictionary = {} 
     while count <= nfolds: 
          x_origin_list = []
          y_origin_list = [] 

          absolute_error_dictionary = {} 
          projected_lat_lon = {}

          station_list = Eval.select_random_station(dictionary_Groups,blocknum,replacement,station_list_used).values()


          if replacement == False: 
               station_list_used.append(list(station_list))
          #print(station_list_used) 

          

                    
          for station_name in Cvar_dict.keys():
               
               if station_name in loc_dict.keys():

                  loc = loc_dict[station_name]
                  latitude = loc[0]
                  longitude = loc[1]
                  Plat, Plon = pyproj.Proj('esri:102001')(longitude,latitude)
                  Plat = float(Plat)
                  Plon = float(Plon)
                  projected_lat_lon[station_name] = [Plat,Plon]


          lat = []
          lon = []
          Cvar = []
          for station_name in sorted(Cvar_dict.keys()):
               if station_name in loc_dict.keys():
                    if station_name not in station_list: #This is the step where we hold back the fold
                         loc = loc_dict[station_name]
                         latitude = loc[0]
                         longitude = loc[1]
                         cvar_val = Cvar_dict[station_name]
                         lat.append(float(latitude))
                         lon.append(float(longitude))
                         Cvar.append(cvar_val)
                    else:
                         pass #Skip the station 
                     
          y = np.array(lat)
          x = np.array(lon)
          z = np.array(Cvar) 
             
          na_map = gpd.read_file(shapefile)
          bounds = na_map.bounds
          xmax = bounds['maxx']
          xmin= bounds['minx']
          ymax = bounds['maxy']
          ymin = bounds['miny']
          pixelHeight = 10000 
          pixelWidth = 10000
                     
          num_col = int((xmax - xmin) / pixelHeight)
          num_row = int((ymax - ymin) / pixelWidth)


               #We need to project to a projected system before making distance matrix
          source_proj = pyproj.Proj(proj='latlong', datum = 'NAD83') 
          xProj, yProj = pyproj.Proj('esri:102001')(x,y)

          df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'var': z})

          yProj_extent=np.append(yProj,[bounds['maxy'],bounds['miny']])
          xProj_extent=np.append(xProj,[bounds['maxx'],bounds['minx']])

          Yi = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row)
          Xi = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col)

          Xi,Yi = np.meshgrid(Xi,Yi)
          Xi,Yi = Xi.flatten(), Yi.flatten()


          maxmin = [np.min(yProj_extent),np.max(yProj_extent),np.max(xProj_extent),np.min(xProj_extent)]


          #Elevation 
          concat = np.array((Xi.flatten(), Yi.flatten())).T #Preparing the coordinates to send to the function that will get the elevation grid 
          send_to_list = concat.tolist()
          send_to_tuple = [tuple(x) for x in send_to_list] #The elevation function takes a tuple 


          Xi1_grd=[]
          Yi1_grd=[]
          elev_grd = []
          elev_grd_dict = GD.finding_data_frm_lookup(send_to_tuple,file_path_elev,idx_list) #Get the elevations from the lookup file 

          for keys in elev_grd_dict.keys(): #The keys are each lat lon pair
              x= keys[0]
              y = keys[1]
              Xi1_grd.append(x)
              Yi1_grd.append(y)
              elev_grd.append(elev_grd_dict[keys]) #Append the elevation data to the empty list 

          elev_array = np.array(elev_grd) #make an elevation array



          elev_dict= GD.finding_data_frm_lookup(zip(xProj, yProj),file_path_elev,idx_list) #Get the elevations for the stations 

          xProj_input=[]
          yProj_input=[]
          e_input = []


          for keys in zip(xProj,yProj): #Repeat process for just the stations not the whole grid
              x= keys[0]
              y = keys[1]
              xProj_input.append(x)
              yProj_input.append(y)
              e_input.append(elev_dict[keys])

          source_elev = np.array(e_input)

          Xi1_grd = np.array(Xi1_grd)
          Yi1_grd = np.array(Yi1_grd)

          df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'elevS':source_elev, 'var': z})

          df_testX = pd.DataFrame({'Xi': Xi1_grd, 'Yi': Yi1_grd, 'elev': elev_array})


          reg = RandomForestRegressor(n_estimators = 100, max_features='sqrt',random_state=1)     


          y = np.array(df_trainX['var']).reshape(-1,1)
          X_train = np.array(df_trainX[['xProj','yProj','elevS']])
          X_test = np.array(df_testX[['Xi','Yi','elev']])

          reg.fit(X_train, y)

          Zi = reg.predict(X_test)

          rf_grid = Zi.reshape(num_row,num_col)

          #Compare at a certain point
          for statLoc in station_list:

               coord_pair = projected_lat_lon[statLoc]

               x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
               y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
               x_origin_list.append(x_orig)
               y_origin_list.append(y_orig)

               interpolated_val = rf_grid[y_orig][x_orig] 

               original_val = Cvar_dict[statLoc]
               absolute_error = abs(interpolated_val-original_val)
               absolute_error_dictionary[statLoc] = absolute_error


          error_dictionary[count]= sum(absolute_error_dictionary.values())/len(absolute_error_dictionary.values()) #average of all the withheld stations
          #print(absolute_error_dictionary)
          count+=1
     overall_error = sum(error_dictionary.values())/nfolds #average of all the runs
     #print(overall_error)
     return overall_error


