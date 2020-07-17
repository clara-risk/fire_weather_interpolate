#coding: latin1

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
    elev_grd_dict = finding_data_frm_lookup(send_to_tuple,file_path_elev,idx_list) #Get the elevations from the lookup file 

    for keys in elev_grd_dict.keys(): #The keys are each lat lon pair 
        x= keys[0]
        y = keys[1]
        Xi1_grd.append(x)
        Yi1_grd.append(y)
        elev_grd.append(elev_grd_dict[keys]) #Append the elevation data to the empty list 

    elev_array = np.array(elev_grd) #make an elevation array

    

    elev_dict= finding_data_frm_lookup(zip(xProj, yProj),file_path_elev,idx_list) #Get the elevations for the stations 

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
        elev_grd_dict = finding_data_frm_lookup(send_to_tuple,file_path_elev,idx_list) #Get the elevations from the lookup file 

        for keys in elev_grd_dict.keys(): #The keys are each lat lon pair 
            x= keys[0]
            y = keys[1]
            Xi1_grd.append(x)
            Yi1_grd.append(y)
            elev_grd.append(elev_grd_dict[keys]) #Append the elevation data to the empty list 

        elev_array = np.array(elev_grd) #make an elevation array

    

        elev_dict= finding_data_frm_lookup(zip(xProj, yProj),file_path_elev,idx_list) #Get the elevations for the stations 

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

        original_val = Cvar_dict[station_name]
        absolute_error = abs(interpolated_val-original_val)
        absolute_error_dictionary[station_name_hold_back] = absolute_error


    return absolute_error_dictionary


        
