#coding: utf-8

"""
Summary
-------
Interpolation functions for thin plate splines using the radial basis function
from SciPy.

References
----------
Flannigan, M. D., & Wotton, B. M. (1989). A study of interpolation methods for forest fire danger rating in Canada.
Canadian Journal of Forest Research, 19(8), 1059–1066. https://doi.org/10.1139/x89-161

"""
    
#import
import geopandas as gpd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

import warnings
warnings.filterwarnings("ignore") #Runtime warning suppress, this suppresses the /0 warning

from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

import cluster_3d as c3d

#functions 
def TPS(latlon_dict,Cvar_dict,input_date,var_name,shapefile,show,phi):
    '''Thin plate splines interpolation implemented using the interpolate radial basis function from 
    SciPy 
    Parameters
        latlon_dict (dict): the latitude and longitudes of the hourly stations, loaded from the 
        .json file
        Cvar_dict (dict): dictionary of weather variable values for each station 
        input_date (str): the date you want to interpolate for 
        var_name (str): name of the variable you are interpolating
        shapefile (str): path to the study area shapefile 
        show (bool): whether you want to plot a map 
        phi (float): smoothing parameter for the thin plate spline, if 0 no smoothing 
    Returns 
        spline (np_array): the array of values for the interpolated surface
        maxmin: the bounds of the array surface, for use in other functions 
    
    '''
    x_origin_list = []
    y_origin_list = []
    z_origin_list = [] 

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

    lat = []
    lon = []
    Cvar = []
    for station_name in Cvar_dict.keys(): #DONT use list of stations, because if there's a no data we delete that in the climate dictionary step
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

    Cvar = []
    for station_name in Cvar_dict.keys(): #DONT use list of stations, because if there's a no data we delete that in the climate dictionary step 

        cvar_val = Cvar_dict[station_name]

        Cvar.append(cvar_val)

    z2 = np.array(Cvar)


    for station_name in station_name_list:

        na_map = gpd.read_file(shapefile)
        bounds = na_map.bounds

        pixelHeight = 10000 
        pixelWidth = 10000


        coord_pair = projected_lat_lon[station_name]

        x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
        y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)
        z_origin_list.append(Cvar_dict[station_name])


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
    

    source_proj = pyproj.Proj(proj='latlong', datum = 'NAD83') #We dont know but assume 
    xProj, yProj = pyproj.Proj('esri:102001')(x,y)

    yProj_extent=np.append(yProj,[bounds['maxy'],bounds['miny']])
    xProj_extent=np.append(xProj,[bounds['maxx'],bounds['minx']])

    maxmin = [np.min(yProj_extent),np.max(yProj_extent),np.max(xProj_extent),np.min(xProj_extent)]

    Yi = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row)
    Xi = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col)

    Xi,Yi = np.meshgrid(Xi,Yi)

    empty_grid = np.empty((num_row,num_col,))*np.nan

    for x,y,z in zip(x_origin_list,y_origin_list,z_origin_list):
        empty_grid[y][x] = z



    vals = ~np.isnan(empty_grid)

    func = interpolate.Rbf(Xi[vals],Yi[vals],empty_grid[vals], function='thin_plate',smooth=phi)
    thin_plate = func(Xi,Yi)
    spline = thin_plate.reshape(num_row,num_col)

    if show: 

        fig, ax = plt.subplots(figsize= (15,15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)
        
      
        plt.imshow(spline,extent=(xProj_extent.min()-1,xProj_extent.max()+1,yProj_extent.max()-1,yProj_extent.min()+1)) 
        na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)
            
        plt.scatter(xProj,yProj,c=z_origin_list,edgecolors='k',linewidth=1)

        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label(var_name) 
        
        title = 'Thin Plate Spline Interpolation for %s on %s'%(var_name,input_date) 
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude') 

        plt.show()

    return spline, maxmin


def cross_validate_tps(latlon_dict,Cvar_dict,shapefile,phi):
    '''Leave-one-out cross-validation for thin plate splines 
    Parameters
        latlon_dict (dict): the latitude and longitudes of the hourly stations, loaded from the 
        .json file
        Cvar_dict (dict): dictionary of weather variable values for each station 
        shapefile (str): path to the study area shapefile 
        phi (float): smoothing parameter for the thin plate spline, if 0 no smoothing 
    Returns 
        absolute_error_dictionary (dict): a dictionary of the absolute error at each station when it
        was left out 
     '''
    x_origin_list = []
    y_origin_list = [] 
    z_origin_list = []
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

        na_map = gpd.read_file(shapefile)
        bounds = na_map.bounds

        pixelHeight = 10000 
        pixelWidth = 10000


        coord_pair = projected_lat_lon[station_name_hold_back]

        x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
        y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)
        z_origin_list.append(Cvar_dict[station_name_hold_back])


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
        source_proj = pyproj.Proj(proj='latlong', datum = 'NAD83') #We dont know but assume 
        xProj, yProj = pyproj.Proj('esri:102001')(x,y)

        yProj_extent=np.append(yProj,[bounds['maxy'],bounds['miny']])
        xProj_extent=np.append(xProj,[bounds['maxx'],bounds['minx']])

        Yi = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row)
        Xi = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col)

        Xi,Yi = np.meshgrid(Xi,Yi)

        empty_grid = np.empty((num_row,num_col,))*np.nan

        for x,y,z in zip(x_origin_list,y_origin_list,z_origin_list):
            empty_grid[y][x] = z



        vals = ~np.isnan(empty_grid)

        func = interpolate.Rbf(Xi[vals],Yi[vals],empty_grid[vals], function='thin_plate',smooth=phi)
        thin_plate = func(Xi,Yi)
        spline = thin_plate.reshape(num_row,num_col)

        #Calc the RMSE, MAE, at the pixel loc
        #Delete at a certain point
        coord_pair = projected_lat_lon[station_name_hold_back]

        x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
        y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)

        interpolated_val = spline[y_orig][x_orig] 

        original_val = Cvar_dict[station_name]
        absolute_error = abs(interpolated_val-original_val)
        absolute_error_dictionary[station_name_hold_back] = absolute_error

    return absolute_error_dictionary

def shuffle_split_tps(latlon_dict,Cvar_dict,shapefile,phi,rep):
    '''Shuffle-split cross-validation for thin plate splines 
    Parameters
        latlon_dict (dict): the latitude and longitudes of the hourly stations, loaded from the 
        .json file
        Cvar_dict (dict): dictionary of weather variable values for each station 
        shapefile (str): path to the study area shapefile 
        phi (float): smoothing parameter for the thin plate spline, if 0 no smoothing
        rep (int): number of replications 
    Returns 
        overall_error (float): MAE average of all the replications
     '''
    count = 1
    error_dictionary = {}
    while count <= rep:
        x_origin_list = []
        y_origin_list = [] 
        z_origin_list = []
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

            na_map = gpd.read_file(shapefile)
            bounds = na_map.bounds

            pixelHeight = 10000 
            pixelWidth = 10000


            coord_pair = projected_lat_lon[station_name_hold_back]

            x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
            y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
            x_origin_list.append(x_orig)
            y_origin_list.append(y_orig)
            z_origin_list.append(Cvar_dict[station_name_hold_back])


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
        source_proj = pyproj.Proj(proj='latlong', datum = 'NAD83') #We dont know but assume 
        xProj, yProj = pyproj.Proj('esri:102001')(x,y)

        yProj_extent=np.append(yProj,[bounds['maxy'],bounds['miny']])
        xProj_extent=np.append(xProj,[bounds['maxx'],bounds['minx']])

        Yi = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row)
        Xi = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col)

        Xi,Yi = np.meshgrid(Xi,Yi)

        empty_grid = np.empty((num_row,num_col,))*np.nan

        for x,y,z in zip(x_origin_list,y_origin_list,z_origin_list):
            empty_grid[y][x] = z



        vals = ~np.isnan(empty_grid)

        func = interpolate.Rbf(Xi[vals],Yi[vals],empty_grid[vals], function='thin_plate',smooth=phi)
        thin_plate = func(Xi,Yi)
        spline = thin_plate.reshape(num_row,num_col)

        #Calc the RMSE, MAE, at the pixel loc
        #Delete at a certain point
        for station_name_hold_back in test_stations: 
            coord_pair = projected_lat_lon[station_name_hold_back]

            x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
            y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
            x_origin_list.append(x_orig)
            y_origin_list.append(y_orig)

            interpolated_val = spline[y_orig][x_orig] 

            original_val = Cvar_dict[station_name]
            absolute_error = abs(interpolated_val-original_val)
            absolute_error_dictionary[station_name_hold_back] = absolute_error
        error_dictionary[count]= sum(absolute_error_dictionary.values())/len(absolute_error_dictionary.values()) #average of all the withheld stations
        count+=1

    overall_error = sum(error_dictionary.values())/rep
    
    return overall_error

def spatial_kfold_tps(loc_dict,Cvar_dict,shapefile,phi,file_path_elev,elev_array,idx_list):
    '''Spatially blocked k-folds cross-validation procedure for thin plate splines 
    Parameters
        loc_dict (dict): the latitude and longitudes of the hourly stations, loaded from the 
        .json file
        Cvar_dict (dict): dictionary of weather variable values for each station 
        shapefile (str): path to the study area shapefile 
        phi (float): smoothing parameter for the thin plate spline, if 0 no smoothing
    Returns 
        MAE (float): MAE average of all the replications
     '''
    groups_complete = [] #If not using replacement, keep a record of what we have done 
    error_dictionary = {} 


    x_origin_list = []
    y_origin_list = [] 
    z_origin_list = []
    absolute_error_dictionary = {} #for plotting
    station_name_list = []
    projected_lat_lon = {}

    #Selecting blocknum
    block_num_ref = [25,16,9] 
    calinski_harabasz = [] 

    label,Xelev,cluster25 = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,25,file_path_elev,idx_list,False,False,True)
    calinski_harabasz.append(metrics.calinski_harabasz_score(Xelev, label)) #Calinski-Harabasz Index --> higher the better
    label,Xelev,cluster16 = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,16,file_path_elev,idx_list,False,False,True)
    calinski_harabasz.append(metrics.calinski_harabasz_score(Xelev, label))
    label,Xelev,cluster9 = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,9,file_path_elev,idx_list,False,False,True)
    calinski_harabasz.append(metrics.calinski_harabasz_score(Xelev, label))

    minIndex = calinski_harabasz.index(min(calinski_harabasz))
    blocknum = block_num_ref[minIndex] #lookup the block size that corresponds

    cluster = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,blocknum,file_path_elev,idx_list,False,False,False)

    for group in cluster.values():
        if group not in groups_complete:
            station_list = [k for k,v in cluster.items() if v == group]
            groups_complete.append(group)

    for station_name in Cvar_dict.keys():
        if station_name in loc_dict.keys():
            station_name_list.append(station_name)

            loc = loc_dict[station_name]
            latitude = loc[0]
            longitude = loc[1]
            Plat, Plon = pyproj.Proj('esri:102001')(longitude,latitude)
            Plat = float(Plat)
            Plon = float(Plon)
            projected_lat_lon[station_name] = [Plat,Plon]


    for station_name_hold_back in station_name_list:

        na_map = gpd.read_file(shapefile)
        bounds = na_map.bounds

        pixelHeight = 10000 
        pixelWidth = 10000


        coord_pair = projected_lat_lon[station_name_hold_back]

        x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
        y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)
        z_origin_list.append(Cvar_dict[station_name_hold_back])



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
    source_proj = pyproj.Proj(proj='latlong', datum = 'NAD83') #We dont know but assume 
    xProj, yProj = pyproj.Proj('esri:102001')(x,y)

    yProj_extent=np.append(yProj,[bounds['maxy'],bounds['miny']])
    xProj_extent=np.append(xProj,[bounds['maxx'],bounds['minx']])

    Yi = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row)
    Xi = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col)

    Xi,Yi = np.meshgrid(Xi,Yi)

    empty_grid = np.empty((num_row,num_col,))*np.nan

    for x,y,z in zip(x_origin_list,y_origin_list,z_origin_list):
        empty_grid[y][x] = z



    vals = ~np.isnan(empty_grid)

    func = interpolate.Rbf(Xi[vals],Yi[vals],empty_grid[vals], function='thin_plate',smooth=phi)
    thin_plate = func(Xi,Yi)
    spline = thin_plate.reshape(num_row,num_col)

    #Calc the RMSE, MAE, at the pixel loc
    #Delete at a certain point
    for station_name_hold_back in station_list: 
        coord_pair = projected_lat_lon[station_name_hold_back]

        x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
        y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)

        interpolated_val = spline[y_orig][x_orig] 

        original_val = Cvar_dict[station_name]
        absolute_error = abs(interpolated_val-original_val)
        absolute_error_dictionary[station_name_hold_back] = absolute_error
        
    MAE= sum(absolute_error_dictionary.values())/len(absolute_error_dictionary.values()) #average of all the withheld stations
     
    return blocknum,MAE
