#coding: utf-8

"""
Summary
-------
Code for inverse distance elevation weighting with a weight of 0.8 for distance (Euclidean) and 0.2 for distance (vertical). 

References
----------
For weights: 

Daly, C., Gibson, W. P., Taylor, G. H., Johnson, G. L., & Pasteris, P. (2002). 
A knowledge-based approach to the statistical mapping of climate. Climate Research, 22(2), 99–113. https://doi.org/10.3354/cr022099
"""
    
#import
import geopandas as gpd
import numpy as np
import pyproj
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore") #Runtime warning suppress, this suppresses the /0 warning

from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

import get_data as GD
import cluster_3d as c3d

#functions 
def IDEW(latlon_dict,Cvar_dict,input_date,var_name,shapefile,show,file_path_elev,idx_list,d):
    '''Inverse distance elevation weighting
    Parameters
        latlon_dict (dict): the latitude and longitudes of the hourly stations, loaded from the 
        .json file
        Cvar_dict (dict): dictionary of weather variable values for each station 
        input_date (str): the date you want to interpolate for 
        shapefile (str): path to the study area shapefile 
        show (bool): whether you want to plot a map 
        file_path_elev (str): file path to the elevation lookup file 
        idx_list (list): the index of the elevation data column in the lookup file 
        d (int): the weighting function for IDW interpolation 
    Returns 
        idew_grid (np_array): the array of values for the interpolated surface
        maxmin (list): the bounds of the array surface, for use in other functions 
        elev_array (np_array): the array of elevation values for the study area, so we can return it
        to the cross-validation function for faster processing 
        
    '''
    #Input: lat lon of station, variable (start day, rainfall, etc), date of interest,variable name (for plotting), show (bool true/false), file path to elevation lookup file
    #idx_list (for the column containing the elevation data), d is the power applied to get the weight 
    lat = [] #Initialize empty lists to store data 
    lon = []
    Cvar = []
    for station_name in Cvar_dict.keys(): #Loop through the list of stations 
        if station_name in latlon_dict.keys(): #Make sure the station is present in the latlon dict 
            loc = latlon_dict[station_name]
            latitude = loc[0]
            longitude = loc[1]
            cvar_val = Cvar_dict[station_name]
            lat.append(float(latitude))
            lon.append(float(longitude))
            Cvar.append(cvar_val)
    y = np.array(lat) #Convert to a numpy array for faster processing speed 
    x = np.array(lon)
    z = np.array(Cvar) 

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
    maxmin = [np.min(yProj_extent),np.max(yProj_extent),np.max(xProj_extent),np.min(xProj_extent)] #We will later return this for use in other functions 

    vals = np.vstack((xProj,yProj)).T #vertically stack station x and y vals and then transpose them so they are in pairs 

    interpol = np.vstack((Xi,Yi)).T #Do the same thing for the grid x and y vals 
    dist_not = np.subtract.outer(vals[:,0], interpol[:,0]) #Length of the triangle side from the cell to the point with data 
    dist_one = np.subtract.outer(vals[:,1], interpol[:,1]) #Length of the triangle side from the cell to the point with data 
    distance_matrix = np.hypot(dist_not,dist_one) #Euclidean distance, getting the hypotenuse


    weights = 1/(distance_matrix**d) #what if distance is 0 --> np.inf? have to account for the pixel underneath
    weights[np.where(np.isinf(weights))] = 1/(1.0E-50) #Making sure to assign the value of the weather station above the pixel directly to the pixel underneath
    weights /= weights.sum(axis = 0) #The weights must add up to 0 


    Zi = np.dot(weights.T, z) #Take the dot product of the weights and the values, in this case the dot product is the sum product over the last axis of Weights.T and z

    idw_grid = Zi.reshape(num_row,num_col) #reshape the array into the proper format for the map 

    #Elevation weights
    #Lon (X) goes in first for a REASON. It has to do with order in the lookup file. 
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


    vals2 = np.vstack(source_elev).T

    interpol2 = np.vstack(elev_array).T

    dist_not2 = np.subtract.outer(vals2[0], interpol2[0]) #Get distance in terms of the elevation (vertical distance) from the station to the point to be interpolated 
    dist_not2 = np.absolute(dist_not2) #Take the absolute value, we just care about what is the difference 
    weights2 = 1/(dist_not2**d) #Get the inverse distance weight 
    weights2[np.where(np.isinf(weights2))] = 1 #In the case of no elevation change 
    weights2 /= weights2.sum(axis = 0) #Make weights add up to 1 


    fin = 0.8*np.dot(weights.T,z) + 0.2*np.dot(weights2.T,z) #Weight distance as 0.8 and elevation as 0.2 

    idew_grid = fin.reshape(num_row,num_col) #Reshape the final array 


    if show: #Plot if show == True 
        fig, ax = plt.subplots(figsize= (15,15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)
        
      
        plt.imshow(elev_array.reshape(num_row,num_col),extent=(xProj_extent.min()-1,xProj_extent.max()+1,yProj_extent.max()-1,yProj_extent.min()+1)) 
        na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)
            
        plt.scatter(xProj,yProj,c=z,edgecolors='k')

        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label(var_name) 
        
        title = 'IDEW Interpolation for %s on %s'%(var_name,input_date)
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude') 

        plt.show()


    return idew_grid, maxmin, elev_array


def cross_validate_IDEW(latlon_dict,Cvar_dict,shapefile,file_path_elev,elev_array,idx_list,d):
    '''Leave-one-out cross-validation procedure for IDEW
    Parameters
        latlon_dict (dict): the latitude and longitudes of the hourly stations, loaded from the 
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
        z = np.array(Cvar) #what if we add the bounding locations to the array??? ==> that would be extrapolation not interpolation? 

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
        Xi,Yi = Xi.flatten(), Yi.flatten()
        maxmin = [np.min(yProj_extent),np.max(yProj_extent),np.max(xProj_extent),np.min(xProj_extent)]

        vals = np.vstack((xProj,yProj)).T
        
        interpol = np.vstack((Xi,Yi)).T
        dist_not = np.subtract.outer(vals[:,0], interpol[:,0]) #Length of the triangle side from the cell to the point with data 
        dist_one = np.subtract.outer(vals[:,1], interpol[:,1]) #Length of the triangle side from the cell to the point with data 
        distance_matrix = np.hypot(dist_not,dist_one) #euclidean distance, getting the hypotenuse
        
        weights = 1/(distance_matrix**d) #what if distance is 0 --> np.inf? have to account for the pixel underneath
        weights[np.where(np.isinf(weights))] = 1/(1.0E-50) #Making sure to assign the value of the weather station above the pixel directly to the pixel underneath
        weights /= weights.sum(axis = 0) 

        Zi = np.dot(weights.T, z)
        idw_grid = Zi.reshape(num_row,num_col)


        
        elev_dict= GD.finding_data_frm_lookup(zip(xProj, yProj),file_path_elev,idx_list)

        xProj_input=[]
        yProj_input=[]
        e_input = []
        

        for keys in zip(xProj,yProj): # in case there are two stations at the same lat\lon 
            x= keys[0]
            y = keys[1]
            xProj_input.append(x)
            yProj_input.append(y)
            e_input.append(elev_dict[keys])

        source_elev = np.array(e_input)


        vals2 = np.vstack(source_elev).T

        interpol2 = np.vstack(elev_array).T

        dist_not2 = np.subtract.outer(vals2[0], interpol2[0])
        dist_not2 = np.absolute(dist_not2)
        weights2 = 1/(dist_not2**d)

        weights2[np.where(np.isinf(weights2))] = 1 
        weights2 /= weights2.sum(axis = 0)

        fin = 0.8*np.dot(weights.T,z) + 0.2*np.dot(weights2.T,z)
        
        fin = fin.reshape(num_row,num_col)

        #Calc the RMSE, MAE, NSE, and MRAE at the pixel loc
        #Delete at a certain point
        coord_pair = projected_lat_lon[station_name_hold_back]

        x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
        y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)

        interpolated_val = fin[y_orig][x_orig] 

        original_val = Cvar_dict[station_name_hold_back] #Get the original value
        absolute_error = abs(interpolated_val-original_val) #Calc the difference
        absolute_error_dictionary[station_name_hold_back] = absolute_error


    return absolute_error_dictionary


def shuffle_split_IDEW(latlon_dict,Cvar_dict,shapefile,file_path_elev,elev_array,idx_list,d,rep):
    '''Leave-one-out cross-validation procedure for IDEW
    Parameters
        latlon_dict (dict): the latitude and longitudes of the hourly stations, loaded from the 
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
    count = 1
    error_dictionary = {}
    while count <= rep:
        x_origin_list = []
        y_origin_list = [] 

        absolute_error_dictionary = {} 
        station_name_list = []
        projected_lat_lon = {}

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
        z = np.array(Cvar) #what if we add the bounding locations to the array??? ==> that would be extrapolation not interpolation? 

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
        Xi,Yi = Xi.flatten(), Yi.flatten()
        maxmin = [np.min(yProj_extent),np.max(yProj_extent),np.max(xProj_extent),np.min(xProj_extent)]

        vals = np.vstack((xProj,yProj)).T
        
        interpol = np.vstack((Xi,Yi)).T
        dist_not = np.subtract.outer(vals[:,0], interpol[:,0]) #Length of the triangle side from the cell to the point with data 
        dist_one = np.subtract.outer(vals[:,1], interpol[:,1]) #Length of the triangle side from the cell to the point with data 
        distance_matrix = np.hypot(dist_not,dist_one) #euclidean distance, getting the hypotenuse
        
        weights = 1/(distance_matrix**d) #what if distance is 0 --> np.inf? have to account for the pixel underneath
        weights[np.where(np.isinf(weights))] = 1/(1.0E-50) #Making sure to assign the value of the weather station above the pixel directly to the pixel underneath
        weights /= weights.sum(axis = 0) 

        Zi = np.dot(weights.T, z)
        idw_grid = Zi.reshape(num_row,num_col)


        
        elev_dict= GD.finding_data_frm_lookup(zip(xProj, yProj),file_path_elev,idx_list)

        xProj_input=[]
        yProj_input=[]
        e_input = []
        

        for keys in zip(xProj,yProj): # in case there are two stations at the same lat\lon 
            x= keys[0]
            y = keys[1]
            xProj_input.append(x)
            yProj_input.append(y)
            e_input.append(elev_dict[keys])

        source_elev = np.array(e_input)


        vals2 = np.vstack(source_elev).T

        interpol2 = np.vstack(elev_array).T

        dist_not2 = np.subtract.outer(vals2[0], interpol2[0])
        dist_not2 = np.absolute(dist_not2)
        weights2 = 1/(dist_not2**d)

        weights2[np.where(np.isinf(weights2))] = 1 
        weights2 /= weights2.sum(axis = 0)

        fin = 0.8*np.dot(weights.T,z) + 0.2*np.dot(weights2.T,z)
        
        fin = fin.reshape(num_row,num_col)

        #Calc the RMSE, MAE, NSE, and MRAE at the pixel loc
        #Delete at a certain point
        for statLoc in test_stations: 
            coord_pair = projected_lat_lon[statLoc]

            x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
            y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
            x_origin_list.append(x_orig)
            y_origin_list.append(y_orig)

            interpolated_val = fin[y_orig][x_orig] 

            original_val = Cvar_dict[statLoc]
            absolute_error = abs(interpolated_val-original_val)
            absolute_error_dictionary[statLoc] = absolute_error

        error_dictionary[count]= sum(absolute_error_dictionary.values())/len(absolute_error_dictionary.values()) #average of all the withheld stations
        count+=1

    overall_error = sum(error_dictionary.values())/rep
    
    return overall_error


def spatial_kfold_IDEW(loc_dict,Cvar_dict,shapefile,file_path_elev,elev_array,idx_list,d):
    '''Spatially blocked k-folds cross-validation procedure for IDEW
    Parameters
        loc_dict (dict): the latitude and longitudes of the hourly stations, loaded from the 
        .json file
        Cvar_dict (dict): dictionary of weather variable values for each station 
        shapefile (str): path to the study area shapefile 
        file_path_elev (str): file path to the elevation lookup file 
        elev_array (np_array): the elevation array for the study area 
        idx_list (list): the index of the elevation data column in the lookup file 
        d (int): the weighting function for IDW interpolation 
    Returns 
        overall_error (float): MAE average of all the replications
    '''
    groups_complete = [] #If not using replacement, keep a record of what we have done 
    error_dictionary = {} 

    x_origin_list = []
    y_origin_list = [] 

    absolute_error_dictionary = {} 
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
    z = np.array(Cvar) #what if we add the bounding locations to the array??? ==> that would be extrapolation not interpolation? 

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
    Xi,Yi = Xi.flatten(), Yi.flatten()
    maxmin = [np.min(yProj_extent),np.max(yProj_extent),np.max(xProj_extent),np.min(xProj_extent)]

    vals = np.vstack((xProj,yProj)).T
    
    interpol = np.vstack((Xi,Yi)).T
    dist_not = np.subtract.outer(vals[:,0], interpol[:,0]) #Length of the triangle side from the cell to the point with data 
    dist_one = np.subtract.outer(vals[:,1], interpol[:,1]) #Length of the triangle side from the cell to the point with data 
    distance_matrix = np.hypot(dist_not,dist_one) #euclidean distance, getting the hypotenuse
    
    weights = 1/(distance_matrix**d) #what if distance is 0 --> np.inf? have to account for the pixel underneath
    weights[np.where(np.isinf(weights))] = 1/(1.0E-50) #Making sure to assign the value of the weather station above the pixel directly to the pixel underneath
    weights /= weights.sum(axis = 0) 

    Zi = np.dot(weights.T, z)
    idw_grid = Zi.reshape(num_row,num_col)


    
    elev_dict= GD.finding_data_frm_lookup(zip(xProj, yProj),file_path_elev,idx_list)

    xProj_input=[]
    yProj_input=[]
    e_input = []
    

    for keys in zip(xProj,yProj): # in case there are two stations at the same lat\lon 
        x= keys[0]
        y = keys[1]
        xProj_input.append(x)
        yProj_input.append(y)
        e_input.append(elev_dict[keys])

    source_elev = np.array(e_input)


    vals2 = np.vstack(source_elev).T

    interpol2 = np.vstack(elev_array).T

    dist_not2 = np.subtract.outer(vals2[0], interpol2[0])
    dist_not2 = np.absolute(dist_not2)
    weights2 = 1/(dist_not2**d)

    weights2[np.where(np.isinf(weights2))] = 1 
    weights2 /= weights2.sum(axis = 0)

    fin = 0.8*np.dot(weights.T,z) + 0.2*np.dot(weights2.T,z)
    
    fin = fin.reshape(num_row,num_col)

    #Calc the RMSE, MAE, NSE, and MRAE at the pixel loc
    #Delete at a certain point
    for statLoc in station_list: 
        coord_pair = projected_lat_lon[statLoc]

        x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
        y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)

        interpolated_val = fin[y_orig][x_orig] 

        original_val = Cvar_dict[statLoc]
        absolute_error = abs(interpolated_val-original_val)
        absolute_error_dictionary[statLoc] = absolute_error



    MAE= sum(absolute_error_dictionary.values())/len(absolute_error_dictionary.values()) #average of all the withheld stations
     
    return blocknum,MAE        
