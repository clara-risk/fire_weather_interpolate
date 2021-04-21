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

# import
import Eval as Eval
import cluster_3d as c3d
import get_data as GD
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit
import geopandas as gpd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import statistics

import warnings
# Runtime warning suppress, this suppresses the /0 warning
warnings.filterwarnings("ignore")


# functions
def IDEW(latlon_dict, Cvar_dict, input_date, var_name, shapefile, show, file_path_elev,
         idx_list, d):
    '''Inverse distance elevation weighting
    Parameters
    ----------
         latlon_dict : dictionary
              the latitude and longitudes of the stations
         Cvar_dict : dictionary
              dictionary of weather variable values for each station
         input_date : string
              the date you want to interpolate for
         var_name : string
              the name of the variable you are interpolating
         shapefile : string
              path to the study area shapefile, including its name
         show : bool
              whether you want to plot a map
         file_path_elev : string
              path to the elevation lookup file
         idx_list : int
              position of the elevation column in the lookup file
         d : int
              the weighting for IDW interpolation

    Returns
    ----------
         ndarray
              - the array of values for the interpolated surface
         list
              - the bounds of the array surface, for use in other functions
         ndarray
              - elevation array (for use in the random forest module 
     '''

    # Input: lat lon of station, variable (start day, rainfall, etc), date of interest,variable name (for plotting), show (bool true/false), file path to elevation lookup file
    # idx_list (for the column containing the elevation data), d is the power applied to get the weight
    lat = []  # Initialize empty lists to store data
    lon = []
    Cvar = []
    for station_name in Cvar_dict.keys():  # Loop through the list of stations
        if station_name in latlon_dict.keys():  # Make sure the station is present in the latlon dict
            loc = latlon_dict[station_name]
            latitude = loc[0]
            longitude = loc[1]
            cvar_val = Cvar_dict[station_name]
            lat.append(float(latitude))
            lon.append(float(longitude))
            Cvar.append(cvar_val)
    y = np.array(lat)  # Convert to a numpy array for faster processing speed
    x = np.array(lon)
    z = np.array(Cvar)

    na_map = gpd.read_file(shapefile)
    bounds = na_map.bounds  # Get the bounding box of the shapefile
    xmax = bounds['maxx']
    xmin = bounds['minx']
    ymax = bounds['maxy']
    ymin = bounds['miny']
    pixelHeight = 10000  # We want a 10 by 10 pixel, or as close as we can get
    pixelWidth = 10000

    # Calculate the number of rows cols to fill the bounding box at that resolution
    num_col = int((xmax - xmin) / pixelHeight)
    num_row = int((ymax - ymin) / pixelWidth)

    # We need to project to a projected system before making distance matrix
    # We dont know but assume NAD83
    source_proj = pyproj.Proj(proj='latlong', datum='NAD83')
    xProj, yProj = pyproj.Proj('esri:102001')(
        x, y)  # Convert to Canada Albers Equal Area

    # Add the bounding box coords to the dataset so we can extrapolate the interpolation to cover whole area
    yProj_extent = np.append(yProj, [bounds['maxy'], bounds['miny']])
    xProj_extent = np.append(xProj, [bounds['maxx'], bounds['minx']])

    # Get the value for lat lon in each cell we just made
    Yi = np.linspace(np.min(yProj_extent), np.max(yProj_extent), num_row)
    Xi = np.linspace(np.min(xProj_extent), np.max(xProj_extent), num_col)

    # Make a rectangular grid (because eventually we will map the values)
    Xi, Yi = np.meshgrid(Xi, Yi)
    # Then we flatten the arrays for easier processing
    Xi, Yi = Xi.flatten(), Yi.flatten()
    maxmin = [np.min(yProj_extent), np.max(yProj_extent), np.max(xProj_extent), np.min(
        xProj_extent)]  # We will later return this for use in other functions

    # vertically stack station x and y vals and then transpose them so they are in pairs
    vals = np.vstack((xProj, yProj)).T

    # Do the same thing for the grid x and y vals
    interpol = np.vstack((Xi, Yi)).T
    # Length of the triangle side from the cell to the point with data
    dist_not = np.subtract.outer(vals[:, 0], interpol[:, 0])
    # Length of the triangle side from the cell to the point with data
    dist_one = np.subtract.outer(vals[:, 1], interpol[:, 1])
    # Euclidean distance, getting the hypotenuse
    distance_matrix = np.hypot(dist_not, dist_one)

    # what if distance is 0 --> np.inf? have to account for the pixel underneath
    weights = 1/(distance_matrix**d)
    # Making sure to assign the value of the weather station above the pixel directly to the pixel underneath
    weights[np.where(np.isinf(weights))] = 1/(1.0E-50)
    weights /= weights.sum(axis=0)  # The weights must add up to 0

    # Take the dot product of the weights and the values, in this case the dot product is the sum product over the last axis of Weights.T and z
    Zi = np.dot(weights.T, z)

    # reshape the array into the proper format for the map
    idw_grid = Zi.reshape(num_row, num_col)

    # Elevation weights
    # Lon (X) goes in first for a REASON. It has to do with order in the lookup file.
    # Preparing the coordinates to send to the function that will get the elevation grid
    concat = np.array((Xi.flatten(), Yi.flatten())).T
    send_to_list = concat.tolist()
    # The elevation function takes a tuple
    send_to_tuple = [tuple(x) for x in send_to_list]

    Xi1_grd = []
    Yi1_grd = []
    elev_grd = []
    # Get the elevations from the lookup file
    elev_grd_dict = GD.finding_data_frm_lookup(
        send_to_tuple, file_path_elev, idx_list)

    for keys in elev_grd_dict.keys():  # The keys are each lat lon pair
        x = keys[0]
        y = keys[1]
        Xi1_grd.append(x)
        Yi1_grd.append(y)
        # Append the elevation data to the empty list
        elev_grd.append(elev_grd_dict[keys])

    elev_array = np.array(elev_grd)  # make an elevation array

    elev_dict = GD.finding_data_frm_lookup(zip(
        xProj, yProj), file_path_elev, idx_list)  # Get the elevations for the stations

    xProj_input = []
    yProj_input = []
    e_input = []

    for keys in zip(xProj, yProj):  # Repeat process for just the stations not the whole grid
        x = keys[0]
        y = keys[1]
        xProj_input.append(x)
        yProj_input.append(y)
        e_input.append(elev_dict[keys])

    source_elev = np.array(e_input)

    vals2 = np.vstack(source_elev).T

    interpol2 = np.vstack(elev_array).T

    # Get distance in terms of the elevation (vertical distance) from the station to the point to be interpolated
    dist_not2 = np.subtract.outer(vals2[0], interpol2[0])
    # Take the absolute value, we just care about what is the difference
    dist_not2 = np.absolute(dist_not2)
    weights2 = 1/(dist_not2**d)  # Get the inverse distance weight
    # In the case of no elevation change
    weights2[np.where(np.isinf(weights2))] = 1
    weights2 /= weights2.sum(axis=0)  # Make weights add up to 1

    # Weight distance as 0.8 and elevation as 0.2
    fin = 0.8*np.dot(weights.T, z) + 0.2*np.dot(weights2.T, z)

    idew_grid = fin.reshape(num_row, num_col)  # Reshape the final array

    if show:  # Plot if show == True
        fig, ax = plt.subplots(figsize=(15, 15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)

        plt.imshow(elev_array.reshape(num_row, num_col), extent=(xProj_extent.min(
        )-1, xProj_extent.max()+1, yProj_extent.max()-1, yProj_extent.min()+1))
        na_map.plot(ax=ax, color='white', edgecolor='k',
                    linewidth=2, zorder=10, alpha=0.1)

        plt.scatter(xProj, yProj, c=z, edgecolors='k')

        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label(var_name)

        title = 'IDEW Interpolation for %s on %s' % (var_name, input_date)
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        plt.show()

    return idew_grid, maxmin, elev_array


def cross_validate_IDEW(latlon_dict, Cvar_dict, shapefile, file_path_elev, elev_array,
                        idx_list, d):
    '''Leave-one-out cross-validation procedure for IDEW
    Parameters
    ----------
         latlon_dict : dictionary
              the latitude and longitudes of the stations
         Cvar_dict : dictionary
              dictionary of weather variable values for each station
         shapefile : string
              path to the study area shapefile, including its name
         file_path_elev : string
              path to the elevation lookup file
         elev_array : ndarray
              array for elevation, create using IDEW interpolation (this is a trick to speed up code)
         idx_list : int
              position of the elevation column in the lookup file
         d : int
              the weighting for IDW interpolation            
    Returns
    ----------
         dictionary
              - a dictionary of the absolute error at each station when it was left out
    '''
    x_origin_list = []
    y_origin_list = []

    absolute_error_dictionary = {}  # for plotting
    station_name_list = []
    projected_lat_lon = {}

    for station_name in Cvar_dict.keys():
        if station_name in latlon_dict.keys():
            station_name_list.append(station_name)

            loc = latlon_dict[station_name]
            latitude = loc[0]
            longitude = loc[1]
            Plat, Plon = pyproj.Proj('esri:102001')(longitude, latitude)
            Plat = float(Plat)
            Plon = float(Plon)
            projected_lat_lon[station_name] = [Plat, Plon]

    # Pre-make the elev_dict to speed up code

    latO = []
    lonO = []
    for station_name in sorted(Cvar_dict.keys()):
        if station_name in latlon_dict.keys():
            loc = latlon_dict[station_name]
            latitude = loc[0]
            longitude = loc[1]
            cvar_val = Cvar_dict[station_name]
            latO.append(float(latitude))
            lonO.append(float(longitude))
        else:
            pass

    yO = np.array(latO)
    xO = np.array(lonO)

    # We need to project to a projected system before making distance matrix
    # We dont know but assume
    source_proj = pyproj.Proj(proj='latlong', datum='NAD83')
    xProjO, yProjO = pyproj.Proj('esri:102001')(xO, yO)
    elev_dict = GD.finding_data_frm_lookup(
        zip(xProjO, yProjO), file_path_elev, idx_list)

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
        # what if we add the bounding locations to the array??? ==> that would be extrapolation not interpolation?
        z = np.array(Cvar)

        na_map = gpd.read_file(shapefile)
        bounds = na_map.bounds
        xmax = bounds['maxx']
        xmin = bounds['minx']
        ymax = bounds['maxy']
        ymin = bounds['miny']
        pixelHeight = 10000
        pixelWidth = 10000

        num_col = int((xmax - xmin) / pixelHeight)
        num_row = int((ymax - ymin) / pixelWidth)

        # We need to project to a projected system before making distance matrix
        # We dont know but assume
        source_proj = pyproj.Proj(proj='latlong', datum='NAD83')
        xProj, yProj = pyproj.Proj('esri:102001')(x, y)

        yProj_extent = np.append(yProj, [bounds['maxy'], bounds['miny']])
        xProj_extent = np.append(xProj, [bounds['maxx'], bounds['minx']])

        Yi = np.linspace(np.min(yProj_extent), np.max(yProj_extent), num_row)
        Xi = np.linspace(np.min(xProj_extent), np.max(xProj_extent), num_col)

        Xi, Yi = np.meshgrid(Xi, Yi)
        Xi, Yi = Xi.flatten(), Yi.flatten()
        maxmin = [np.min(yProj_extent), np.max(yProj_extent),
                  np.max(xProj_extent), np.min(xProj_extent)]

        vals = np.vstack((xProj, yProj)).T

        interpol = np.vstack((Xi, Yi)).T
        # Length of the triangle side from the cell to the point with data
        dist_not = np.subtract.outer(vals[:, 0], interpol[:, 0])
        # Length of the triangle side from the cell to the point with data
        dist_one = np.subtract.outer(vals[:, 1], interpol[:, 1])
        # euclidean distance, getting the hypotenuse
        distance_matrix = np.hypot(dist_not, dist_one)

        # what if distance is 0 --> np.inf? have to account for the pixel underneath
        weights = 1/(distance_matrix**d)
        # Making sure to assign the value of the weather station above the pixel directly to the pixel underneath
        weights[np.where(np.isinf(weights))] = 1/(1.0E-50)
        weights /= weights.sum(axis=0)

        Zi = np.dot(weights.T, z)
        idw_grid = Zi.reshape(num_row, num_col)

        #elev_dict= GD.finding_data_frm_lookup(zip(xProj, yProj),file_path_elev,idx_list)

        xProj_input = []
        yProj_input = []
        e_input = []

        for keys in zip(xProj, yProj):  # in case there are two stations at the same lat\lon
            x = keys[0]
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
        weights2 /= weights2.sum(axis=0)

        fin = 0.8*np.dot(weights.T, z) + 0.2*np.dot(weights2.T, z)

        fin = fin.reshape(num_row, num_col)

        # Calc the RMSE, MAE, NSE, and MRAE at the pixel loc
        # Delete at a certain point
        coord_pair = projected_lat_lon[station_name_hold_back]

        x_orig = int(
            (coord_pair[0] - float(bounds['minx']))/pixelHeight)  # lon
        y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth)  # lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)

        interpolated_val = fin[y_orig][x_orig]

        # Get the original value
        original_val = Cvar_dict[station_name_hold_back]
        # Calc the difference
        absolute_error = abs(interpolated_val-original_val)
        absolute_error_dictionary[station_name_hold_back] = absolute_error

    return absolute_error_dictionary


def shuffle_split_IDEW(latlon_dict, Cvar_dict, shapefile, file_path_elev, elev_array, idx_list,
                       d, rep):
    '''Shuffle-split cross-validation with 50/50 training test split
   Parameters
   ----------
        loc_dict : dictionary
             the latitude and longitudes of the daily/hourly stations
        Cvar_dict : dictionary
             dictionary of weather variable values for each station
        shapefile : string
             path to the study area shapefile
        file_path_elev : string
            path to the elevation lookup file
        elev_array : ndarray
            array for elevation, create using IDEW interpolation (this is a trick to speed up code)
        idx_list : int
            position of the elevation column in the lookup file
        d : int
             the weighting for IDW interpolation
        rep : int
             number of replications
        show : bool
             if you want to show a map of the clusters
   Returns
   ----------
        float
             - MAE estimate for entire surface (average of replications)
   '''
    count = 1
    error_dictionary = {}
    while count <= rep:
        x_origin_list = []
        y_origin_list = []

        absolute_error_dictionary = {}
        station_name_list = []
        projected_lat_lon = {}

        # we can't just use Cvar_dict.keys() because some stations do not have valid lat/lon
        stations_input = []
        for station_code in Cvar_dict.keys():
            if station_code in latlon_dict.keys():
                stations_input.append(station_code)
        # Split the stations in two
        stations = np.array(stations_input)
        # Won't be exactly 50/50 if uneven num stations
        splits = ShuffleSplit(n_splits=1, train_size=.5)

        for train_index, test_index in splits.split(stations):

            train_stations = stations[train_index]
            # print(train_stations)
            test_stations = stations[test_index]
            # print(test_stations)

       # They can't overlap

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
                Plat, Plon = pyproj.Proj('esri:102001')(longitude, latitude)
                Plat = float(Plat)
                Plon = float(Plon)
                projected_lat_lon[station_name] = [Plat, Plon]

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
        # what if we add the bounding locations to the array??? ==> that would be extrapolation not interpolation?
        z = np.array(Cvar)

        na_map = gpd.read_file(shapefile)
        bounds = na_map.bounds
        xmax = bounds['maxx']
        xmin = bounds['minx']
        ymax = bounds['maxy']
        ymin = bounds['miny']
        pixelHeight = 10000
        pixelWidth = 10000

        num_col = int((xmax - xmin) / pixelHeight)
        num_row = int((ymax - ymin) / pixelWidth)

        # We need to project to a projected system before making distance matrix
        # We dont know but assume
        source_proj = pyproj.Proj(proj='latlong', datum='NAD83')
        xProj, yProj = pyproj.Proj('esri:102001')(x, y)

        yProj_extent = np.append(yProj, [bounds['maxy'], bounds['miny']])
        xProj_extent = np.append(xProj, [bounds['maxx'], bounds['minx']])

        Yi = np.linspace(np.min(yProj_extent), np.max(yProj_extent), num_row)
        Xi = np.linspace(np.min(xProj_extent), np.max(xProj_extent), num_col)

        Xi, Yi = np.meshgrid(Xi, Yi)
        Xi, Yi = Xi.flatten(), Yi.flatten()
        maxmin = [np.min(yProj_extent), np.max(yProj_extent),
                  np.max(xProj_extent), np.min(xProj_extent)]

        vals = np.vstack((xProj, yProj)).T

        interpol = np.vstack((Xi, Yi)).T
        # Length of the triangle side from the cell to the point with data
        dist_not = np.subtract.outer(vals[:, 0], interpol[:, 0])
        # Length of the triangle side from the cell to the point with data
        dist_one = np.subtract.outer(vals[:, 1], interpol[:, 1])
        # euclidean distance, getting the hypotenuse
        distance_matrix = np.hypot(dist_not, dist_one)

        # what if distance is 0 --> np.inf? have to account for the pixel underneath
        weights = 1/(distance_matrix**d)
        # Making sure to assign the value of the weather station above the pixel directly to the pixel underneath
        weights[np.where(np.isinf(weights))] = 1/(1.0E-50)
        weights /= weights.sum(axis=0)

        Zi = np.dot(weights.T, z)
        idw_grid = Zi.reshape(num_row, num_col)

        elev_dict = GD.finding_data_frm_lookup(
            zip(xProj, yProj), file_path_elev, idx_list)

        xProj_input = []
        yProj_input = []
        e_input = []

        for keys in zip(xProj, yProj):  # in case there are two stations at the same lat\lon
            x = keys[0]
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
        weights2 /= weights2.sum(axis=0)

        fin = 0.8*np.dot(weights.T, z) + 0.2*np.dot(weights2.T, z)

        fin = fin.reshape(num_row, num_col)

        # Calc the RMSE, MAE, NSE, and MRAE at the pixel loc
        # Delete at a certain point
        for statLoc in test_stations:
            coord_pair = projected_lat_lon[statLoc]

            x_orig = int(
                (coord_pair[0] - float(bounds['minx']))/pixelHeight)  # lon
            y_orig = int(
                (coord_pair[1] - float(bounds['miny']))/pixelWidth)  # lat
            x_origin_list.append(x_orig)
            y_origin_list.append(y_orig)

            interpolated_val = fin[y_orig][x_orig]

            original_val = Cvar_dict[statLoc]
            absolute_error = abs(interpolated_val-original_val)
            absolute_error_dictionary[statLoc] = absolute_error

        error_dictionary[count] = sum(absolute_error_dictionary.values(
        ))/len(absolute_error_dictionary.values())  # average of all the withheld stations
        count += 1

    overall_error = sum(error_dictionary.values())/rep

    return overall_error


def spatial_kfold_IDEW(loc_dict, Cvar_dict, shapefile, file_path_elev, elev_array, idx_list, d, clusterNum):
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
        clusterNum (int): the number of clusters that the user wants to use 
    Returns 
        overall_error (float): MAE average of all the replications
    '''
    groups_complete = []  # If not using replacement, keep a record of what we have done
    error_dictionary = {}

    x_origin_list = []
    y_origin_list = []

    absolute_error_dictionary = {}
    projected_lat_lon = {}

    cluster = c3d.spatial_cluster(
        loc_dict, Cvar_dict, shapefile, clusterNum, file_path_elev, idx_list, False, False, False)

    for group in cluster.values():
        if group not in groups_complete:
            station_list = [k for k, v in cluster.items() if v == group]
            groups_complete.append(group)

    for station_name in Cvar_dict.keys():
        if station_name in loc_dict.keys():

            loc = loc_dict[station_name]
            latitude = loc[0]
            longitude = loc[1]
            Plat, Plon = pyproj.Proj('esri:102001')(longitude, latitude)
            Plat = float(Plat)
            Plon = float(Plon)
            projected_lat_lon[station_name] = [Plat, Plon]

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
    # what if we add the bounding locations to the array??? ==> that would be extrapolation not interpolation?
    z = np.array(Cvar)

    na_map = gpd.read_file(shapefile)
    bounds = na_map.bounds
    xmax = bounds['maxx']
    xmin = bounds['minx']
    ymax = bounds['maxy']
    ymin = bounds['miny']
    pixelHeight = 10000
    pixelWidth = 10000

    num_col = int((xmax - xmin) / pixelHeight)
    num_row = int((ymax - ymin) / pixelWidth)

    # We need to project to a projected system before making distance matrix
    # We dont know but assume
    source_proj = pyproj.Proj(proj='latlong', datum='NAD83')
    xProj, yProj = pyproj.Proj('esri:102001')(x, y)

    yProj_extent = np.append(yProj, [bounds['maxy'], bounds['miny']])
    xProj_extent = np.append(xProj, [bounds['maxx'], bounds['minx']])

    Yi = np.linspace(np.min(yProj_extent), np.max(yProj_extent), num_row)
    Xi = np.linspace(np.min(xProj_extent), np.max(xProj_extent), num_col)

    Xi, Yi = np.meshgrid(Xi, Yi)
    Xi, Yi = Xi.flatten(), Yi.flatten()
    maxmin = [np.min(yProj_extent), np.max(yProj_extent),
              np.max(xProj_extent), np.min(xProj_extent)]

    vals = np.vstack((xProj, yProj)).T

    interpol = np.vstack((Xi, Yi)).T
    # Length of the triangle side from the cell to the point with data
    dist_not = np.subtract.outer(vals[:, 0], interpol[:, 0])
    # Length of the triangle side from the cell to the point with data
    dist_one = np.subtract.outer(vals[:, 1], interpol[:, 1])
    # euclidean distance, getting the hypotenuse
    distance_matrix = np.hypot(dist_not, dist_one)

    # what if distance is 0 --> np.inf? have to account for the pixel underneath
    weights = 1/(distance_matrix**d)
    # Making sure to assign the value of the weather station above the pixel directly to the pixel underneath
    weights[np.where(np.isinf(weights))] = 1/(1.0E-50)
    weights /= weights.sum(axis=0)

    Zi = np.dot(weights.T, z)
    idw_grid = Zi.reshape(num_row, num_col)

    elev_dict = GD.finding_data_frm_lookup(
        zip(xProj, yProj), file_path_elev, idx_list)

    xProj_input = []
    yProj_input = []
    e_input = []

    for keys in zip(xProj, yProj):  # in case there are two stations at the same lat\lon
        x = keys[0]
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
    weights2 /= weights2.sum(axis=0)

    fin = 0.8*np.dot(weights.T, z) + 0.2*np.dot(weights2.T, z)

    fin = fin.reshape(num_row, num_col)

    # Calc the RMSE, MAE, NSE, and MRAE at the pixel loc
    # Delete at a certain point
    for statLoc in station_list:
        coord_pair = projected_lat_lon[statLoc]

        x_orig = int(
            (coord_pair[0] - float(bounds['minx']))/pixelHeight)  # lon
        y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth)  # lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)

        interpolated_val = fin[y_orig][x_orig]

        original_val = Cvar_dict[statLoc]
        absolute_error = abs(interpolated_val-original_val)
        absolute_error_dictionary[statLoc] = absolute_error

    # average of all the withheld stations
    MAE = sum(absolute_error_dictionary.values()) / \
        len(absolute_error_dictionary.values())

    return clusterNum, MAE


def select_block_size_IDEW(nruns, group_type, loc_dict, Cvar_dict, idw_example_grid, shapefile, file_path_elev, idx_list, elev_array, d):
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

    # Get group dictionaries

    if group_type == 'blocks':

        folds25 = mbk.make_block(idw_example_grid, 25)
        dictionaryGroups25 = mbk.sorting_stations(
            folds25, shapefile, Cvar_dict)
        folds16 = mbk.make_block(idw_example_grid, 16)
        dictionaryGroups16 = mbk.sorting_stations(
            folds16, shapefile, Cvar_dict)
        folds9 = mbk.make_block(idw_example_grid, 9)
        dictionaryGroups9 = mbk.sorting_stations(folds9, shapefile, Cvar_dict)

    elif group_type == 'clusters':

        dictionaryGroups25 = c3d.spatial_cluster(
            loc_dict, Cvar_dict, shapefile, 25, file_path_elev, idx_list, False, False, False)
        dictionaryGroups16 = c3d.spatial_cluster(
            loc_dict, Cvar_dict, shapefile, 16, file_path_elev, idx_list, False, False, False)
        dictionaryGroups9 = c3d.spatial_cluster(
            loc_dict, Cvar_dict, shapefile, 9, file_path_elev, idx_list, False, False, False)

    else:
        print('Thats not a valid group type')
        sys.exit()

    block25_error = []
    block16_error = []
    block9_error = []
    if nruns <= 1:
        print('That is not enough runs to calculate the standard deviation!')
        sys.exit()

    for n in range(0, nruns):

        block25 = spatial_groups_IDEW(idw_example_grid, loc_dict, Cvar_dict, shapefile,
                                      d, 25, 5, True, dictionaryGroups25, file_path_elev, idx_list, elev_array)
        block25_error.append(block25)

        block16 = spatial_groups_IDEW(idw_example_grid, loc_dict, Cvar_dict, shapefile,
                                      d, 16, 8, True, dictionaryGroups16, file_path_elev, idx_list, elev_array)
        block16_error.append(block16)

        block9 = spatial_groups_IDEW(idw_example_grid, loc_dict, Cvar_dict, shapefile,
                                     d, 9, 14, True, dictionaryGroups9, file_path_elev, idx_list, elev_array)
        block9_error.append(block9)

    stdev25 = statistics.stdev(block25_error)
    stdev16 = statistics.stdev(block16_error)
    stdev9 = statistics.stdev(block9_error)

    list_stdev = [stdev25, stdev16, stdev9]
    list_block_name = [25, 16, 9]
    list_error = [block25_error, block16_error, block9_error]
    index_min = list_stdev.index(min(list_stdev))
    lowest_stdev = list_block_name[index_min]

    ave_MAE = sum(list_error[index_min])/len(list_error[index_min])

    print(lowest_stdev)
    print(ave_MAE)
    return lowest_stdev, ave_MAE


def spatial_groups_IDEW(idw_example_grid, loc_dict, Cvar_dict, shapefile, d, blocknum, nfolds, replacement, dictionary_Groups, file_path_elev, idx_list, elev_array):
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
    error_dictionary (dict): a dictionary of the absolute error at each fold when it was left out
    '''
    station_list_used = []  # If not using replacement, keep a record of what we have done
    count = 1
    error_dictionary = {}

    # Premake elevation dictionary to speed up code

    latO = []
    lonO = []
    for station_name in sorted(Cvar_dict.keys()):
        if station_name in loc_dict.keys():
            loc = loc_dict[station_name]
            latitude = loc[0]
            longitude = loc[1]
            cvar_val = Cvar_dict[station_name]
            latO.append(float(latitude))
            lonO.append(float(longitude))
        else:
            pass

    yO = np.array(latO)
    xO = np.array(lonO)

    # We need to project to a projected system before making distance matrix
    # We dont know but assume
    source_proj = pyproj.Proj(proj='latlong', datum='NAD83')
    xProjO, yProjO = pyproj.Proj('esri:102001')(xO, yO)
    elev_dict = GD.finding_data_frm_lookup(
        zip(xProjO, yProjO), file_path_elev, idx_list)
    while count <= nfolds:
        x_origin_list = []
        y_origin_list = []

        absolute_error_dictionary = {}
        projected_lat_lon = {}

        station_list = Eval.select_random_station(
            dictionary_Groups, blocknum, replacement, station_list_used).values()
        if replacement == False:
            station_list_used.append(list(station_list))
        # print(station_list_used)

        for station_name in Cvar_dict.keys():

            if station_name in loc_dict.keys():

                loc = loc_dict[station_name]
                latitude = loc[0]
                longitude = loc[1]
                Plat, Plon = pyproj.Proj('esri:102001')(longitude, latitude)
                Plat = float(Plat)
                Plon = float(Plon)
                projected_lat_lon[station_name] = [Plat, Plon]

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
        # what if we add the bounding locations to the array??? ==> that would be extrapolation not interpolation?
        z = np.array(Cvar)

        na_map = gpd.read_file(shapefile)
        bounds = na_map.bounds
        xmax = bounds['maxx']
        xmin = bounds['minx']
        ymax = bounds['maxy']
        ymin = bounds['miny']
        pixelHeight = 10000
        pixelWidth = 10000

        num_col = int((xmax - xmin) / pixelHeight)
        num_row = int((ymax - ymin) / pixelWidth)

        # We need to project to a projected system before making distance matrix
        # We dont know but assume
        source_proj = pyproj.Proj(proj='latlong', datum='NAD83')
        xProj, yProj = pyproj.Proj('esri:102001')(x, y)

        yProj_extent = np.append(yProj, [bounds['maxy'], bounds['miny']])
        xProj_extent = np.append(xProj, [bounds['maxx'], bounds['minx']])

        Yi = np.linspace(np.min(yProj_extent), np.max(yProj_extent), num_row)
        Xi = np.linspace(np.min(xProj_extent), np.max(xProj_extent), num_col)

        Xi, Yi = np.meshgrid(Xi, Yi)
        Xi, Yi = Xi.flatten(), Yi.flatten()
        maxmin = [np.min(yProj_extent), np.max(yProj_extent),
                  np.max(xProj_extent), np.min(xProj_extent)]

        vals = np.vstack((xProj, yProj)).T

        interpol = np.vstack((Xi, Yi)).T
        # Length of the triangle side from the cell to the point with data
        dist_not = np.subtract.outer(vals[:, 0], interpol[:, 0])
        # Length of the triangle side from the cell to the point with data
        dist_one = np.subtract.outer(vals[:, 1], interpol[:, 1])
        # euclidean distance, getting the hypotenuse
        distance_matrix = np.hypot(dist_not, dist_one)

        # what if distance is 0 --> np.inf? have to account for the pixel underneath
        weights = 1/(distance_matrix**d)
        # Making sure to assign the value of the weather station above the pixel directly to the pixel underneath
        weights[np.where(np.isinf(weights))] = 1/(1.0E-50)
        weights /= weights.sum(axis=0)

        Zi = np.dot(weights.T, z)
        idw_grid = Zi.reshape(num_row, num_col)

        elev_dict = GD.finding_data_frm_lookup(
            zip(xProj, yProj), file_path_elev, idx_list)

        xProj_input = []
        yProj_input = []
        e_input = []

        for keys in zip(xProj, yProj):  # in case there are two stations at the same lat\lon
            x = keys[0]
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
        weights2 /= weights2.sum(axis=0)

        fin = 0.8*np.dot(weights.T, z) + 0.2*np.dot(weights2.T, z)

        fin = fin.reshape(num_row, num_col)

        # Compare at a certain point
        for statLoc in station_list:

            coord_pair = projected_lat_lon[statLoc]

            x_orig = int(
                (coord_pair[0] - float(bounds['minx']))/pixelHeight)  # lon
            y_orig = int(
                (coord_pair[1] - float(bounds['miny']))/pixelWidth)  # lat
            x_origin_list.append(x_orig)
            y_origin_list.append(y_orig)

            interpolated_val = fin[y_orig][x_orig]

            original_val = Cvar_dict[statLoc]
            absolute_error = abs(interpolated_val-original_val)
            absolute_error_dictionary[statLoc] = absolute_error

        error_dictionary[count] = sum(absolute_error_dictionary.values(
        ))/len(absolute_error_dictionary.values())  # average of all the withheld stations
        # print(absolute_error_dictionary)
        count += 1
    overall_error = sum(error_dictionary.values()) / \
        nfolds  # average of all the runs
    # print(overall_error)
    return overall_error
