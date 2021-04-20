# coding: utf-8

"""
Summary
-------
Code for interpolating weather data using inverse distance weighting interpolation.

References
----------
Code for IDW modified from https://stackoverflow.com/questions/3104781/inverse-distance-weighted-idw-interpolation-with-python?rq=1 (answer by Joe Kington)
[1] Kington, J. "Inverse Distance Weighted (IDW) Interpolation with Python" https://stackoverflow.com/questions/3104781/inverse-distance-weighted-idw-interpolation-with-python?rq=1. Jun 2010.
Last accessed: 2020-12-26.
"""

# import

import cluster_3d as c3d
import get_data as GD
import Eval as Eval
import make_blocks as mbk
import geopandas as gpd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import os
import sys
import math
import statistics

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

import warnings
# Runtime warning suppress, this suppresses the /0 warning
warnings.filterwarnings("ignore")


def IDW(latlon_dict, Cvar_dict, input_date,
        var_name, shapefile, show, d, expand_area):
    '''Inverse distance weighting interpolation
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
        d : int
             the weighting for IDW interpolation
        expand_area : bool
             function will expand the study area so that more stations are taken into account (200 km)
   Returns
   ----------
        ndarray
             - the array of values for the interpolated surface
        list
             - the bounds of the array surface, for use in other functions
    '''
    plotting_dictionary = {
    }  # if expanding the study area, we need to return a dictionary of the stations used
    lat = []
    lon = []
    Cvar = []

    source_proj = pyproj.Proj(proj='latlong', datum='NAD83')
    na_map = gpd.read_file(shapefile)
    bounds = na_map.bounds
    if expand_area:
        xmax = bounds['maxx'] + 200000
        xmin = bounds['minx'] - 200000
        ymax = bounds['maxy'] + 200000
        ymin = bounds['miny'] - 200000
    else:
        xmax = bounds['maxx']
        xmin = bounds['minx']
        ymax = bounds['maxy']
        ymin = bounds['miny']

    for station_name in Cvar_dict.keys():

        if station_name in latlon_dict.keys():

            loc = latlon_dict[station_name]
            latitude = loc[0]
            longitude = loc[1]
            proj_coord = pyproj.Proj('esri:102001')(
                longitude, latitude)  # Filter out stations outside of grid
            if (proj_coord[1] <= float(ymax[0]) and proj_coord[1] >= float(
                    ymin[0]) and proj_coord[0] <= float(xmax[0]) and proj_coord[0] >= float(xmin[0])):
                cvar_val = Cvar_dict[station_name]
                lat.append(float(latitude))
                lon.append(float(longitude))
                Cvar.append(cvar_val)
                plotting_dictionary[station_name] = cvar_val
    y = np.array(lat)
    x = np.array(lon)
    z = np.array(Cvar)

    pixelHeight = 10000
    pixelWidth = 10000

    num_col = int((xmax - xmin) / pixelHeight)
    num_row = int((ymax - ymin) / pixelWidth)

    # We need to project to a projected system before making distance matrix
    xProj, yProj = pyproj.Proj('esri:102001')(x, y)

    if expand_area:

        yProj_extent = np.append(
            yProj, [bounds['maxy'] + 200000, bounds['miny'] - 200000])
        xProj_extent = np.append(
            xProj, [bounds['maxx'] + 200000, bounds['minx'] - 200000])
    else:
        yProj_extent = np.append(yProj, [bounds['maxy'], bounds['miny']])
        xProj_extent = np.append(xProj, [bounds['maxx'], bounds['minx']])

    Yi = np.linspace(np.min(yProj_extent), np.max(yProj_extent), num_row + 1)
    Xi = np.linspace(np.min(xProj_extent), np.max(xProj_extent), num_col + 1)

    Xi, Yi = np.meshgrid(Xi, Yi)
    Xi, Yi = Xi.flatten(), Yi.flatten()
    maxmin = [
        np.min(yProj_extent),
        np.max(yProj_extent),
        np.max(xProj_extent),
        np.min(xProj_extent)]

    vals = np.vstack((xProj, yProj)).T

    interpol = np.vstack((Xi, Yi)).T
    dist_not = np.subtract.outer(vals[:, 0], interpol[:, 0])
    dist_one = np.subtract.outer(vals[:, 1], interpol[:, 1])
    distance_matrix = np.hypot(dist_not, dist_one)

    weights = 1 / (distance_matrix ** d)
    weights[np.where(np.isinf(weights))] = 1 / (1.0E-50)
    weights /= weights.sum(axis=0)

    Zi = np.dot(weights.T, z)
    # +1 because of lines issue
    idw_grid = Zi.reshape(num_row + 1, num_col + 1)

    if show:
        fig, ax = plt.subplots(figsize=(15, 15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)

        plt.imshow(
            idw_grid,
            extent=(
                xProj_extent.min() - 1,
                xProj_extent.max() + 1,
                yProj_extent.max() - 1,
                yProj_extent.min() + 1))
        na_map.plot(
            ax=ax,
            color='white',
            edgecolor='k',
            linewidth=2,
            zorder=10,
            alpha=0.1)

        plt.scatter(xProj, yProj, c=z, edgecolors='k')

        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label(var_name)

        title = 'IDW Interpolation for %s on %s' % (var_name, input_date)
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        plt.show()

    return idw_grid, maxmin


def cross_validate_IDW(latlon_dict, Cvar_dict, shapefile,
                       d, pass_to_plot, expand_area):
    '''Leave-one-out cross-validation procedure for IDW
    Parameters
    ----------
         latlon_dict : dictionary
              the latitude and longitudes of the stations
         Cvar_dict : dictionary
              dictionary of weather variable values for each station
         shapefile : string
              path to the study area shapefile, including its name
         d : int
              the weighting for IDW interpolation
         pass_to_plot : bool
              whether you will be plotting the error and need a version without absolute value error (i.e. fire season days)
         expand_area : bool
              function will expand the study area so that more stations are taken into account (200 km)
    Returns
    ----------
         dictionary
              - a dictionary of the absolute error at each station when it was left out
         dictionary
              - if pass_to_plot = True, returns a dictionary without the absolute value of the error, for example for plotting fire season error
     '''

    x_origin_list = []
    y_origin_list = []

    absolute_error_dictionary = {}  # for plotting
    no_absolute_value_dict = {}
    station_name_list = []
    projected_lat_lon = {}

    na_map = gpd.read_file(shapefile)
    bounds = na_map.bounds
    if expand_area:
        xmax = bounds['maxx'] + 200000
        xmin = bounds['minx'] - 200000
        ymax = bounds['maxy'] + 200000
        ymin = bounds['miny'] - 200000
    else:
        xmax = bounds['maxx']
        xmin = bounds['minx']
        ymax = bounds['maxy']
        ymin = bounds['miny']

    for station_name in Cvar_dict.keys():

        if station_name in latlon_dict.keys():

            loc = latlon_dict[station_name]
            latitude = loc[0]
            longitude = loc[1]
            Plat, Plon = pyproj.Proj('esri:102001')(longitude, latitude)
            proj_coord = pyproj.Proj('esri:102001')(
                longitude, latitude)  # Filter out stations outside of grid
            if (proj_coord[1] <= float(ymax[0]) and proj_coord[1] >= float(
                    ymin[0]) and proj_coord[0] <= float(xmax[0]) and proj_coord[0] >= float(xmin[0])):
                Plat = float(Plat)
                Plon = float(Plon)
                projected_lat_lon[station_name] = [Plat, Plon]
                # Only append if it falls inside the generated grid
                station_name_list.append(station_name)

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
                    proj_coord = pyproj.Proj('esri:102001')(
                        longitude, latitude)  # Filter out stations outside of grid
                    if (proj_coord[1] <= float(ymax[0]) and proj_coord[1] >= float(
                            ymin[0]) and proj_coord[0] <= float(xmax[0]) and proj_coord[0] >= float(xmin[0])):
                        cvar_val = Cvar_dict[station_name]
                        lat.append(float(latitude))
                        lon.append(float(longitude))
                        Cvar.append(cvar_val)
                    else:

                        pass

        y = np.array(lat)
        x = np.array(lon)
        z = np.array(Cvar)

        pixelHeight = 10000
        pixelWidth = 10000

        num_col = int((xmax - xmin) / pixelHeight) + 1
        num_row = int((ymax - ymin) / pixelWidth) + 1

        # We need to project to a projected system before making distance
        # matrix
        # We dont know but assume
        source_proj = pyproj.Proj(proj='latlong', datum='NAD83')
        xProj, yProj = pyproj.Proj('esri:102001')(x, y)
        if expand_area:
            yProj_extent = np.append(
                yProj, [bounds['maxy'] + 200000, bounds['miny'] - 200000])
            xProj_extent = np.append(
                xProj, [bounds['maxx'] + 200000, bounds['minx'] - 200000])
        else:
            yProj_extent = np.append(yProj, [bounds['maxy'], bounds['miny']])
            xProj_extent = np.append(xProj, [bounds['maxx'], bounds['minx']])

        Yi = np.linspace(
            np.min(yProj_extent),
            np.max(yProj_extent),
            num_row)
        Xi = np.linspace(
            np.min(xProj_extent),
            np.max(xProj_extent),
            num_col)

        Xi, Yi = np.meshgrid(Xi, Yi)
        Xi, Yi = Xi.flatten(), Yi.flatten()
        maxmin = [
            np.min(yProj_extent),
            np.max(yProj_extent),
            np.max(xProj_extent),
            np.min(xProj_extent)]

        vals = np.vstack((xProj, yProj)).T

        interpol = np.vstack((Xi, Yi)).T
        # Length of the triangle side from the cell to the point with data
        dist_not = np.subtract.outer(vals[:, 0], interpol[:, 0])
        # Length of the triangle side from the cell to the point with data
        dist_one = np.subtract.outer(vals[:, 1], interpol[:, 1])
        # euclidean distance, getting the hypotenuse
        distance_matrix = np.hypot(dist_not, dist_one)

        weights = 1 / (distance_matrix**d)

        weights[np.where(np.isinf(weights))] = 1 / (1.0E-50)
        weights /= weights.sum(axis=0)

        Zi = np.dot(weights.T, z)
        idw_grid = Zi.reshape(num_row, num_col)

        # Delete at a certain point
        coord_pair = projected_lat_lon[station_name_hold_back]

        x_orig = int((coord_pair[0] - float(xmin)) / pixelHeight)  # lon
        y_orig = int((coord_pair[1] - float(ymin)) / pixelWidth)  # lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)

        interpolated_val = idw_grid[y_orig][x_orig]

        original_val = Cvar_dict[station_name_hold_back]
        absolute_error = abs(interpolated_val - original_val)
        absolute_error_dictionary[station_name_hold_back] = absolute_error
        no_absolute_value_dict[station_name_hold_back] = interpolated_val - original_val
    if pass_to_plot:
        return absolute_error_dictionary, no_absolute_value_dict
    else:
        return absolute_error_dictionary


def select_block_size_IDW(nruns, group_type, loc_dict, Cvar_dict, idw_example_grid, shapefile, file_path_elev, idx_list, d,
                          cluster_num1, cluster_num2, cluster_num3, expand_area, boreal_shapefile):
    '''Evaluate the standard deviation of MAE values based on consective runs of the cross-valiation, in order to select the block/cluster size
    Parameters
    ----------
         nruns : int
              number of repetitions
         group_type : string
              whether using 'clusters' or 'blocks'
         loc_dict : dictionary
              the latitude and longitudes of the daily/hourly stations
         Cvar_dict : dictionary
              dictionary of weather variable values for each station
         idw_example_grid  : ndarray
              used for reference of study area grid size
         shapefile : string
              path to the study area shapefile
         file_path_elev : string
              path to the elevation lookup file
         idx_list : int
              position of the elevation column in the lookup file
         cluster_num : list
              three cluster numbers to test, for blocking this must be one of three:25, 16, 9
              you can enter 'None' and it will automatically test 25, 16, 9
         expand_area : bool
              expand area by 200km
         boreal_shapefile : string
              path to shapefile with the boreal zone
    Returns
    ----------
         int
              - block/cluster number with lowest stdev
         float
              - average MAE of all the runs for that cluster/block number
     '''

    # Get group dictionaries
    if group_type == 'blocks':

        folds25 = mbk.make_block(idw_example_grid, 25)
        dictionaryGroups25 = mbk.sorting_stations(
            folds25, shapefile, loc_dict, Cvar_dict)
        folds16 = mbk.make_block(idw_example_grid, 16)
        dictionaryGroups16 = mbk.sorting_stations(
            folds16, shapefile, loc_dict, Cvar_dict)
        folds9 = mbk.make_block(idw_example_grid, 9)
        dictionaryGroups9 = mbk.sorting_stations(
            folds9, shapefile, loc_dict, Cvar_dict)

    elif group_type == 'clusters':
        # if we expand the area, re-make the dictionary with only the
        # stations in the boreal region (Project 2)
        if expand_area:
            inBoreal = GD.is_station_in_boreal(
                loc_dict, Cvar_dict, boreal_shapefile)
            # Overwrite cvar_dict
            Cvar_dict = {k: v for k, v in Cvar_dict.items() if k in inBoreal}
            dictionaryGroups25 = c3d.spatial_cluster(loc_dict, Cvar_dict, shapefile, cluster_num1,
                                                     file_path_elev, idx_list, False, False, False)
            dictionaryGroups16 = c3d.spatial_cluster(loc_dict, Cvar_dict, shapefile, cluster_num2,
                                                     file_path_elev, idx_list, False, False, False)
            dictionaryGroups9 = c3d.spatial_cluster(loc_dict, Cvar_dict, shapefile, cluster_num3,
                                                    file_path_elev, idx_list, False, False, False)
        else:
            dictionaryGroups25 = c3d.spatial_cluster(loc_dict, Cvar_dict, shapefile, cluster_num1,
                                                     file_path_elev, idx_list, False, False, False)
            dictionaryGroups16 = c3d.spatial_cluster(loc_dict, Cvar_dict, shapefile, cluster_num2,
                                                     file_path_elev, idx_list, False, False, False)
            dictionaryGroups9 = c3d.spatial_cluster(loc_dict, Cvar_dict, shapefile, cluster_num3,
                                                    file_path_elev, idx_list, False, False, False)
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
        # We want same number of stations selected for each cluster number
        # We need to calculate, 5 folds x 25 clusters = 125 stations; 8
        # folds x 16 clusters = 128 stations, etc.
        target_stations = len(Cvar_dict.keys()) * \
            0.3  # What is 30% of the stations
        fold_num1 = int(round(target_stations / cluster_num1))
        fold_num2 = int(round(target_stations / cluster_num2))
        fold_num3 = int(round(target_stations / cluster_num3))

        # For our first project, this is what we did
        # fold_num1 = 5
        # fold_num2 = 8
        # fold_num3 = 14
        # Just so there is a record of that

        block25 = spatial_groups_IDW(idw_example_grid, loc_dict, Cvar_dict, shapefile, d, cluster_num1, fold_num1,
                                     True, False, dictionaryGroups25, expand_area)
        block25_error.append(block25)

        block16 = spatial_groups_IDW(idw_example_grid, loc_dict, Cvar_dict, shapefile, d, cluster_num2, fold_num2,
                                     True, False, dictionaryGroups16, expand_area)
        block16_error.append(block16)

        block9 = spatial_groups_IDW(idw_example_grid, loc_dict, Cvar_dict, shapefile, d, cluster_num3, fold_num3,
                                    True, False, dictionaryGroups9, expand_area)
        block9_error.append(block9)

    stdev25 = statistics.stdev(block25_error)
    stdev16 = statistics.stdev(block16_error)
    stdev9 = statistics.stdev(block9_error)

    list_stdev = [stdev25, stdev16, stdev9]
    list_block_name = [cluster_num1, cluster_num2, cluster_num3]
    list_error = [block25_error, block16_error, block9_error]
    index_min = list_stdev.index(min(list_stdev))
    lowest_stdev = statistics.stdev(list_error[index_min])

    ave_MAE = sum(list_error[index_min]) / len(list_error[index_min])
    cluster_select = list_block_name[index_min]

    print(list_error[index_min])
    print(ave_MAE)
    print(lowest_stdev)
    print(cluster_select)
    return cluster_select, ave_MAE, lowest_stdev


def spatial_groups_IDW(idw_example_grid, loc_dict, Cvar_dict, shapefile, d, blocknum, nfolds, replacement,
                       show, dictionary_Groups, expand_area):
    '''Stratified shuffle-split cross-validation procedure
    Parameters
    ----------
         idw_example_grid  : ndarray
              used for reference of study area grid size
         loc_dict : dictionary
              the latitude and longitudes of the daily/hourly stations
         Cvar_dict : dictionary
              dictionary of weather variable values for each station
         shapefile : string
              path to the study area shapefile
         d : int
              the weighting for IDW interpolation
         blocknum : int
              number of blocks/clusters
         nfolds : int
              number of folds to create (essentially repetitions)
         replacement : bool
              whether or not to use replacement between folds, should usually be true
         dictionary_Groups : dictionary
              dictionary of what groups (clusters) the stations belong to
         expand_area : bool
              function will expand the study area so that more stations are taken into account (200 km)
    Returns
    ----------
         dictionary
              - a dictionary of the absolute error at each fold when it was left out
    '''
    station_list_used = []  # If not using replacement, keep a record of what we have done
    count = 1
    error_dictionary = {}

    na_map = gpd.read_file(shapefile)
    bounds = na_map.bounds
    if expand_area:
        xmax = bounds['maxx'] + 200000
        xmin = bounds['minx']-200000
        ymax = bounds['maxy'] + 200000
        ymin = bounds['miny'] - 200000
    else:
        xmax = bounds['maxx']
        xmin = bounds['minx']
        ymax = bounds['maxy']
        ymin = bounds['miny']

    while count <= nfolds:
        x_origin_list = []
        y_origin_list = []

        absolute_error_dictionary = {}
        projected_lat_lon = {}

        # if we expand the area, re-select station only if it is in the
        # boreal region

        station_list = Eval.select_random_station(
            dictionary_Groups, blocknum, replacement, station_list_used).values()

        if replacement == False:
            station_list_used.append(list(station_list))

        for station_name in Cvar_dict.keys():

            if station_name in loc_dict.keys():

                loc = loc_dict[station_name]
                latitude = loc[0]
                longitude = loc[1]
                Plat, Plon = pyproj.Proj(
                    'esri:102001')(longitude, latitude)
                Plat = float(Plat)
                Plon = float(Plon)
                # Filter out stations outside of grid
                proj_coord = pyproj.Proj('esri:102001')(longitude, latitude)
                if (proj_coord[1] <= float(ymax[0]) and proj_coord[1] >= float(
                        ymin[0]) and proj_coord[0] <= float(xmax[0]) and proj_coord[0] >= float(xmin[0])):
                    projected_lat_lon[station_name] = [Plat, Plon]

        lat = []
        lon = []
        Cvar = []
        for station_name in sorted(Cvar_dict.keys()):
            if station_name in loc_dict.keys():
                if station_name not in station_list:  # This is the step where we hold back the fold
                    loc = loc_dict[station_name]
                    latitude = loc[0]
                    longitude = loc[1]
                    cvar_val = Cvar_dict[station_name]
                    # Filter out stations outside of grid
                    proj_coord = pyproj.Proj(
                        'esri:102001')(longitude, latitude)
                    if (proj_coord[1] <= float(ymax[0]) and proj_coord[1] >= float(
                            ymin[0]) and proj_coord[0] <= float(xmax[0]) and proj_coord[0] >= float(xmin[0])):
                        lat.append(float(latitude))
                        lon.append(float(longitude))
                        Cvar.append(cvar_val)
                else:
                    pass  # Skip the station

            y = np.array(lat)
            x = np.array(lon)
            z = np.array(Cvar)

            pixelHeight = 10000
            pixelWidth = 10000

            num_col = int((xmax - xmin) / pixelHeight)
            num_row = int((ymax - ymin) / pixelWidth)

            # We need to project to a projected system before making
            # distance matrix
            # We dont know but assume
            source_proj = pyproj.Proj(proj='latlong', datum='NAD83')
            xProj, yProj = pyproj.Proj('esri:102001')(x, y)

            if expand_area:

                yProj_extent = np.append(
                    yProj, [bounds['maxy']+200000, bounds['miny']-200000])
                xProj_extent = np.append(
                    xProj, [bounds['maxx']+200000, bounds['minx']-200000])
            else:
                yProj_extent = np.append(
                    yProj, [bounds['maxy'], bounds['miny']])
                xProj_extent = np.append(
                    xProj, [bounds['maxx'], bounds['minx']])

            Yi = np.linspace(np.min(yProj_extent),
                             np.max(yProj_extent), num_row+1)
            Xi = np.linspace(np.min(xProj_extent),
                             np.max(xProj_extent), num_col+1)

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
            weights = 1 / (distance_matrix**d)
            # Making sure to assign the value of the weather station above the pixel directly to the pixel underneath
            weights[np.where(np.isinf(weights))] = 1 / (1.0E-50)
            weights /= weights.sum(axis=0)

            Zi = np.dot(weights.T, z)
            idw_grid = Zi.reshape(num_row + 1, num_col + 1)
            if show and (count == 1):
                fig, ax = plt.subplots(figsize=(15, 15))
                crs = {'init': 'esri:102001'}

                na_map = gpd.read_file(shapefile)

                im = plt.imshow(folds, extent=(xProj_extent.min(
                )-1, xProj_extent.max()+1, yProj_extent.max()-1, yProj_extent.min()+1), cmap='tab20b')
                na_map.plot(ax=ax, color='white', edgecolor='k',
                            linewidth=2, zorder=10, alpha=0.1)

                plt.scatter(xProj, yProj, c='black', s=2)

                plt.gca().invert_yaxis()
                cbar = plt.colorbar(im, ax=ax, cmap='tab20b')
                cbar.set_label('Block Number')

                title = 'Group selection'
                fig.suptitle(title, fontsize=14)
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')

                plt.show()

            # Compare at a certain point
            for statLoc in station_list:

                coord_pair = projected_lat_lon[statLoc]

                x_orig = int((coord_pair[0] - float(xmin))/pixelHeight)  # lon
                y_orig = int((coord_pair[1] - float(ymin)) / pixelWidth)  # lat
                x_origin_list.append(x_orig)
                y_origin_list.append(y_orig)

                interpolated_val = idw_grid[y_orig][x_orig]

                original_val = Cvar_dict[statLoc]
                absolute_error = abs(interpolated_val - original_val)
                absolute_error_dictionary[statLoc] = absolute_error

            error_dictionary[count] = sum(absolute_error_dictionary.values(
            ))/len(absolute_error_dictionary.values())  # average of all the withheld stations

            count += 1

    overall_error = sum(error_dictionary.values()) / \
        nfolds  # average of all the runs

    return overall_error


def spatial_kfold_idw(idw_example_grid, loc_dict, Cvar_dict, shapefile, d, file_path_elev, idx_list, block_num, return_error):
    '''Spatially blocked k-fold cross-validation procedure for IDW
    Parameters
    ----------
         idw_example_grid  : ndarray
              used for reference of study area grid size
         loc_dict : dictionary
              the latitude and longitudes of the daily/hourly stations
         Cvar_dict : dictionary
              dictionary of weather variable values for each station
         shapefile : string
              path to the study area shapefile
         d : int
              the weighting for IDW interpolation
         file_path_elev : string
              path to the elevation lookup file
         idx_list : int
              position of the elevation column in the lookup file
         block_num : int
              number of blocks/clusters
         return_error : bool
              whether or not to return the error dictionary
    Returns
    ----------
         float
              - MAE estimate for entire surface
         int
              - Return the block number just so we can later write it into the file to keep track
         dictionary
              - if return_error = True, a dictionary of the absolute error at each fold when it was left out
    '''

    groups_complete = []
    error_dictionary = {}
    x_origin_list = []
    y_origin_list = []

    absolute_error_dictionary = {}
    projected_lat_lon = {}

    cluster = c3d.spatial_cluster(
        loc_dict, Cvar_dict, shapefile, block_num, file_path_elev, idx_list, False, False, False)

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
            if station_name not in station_list:  # This is the step where we hold back the fold
                loc = loc_dict[station_name]
                latitude = loc[0]
                longitude = loc[1]
                cvar_val = Cvar_dict[station_name]
                lat.append(float(latitude))
                lon.append(float(longitude))
                Cvar.append(cvar_val)
            else:
                pass  # Skip the station

    y = np.array(lat)
    x = np.array(lon)
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
    weights = 1 / (distance_matrix**d)
    # Making sure to assign the value of the weather station above the pixel directly to the pixel underneath
    weights[np.where(np.isinf(weights))] = 1 / (1.0E-50)
    weights /= weights.sum(axis=0)

    Zi = np.dot(weights.T, z)
    idw_grid = Zi.reshape(num_row, num_col)

    # Compare at a certain point
    for statLoc in station_list:
        coord_pair = projected_lat_lon[statLoc]

        x_orig = int(
            (coord_pair[0] - float(bounds['minx']))/pixelHeight)  # lon
        y_orig = int(
            (coord_pair[1] - float(bounds['miny'])) / pixelWidth)  # lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)

        interpolated_val = idw_grid[y_orig][x_orig]

        original_val = Cvar_dict[statLoc]
        absolute_error = abs(interpolated_val - original_val)
        absolute_error_dictionary[statLoc] = absolute_error

    # average of all the withheld stations
    MAE = sum(absolute_error_dictionary.values()) / \
        len(absolute_error_dictionary.values())
    if return_error:
        return block_num, MAE, absolute_error_dictionary
    else:
        return block_num, MAE


def shuffle_split(loc_dict, Cvar_dict, shapefile, d, rep, show):
    '''Shuffle-split cross-validation with 50/50 training test split
   Parameters
   ----------
        loc_dict : dictionary
             the latitude and longitudes of the daily/hourly stations
        Cvar_dict : dictionary
             dictionary of weather variable values for each station
        shapefile : string
             path to the study area shapefile
        d : int
             the weighting for IDW interpolation
        rep : int
             number of replications
        show : bool
             if you want to show a map of the clusters
        block_num : int
             number of blocks/clusters
   Returns
   ----------
        float
             - MAE estimate for entire surface (average of replications)
   '''
    count = 1
    error_dictionary = {}
    while count <= rep:  # Loop through each block/cluster, leaving whole cluster out
        x_origin_list = []
        y_origin_list = []

        absolute_error_dictionary = {}
        projected_lat_lon = {}

        # we can't just use Cvar_dict.keys() because some stations do not have valid lat/lon
        stations_input = []
        for station_code in Cvar_dict.keys():
            if station_code in loc_dict.keys():
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
                if station_name not in test_stations:  # This is the step where we hold back the fold
                    loc = loc_dict[station_name]
                    latitude = loc[0]
                    longitude = loc[1]
                    cvar_val = Cvar_dict[station_name]
                    lat.append(float(latitude))
                    lon.append(float(longitude))
                    Cvar.append(cvar_val)
                else:
                    pass  # Skip the station

        y = np.array(lat)
        x = np.array(lon)
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
        dist_not = np.subtract.outer(vals[:, 0], interpol[:, 0])
        dist_one = np.subtract.outer(vals[:, 1], interpol[:, 1])
        distance_matrix = np.hypot(dist_not, dist_one)

        weights = 1 / (distance_matrix**d)
        weights[np.where(np.isinf(weights))] = 1 / (1.0E-50)
        weights /= weights.sum(axis=0)

        Zi = np.dot(weights.T, z)
        idw_grid = Zi.reshape(num_row, num_col)
        if show and (count == 1):
            fig, ax = plt.subplots(figsize=(15, 15))
            crs = {'init': 'esri:102001'}

            na_map = gpd.read_file(shapefile)

            im = plt.imshow(folds, extent=(xProj_extent.min(
            )-1, xProj_extent.max()+1, yProj_extent.max()-1, yProj_extent.min()+1), cmap='tab20b')
            na_map.plot(ax=ax, color='white', edgecolor='k',
                        linewidth=2, zorder=10, alpha=0.1)

            plt.scatter(xProj, yProj, c='black', s=2)

            plt.gca().invert_yaxis()
            cbar = plt.colorbar(im, ax=ax, cmap='tab20b')
            cbar.set_label('Block Number')

            title = 'Group selection'
            fig.suptitle(title, fontsize=14)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')

            plt.show()

        # Compare at a certain point
        for statLoc in test_stations:
            coord_pair = projected_lat_lon[statLoc]

            x_orig = int(
                (coord_pair[0] - float(bounds['minx']))/pixelHeight)  # lon
            y_orig = int(
                (coord_pair[1] - float(bounds['miny'])) / pixelWidth)  # lat
            x_origin_list.append(x_orig)
            y_origin_list.append(y_orig)

            interpolated_val = idw_grid[y_orig][x_orig]

            # Previous, this line had a large error.
            original_val = Cvar_dict[statLoc]
            absolute_error = abs(interpolated_val - original_val)
            absolute_error_dictionary[statLoc] = absolute_error

        error_dictionary[count] = sum(absolute_error_dictionary.values(
        ))/len(absolute_error_dictionary.values())  # average of all the withheld stations

        count += 1
    overall_error = sum(error_dictionary.values()) / \
        rep  # average of all the runs

    return overall_error
