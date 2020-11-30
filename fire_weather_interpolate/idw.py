#coding: utf-8

"""
Summary
-------
Code for interpolating weather data using inverse distance weighting interpolation. 
"""
    
#import

import geopandas as gpd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import os,sys
import math, statistics

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore") #Runtime warning suppress, this suppresses the /0 warning

import cluster_3d as c3d 
import make_blocks as mbk
import Eval as Eval 

def IDW(latlon_dict,Cvar_dict,input_date,var_name,shapefile,show,d): 
     '''Inverse distance weighting interpolation
     Parameters
         latlon_dict (dict): the latitude and longitudes of the hourly stations, loaded from the 
         .json file
         Cvar_dict (dict): dictionary of weather variable values for each station 
         input_date (str): the date you want to interpolate for 
         shapefile (str): path to the study area shapefile 
         show (bool): whether you want to plot a map 
         d (int): the weighting function for IDW interpolation 
     Returns
         idw_grid (np_array): the array of values for the interpolated surface
         maxmin: the bounds of the array surface, for use in other functions 
     '''
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
     y = np.array(lat)
     x = np.array(lon)
     z = np.array(Cvar) 

     pixelHeight = 10000 
     pixelWidth = 10000
            
     num_col = int((xmax - xmin) / pixelHeight)
     num_row = int((ymax - ymin) / pixelWidth)


     #We need to project to a projected system before making distance matrix
     xProj, yProj = pyproj.Proj('esri:102001')(x,y)
               

     yProj_extent=np.append(yProj,[bounds['maxy']+200000,bounds['miny']-200000])
     xProj_extent=np.append(xProj,[bounds['maxx']+200000,bounds['minx']-200000])

     Yi = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row)
     Xi = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col)

     Xi,Yi = np.meshgrid(Xi,Yi)
     Xi,Yi = Xi.flatten(), Yi.flatten()
     maxmin = [np.min(yProj_extent),np.max(yProj_extent),np.max(xProj_extent),np.min(xProj_extent)]

     vals = np.vstack((xProj,yProj)).T

     interpol = np.vstack((Xi,Yi)).T
     dist_not = np.subtract.outer(vals[:,0], interpol[:,0]) 
     dist_one = np.subtract.outer(vals[:,1], interpol[:,1]) 
     distance_matrix = np.hypot(dist_not,dist_one) 

     weights = 1/(distance_matrix **d)
     weights[np.where(np.isinf(weights))] = 1/(1.0E-50) 
     weights /= weights.sum(axis = 0) 

     Zi = np.dot(weights.T, z)
     idw_grid = Zi.reshape(num_row,num_col)

     if show:
        fig, ax = plt.subplots(figsize= (15,15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)
        
      
        plt.imshow(idw_grid,extent=(xProj_extent.min()-1,xProj_extent.max()+1,yProj_extent.max()-1,yProj_extent.min()+1)) 
        na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)
            
        plt.scatter(xProj,yProj,c=z,edgecolors='k')

        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label(var_name) 
        
        title = 'IDW Interpolation for %s on %s'%(var_name,input_date)
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude') 

        plt.show()


     return idw_grid, maxmin

def cross_validate_IDW(latlon_dict,Cvar_dict,shapefile,d,pass_to_plot):
     '''Leave-one-out cross-validation procedure for IDW 
     Parameters
         latlon_dict (dict): the latitude and longitudes of the hourly stations, loaded from the 
         .json file
         Cvar_dict (dict): dictionary of weather variable values for each station 
         shapefile (str): path to the study area shapefile 
         d (int): the weighting function for IDW interpolation
         pass_to_plot (bool):whether you will be plotting the error and need a version without absolute
         value error 
     Returns 
         absolute_error_dictionary (dict): a dictionary of the absolute error at each station when it
         was left out 
     '''
     x_origin_list = []
     y_origin_list = [] 

     absolute_error_dictionary = {} #for plotting
     no_absolute_value_dict = {} 
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

        #Delete at a certain point
        coord_pair = projected_lat_lon[station_name_hold_back]

        x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
        y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)

        interpolated_val = idw_grid[y_orig][x_orig] 

        original_val = Cvar_dict[station_name_hold_back]
        absolute_error = abs(interpolated_val-original_val)
        absolute_error_dictionary[station_name_hold_back] = absolute_error
        no_absolute_value_dict[station_name_hold_back] = interpolated_val-original_val
     if pass_to_plot:
         return absolute_error_dictionary, no_absolute_value_dict
     else:
         return absolute_error_dictionary
     

def select_block_size_IDW(nruns,group_type,loc_dict,Cvar_dict,idw_example_grid,shapefile,file_path_elev,idx_list,d,cluster_num1,cluster_num2,cluster_num3):
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
         cluster_num: three cluster numbers to test, for blocking this must be one of three:25, 16, 9 
         For blocking you can enter 'None' and it will automatically test 25, 16, 9 
     Returns 
         lowest_stdev,ave_MAE (int,float): block/cluster number w/ lowest stdev, associated
         ave_MAE of all the runs 
     '''
     
     #Get group dictionaries

     if group_type == 'blocks': 

          folds25 = mbk.make_block(idw_example_grid,25)
          dictionaryGroups25 = mbk.sorting_stations(folds25,shapefile,loc_dict,Cvar_dict)
          folds16 = mbk.make_block(idw_example_grid,16)
          dictionaryGroups16 = mbk.sorting_stations(folds16,shapefile,loc_dict,Cvar_dict)
          folds9 = mbk.make_block(idw_example_grid,9)
          dictionaryGroups9 = mbk.sorting_stations(folds9,shapefile,loc_dict,Cvar_dict)

     elif group_type == 'clusters':

          dictionaryGroups25 = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,cluster_num1,file_path_elev,idx_list,False,False,False)
          dictionaryGroups16 = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,cluster_num2,file_path_elev,idx_list,False,False,False)
          dictionaryGroups9 = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,cluster_num3,file_path_elev,idx_list,False,False,False)

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
          #We want same number of stations selected for each cluster number
          #We need to calculate, 5 folds x 25 clusters = 125 stations; 8 folds x 16 clusters = 128 stations, etc. 
          target_stations = len(Cvar_dict.keys())*0.3 # What is 30% of the stations
          fold_num1 = int(round(target_stations/cluster_num1)) 
          fold_num2 = int(round(target_stations/cluster_num2))
          fold_num3 = int(round(target_stations/cluster_num3)) 
            
          #For our first project, this is what we did 
          #fold_num1 = 5
          #fold_num2 = 8
          #fold_num3 = 14 
          #Just so there is a record of that
          

          block25 = spatial_groups_IDW(idw_example_grid,loc_dict,Cvar_dict,shapefile,d,cluster_num1,fold_num1,True,False,dictionaryGroups25)
          block25_error.append(block25) 

          block16 = spatial_groups_IDW(idw_example_grid,loc_dict,Cvar_dict,shapefile,d,cluster_num2,fold_num2,True,False,dictionaryGroups16)
          block16_error.append(block16)
          
          block9 = spatial_groups_IDW(idw_example_grid,loc_dict,Cvar_dict,shapefile,d,cluster_num3,fold_num3,True,False,dictionaryGroups9)
          block9_error.append(block9)

     stdev25 = statistics.stdev(block25_error) 
     stdev16 = statistics.stdev(block16_error)
     stdev9 = statistics.stdev(block9_error)

     list_stdev = [stdev25,stdev16,stdev9]
     list_block_name = [cluster_num1,cluster_num2,cluster_num3]
     list_error = [block25_error,block16_error,block9_error]
     index_min = list_stdev.index(min(list_stdev))
     lowest_stdev = list_block_name[index_min]

     ave_MAE = sum(list_error[index_min])/len(list_error[index_min]) 

     print(lowest_stdev)
     #print(ave_MAE) 
     return lowest_stdev,ave_MAE
               
          
def spatial_groups_IDW(idw_example_grid,loc_dict,Cvar_dict,shapefile,d,blocknum,nfolds,replacement,show,dictionary_Groups):
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
          if show and (count == 1):
               fig, ax = plt.subplots(figsize= (15,15))
               crs = {'init': 'esri:102001'}

               na_map = gpd.read_file(shapefile)


               im = plt.imshow(folds,extent=(xProj_extent.min()-1,xProj_extent.max()+1,yProj_extent.max()-1,yProj_extent.min()+1),cmap='tab20b') 
               na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)
                 
               plt.scatter(xProj,yProj,c='black',s=2)

               plt.gca().invert_yaxis()
               cbar = plt.colorbar(im,ax=ax,cmap='tab20b')
               cbar.set_label('Block Number') 

               title = 'Group selection'
               fig.suptitle(title, fontsize=14)
               plt.xlabel('Longitude')
               plt.ylabel('Latitude') 

               plt.show()

          #Compare at a certain point
          for statLoc in station_list:

               coord_pair = projected_lat_lon[statLoc]

               x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
               y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
               x_origin_list.append(x_orig)
               y_origin_list.append(y_orig)

               interpolated_val = idw_grid[y_orig][x_orig] 

               original_val = Cvar_dict[statLoc]
               absolute_error = abs(interpolated_val-original_val)
               absolute_error_dictionary[statLoc] = absolute_error


          error_dictionary[count]= sum(absolute_error_dictionary.values())/len(absolute_error_dictionary.values()) #average of all the withheld stations
          #print(absolute_error_dictionary)
          count+=1
     overall_error = sum(error_dictionary.values())/nfolds #average of all the runs
     #print(overall_error)
     return overall_error


def spatial_kfold_idw(idw_example_grid,loc_dict,Cvar_dict,shapefile,d,file_path_elev,idx_list,BlockNum,return_error):
     '''Spatially blocked k-folds cross-validation procedure for IDW 
     Parameters
         idw_example_grid (numpy array): the example idw grid to base the size of the group array off of 
         loc_dict (dict): the latitude and longitudes of the hourly/daily stations, loaded from the 
         .json file
         Cvar_dict (dict): dictionary of weather variable values for each station 
         shapefile (str): path to the study area shapefile 
         d (int): the weighting function for IDW interpolation
     Returns 
         error_dictionary (dict): a dictionary of the absolute error at each fold when it
         was left out 
     '''
     groups_complete = [] #If not using replacement, keep a record of what we have done 
     error_dictionary = {} 
     x_origin_list = []
     y_origin_list = [] 

     absolute_error_dictionary = {} 
     projected_lat_lon = {}

     cluster = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,BlockNum,file_path_elev,idx_list,False,False,False)

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

     #Compare at a certain point
     for statLoc in station_list: 
          coord_pair = projected_lat_lon[statLoc]

          x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
          y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
          x_origin_list.append(x_orig)
          y_origin_list.append(y_orig)

          interpolated_val = idw_grid[y_orig][x_orig] 

          original_val = Cvar_dict[statLoc]
          absolute_error = abs(interpolated_val-original_val)
          absolute_error_dictionary[statLoc] = absolute_error

     MAE= sum(absolute_error_dictionary.values())/len(absolute_error_dictionary.values()) #average of all the withheld stations
     if return_error:
         return BlockNum,MAE,absolute_error_dictionary
     else:
         return BlockNum,MAE

def shuffle_split(loc_dict,Cvar_dict,shapefile,d,rep,show):
     '''Shuffle-split cross-validation with 50/50 training test split 
     Parameters
         loc_dict (dict): the latitude and longitudes of the hourly/daily stations, loaded from the 
         .json file
         Cvar_dict (dict): dictionary of weather variable values for each station 
         shapefile (str): path to the study area shapefile 
         d (int): the weighting function for IDW interpolation
         rep (int): number of replications 
         show (bool): if you want to show a map of the clusters
     Returns 
         overall_error (float): average MAE of all the replications 
     '''
     count = 1
     error_dictionary = {} 
     while count <= rep: #Loop through each block/cluster, leaving whole cluster out 
          x_origin_list = []
          y_origin_list = [] 

          absolute_error_dictionary = {} 
          projected_lat_lon = {}

          stations_input = [] #we can't just use Cvar_dict.keys() because some stations do not have valid lat/lon
          for station_code in Cvar_dict.keys():
               if station_code in loc_dict.keys():
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
                    if station_name not in test_stations: #This is the step where we hold back the fold
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
          if show and (count == 1):
               fig, ax = plt.subplots(figsize= (15,15))
               crs = {'init': 'esri:102001'}

               na_map = gpd.read_file(shapefile)


               im = plt.imshow(folds,extent=(xProj_extent.min()-1,xProj_extent.max()+1,yProj_extent.max()-1,yProj_extent.min()+1),cmap='tab20b') 
               na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)
                 
               plt.scatter(xProj,yProj,c='black',s=2)

               plt.gca().invert_yaxis()
               cbar = plt.colorbar(im,ax=ax,cmap='tab20b')
               cbar.set_label('Block Number') 

               title = 'Group selection'
               fig.suptitle(title, fontsize=14)
               plt.xlabel('Longitude')
               plt.ylabel('Latitude') 

               plt.show()

          #Compare at a certain point
          for statLoc in test_stations: 
               coord_pair = projected_lat_lon[statLoc]

               x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
               y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
               x_origin_list.append(x_orig)
               y_origin_list.append(y_orig)

               interpolated_val = idw_grid[y_orig][x_orig] 

               original_val = Cvar_dict[statLoc] #Previous, this line had a large error. 
               absolute_error = abs(interpolated_val-original_val)
               absolute_error_dictionary[statLoc] = absolute_error

          error_dictionary[count]= sum(absolute_error_dictionary.values())/len(absolute_error_dictionary.values()) #average of all the withheld stations
          
          count+=1
     overall_error = sum(error_dictionary.values())/rep #average of all the runs
     
     return overall_error

