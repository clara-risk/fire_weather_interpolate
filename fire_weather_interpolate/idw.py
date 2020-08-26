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

import warnings
warnings.filterwarnings("ignore") #Runtime warning suppress, this suppresses the /0 warning

import cluster_3d as c3d 

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

     yProj_extent=np.append(yProj,[bounds['maxy'],bounds['miny']])
     xProj_extent=np.append(xProj,[bounds['maxx'],bounds['minx']])

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

def cross_validate_IDW(latlon_dict,Cvar_dict,shapefile,d):
     '''Leave-one-out cross-validation procedure for IDW 
     Parameters
         latlon_dict (dict): the latitude and longitudes of the hourly stations, loaded from the 
         .json file
         Cvar_dict (dict): dictionary of weather variable values for each station 
         shapefile (str): path to the study area shapefile 
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

        original_val = Cvar_dict[station_name]
        absolute_error = abs(interpolated_val-original_val)
        absolute_error_dictionary[station_name_hold_back] = absolute_error


     return absolute_error_dictionary
     
def make_block(idw_grid,blocknum):
     '''Function to create an array delineating the groups. 
     '''
     if blocknum == 4:
          shape = idw_grid.shape
          blocks = np.array_split(idw_grid,2) #We need to make like a quilt of values of the blocks and then search for stations that overlay a block
          blocks[0] = np.zeros(blocks[0].shape)+1
          blocks[1] = np.zeros(blocks[1].shape)+3

          blocks[0] = [x.T for x in np.array_split(blocks[0].T, 2)] #transpose the array so we can split by column
          blocks[0][1] = np.zeros(blocks[0][1].shape)+2
 
          blocks[1] = [x.T for x in np.array_split(blocks[1].T, 2)]
          blocks[1][1] = np.zeros(blocks[1][1].shape)+4

          #Ok now we knit it back together lengthwise
          topBlock = np.concatenate([blocks[0][0],blocks[0][1]],axis=1)
          bottomBlock = np.concatenate([blocks[1][0],blocks[1][1]],axis=1)
          #Now widthwise
          blocks = np.concatenate([topBlock,bottomBlock],axis=0)

     if blocknum == 9:
          shape = idw_grid.shape
          blocks = np.array_split(idw_grid,3) #We need to make like a quilt of values of the blocks and then search for stations that overlay a block
          blocks[0] = np.zeros(blocks[0].shape)+1
          blocks[1] = np.zeros(blocks[1].shape)+4
          blocks[2] = np.zeros(blocks[2].shape)+7

          blocks[0] = [x.T for x in np.array_split(blocks[0].T, 3)] #transpose the array so we can split by column
          blocks[0][1] = np.zeros(blocks[0][1].shape)+2
          blocks[0][2] = np.zeros(blocks[0][2].shape)+3
 
          blocks[1] = [x.T for x in np.array_split(blocks[1].T, 3)]
          blocks[1][1] = np.zeros(blocks[1][1].shape)+5
          blocks[1][2] = np.zeros(blocks[1][2].shape)+6

          blocks[2] = [x.T for x in np.array_split(blocks[2].T, 3)]
          blocks[2][1] = np.zeros(blocks[2][1].shape)+8
          blocks[2][2] = np.zeros(blocks[2][2].shape)+9

          
          #Ok now we knit it back together lengthwise
          topBlock = np.concatenate([blocks[0][0],blocks[0][1],blocks[0][2]],axis=1)
          secondBlock = np.concatenate([blocks[1][0],blocks[1][1],blocks[1][2]],axis=1)
          bottomBlock = np.concatenate([blocks[2][0],blocks[2][1],blocks[2][2]],axis=1)
          #Now widthwise
          blocks = np.concatenate([topBlock,secondBlock,bottomBlock],axis=0)
     if blocknum == 16: 
          shape = idw_grid.shape
          blocks = np.array_split(idw_grid,4) #We need to make like a quilt of values of the blocks and then search for stations that overlay a block
          blocks[0] = np.zeros(blocks[0].shape)+1
          blocks[1] = np.zeros(blocks[1].shape)+5
          blocks[2] = np.zeros(blocks[2].shape)+9
          blocks[3] = np.zeros(blocks[3].shape)+13
          blocks[0] = [x.T for x in np.array_split(blocks[0].T, 4)] #transpose the array so we can split by column
          blocks[0][1] = np.zeros(blocks[0][1].shape)+2
          blocks[0][2] = np.zeros(blocks[0][2].shape)+3
          blocks[0][3] = np.zeros(blocks[0][3].shape)+4
          blocks[1] = [x.T for x in np.array_split(blocks[1].T, 4)]
          blocks[1][1] = np.zeros(blocks[1][1].shape)+6
          blocks[1][2] = np.zeros(blocks[1][2].shape)+7
          blocks[1][3] = np.zeros(blocks[1][3].shape)+8
          blocks[2] = [x.T for x in np.array_split(blocks[2].T, 4)]
          blocks[2][1] = np.zeros(blocks[2][1].shape)+10
          blocks[2][2] = np.zeros(blocks[2][2].shape)+11
          blocks[2][3] = np.zeros(blocks[2][3].shape)+12
          blocks[3] = [x.T for x in np.array_split(blocks[3].T, 4)]
          blocks[3][1] = np.zeros(blocks[3][1].shape)+14
          blocks[3][2] = np.zeros(blocks[3][2].shape)+15
          blocks[3][3] = np.zeros(blocks[3][3].shape)+16
          
          #Ok now we knit it back together lengthwise
          topBlock = np.concatenate([blocks[0][0],blocks[0][1],blocks[0][2],blocks[0][3]],axis=1)
          secondBlock = np.concatenate([blocks[1][0],blocks[1][1],blocks[1][2],blocks[1][3]],axis=1)
          thirdBlock = np.concatenate([blocks[2][0],blocks[2][1],blocks[2][2],blocks[2][3]],axis=1)
          bottomBlock = np.concatenate([blocks[3][0],blocks[3][1],blocks[3][2],blocks[3][3]],axis=1)
          #Now widthwise
          blocks = np.concatenate([topBlock,secondBlock,thirdBlock,bottomBlock],axis=0)
     if blocknum == 25:
          shape = idw_grid.shape
          blocks = np.array_split(idw_grid,5) #We need to make like a quilt of values of the blocks and then search for stations that overlay a block
          blocks[0] = np.zeros(blocks[0].shape)+1
          blocks[1] = np.zeros(blocks[1].shape)+6
          blocks[2] = np.zeros(blocks[2].shape)+11
          blocks[3] = np.zeros(blocks[3].shape)+16
          blocks[4] = np.zeros(blocks[3].shape)+21
          blocks[0] = [x.T for x in np.array_split(blocks[0].T, 5)] #transpose the array so we can split by column
          blocks[0][1] = np.zeros(blocks[0][1].shape)+2
          blocks[0][2] = np.zeros(blocks[0][2].shape)+3
          blocks[0][3] = np.zeros(blocks[0][3].shape)+4
          blocks[0][4] = np.zeros(blocks[0][4].shape)+5
          blocks[1] = [x.T for x in np.array_split(blocks[1].T, 5)]
          blocks[1][1] = np.zeros(blocks[1][1].shape)+7
          blocks[1][2] = np.zeros(blocks[1][2].shape)+8
          blocks[1][3] = np.zeros(blocks[1][3].shape)+9
          blocks[1][4] = np.zeros(blocks[1][4].shape)+10
          blocks[2] = [x.T for x in np.array_split(blocks[2].T, 5)]
          blocks[2][1] = np.zeros(blocks[2][1].shape)+11
          blocks[2][2] = np.zeros(blocks[2][2].shape)+12
          blocks[2][3] = np.zeros(blocks[2][3].shape)+13
          blocks[2][4] = np.zeros(blocks[2][4].shape)+14
          blocks[3] = [x.T for x in np.array_split(blocks[3].T, 5)]
          blocks[3][1] = np.zeros(blocks[3][1].shape)+15
          blocks[3][2] = np.zeros(blocks[3][2].shape)+16
          blocks[3][3] = np.zeros(blocks[3][3].shape)+17
          blocks[3][4] = np.zeros(blocks[3][4].shape)+18
          blocks[4] = [x.T for x in np.array_split(blocks[4].T, 5)]
          blocks[4][1] = np.zeros(blocks[4][1].shape)+19
          blocks[4][2] = np.zeros(blocks[4][2].shape)+20
          blocks[4][3] = np.zeros(blocks[4][3].shape)+21
          blocks[4][4] = np.zeros(blocks[4][4].shape)+22

          
          #Ok now we knit it back together lengthwise
          topBlock = np.concatenate([blocks[0][0],blocks[0][1],blocks[0][2],blocks[0][3],blocks[0][4]],axis=1)
          secondBlock = np.concatenate([blocks[1][0],blocks[1][1],blocks[1][2],blocks[1][3],blocks[1][4]],axis=1)
          thirdBlock = np.concatenate([blocks[2][0],blocks[2][1],blocks[2][2],blocks[2][3],blocks[2][4]],axis=1)
          fourthBlock = np.concatenate([blocks[3][0],blocks[3][1],blocks[3][2],blocks[3][3],blocks[3][4]],axis=1)
          bottomBlock = np.concatenate([blocks[4][0],blocks[4][1],blocks[4][2],blocks[4][3],blocks[4][4]],axis=1)
          #Now widthwise
          blocks = np.concatenate([topBlock,secondBlock,thirdBlock,fourthBlock,bottomBlock],axis=0)         

     return blocks


def sorting_stations(blocks,shapefile,Cvar_dict):
     '''Here we are sorting the stations based on their position on the array. 
     '''
     x_origin_list = []
     y_origin_list = [] 

     groups = {} 
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
        xmax = bounds['maxx']
        xmin= bounds['minx']
        ymax = bounds['maxy']
        ymin = bounds['miny']
        pixelHeight = 10000 
        pixelWidth = 10000

        #Delete at a certain point
        coord_pair = projected_lat_lon[station_name_hold_back]

        x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
        y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat

        group = blocks[y_orig][x_orig] 
        groups[station_name_hold_back] = group 

     return groups

def select_block_size(nruns,group_type):
     block25_error = []
     block16_error = []
     block9_error = []
     if nruns <= 1:
          print('That is not enough runs to calculate the standard deviation!')
          sys.exit() 
     
     for n in range(0,nruns):

          block25 = spatial_groups_IDW(idw1_grid,latlon_dict,temp_dict,shapefile,1,25,5,True,False,group_type)
          block25_error.append(block25) 

          block16 = spatial_groups_IDW(idw1_grid,latlon_dict,temp_dict,shapefile,1,16,8,True,False,group_type)
          block16_error.append(block16)
          
          block9 = spatial_groups_IDW(idw1_grid,latlon_dict,temp_dict,shapefile,1,9,14,True,False,group_type)
          block9_error.append(block9)

     stdev25 = statistics.stdev(block25_error) 
     stdev16 = statistics.stdev(block16_error)
     stdev9 = statistics.stdev(block9_error)

     list_stdev = [stdev25,stdev16,stdev9]
     list_block_name = [25,16,9]
     index_min = list_stdev.index(min(list_stdev))
     lowest_stdev = list_block_name[index_min]

     print(lowest_stdev) 

     return lowest_stdev
               
          
def spatial_groups_IDW(idw_example_grid,latlon_dict,Cvar_dict,shapefile,d,blocknum,nfolds,replacement,show):
     '''Spatially blocked k-folds cross-validation procedure for IDW 
     Parameters
         idw_example_grid (numpy array): the example idw grid to base the size of the group array off of 
         latlon_dict (dict): the latitude and longitudes of the hourly stations, loaded from the 
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

          if group_type == 'blocks': 
               folds = make_block(idw1_grid,blocknum)
               dictionaryGroups = sorting_stations(folds,shapefile,Cvar_dict)
               station_list = select_random_station(dictionaryGroups,blocknum,replacement,station_list_used).values()
          elif group_type == 'clusters':
               cluster = c3d.spatial_cluster(latlon_dict,Cvar_dict,shapefile,blocknum,False,False)
               station_list = select_random_station(cluster,blocknum,replacement,station_list_used).values()

          else:
               print('That is not a valid group type.')
               sys.exit() 

          if replacement == False: 
               station_list_used.append(list(station_list))
          #print(station_list_used) 
          #print(station_list) 
          

          for station_name in Cvar_dict.keys():
               
               if station_name in latlon_dict.keys():

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
                    if station_name not in station_list: #This is the step where we hold back the fold
                         loc = latlon_dict[station_name]
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
          for station_name_hold_back in station_list: 
               coord_pair = projected_lat_lon[station_name_hold_back]

               x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
               y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
               x_origin_list.append(x_orig)
               y_origin_list.append(y_orig)

               interpolated_val = idw_grid[y_orig][x_orig] 

               original_val = Cvar_dict[station_name]
               absolute_error = abs(interpolated_val-original_val)
               absolute_error_dictionary[station_name_hold_back] = absolute_error

          error_dictionary[count]= sum(absolute_error_dictionary.values())/len(absolute_error_dictionary.values()) #average of all the withheld stations
          #print(absolute_error_dictionary)
          count+=1
     overall_error = sum(error_dictionary.values())/nfolds #average of all the runs
     print(overall_error)
     return overall_error


def spatial_kfold_idw(idw_example_grid,loc_dict,Cvar_dict,shapefile,d,blocknum,nfolds,show,group_type):
     '''Spatially blocked k-folds cross-validation procedure for IDW 
     Parameters
         idw_example_grid (numpy array): the example idw grid to base the size of the group array off of 
         loc_dict (dict): the latitude and longitudes of the hourly/daily stations, loaded from the 
         .json file
         Cvar_dict (dict): dictionary of weather variable values for each station 
         shapefile (str): path to the study area shapefile 
         d (int): the weighting function for IDW interpolation
         blocknum (int): number of clusters/blocks you want to use 
         nfolds (int): # number of folds. For 10-fold we use 10, etc. 
         show (bool): if you want to show a map of the clusters
         group_type (str): specify one of two options, 'clusters' or 'blocks'
     Returns 
         error_dictionary (dict): a dictionary of the absolute error at each fold when it
         was left out 
     '''
     groups_complete = [] #If not using replacement, keep a record of what we have done 
     count = 1
     error_dictionary = {} 
     while count <= nfolds: #Loop through each block/cluster, leaving whole cluster out 
          x_origin_list = []
          y_origin_list = [] 

          absolute_error_dictionary = {} 
          projected_lat_lon = {}

          if group_type == 'blocks': 
               folds = make_block(idw_example_grid,blocknum)
               dictionaryGroups = sorting_stations(folds,shapefile,Cvar_dict)
               for group in dictionaryGroups.values():
                    if group not in groups_complete: #if it has already been done 
                         station_list = [k for k,v in dictionaryGroups.items() if v == group]
                         groups_complete.append(group)
                         
               
          elif group_type == 'clusters':
               cluster = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,blocknum,False,False)
               for group in cluster.values():
                    if group not in groups_complete:
                         station_list = [k for k,v in cluster.items() if v == group]
                         groups_complete.append(group)
                         print(station_list) 

          else:
               print('That is not a valid group type.')
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
          for station_name_hold_back in station_list: 
               coord_pair = projected_lat_lon[station_name_hold_back]

               x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
               y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
               x_origin_list.append(x_orig)
               y_origin_list.append(y_orig)

               interpolated_val = idw_grid[y_orig][x_orig] 

               original_val = Cvar_dict[station_name]
               absolute_error = abs(interpolated_val-original_val)
               absolute_error_dictionary[station_name_hold_back] = absolute_error

          error_dictionary[count]= sum(absolute_error_dictionary.values())/len(absolute_error_dictionary.values()) #average of all the withheld stations
          
          count+=1
     overall_error = sum(error_dictionary.values())/nfolds #average of all the runs
     
     return overall_error
