#coding: utf-8

"""
Summary
-------
Code for creating spatial blocks for cross-validation. 
"""
    
#import
import geopandas as gpd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import warnings
import os,sys
import math,statistics

def make_block(idw_grid,blocknum):
     '''Divide the study area into blocks 
     Parameters
         idw_grid (numpy array): the example idw grid to base the size of the group array off of 
         blocknum (int): number of blocks to create, either 4,9,16,25
     Returns 
         blocks (numpy array): an array with the block value contained in each pixel 
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
     '''Find the stations in each block and create a reference dictionary for this information 
     Parameters
         blocks (numpy array): output of the make_block function 
         shapefile (str): path to shapefile of study area 
         Cvar_dict (dict): the dictionary keyed by station with the weather data inside 
     Returns 
         groups (dict): dictionary of the block for each station 
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
