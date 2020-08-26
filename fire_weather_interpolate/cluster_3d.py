#coding: utf-8

"""
Summary
-------
Code for making spatial clusters for cross-validation. 
"""
    
#import
import geopandas as gpd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import warnings
import os,sys
import math,statistics

warnings.filterwarnings("ignore") #Runtime warning suppress, this suppresses the /0 warning

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import mpl_toolkits.mplot3d.axes3d as p3

import get_data as GD

def spatial_cluster(loc_dict,Cvar_dict,shapefile,cluster_num,file_path_elev,idx_list,plot_2D,plot_3D):
     '''Spatial clustering based on scikit learn's agglomerative clustering
     Parameters
         loc_dict (dict): the latitude and longitudes of the hourly/daily stations, loaded from the 
         .json file
         Cvar_dict (dict): dictionary of weather variable values for each station 
         shapefile (str): path to the study area shapefile 
         cluster_num (int): number of clusters to form 
         file_path_elev (str): path to the elevation lookup file 
         idx_list (int): column of elevation in lookup file 
         plot_2D/plot_3D (bool): whether to plot maps of the clusters in either 2d or 3d
     Returns 
         station_class (dict): a dictionary of cluster that each station is in 
     '''     

     x = []
     y = []
     proj_stations = {} 
     for station in Cvar_dict.keys():
          if station in loc_dict.keys():
               coord = loc_dict[station]
               Plon1, Plat1 = pyproj.Proj('esri:102001')(coord[1],coord[0]) #longitude,lat
               Plat = float(Plat1)
               Plon = float(Plon1)
               x.append([Plon]) 
               y.append([Plat])
               proj_stations[station] = [Plat,Plon]
     X = [val+y[i] for i,val in enumerate(x)]
     X = np.array(X)
     #print(X)
     #Make the longitudinal transect of distance (lon, elev)

     Xi1_grd=[]
     Yi1_grd=[]
     elev_grd = []
     concat = np.array((x,y)).T #Preparing the coordinates to send to the function that will get the elevation grid 
     send_to_list = concat[0].tolist()
     send_to_tuple = [tuple(x) for x in send_to_list]
     elev_grd_dict = GD.finding_data_frm_lookup(send_to_tuple,file_path_elev,idx_list) #Get the elevations from the lookup file 

     for keys in elev_grd_dict.keys(): #The keys are each lat lon pair 
          x= keys[0]
          y = keys[1]
          Xi1_grd.append(x)
          Yi1_grd.append(y)
          elev_grd.append(elev_grd_dict[keys]) #Append the elevation data to the empty list 

     lon = [i for i in Xi1_grd] #list of 0
     lon_list = [[i] for i in lon]
     lat_list = [[i] for i in Yi1_grd]
     elev = [[i] for i in elev_grd] # put into sublist so you can make pairs 
     Xelev = [val+lat_list[i]+elev[i] for i,val in enumerate(lon_list)]
     Xelev = np.array(Xelev)
     

     #This is where we make the connectivity graph based on elevation 

     knn_graph = kneighbors_graph(Xelev, 10, include_self=False)
     connectivity = knn_graph
     n_clusters = cluster_num
     
     linkage = 'ward'
               
     model = AgglomerativeClustering(linkage=linkage,connectivity=connectivity,n_clusters=n_clusters)
                 
     model.fit(Xelev) #fit with lat lon elev
     label = model.labels_

     if plot_3D: 
          fig = plt.figure()
          ax = p3.Axes3D(fig)
          ax.view_init(7, -80)
          for l in np.unique(label):
              ax.scatter(Xelev[label == l, 0], Xelev[label == l, 1], Xelev[label == l, 2],
                    color=plt.cm.jet(float(l) / np.max(label + 1)),
                    s=20, edgecolor='k')
          plt.title('With connectivity constraints, Elevation inc.')
          ax.set_xlabel('Longitude')
          ax.set_ylabel('Latitude')
          ax.set_zlabel('Elevation (m)')

          plt.show()
     
     #This is where we make the connectivity graph where we can see on the map  
     if plot_2D: 
          fig, ax = plt.subplots(figsize= (15,15))
          crs = {'init': 'esri:102001'}
          na_map = gpd.read_file(shapefile)

          na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)
     
          plt.scatter(Xelev[:, 0], Xelev[:, 1], c=model.labels_,
                        cmap=plt.cm.nipy_spectral)
            
          plt.axis('equal')
          plt.axis('off')

          plt.subplots_adjust(bottom=0, top=.83, wspace=0,
                                left=0, right=1)
          plt.suptitle('n_cluster=%i, connectivity=%r' %
                         (n_clusters, connectivity is not None), size=17)

          ax.set_xlabel('Longitude')
          ax.set_ylabel('Latitude')

          plt.show()

     #Make a dictionary with each class
     station_class = {}

     count = 0 
     for val in Xelev:
          key = [key for key, value in proj_stations.items() if value == [val[1],val[0]]]
          if len(key) == 1: 
               station_class[key[0]] = label[count] +1 #We add 1, because for the random selection the groups start at 1
          elif len(key) == 2:
               station_class[key[0]] = label[count] +1
               station_class[key[1]] = label[count] +1
          elif len(key) == 3:
               station_class[key[0]] = label[count] +1
               station_class[key[1]] = label[count] +1
               station_class[key[2]] = label[count] +1
          else:
               print('Too many stations have the same lat lon.') 
          count+=1

     if count != label.shape[0]:
          print('The groups and label matrix do not match')

     return station_class
