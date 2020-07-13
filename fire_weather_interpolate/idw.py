#coding: latin1

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
import warnings
warnings.filterwarnings("ignore") #Runtime warning suppress, this suppresses the /0 warning


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
     
