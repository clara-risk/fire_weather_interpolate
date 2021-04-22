#coding: utf-8

"""
Summary
-------
Spatial interpolation functions for gaussian process regression interpolation using the scikit-learn package.

References
----------

scikit-learn developers. (2019). 1.7.1. Gaussian Process Regression (GPR). Retrieved September 26, 2020, from https://scikit-learn.org/stable/modules/gaussian_process.html
scikit-learn developers. (2019a). sklearn.gaussian_process.GaussianProcessRegressor. Retrieved 	September 26, 2020, from https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor
Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes for machine learning. 2006. In The MIT Press, Cambridge, MA, USA (Vol. 38, Issue 2). 
"""
    
#import
import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") #Runtime warning suppress, this suppresses the /0 warning

from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)


import get_data as GD
import cluster_3d as c3d
import make_blocks as mbk
import Eval as Eval
import statistics 

def GPR_interpolator(latlon_dict,Cvar_dict,input_date,var_name,shapefile,show,\
                     file_path_elev,idx_list,expand_area,kernel_object,restarts,report_params,optimizer,param_initiate=None,cov_type='RBF'):
    '''Base interpolator function for gaussian process regression

    Parameters
    ----------
    latlon_dict : dictionary
        the latitude and longitudes of the stations
    Cvar_dict : dictionary
        dictionary of weather variable values for each station
    input_date : string
        the date you want to interpolate for
    shapefile : string
        path to the study area shapefile, including its name
    show : bool
        whether you want to plot a map    
    file_path_elev : string
        file path to the elevation lookup file 
    idx_list : list
        the index of the elevation data column in the lookup file 
    expand_area : bool
        function will expand the study area so that more stations are taken into account (200 km)   
    kernel_object : list
        kernel object describing input kernel you want to use, if optimizing a set of parameters, can input empty list
    restarts : int
        number of times to restart to avoid local optima
    report_params : bool
        if True, outputs optimized values for kernel hyperparameters
    optimizer : bool
        if False, fix parameters of covariance function
    param_initiate : list
        input parameters needed to start optimization, controls extent of the spatial autocorrelation modelled by the process
        whether the spatial autocorrelation is the same in all directions will depend on the inputs for parameters,
        you need to input the parameters of the function (distribution) as a vector not a scalar
        since we are working in 3d (latitude, longitude, elevation) the vector must be len=3 because this corresponds to the [x,y,z]
        if we are using an anisotropic distribution
        ...for isotropic 1d, [1] (or if 2 parameters, [[1],[1]]), for anisotropic, will be [1,1,1] or [[1,1],[1,1],[1,1]]
    cov_type : str
        type of covariance function to use if have not specified a kernel object

    Returns
    ----------
    ndarray
        - an array of the interpolated values
    '''
    lat = []
    lon = []
    Cvar = []

    na_map = gpd.read_file(shapefile)
    bounds = na_map.bounds
    if expand_area: 
        xmax = bounds['maxx']+200000 
        xmin= bounds['minx']-200000 
        ymax = bounds['maxy']+200000 
        ymin = bounds['miny']-200000
    else:
        xmax = bounds['maxx']
        xmin= bounds['minx']
        ymax = bounds['maxy']
        ymin = bounds['miny']
        
    for station_name in Cvar_dict.keys():
        if station_name in latlon_dict.keys():
            
            loc = latlon_dict[station_name]
            latitude = loc[0]
            longitude = loc[1]
            proj_coord = pyproj.Proj('esri:102001')(longitude,latitude) #Filter out stations outside of grid
            if (proj_coord[1] <= float(ymax[0]) and proj_coord[1] >= \
                float(ymin[0]) and proj_coord[0] <= float(xmax[0]) and \
                proj_coord[0] >= float(xmin[0])):
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
    source_proj = pyproj.Proj(proj='latlong', datum = 'NAD83') 
    xProj, yProj = pyproj.Proj('esri:102001')(x,y)
    
    df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'var': z})

    if expand_area: 

        yProj_extent=np.append(yProj,[bounds['maxy']+200000,bounds['miny']-200000])
        xProj_extent=np.append(xProj,[bounds['maxx']+200000,bounds['minx']-200000])

    else:
        yProj_extent=np.append(yProj,[bounds['maxy'],bounds['miny']])
        xProj_extent=np.append(xProj,[bounds['maxx'],bounds['minx']])

    Yi = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row+1)
    Xi = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col+1)

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
    
    Xi1_grd = np.array(Xi1_grd)
    Yi1_grd = np.array(Yi1_grd)
    
    df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'elevS':source_elev, 'var': z})
    
    df_testX = pd.DataFrame({'Xi': Xi1_grd, 'Yi': Yi1_grd, 'elev': elev_array})

    if param_initiate is not None: 

        if len(param_initiate) > 1: 
        
            kernels = [1.0 * RBF(length_scale=param_initiate[0]), 1.0 * RationalQuadratic(length_scale=param_initiate[0][0], alpha=param_initiate[0][1]), \
                       1.0 * Matern(length_scale=param_initiate[0],nu=param_initiate[1],length_scale_bounds=(1000,500000))] #Temp =(100,500000) #RH = (1000,500000)
        #Optimizer =  ‘L-BGFS-B’ algorithm
        else:
            
            kernels = [1.0 * RBF(length_scale=param_initiate[0])]

        if cov == 'RationalQuadratic':
            if optimizer:
                reg = GaussianProcessRegressor(kernel=kernels[1],normalize_y=True,n_restarts_optimizer=restarts) #Updated Nov 23 for fire season manuscript to make 3 restarts, Dec 9 = 5
            else:
                reg = GaussianProcessRegressor(kernel=kernels[1],normalize_y=True,n_restarts_optimizer=restarts,optimizer = None)
        elif cov == 'RBF':
            if optimizer:
                reg = GaussianProcessRegressor(kernel=kernels[0],normalize_y=True,n_restarts_optimizer=restarts) #Updated Nov 23 for fire season manuscript to make 3 restarts, Dec 9 = 5
            else:
                reg = GaussianProcessRegressor(kernel=kernels[0],normalize_y=True,n_restarts_optimizer=restarts,optimizer = None) 
        elif cov == 'Matern':

            if optimizer:
                reg = GaussianProcessRegressor(kernel=kernels[2],normalize_y=True,n_restarts_optimizer=restarts) #Updated Nov 23 for fire season manuscript to make 3 restarts, Dec 9 = 5
            else:
                #kernels = [307**2 * Matern(length_scale=[5e+05, 6.62e+04, 1.07e+04], nu=0.5)]
                #kernels = [316**2 * Matern(length_scale=[5e+05, 5e+05, 6.01e+03], nu=0.5)]
                #kernels = [316**2 * Matern(length_scale=[5e+05, 5e+05, 4.67e+05], nu=0.5)]
                reg = GaussianProcessRegressor(kernel=kernels[0],normalize_y=True,n_restarts_optimizer=restarts,optimizer = None)
    else:
        kernels = [eval(kernel_object[0])]
        reg = GaussianProcessRegressor(kernel=kernels[0],normalize_y=True,n_restarts_optimizer=0,optimizer = None)
            
    y = np.array(df_trainX['var']).reshape(-1,1)
    X_train = np.array(df_trainX[['xProj','yProj','elevS']])
    X_test = np.array(df_testX[['Xi','Yi','elev']])
    
    reg.fit(X_train, y)
    fitted_params = reg.kernel_
    score = reg.score(X_train, y)
    print(fitted_params)
    print(score)
    
    Zi = reg.predict(X_test)
    
    gpr_grid = Zi.reshape(num_row+1,num_col+1)

    if show:
        fig, ax = plt.subplots(figsize= (15,15))
        crs = {'init': 'esri:102001'}

        na_map = gpd.read_file(shapefile)
        
      
        plt.imshow(gpr_grid,extent=(xProj_extent.min()-1,xProj_extent.max()+1,yProj_extent.max()-1,yProj_extent.min()+1)) 
        na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)
            
        plt.scatter(xProj,yProj,c=z,edgecolors='k')

        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.set_label(var_name) 
        
        title = 'GPR Interpolation for %s on %s'%(var_name,input_date)
        fig.suptitle(title, fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude') 

        plt.show()

    if report_params:
        return fitted_params

    else: 
        return gpr_grid, maxmin

def cross_validate_gpr(latlon_dict,Cvar_dict,shapefile,file_path_elev,elev_array,idx_list,cov_function):
    '''Leave-one-out cross-validation procedure for GPR

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
    cov_function : list
        list containing a string that describes the input covariance function, similar to: ['316**2 * Matern(length_scale=[5e+05, 5e+05, 6.01e+03], nu=0.5)']
      
    Returns
    ----------
    dictionary
        - a dictionary of the absolute error at each station when it was left out
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
            
        num_col = int((xmax - xmin) / pixelHeight)+1
        num_row = int((ymax - ymin) / pixelWidth)+1


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
    
        Xi1_grd = np.array(Xi1_grd)
        Yi1_grd = np.array(Yi1_grd)
    
        df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'elevS':source_elev, 'var': z})
    
        df_testX = pd.DataFrame({'Xi': Xi1_grd, 'Yi': Yi1_grd, 'elev': elev_array})
    
    
        #kernels = [1.0 * RationalQuadratic(length_scale=1.0, alpha=alpha_input)]
        #kernels = [multiplier**exponent * Matern(length_scale=length_scale_list,nu=param_initiate[1],length_scale_bounds='fixed')]
        #kernels = [params]

        #Temperature 
        #kernels = [316**2 * Matern(length_scale=[5e+05, 5e+05, 6.01e+03], nu=0.5)]

        #RH
        #kernels = [307**2 * Matern(length_scale=[9.51e+04, 9.58e+04, 3.8e+05], nu=0.5)]

        #Wind =

        #kernels = [316**2 * Matern(length_scale=[5e+05, 6.62e+04, 1.07e+04], nu=0.5)]
        kernels = [eval(cov_function[0])]
        reg = GaussianProcessRegressor(kernel=kernels[0],normalize_y=True,n_restarts_optimizer=0,optimizer=None)     
    
    
        y = np.array(df_trainX['var']).reshape(-1,1)
        X_train = np.array(df_trainX[['xProj','yProj','elevS']])
        X_test = np.array(df_testX[['Xi','Yi','elev']])
    
        reg.fit(X_train, y)
    
        Zi = reg.predict(X_test)
    
        gpr_grid = Zi.reshape(num_row,num_col)

        #Calc the RMSE, MAE at the pixel loc
        #Delete at a certain point
        coord_pair = projected_lat_lon[station_name_hold_back]

        x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
        y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)

        interpolated_val = gpr_grid[y_orig][x_orig] 

        original_val = Cvar_dict[station_name_hold_back]
        absolute_error = abs(interpolated_val-original_val)
        absolute_error_dictionary[station_name_hold_back] = absolute_error

    return absolute_error_dictionary


def shuffle_split_gpr(latlon_dict,Cvar_dict,shapefile,file_path_elev,elev_array,idx_list,cov_function,rep):
    '''Shuffle split cross-validation procedure for GPR
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
    cov_function : list
        list containing a string that describes the input covariance function, similar to: ['316**2 * Matern(length_scale=[5e+05, 5e+05, 6.01e+03], nu=0.5)']
    rep : int
        number of replications
        
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
    
        Xi1_grd = np.array(Xi1_grd)
        Yi1_grd = np.array(Yi1_grd)
    
        df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'elevS':source_elev, 'var': z})
    
        df_testX = pd.DataFrame({'Xi': Xi1_grd, 'Yi': Yi1_grd, 'elev': elev_array})
    
    
        #kernels = [1.0 * RationalQuadratic(length_scale=1.0, alpha=alpha_input)]

        #Temperature 
        #kernels = [316**2 * Matern(length_scale=[5e+05, 5e+05, 6.01e+03], nu=0.5)]

        #RH
        #kernels = [307**2 * Matern(length_scale=[9.51e+04, 9.58e+04, 3.8e+05], nu=0.5)]

        #Wind

        #kernels = [316**2 * Matern(length_scale=[5e+05, 6.62e+04, 1.07e+04], nu=0.5)]

        #Okay, I know we are not supposed to do this, and it's a hack, but let's try Eval 
        kernels = [eval(cov_function[0])]
        
        reg = GaussianProcessRegressor(kernel=kernels[0],normalize_y=True,n_restarts_optimizer=0,optimizer=None)     
    
    
        y = np.array(df_trainX['var']).reshape(-1,1)
        X_train = np.array(df_trainX[['xProj','yProj','elevS']])
        X_test = np.array(df_testX[['Xi','Yi','elev']])
    
        reg.fit(X_train, y)
    
        Zi = reg.predict(X_test)
    
        gpr_grid = Zi.reshape(num_row,num_col)

        #Calc the RMSE, MAE at the pixel loc
        #Delete at a certain point
        for statLoc in test_stations: 
            coord_pair = projected_lat_lon[statLoc]

            x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
            y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
            x_origin_list.append(x_orig)
            y_origin_list.append(y_orig)

            interpolated_val = gpr_grid[y_orig][x_orig] 

            original_val = Cvar_dict[statLoc]
            absolute_error = abs(interpolated_val-original_val)
            absolute_error_dictionary[statLoc] = absolute_error
        error_dictionary[count]= sum(absolute_error_dictionary.values())/len(absolute_error_dictionary.values()) #average of all the withheld stations
        count+=1

    overall_error = sum(error_dictionary.values())/rep


    return overall_error
        

def spatial_kfold_gpr(idw_example_grid,loc_dict,Cvar_dict,shapefile,file_path_elev,elev_array,\
                      idx_list,cov_function,block_num,blocking_type):
    '''Spatially blocked k-folds cross-validation procedure for gpr 

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
    file_path_elev : string
        path to the elevation lookup file
    elev_array : ndarray
        array for elevation, create using IDEW interpolation (this is a trick to speed up code)         
    idx_list : int
        position of the elevation column in the lookup file
    cov_function : list
        list containing a string that describes the input covariance function, similar to: ['316**2 * Matern(length_scale=[5e+05, 5e+05, 6.01e+03], nu=0.5)']
    block_num : int
        number of blocks/clusters
    blocking_type : string
        whether to use clusters or blocks
        
    Returns
    ----------
    float
        - MAE estimate for entire surface
    int
        - Return the block number just so we can later write it into the file to keep track
    '''
    groups_complete = [] #If not using replacement, keep a record of what we have done 
    error_dictionary = {} 

    x_origin_list = []
    y_origin_list = [] 

    absolute_error_dictionary = {} 
    projected_lat_lon = {}

    #Selecting blocknum
    if blocking_type == 'cluster':
        cluster = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,block_num,file_path_elev,idx_list,False,False,False)
    elif blocking_type == 'block':
        np_array_blocks = mbk.make_block(idw_example_grid,block_num) #Get the numpy array that delineates the blocks
        cluster = mbk.sorting_stations(np_array_blocks,shapefile,loc_dict,Cvar_dict) #Now get the dictionary
    else:
        print('That is not a valid blocking method')
        sys.exit()
        
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

    Xi1_grd = np.array(Xi1_grd)
    Yi1_grd = np.array(Yi1_grd)

    df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'elevS':source_elev, 'var': z})

    df_testX = pd.DataFrame({'Xi': Xi1_grd, 'Yi': Yi1_grd, 'elev': elev_array})


    #kernels = [1.0 * RationalQuadratic(length_scale=1.0, alpha=alpha_input)]
    kernels = [eval(cov_function[0])]
    reg = GaussianProcessRegressor(kernel=kernels[0],normalize_y=True,n_restarts_optimizer=0,optimizer=None)    


    y = np.array(df_trainX['var']).reshape(-1,1)
    X_train = np.array(df_trainX[['xProj','yProj','elevS']])
    X_test = np.array(df_testX[['Xi','Yi','elev']])

    reg.fit(X_train, y)

    Zi = reg.predict(X_test)

    gpr_grid = Zi.reshape(num_row,num_col)

    #Calc the RMSE, MAE at the pixel loc
    #Delete at a certain point
    for statLoc in station_list: 
        coord_pair = projected_lat_lon[statLoc]

        x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
        y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)

        interpolated_val = gpr_grid[y_orig][x_orig] 

        original_val = Cvar_dict[statLoc]
        absolute_error = abs(interpolated_val-original_val)
        absolute_error_dictionary[statLoc] = absolute_error
        
    MAE= sum(absolute_error_dictionary.values())/len(absolute_error_dictionary.values()) #average of all the withheld stations
     
    return block_num, MAE

def select_block_size_gpr(nruns,group_type,loc_dict,Cvar_dict,idw_example_grid,shapefile,\
                          file_path_elev,idx_list,cluster_num1,cluster_num2,cluster_num3,\
                          expand_area,boreal_shapefile,cov_function):
     '''Evaluate the standard deviation of MAE values based on consective runs of the cross-valiation, 
     in order to select the block/cluster size
     
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
    cluster_num1-3 : int
        three cluster numbers to test, for blocking this must be one of three:25, 16, 9
        you can enter 'None' and it will automatically test 25, 16, 9
    expand_area : bool
        expand area by 200km
    boreal_shapefile : string
        path to shapefile with the boreal zone
    cov_function : list
        list containing a string that describes the input covariance function, similar to: ['316**2 * Matern(length_scale=[5e+05, 5e+05, 6.01e+03], nu=0.5)']
     
    Returns
    ----------
    int
        - block/cluster number with lowest stdev
    float
        - average MAE of all the runs for that cluster/block number
     '''
     
     #Get group dictionaries

     if group_type == 'blocks':
         #No support for expand_area
          folds25 = mbk.make_block(idw_example_grid,25)
          dictionaryGroups25 = mbk.sorting_stations(folds25,shapefile,Cvar_dict)
          folds16 = mbk.make_block(idw_example_grid,16)
          dictionaryGroups16 = mbk.sorting_stations(folds16,shapefile,Cvar_dict)
          folds9 = mbk.make_block(idw_example_grid,9)
          dictionaryGroups9 = mbk.sorting_stations(folds9,shapefile,Cvar_dict)

     elif group_type == 'clusters':
         if expand_area: 
               inBoreal = GD.is_station_in_boreal(loc_dict,Cvar_dict,boreal_shapefile)
               Cvar_dict = {k: v for k, v in Cvar_dict.items() if k in inBoreal} #Overwrite cvar_dict
               dictionaryGroups25 = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,cluster_num1,\
                                                        file_path_elev,idx_list,False,False,False)
               dictionaryGroups16 = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,cluster_num2,\
                                                        file_path_elev,idx_list,False,False,False)
               dictionaryGroups9 = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,cluster_num3,\
                                                       file_path_elev,idx_list,False,False,False)
         else: 
               dictionaryGroups25 = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,cluster_num1,\
                                                        file_path_elev,idx_list,False,False,False)
               dictionaryGroups16 = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,cluster_num2,\
                                                        file_path_elev,idx_list,False,False,False)
               dictionaryGroups9 = c3d.spatial_cluster(loc_dict,Cvar_dict,shapefile,cluster_num3,\
                                                       file_path_elev,idx_list,False,False,False)

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
##         target_stations = len(Cvar_dict.keys())*0.3 # What is 30% of the stations
##         fold_num1 = int(round(target_stations/cluster_num1))
##         fold_num2 = int(round(target_stations/cluster_num2))
##         fold_num3 = int(round(target_stations/cluster_num3)) 
            
          #For our first project, this is what we did
         fold_num1 = 5
         fold_num2 = 8
         fold_num3 = 14 
          #Just so there is a record of that


         block25 = spatial_groups_gpr(idw_example_grid,loc_dict,Cvar_dict,shapefile,file_path_elev,idx_list,\
                                      cluster_num1,fold_num1,\
                                      True,dictionaryGroups25,cov_function,expand_area)
         block25_error.append(block25)
         block16 = spatial_groups_gpr(idw_example_grid,loc_dict,Cvar_dict,shapefile,file_path_elev,idx_list,\
                                      cluster_num2,fold_num2,\
                                      True,dictionaryGroups16,cov_function,expand_area)
         block16_error.append(block16)
         block9 = spatial_groups_gpr(idw_example_grid,loc_dict,Cvar_dict,shapefile,file_path_elev,idx_list,\
                                     cluster_num3,fold_num3,\
                                     True,dictionaryGroups9,cov_function,expand_area)
         block9_error.append(block9)

     stdev25 = statistics.stdev(block25_error) 
     stdev16 = statistics.stdev(block16_error)
     stdev9 = statistics.stdev(block9_error)

     list_stdev = [stdev25,stdev16,stdev9]
     list_block_name = [cluster_num1,cluster_num2,cluster_num3]
     list_error = [block25_error,block16_error,block9_error]
     index_min = list_stdev.index(min(list_stdev))
     stdev_number = min(list_stdev)
     lowest_stdev = list_block_name[index_min]

     ave_MAE = sum(list_error[index_min])/len(list_error[index_min]) 

     print(lowest_stdev)
     #print(ave_MAE) 
     return lowest_stdev,ave_MAE
    
def spatial_groups_gpr(idw_example_grid,loc_dict,Cvar_dict,shapefile,file_path_elev,idx_list,blocknum,\
                       nfolds,replacement,dictionary_Groups,cov_function,expand_area):
     '''Stratified shuffle-split cross-validation procedure for gpr

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
    blocknum : int
        number of blocks/clusters
    nfolds : int
        number of folds to create (essentially repetitions)
    replacement : bool
        whether or not to use replacement between folds, should usually be true
    dictionary_Groups : dictionary
        dictionary of what groups (clusters) the stations belong to
    cov_function : list
        list containing a string that describes the input covariance function, similar to: ['316**2 * Matern(length_scale=[5e+05, 5e+05, 6.01e+03], nu=0.5)']
    expand_area : bool
        function will expand the study area so that more stations are taken into account (200 km)

    Returns
    ----------
    dictionary
        - a dictionary of the absolute error at each fold when it was left out
     '''
     station_list_used = [] #If not using replacement, keep a record of what we have done 
     count = 1
     error_dictionary = {}

     na_map = gpd.read_file(shapefile)
     bounds = na_map.bounds
     if expand_area: 
        xmax = bounds['maxx']+200000 
        xmin= bounds['minx']-200000 
        ymax = bounds['maxy']+200000 
        ymin = bounds['miny']-200000
     else:
        xmax = bounds['maxx']
        xmin= bounds['minx']
        ymax = bounds['maxy']
        ymin = bounds['miny']  
     
     while count <= nfolds: 
          x_origin_list = []
          y_origin_list = [] 

          absolute_error_dictionary = {} 
          projected_lat_lon = {}

          station_list = Eval.select_random_station(dictionary_Groups,blocknum,replacement,station_list_used).values()


          if replacement == False: 
               station_list_used.append(list(station_list))
          #print(station_list_used) 

          

                    
          for station_name in Cvar_dict.keys():
               
               if station_name in loc_dict.keys():

                  loc = loc_dict[station_name]
                  latitude = loc[0]
                  longitude = loc[1]
                  Plat, Plon = pyproj.Proj('esri:102001')(longitude,latitude)
                  Plat = float(Plat)
                  Plon = float(Plon)
                  proj_coord = pyproj.Proj('esri:102001')(longitude,latitude) #Filter out stations outside of grid
                  if (proj_coord[1] <= float(ymax[0]) and proj_coord[1] >= float(ymin[0]) and proj_coord[0] <= float(xmax[0]) and proj_coord[0] >= float(xmin[0])):
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
                         proj_coord = pyproj.Proj('esri:102001')(longitude,latitude) #Filter out stations outside of grid
                         if (proj_coord[1] <= float(ymax[0]) and proj_coord[1] >= float(ymin[0]) and proj_coord[0] <= float(xmax[0]) and proj_coord[0] >= float(xmin[0])):
                              lat.append(float(latitude))
                              lon.append(float(longitude))
                              Cvar.append(cvar_val)
                    else:
                         pass #Skip the station 
                     
          y = np.array(lat)
          x = np.array(lon)
          z = np.array(Cvar) 
             

          pixelHeight = 10000 
          pixelWidth = 10000
                     
          num_col = int((xmax - xmin) / pixelHeight)
          num_row = int((ymax - ymin) / pixelWidth)


               #We need to project to a projected system before making distance matrix
          source_proj = pyproj.Proj(proj='latlong', datum = 'NAD83') 
          xProj, yProj = pyproj.Proj('esri:102001')(x,y)

          df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'var': z})

          if expand_area: 

             yProj_extent=np.append(yProj,[bounds['maxy']+200000,bounds['miny']-200000])
             xProj_extent=np.append(xProj,[bounds['maxx']+200000,bounds['minx']-200000])
          else:
             yProj_extent=np.append(yProj,[bounds['maxy'],bounds['miny']])
             xProj_extent=np.append(xProj,[bounds['maxx'],bounds['minx']])    


          Yi = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row+1)
          Xi = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col+1)

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

          Xi1_grd = np.array(Xi1_grd)
          Yi1_grd = np.array(Yi1_grd)

          df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'elevS':source_elev, 'var': z})

          df_testX = pd.DataFrame({'Xi': Xi1_grd, 'Yi': Yi1_grd, 'elev': elev_array})

          kernels = [eval(cov_function[0])]
          reg = GaussianProcessRegressor(kernel=kernels[0],normalize_y=True,n_restarts_optimizer=0,optimizer=None)   


          y = np.array(df_trainX['var']).reshape(-1,1)
          X_train = np.array(df_trainX[['xProj','yProj','elevS']])
          X_test = np.array(df_testX[['Xi','Yi','elev']])

          reg.fit(X_train, y)

          Zi = reg.predict(X_test)

          gpr_grid = Zi.reshape(num_row+1,num_col+1)

          #Compare at a certain point
          for statLoc in station_list:

               coord_pair = projected_lat_lon[statLoc]

               x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
               y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
               x_origin_list.append(x_orig)
               y_origin_list.append(y_orig)

               interpolated_val = gpr_grid[y_orig][x_orig] 

               original_val = Cvar_dict[statLoc]
               absolute_error = abs(interpolated_val-original_val)
               absolute_error_dictionary[statLoc] = absolute_error


          error_dictionary[count]= sum(absolute_error_dictionary.values())/len(absolute_error_dictionary.values()) #average of all the withheld stations
          #print(absolute_error_dictionary)
          count+=1
     overall_error = sum(error_dictionary.values())/nfolds #average of all the runs
     #print(overall_error)
     return overall_error

