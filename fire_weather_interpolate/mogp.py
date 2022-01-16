#coding: utf-8

"""
Summary
-------
Spatial interpolation functions for mogp interpolation using the scikit-learn package.

Requires installation of IPython if it is not already installed 
References
----------
https://github.com/GAMES-UChile/mogptk/blob/master/examples/example_human_activity_recognition.ipynb
 
"""

# import
import statistics
import Eval as Eval
import make_blocks as mbk
import cluster_3d as c3d
import get_data as GD
import mogptk #Requires user to comment out line 'X = self.dataset.get_prediction()' in def _to_kernel_format(self, X, Y=None) in model.py

import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import warnings
# Runtime warning suppress, this suppresses the /0 warning
warnings.filterwarnings("ignore")


def MOGP_interpolator(latlon_dict,daily_dict, Cvar_dict, Cvar_dict2, Cvar_dict3, Cvar_dict4, input_date, var_name, shapefile, show,
                     file_path_elev, idx_list, expand_area, res=10000):
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
    Cvar2 = []
    Cvar3 = []
    Cvar4 = [] 

    na_map = gpd.read_file(shapefile)
    bounds = na_map.bounds
    if expand_area:
        xmax = bounds['maxx']+200000
        xmin = bounds['minx']-200000
        ymax = bounds['maxy']+200000
        ymin = bounds['miny']-200000
    else:
        xmax = bounds['maxx']
        xmin = bounds['minx']
        ymax = bounds['maxy']
        ymin = bounds['miny']

    list_of_stations = [Cvar_dict.keys(),Cvar_dict2.keys(),Cvar_dict3.keys(),Cvar_dict4.keys()]

    combo_keys = [j for i in list_of_stations for j in i]

    for station_name in combo_keys:
        if station_name in latlon_dict.keys() or station_name in daily_dict.keys(): #or daily_dict
            if station_name in latlon_dict.keys():

                loc = latlon_dict[station_name]
                latitude = loc[0]
                longitude = loc[1]
            elif station_name in daily_dict.keys():
                loc = daily_dict[station_name]
                latitude = loc[0]
                longitude = loc[1]
            else:
                pass 
            
            # Filter out stations outside of grid
            proj_coord = pyproj.Proj('esri:102001')(longitude, latitude)
            if (proj_coord[1] <= float(ymax[0]) and proj_coord[1] >=
                float(ymin[0]) and proj_coord[0] <= float(xmax[0]) and
                    proj_coord[0] >= float(xmin[0])):
                if station_name in Cvar_dict.keys(): 
                    cvar_val = Cvar_dict[station_name]
                else:
                    cvar_val = np.nan
                if station_name in Cvar_dict2.keys(): 
                    cvar2_val = Cvar_dict2[station_name]
                else:
                    cvar2_val = np.nan
                if station_name in Cvar_dict3.keys():
                    cvar3_val = Cvar_dict3[station_name]
                else:
                    cvar3_val = np.nan
                if station_name in Cvar_dict4.keys():
                    cvar4_val = Cvar_dict4[station_name]
                else:
                    cvar4_val = np.nan
                    
                lat.append(float(latitude))
                lon.append(float(longitude))
                Cvar.append(cvar_val)
                Cvar2.append(cvar2_val)
                Cvar3.append(cvar3_val)
                Cvar4.append(cvar4_val)

    y = np.array(lat)
    x = np.array(lon)
    z = np.array(Cvar)
    z2 = np.array(Cvar2)
    z3 = np.array(Cvar3)
    z4 = np.array(Cvar4)

    pixelHeight = res
    pixelWidth = res

    num_col = int((xmax - xmin) / pixelHeight)
    num_row = int((ymax - ymin) / pixelWidth)

    #print(num_col)

    # We need to project to a projected system before making distance matrix
    source_proj = pyproj.Proj(proj='latlong', datum='NAD83')
    xProj, yProj = pyproj.Proj('esri:102001')(x, y)

    df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'var': z,'var2':z2,'var3':z3,'var4':z4})

    if expand_area:

        yProj_extent = np.append(
            yProj, [bounds['maxy']+200000, bounds['miny']-200000])
        xProj_extent = np.append(
            xProj, [bounds['maxx']+200000, bounds['minx']-200000])

    else:
        yProj_extent = np.append(yProj, [bounds['maxy'], bounds['miny']])
        xProj_extent = np.append(xProj, [bounds['maxx'], bounds['minx']])

    Yi = np.linspace(np.min(yProj_extent), np.max(yProj_extent), num_row+1)
    Xi = np.linspace(np.min(xProj_extent), np.max(xProj_extent), num_col+1)

    Xi, Yi = np.meshgrid(Xi, Yi)
    Xi, Yi = Xi.flatten(), Yi.flatten()

    maxmin = [np.min(yProj_extent), np.max(yProj_extent),
              np.max(xProj_extent), np.min(xProj_extent)]

    # Elevation
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

    Xi1_grd = np.array(Xi1_grd)
    Yi1_grd = np.array(Yi1_grd)

    df_trainX = pd.DataFrame(
        {'xProj': xProj, 'yProj': yProj, 'elev': source_elev, 'var': z,'var2':z2,'var3':z3,'var4':z4})

    df_testX = pd.DataFrame({'xProj': Xi1_grd, 'yProj': Yi1_grd, 'elev': elev_array})
    
    df_testC = pd.DataFrame({0: Xi1_grd, 1: Yi1_grd, 2: elev_array})
    
    df_testX['var'] = np.nan
    df_testX['var2'] = np.nan
    df_testX['var3'] = np.nan
    df_testX['var4'] = np.nan

    trainer = pd.concat([df_trainX[['xProj', 'yProj', 'elev','var','var2','var3','var4']],\
                        df_testX[['xProj', 'yProj', 'elev','var','var2','var3','var4']]])

    len_trainer_1 = len(df_trainX)
    

    y = np.array(trainer[['var','var2','var3','var4']])
    y_rem = np.argwhere(np.isnan(y))
    y_test = np.array(df_testX[['var','var2','var3','var4']])
    #print(y)
    X_train = np.array(trainer[['xProj', 'yProj', 'elev']])
    #X_train_coords = [(x,y,z,) for x,y,z in zip(xProj,yProj,source_elev)]
    #print(X_train)
    X_test = list(df_testX[['xProj', 'yProj', 'elev']])

    #new_train = [X_train[:]]
    new_train = [X_train[:,0],X_train[:,1],X_train[:,2]]
    #print(new_train)
    #print(len(new_train))

    data = mogptk.DataSet()
    cols = ['Temp','RH','Wind','Pcp']
    for i in range(4):
        inst = mogptk.Data(X_train[:],y[:, i], name=cols[i]) #X_train[:,0] for 1 input (lon)
        y_rem = np.argwhere(np.isnan(np.array(y[:, i])))
        inst.remove_index(y_rem)
        data.append(inst)
    

    #print(data)
    #data.transform(mogptk.TransformDetrend()) #Cannot detrend on 2-3d input data
    #data.plot(title='Untrained model | Known Set')
    #plt.show()
    print('Data initialized') 
    model = mogptk.SM(data, Q=12) #Q = n* original Q for 1 input 
    model.init_parameters('LS')
    model.train(verbose=True,iters=100,lr=0.1) #Cannot be verbose in the IDLE, unless you edit the source code


    vals = model.predict()

    temp = vals[0][0][len_trainer_1:]
    rh = vals[0][1][len_trainer_1:]
    wind = vals[0][2][len_trainer_1:]
    pcp = vals[0][3][len_trainer_1:]
    
    #print(X_train)
##    np.set_printoptions(suppress=True)
##
##    results = pd.DataFrame()
##
##    
##
##    results['Lon'] = list(trainer['xProj'])
##    results['Lat'] = list(trainer['yProj'])
##    results['Temp'] = vals[0][0]
##    results['Rh'] = vals[0][1]
##    #results['Pred3'] = vals[0][2]
##    results['TempLower-Conf'] = vals[1][0]
##    results['RhLower-Conf'] = vals[1][1]
##    results['TempUpper-Conf'] = vals[2][0]
##    results['RhUpper-Conf'] = vals[2][1]
##
##    print(np.shape(vals))
##    
##    print(results)
##    results.to_csv('Data.txt',sep=',') 
    
    #data.plot(title='Trained model on Unknown Set')
    #plt.show()

    #This part is what we do for the xval, after initial model training! 
    
##    training_triple = [new_train,new_train,new_train,new_train] #,X_train[:,0],X_train[:,0]] #[new_train] for single output
##    print(len(training_triple))
##    print(training_triple)
##    vals = model.predict(training_triple)
##
##    print(vals)
    
    

def cross_validate_mogp(latlon_dict, daily_dict,  Cvar_dict, Cvar_dict2, \
                        Cvar_dict3, Cvar_dict4, shapefile,file_path_elev, idx_list, res=10000):
    '''Leave-one-out cross-validation procedure for MOGP

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
    '''

    lat = []
    lon = []
    Cvar = []
    Cvar2 = []
    Cvar3 = []
    Cvar4 = []

    temp_absolute_error_dictionary = {}
    rh_absolute_error_dictionary = {}
    wind_absolute_error_dictionary = {}
    pcp_absolute_error_dictionary = {}
    
    projected_lat_lon = {}
    
    #station lists for xval

    na_map = gpd.read_file(shapefile)
    bounds = na_map.bounds
    xmax = bounds['maxx']
    xmin = bounds['minx']
    ymax = bounds['maxy']
    ymin = bounds['miny']

    list_of_stations = [Cvar_dict.keys(),Cvar_dict2.keys(),Cvar_dict3.keys(),Cvar_dict4.keys()] 

    combo_keys = [j for i in list_of_stations for j in i]
    station_set = list(set(combo_keys))

    for station_name in station_set:
        if station_name in latlon_dict.keys() or station_name in daily_dict.keys(): #or daily_dict
            if station_name in latlon_dict.keys():

                loc = latlon_dict[station_name]
                latitude = loc[0]
                longitude = loc[1]
            elif station_name in daily_dict.keys():
                loc = daily_dict[station_name]
                latitude = loc[0]
                longitude = loc[1]
            else:
                pass
            
            # Filter out stations outside of grid
            Plat, Plon = pyproj.Proj('esri:102001')(longitude, latitude)
            proj_coord = pyproj.Proj('esri:102001')(longitude, latitude)
            if (proj_coord[1] <= float(ymax[0]) and proj_coord[1] >=
                float(ymin[0]) and proj_coord[0] <= float(xmax[0]) and
                    proj_coord[0] >= float(xmin[0])):
                Plat = float(Plat)
                Plon = float(Plon)
                projected_lat_lon[station_name] = [Plat, Plon]

    for station_name_hold_back in station_set:

        lat = []
        lon = []
        Cvar = []
        Cvar2 = []
        Cvar3 = []
        Cvar4 = [] 
        for station_name in station_set:
            if station_name in latlon_dict.keys() or station_name in daily_dict.keys(): #or daily_dict
                if station_name != station_name_hold_back:
                    if station_name in latlon_dict.keys():
                        loc = latlon_dict[station_name]
                        latitude = loc[0]
                        longitude = loc[1]
                    elif station_name in daily_dict.keys():
                        loc = daily_dict[station_name]
                        latitude = loc[0]
                        longitude = loc[1]
                    else:
                        pass
                
                    # Filter out stations outside of grid
                    proj_coord = pyproj.Proj('esri:102001')(longitude, latitude)
                    if (proj_coord[1] <= float(ymax[0]) and proj_coord[1] >=
                        float(ymin[0]) and proj_coord[0] <= float(xmax[0]) and
                            proj_coord[0] >= float(xmin[0])):
                        if station_name in Cvar_dict.keys(): 
                            cvar_val = Cvar_dict[station_name]
                        else:
                            cvar_val = np.nan
                        if station_name in Cvar_dict2.keys(): 
                            cvar2_val = Cvar_dict2[station_name]
                        else:
                            cvar2_val = np.nan
                        if station_name in Cvar_dict3.keys():
                            cvar3_val = Cvar_dict3[station_name]
                        else:
                            cvar3_val = np.nan
                        if station_name in Cvar_dict4.keys():
                            cvar4_val = Cvar_dict4[station_name]
                        else:
                            cvar4_val = np.nan
                            
                        lat.append(float(latitude))
                        lon.append(float(longitude))
                        Cvar.append(cvar_val)
                        Cvar2.append(cvar2_val)
                        Cvar3.append(cvar3_val)
                        Cvar4.append(cvar4_val)



        y = np.array(lat)
        x = np.array(lon)
        z = np.array(Cvar)
        z2 = np.array(Cvar2)
        z3 = np.array(Cvar3)
        z4 = np.array(Cvar4)

        pixelHeight = res
        pixelWidth = res

        num_col = int((xmax - xmin) / pixelHeight)
        num_row = int((ymax - ymin) / pixelWidth)

        #print(num_col)

        # We need to project to a projected system before making distance matrix
        source_proj = pyproj.Proj(proj='latlong', datum='NAD83')
        xProj, yProj = pyproj.Proj('esri:102001')(x, y)

        df_trainX = pd.DataFrame({'xProj': xProj, 'yProj': yProj, 'var': z,'var2':z2,'var3':z3,'var4':z4})

        yProj_extent = np.append(yProj, [bounds['maxy'], bounds['miny']])
        xProj_extent = np.append(xProj, [bounds['maxx'], bounds['minx']])

        Yi = np.linspace(np.min(yProj_extent), np.max(yProj_extent), num_row+1)
        Xi = np.linspace(np.min(xProj_extent), np.max(xProj_extent), num_col+1)

        Xi, Yi = np.meshgrid(Xi, Yi)
        Xi, Yi = Xi.flatten(), Yi.flatten()

        maxmin = [np.min(yProj_extent), np.max(yProj_extent),
                  np.max(xProj_extent), np.min(xProj_extent)]

        # Elevation
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

        Xi1_grd = np.array(Xi1_grd)
        Yi1_grd = np.array(Yi1_grd)

        df_trainX = pd.DataFrame(
            {'xProj': xProj, 'yProj': yProj, 'elev': source_elev, 'var': z,'var2':z2,'var3':z3,'var4':z4})

        df_testX = pd.DataFrame({'xProj': Xi1_grd, 'yProj': Yi1_grd, 'elev': elev_array})
        
        df_testC = pd.DataFrame({0: Xi1_grd, 1: Yi1_grd, 2: elev_array})
        
        df_testX['var'] = np.nan
        df_testX['var2'] = np.nan
        df_testX['var3'] = np.nan
        df_testX['var4'] = np.nan

        trainer = pd.concat([df_trainX[['xProj', 'yProj', 'elev','var','var2','var3','var4']],\
                            df_testX[['xProj', 'yProj', 'elev','var','var2','var3','var4']]])

        len_trainer_1 = len(df_trainX)
        

        y = np.array(trainer[['var','var2','var3','var4']])
        y_rem = np.argwhere(np.isnan(y))
        y_test = np.array(df_testX[['var','var2','var3','var4']])
        #print(y)
        X_train = np.array(trainer[['xProj', 'yProj', 'elev']])
        #X_train_coords = [(x,y,z,) for x,y,z in zip(xProj,yProj,source_elev)]
        #print(X_train)
        X_test = np.array(df_testX[['xProj', 'yProj', 'elev']])

        #new_train = [X_train[:]]
        new_train = [X_train[:,0],X_train[:,1],X_train[:,2]]
        total_len= len(X_test[:,0])

        eights = int(total_len/8)
        e1 = eights+len_trainer_1
        e2 = eights*2
        e3 = eights*3
        e4 = eights*4
        e5 = eights*5
        e6 = eights*6
        e7 = eights*7
       
        new_train1 =[X_train[:len_trainer_1,0],X_train[:len_trainer_1,1],X_train[:len_trainer_1,2]]
        new_train2 = [X_train[len_trainer_1:e1,0],X_train[len_trainer_1:e1,1],X_train[len_trainer_1:e1,2]] #does the trainer need to be in there?
        new_train3 = [X_train[e1:e2,0],X_train[e1:e2,1],X_train[e1:e2,2]]

        new_train4 = [X_train[e1:e2,0],X_train[e1:e2,1],X_train[e1:e2,2]]
        new_train5 = [X_train[e2:e3,0],X_train[e2:e3,1],X_train[e2:e3,2]]
        new_train6 = [X_train[e3:e4,0],X_train[e3:e4,1],X_train[e3:e4,2]]
        new_train7 = [X_train[e4:e5,0],X_train[e4:e5,1],X_train[e4:e5,2]]
        new_train8 = [X_train[e5:e6,0],X_train[e5:e6,1],X_train[e5:e6,2]]
        new_train9 = [X_train[e6:e7,0],X_train[e6:e7,1],X_train[e6:e7,2]]
        new_train10 = [X_train[e7:,0],X_train[e7:,1],X_train[e7:,2]]
        

        training1 = [new_train1,new_train1,new_train1,new_train1]
        training2 = [new_train2,new_train2,new_train2,new_train2]
        training3 = [new_train3,new_train3,new_train3,new_train3]
        training4 = [new_train4,new_train4,new_train4,new_train4]
        training5 = [new_train5,new_train5,new_train5,new_train5]
        training6 = [new_train6,new_train6,new_train6,new_train6]
        training7 = [new_train7,new_train7,new_train7,new_train7]
        training8 = [new_train8,new_train8,new_train8,new_train8]
        training9 = [new_train9,new_train9,new_train9,new_train9]
        training10 = [new_train10,new_train10,new_train10,new_train10]
        
        data = mogptk.DataSet()
        cols = ['Temp','RH','Wind','Pcp']
        for i in range(4):
            inst = mogptk.Data(X_train[:],y[:, i], name=cols[i]) #X_train[:,0] for 1 input (lon)
            y_rem = np.argwhere(np.isnan(np.array(y[:, i])))
            inst.remove_index(y_rem)
            data.append(inst)
        

        #print(data)
        #data.transform(mogptk.TransformDetrend()) #Cannot detrend on 2-3d input data
        #data.plot(title='Untrained model | Known Set')
        #plt.show()
        print('Data initialized') 
        model = mogptk.SM(data, Q=12) #Q = n* original Q for 1 input 
        model.init_parameters('LS')
        model.train(verbose=True,iters=100,lr=0.1) #Cannot be verbose in the IDLE, unless you edit the source code
        #surfaces = model.predict(training_triple)
        surfaces1 = model.predict(training2)
        surfaces2 = model.predict(training3)
        surfaces3 = model.predict(training4)
        surfaces4 = model.predict(training5)
        surfaces5 = model.predict(training6)
        surfaces6 = model.predict(training7)
        surfaces7 = model.predict(training8)
        surfaces8 = model.predict(training9)
        surfaces9 = model.predict(training10)
        
        
        #surfaces = model.predict()
        #gtemp = surfaces[0][0][len_trainer_1:] + surfaces1[0][0]
        gtemp = np.array(list(surfaces1[0][0]) + list(surfaces2[0][0]) + list(surfaces3[0][0])\
                         + list(surfaces4[0][0])+ list(surfaces5[0][0])+ list(surfaces6[0][0])\
                         + list(surfaces7[0][0])+ list(surfaces8[0][0])+ list(surfaces9[0][0]))
        gtemp = gtemp.reshape(num_row + 1, num_col + 1)

        grh = np.array(list(surfaces1[0][1]) + list(surfaces2[0][1]) + list(surfaces3[0][1])\
                         + list(surfaces4[0][1])+ list(surfaces5[0][1])+ list(surfaces6[0][1])\
                         + list(surfaces7[0][1])+ list(surfaces8[0][1])+ list(surfaces9[0][1]))
        grh = grh.reshape(num_row + 1, num_col + 1)
        
        
        gwind = np.array(list(surfaces1[0][2]) + list(surfaces2[0][2]) + list(surfaces3[0][2])\
                         + list(surfaces4[0][2])+ list(surfaces5[0][2])+ list(surfaces6[0][2])\
                         + list(surfaces7[0][2])+ list(surfaces8[0][2])+ list(surfaces9[0][2]))
        gwind = gwind.reshape(num_row + 1, num_col + 1)

        gpcp = np.array(list(surfaces1[0][3]) + list(surfaces2[0][3]) + list(surfaces3[0][3])\
                         + list(surfaces4[0][3])+ list(surfaces5[0][3])+ list(surfaces6[0][3])\
                         + list(surfaces7[0][3])+ list(surfaces8[0][3])+ list(surfaces9[0][3]))
        gpcp = gwind.reshape(num_row + 1, num_col + 1)
        
        
        coord_pair = projected_lat_lon[station_name_hold_back]
        x_orig = int((coord_pair[0] - float(xmin)) / pixelHeight)  # lon
        y_orig = int((coord_pair[1] - float(ymin)) / pixelWidth)  # lat
        
        #Temp
        interpolated_temp = gtemp[y_orig][x_orig]

        #RH
        interpolated_rh = grh[y_orig][x_orig]

        #Wind
        interpolated_wind = gwind[y_orig][x_orig]

        #pcp
        interpolated_pcp = gpcp[y_orig][x_orig]

        if station_name_hold_back in Cvar_dict.keys():
            original_val = Cvar_dict[station_name_hold_back]
            absolute_error = abs(interpolated_temp - original_val)
            temp_absolute_error_dictionary[station_name_hold_back] = absolute_error
        if station_name_hold_back in Cvar_dict2.keys():
            original_val = Cvar_dict2[station_name_hold_back]
            absolute_error = abs(interpolated_rh - original_val)
            rh_absolute_error_dictionary[station_name_hold_back] = absolute_error            
        if station_name_hold_back in Cvar_dict3.keys():
            original_val = Cvar_dict3[station_name_hold_back]
            absolute_error = abs(interpolated_wind - original_val)
            wind_absolute_error_dictionary[station_name_hold_back] = absolute_error
        if station_name_hold_back in Cvar_dict4.keys():
            original_val = Cvar_dict4[station_name_hold_back]
            absolute_error = abs(interpolated_pcp - original_val)
            pcp_absolute_error_dictionary[station_name_hold_back] = absolute_error

    
    return temp_absolute_error_dictionary,rh_absolute_error_dictionary,wind_absolute_error_dictionary,\
           pcp_absolute_error_dictionary

