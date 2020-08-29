#coding: utf-8

"""
Summary
-------
Spatial interpolation functions for ordinary kriging using the PyKrige package. 

"""
    
#import
import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
from pykrige.ok import OrdinaryKriging
from skgstat import Variogram
import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

import cluster_3d as c3d

import warnings
warnings.filterwarnings("ignore") #Runtime warning suppress, this suppresses the /0 warning

def get_pandas_df(Cvar_dict,latlon_dict):
    '''Make a pandas dataframe for input into the check_variogram function
    Parameters
        Cvar_dict (dict): dictionary of weather variable values at each active
        station
        latlon_dict (dict): dictionary of station locations 
    Returns
        df (pandas dataframe): pandas dataframe, containes value & location 
    '''
    station_name_list = []
    projected_lat_lon = {} 
    Cvar_lookup = {} 
    for station_name in Cvar_dict.keys():
        station_name_list.append(station_name)

        loc = latlon_dict[station_name]
        latitude = loc[0]
        longitude = loc[1]
        Plat, Plon = pyproj.Proj('esri:102001')(longitude,latitude)
        Plat = float(Plat)
        Plon = float(Plon)
        projected_lat_lon[station_name] = [Plat,Plon]
        Cvar_lookup[tuple([Plat,Plon])] = float(Cvar_dict[station_name])

    xProj_input=[]
    yProj_input=[]

    cvar_input = []
    

    for keys in Cvar_lookup.keys():
        x= keys[0]
        y = keys[1]
        xProj_input.append(x)
        yProj_input.append(y)

        cvar_input.append(Cvar_lookup[keys])


    model = {
        'z': np.array(cvar_input),
        'x': np.array(xProj_input),
        'y': np.array(yProj_input),
        }

    df = pd.DataFrame(model, columns=['z','x','y'])

    num_stations = str(len(station_name_list)) 
    print('Number of stations with valid data on the study date: %s'%(num_stations))

    return df


def check_variogram(df):
    '''Make a pandas dataframe for input into the check_variogram function
    Parameters
        df (pandas dataframe): pandas dataframe containing value & location
    Returns
        RMSE_best (list): list containing the best semivariogram model(s) for
        the data contained in the dataframe (values for active stations on a
        certain date) 
    '''
    RMSE_list = []
    RMSE_best = [] 
    try: 
        data = df
        x = df['x'].astype(object)
        y = df['y'].astype(object)
        z = df['z'].astype(object)
        V = Variogram(list(zip(x, y)), z, normalize=False)
        V.model = 'exponential'
        RMSE_list.append(V.rmse)
        V2 = Variogram(list(zip(x, y)), z, normalize=False)
        V2.model = 'spherical'

        RMSE_list.append(V2.rmse)
        V3 = Variogram(list(zip(x, y)), z, normalize=False)
        V3.model = 'gaussian'
        RMSE_list.append(V3.rmse)

    except ValueError:
        print('Variogram failed, likely a day with too little records.')
        #default selection will be exponential
        RMSE_best.append('exponential') 
    listm = ['exponential','spherical','gaussian']
    idx = [i for i, x in enumerate(RMSE_list) if x == min(RMSE_list)]
    if len(set(RMSE_list)) == 1: #if all equal, also pick exponential 
        RMSE_best.append('exponential')
    elif len(idx) == 2: #if two are equal, just take the first 
        RMSE_best.append(str(listm[idx[0]]))
    elif len(idx) == 1: 
        RMSE_best.append(listm[idx[0]])

    print(RMSE_best) 

    return RMSE_best

#functions 
def OKriging(latlon_dict,Cvar_dict,input_date,var_name,shapefile,model,show):
    '''Implement ordinary kriging 
    Parameters
        latlon_dict (dict): the latitude and longitudes of the hourly stations, loaded from the 
        .json file
        Cvar_dict (dict): dictionary of weather variable values for each station 
        input_date (str): the date you want to interpolate for 
        var_name (str): name of the variable you are interpolating
        shapefile (str): path to the study area shapefile 
        model (str): semivariogram model name, ex. 'gaussian'
        show (bool): whether you want to plot a map 
    Returns 
        kriging_surface (np_array): the array of values for the interpolated surface
        maxmin: the bounds of the array surface, for use in other functions 
        
        if there is an error when interpolating the surface, it does not return anything 
    '''
    x_origin_list = []
    y_origin_list = []
    z_origin_list = [] 

    absolute_error_dictionary = {} #for plotting
    station_name_list = []
    projected_lat_lon = {}

    for station_name in Cvar_dict.keys():
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
    for station_name in Cvar_dict.keys(): #DONT use list of stations, because if there's a no data we delete that in the climate dictionary step 
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

    Cvar = []
    for station_name in Cvar_dict.keys(): #DONT use list of stations, because if there's a no data we delete that in the climate dictionary step 

        cvar_val = Cvar_dict[station_name]

        Cvar.append(cvar_val)

    z2 = np.array(Cvar)


    #for station_name in sorted(Cvar_dict.keys()): #DONT use list of stations, because if there's a no data we delete that in the climate dictionary step
    for station_name_hold_back in station_name_list:

        na_map = gpd.read_file(shapefile)
        bounds = na_map.bounds

        pixelHeight = 10000 
        pixelWidth = 10000


        coord_pair = projected_lat_lon[station_name_hold_back]

        x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
        y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)
        z_origin_list.append(Cvar_dict[station_name_hold_back])


    #Uncomment to check if the correct stations are being accessed
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
    

    source_proj = pyproj.Proj(proj='latlong', datum = 'NAD83') #We dont know but assume 
    xProj, yProj = pyproj.Proj('esri:102001')(x,y)

    yProj_extent=np.append(yProj,[bounds['maxy'],bounds['miny']])
    xProj_extent=np.append(xProj,[bounds['maxx'],bounds['minx']])

    maxmin = [np.min(yProj_extent),np.max(yProj_extent),np.max(xProj_extent),np.min(xProj_extent)]

    Yi1 = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row)
    Xi1 = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col)

    Xi,Yi = np.meshgrid(Xi1,Yi1)

    empty_grid = np.empty((num_row,num_col,))*np.nan

    for x3,y3,z3 in zip(x_origin_list,y_origin_list,z_origin_list):
        empty_grid[y3][x3] = z3



    vals = ~np.isnan(empty_grid)



    OK = OrdinaryKriging(xProj,yProj,z,variogram_model=model,verbose=False,enable_plotting=False)
    try: 
        z1,ss1 = OK.execute('grid',Xi1,Yi1,n_closest_points=10,backend='C') #n_closest_points=10
        
        kriging_surface = z1.reshape(num_row,num_col)
        if show:
            fig, ax = plt.subplots(figsize= (15,15))
            crs = {'init': 'esri:102001'}

            na_map = gpd.read_file(shapefile)
            
          
            plt.imshow(kriging_surface,extent=(xProj_extent.min()-1,xProj_extent.max()+1,yProj_extent.max()-1,yProj_extent.min()+1)) 
            na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)
                
            plt.scatter(xProj,yProj,c=z_origin_list,edgecolors='k',linewidth=1)

            plt.gca().invert_yaxis()
            cbar = plt.colorbar()
            cbar.set_label(var_name) 
            
            title = 'Ordinary Kriging Interpolation for %s on %s'%(var_name,input_date) 
            fig.suptitle(title, fontsize=14)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude') 

            plt.show()

        return kriging_surface, maxmin

    except:
        pass 

    

def cross_validate_OK(latlon_dict,Cvar_dict,shapefile,model):
    '''Cross_validate the ordinary kriging 
    Parameters 
        latlon_dict (dict): the latitude and longitudes of the hourly stations, loaded from the 
        .json file
        Cvar_dict (dict): dictionary of weather variable values for each station 
        shapefile (str): path to the study area shapefile 
        model (str): semivariogram model name, ex. 'gaussian'
    Returns 
        absolute_error_dictionary (dict): a dictionary of the absolute error at each station when it
        was left out 
    '''
    x_origin_list = []
    y_origin_list = [] 
    z_origin_list = []
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

        na_map = gpd.read_file(shapefile)
        bounds = na_map.bounds

        pixelHeight = 10000 
        pixelWidth = 10000


        coord_pair = projected_lat_lon[station_name_hold_back]

        x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
        y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)
        z_origin_list.append(Cvar_dict[station_name_hold_back])




    #for station_name in sorted(Cvar_dict.keys()): #DONT use list of stations, because if there's a no data we delete that in the climate dictionary step
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
                    #print('Skipping station!')
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

        Yi1 = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row)
        Xi1 = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col)

        Xi,Yi = np.meshgrid(Xi1,Yi1)
        
        empty_grid = np.empty((num_row,num_col,))*np.nan

        for x3,y3,z3 in zip(x_origin_list,y_origin_list,z_origin_list):
            empty_grid[y3][x3] = z3



        vals = ~np.isnan(empty_grid)

        OK = OrdinaryKriging(xProj,yProj,z,variogram_model=model,verbose=False,enable_plotting=False)
        try: 
            z1,ss1 = OK.execute('grid',Xi1,Yi1,n_closest_points=10,backend='C') #n_closest_points=10
    
            kriging_surface = z1.reshape(num_row,num_col)

        #Calc the RMSE, MAE at the pixel loc
        #Delete at a certain point
            coord_pair = projected_lat_lon[station_name_hold_back]

            x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
            y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
            x_origin_list.append(x_orig)
            y_origin_list.append(y_orig)

            interpolated_val = kriging_surface[y_orig][x_orig] #which comes first?

            original_val = Cvar_dict[station_name_hold_back]
            absolute_error = abs(interpolated_val-original_val)
            absolute_error_dictionary[station_name_hold_back] = absolute_error
        except:
            pass 


    return absolute_error_dictionary  


def leave_p_out_crossval(latlon_dict,Cvar_dict,shapefile,model,nruns,p):
    '''Cross_validate the ordinary kriging, but only with p weather stations for faster selection of the semivariogram
    Parameters 
        latlon_dict (dict): the latitude and longitudes of the hourly stations, loaded from the 
        .json file
        Cvar_dict (dict): dictionary of weather variable values for each station 
        shapefile (str): path to the study area shapefile 
        model (str): semivariogram model name, ex. 'gaussian'
        nruns (int): how many times to run the procedure, for bootstrapping, if nruns = 10 it will take the average of 10 runs
        p (int): number of weather stations to randomly sample for cross-validation with replacement
    Returns 
        MAE2 (float): the MAE resulting from the cross-validation procedure, to be used to evaluate different semivariograms 
    '''
    MAEs = {} 
    for n in range(0,nruns): 
        x_origin_list = []
        y_origin_list = [] 
        z_origin_list = []
        absolute_error_dictionary = {} 
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

            pixelHeight = 10000 
            pixelWidth = 10000


            coord_pair = projected_lat_lon[station_name_hold_back]

            x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
            y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
            x_origin_list.append(x_orig)
            y_origin_list.append(y_orig)
            z_origin_list.append(Cvar_dict[station_name_hold_back])

        #Making the groups of randomly selected, bootstrapping with replacement
        #weather stations to be held out together
        bootstrap_p = np.random.choice(station_name_list,p) 
        
        for station_name_hold_back in bootstrap_p:
             
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
                        #print('Skipping station!')
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

            Yi1 = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row)
            Xi1 = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col)

            Xi,Yi = np.meshgrid(Xi1,Yi1)
            
            empty_grid = np.empty((num_row,num_col,))*np.nan

            for x3,y3,z3 in zip(x_origin_list,y_origin_list,z_origin_list):
                empty_grid[y3][x3] = z3



            vals = ~np.isnan(empty_grid)

            OK = OrdinaryKriging(xProj,yProj,z,variogram_model=model,verbose=False,enable_plotting=False)
            try: 
                z1,ss1 = OK.execute('grid',Xi1,Yi1,backend='C') #n_closest_points=10
        
                kriging_surface = z1.reshape(num_row,num_col)

            #Calc the RMSE, MAE at the pixel loc
            #Delete at a certain point
                coord_pair = projected_lat_lon[station_name_hold_back]

                x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
                y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
                x_origin_list.append(x_orig)
                y_origin_list.append(y_orig)

                interpolated_val = kriging_surface[y_orig][x_orig] #which comes first?

                original_val = Cvar_dict[station_name_hold_back]
                absolute_error = abs(interpolated_val-original_val)
                absolute_error_dictionary[station_name_hold_back] = absolute_error
            except:
                pass
        try: 
            MAE = sum(absolute_error_dictionary.values())/len(absolute_error_dictionary.values())
        except ZeroDivisionError:
            MAE = 10000000000000000000 #If it fails, we dont want to use that semivariogram, so make the value large
            
        #print(MAE)
        MAEs[n] = MAE
    MAE2 = sum(MAEs.values())/len(MAEs.values()) #Sum of the results for all the runs 
    
    return MAE2 

def get_best_model(models,latlon_dict,Cvar_dict,shapefile,nruns,p):
    '''Select the best semivariogram using the leave_p_out cross-validation procedure 
    Parameters 
        models (list): the list of models you want to test, i.e. ['gaussian','exponenital','linear']
        latlon_dict (dict): the latitude and longitudes of the hourly stations, loaded from the .json file
        Cvar_dict (dict): dictionary of weather variable values for each station 
        shapefile (str): path to the study area shapefile 
        nruns (int): how many times to run the procedure, for bootstrapping, if nruns = 10 it will take the average of 10 runs
        p (int): number of weather stations to randomly sample for cross-validation with replacement
    Returns 
        selected (str): the best semivariogram according to the procedure 
    '''
    results = {} 
    for model in models:
        #print(model)
        #start = time.time() 
        MAE = leave_p_out_crossval(latlon_dict,Cvar_dict,shapefile,model,nruns,p)
        results[model] = MAE
        #end = time.time()
        #print('Time taken = %s'%(end-start))
    selected = min(results, key=results.get)
    
    return selected         


def shuffle_split_OK(latlon_dict,Cvar_dict,shapefile,model,rep):
    '''Shuffle-split cross_validate the ordinary kriging 
    Parameters 
        latlon_dict (dict): the latitude and longitudes of the hourly stations, loaded from the 
        .json file
        Cvar_dict (dict): dictionary of weather variable values for each station 
        shapefile (str): path to the study area shapefile 
        model (str): semivariogram model name, ex. 'gaussian'
    Returns 
        absolute_error_dictionary (dict): a dictionary of the absolute error at each station when it
        was left out 
    '''
    count = 1
    error_dictionary = {}
    while count <= rep:
        x_origin_list = []
        y_origin_list = [] 
        z_origin_list = []
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

        for station in station_name_list:

            na_map = gpd.read_file(shapefile)
            bounds = na_map.bounds

            pixelHeight = 10000 
            pixelWidth = 10000


            coord_pair = projected_lat_lon[station]

            x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
            y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
            x_origin_list.append(x_orig)
            y_origin_list.append(y_orig)
            z_origin_list.append(Cvar_dict[station])
            
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
                    #print('Skipping station!')
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

        Yi1 = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row)
        Xi1 = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col)

        Xi,Yi = np.meshgrid(Xi1,Yi1)
        
        empty_grid = np.empty((num_row,num_col,))*np.nan

        for x3,y3,z3 in zip(x_origin_list,y_origin_list,z_origin_list):
            empty_grid[y3][x3] = z3



        vals = ~np.isnan(empty_grid)

        OK = OrdinaryKriging(xProj,yProj,z,variogram_model=model,verbose=False,enable_plotting=False)
        try: 
            z1,ss1 = OK.execute('grid',Xi1,Yi1,n_closest_points=10,backend='C') #n_closest_points=10
    
            kriging_surface = z1.reshape(num_row,num_col)

        #Calc the RMSE, MAE at the pixel loc
        #Delete at a certain point
            for statLoc in test_stations: 
                coord_pair = projected_lat_lon[statLoc]

                x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
                y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
                x_origin_list.append(x_orig)
                y_origin_list.append(y_orig)

                interpolated_val = kriging_surface[y_orig][x_orig] #which comes first?

                original_val = Cvar_dict[statLoc]
                absolute_error = abs(interpolated_val-original_val)
                absolute_error_dictionary[statLoc] = absolute_error
            error_dictionary[count]= sum(absolute_error_dictionary.values())/len(absolute_error_dictionary.values()) #average of all the withheld stations
            count+=1
        except:
            pass 

    overall_error = sum(error_dictionary.values())/rep
    return overall_error

def spatial_kfold_OK(loc_dict,Cvar_dict,shapefile,phi,file_path_elev,elev_array,idx_list):
    '''Cross_validate the ordinary kriging 
    Parameters 
        latlon_dict (dict): the latitude and longitudes of the hourly stations, loaded from the 
        .json file
        Cvar_dict (dict): dictionary of weather variable values for each station 
        shapefile (str): path to the study area shapefile 
        model (str): semivariogram model name, ex. 'gaussian'
    Returns 
        absolute_error_dictionary (dict): a dictionary of the absolute error at each station when it
        was left out 
    '''
    groups_complete = [] #If not using replacement, keep a record of what we have done 
    error_dictionary = {} 

    x_origin_list = []
    y_origin_list = [] 
    z_origin_list = []
    absolute_error_dictionary = {} #for plotting
    station_name_list = []
    projected_lat_lon = {}

    for station_name in Cvar_dict.keys():
        if station_name in loc_dict.keys():
            station_name_list.append(station_name)

            loc = loc_dict[station_name]
            latitude = loc[0]
            longitude = loc[1]
            Plat, Plon = pyproj.Proj('esri:102001')(longitude,latitude)
            Plat = float(Plat)
            Plon = float(Plon)
            projected_lat_lon[station_name] = [Plat,Plon]

    for station in station_name_list:

        na_map = gpd.read_file(shapefile)
        bounds = na_map.bounds

        pixelHeight = 10000 
        pixelWidth = 10000


        coord_pair = projected_lat_lon[station]

        x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
        y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
        x_origin_list.append(x_orig)
        y_origin_list.append(y_orig)
        z_origin_list.append(Cvar_dict[station])
            
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
                #print('Skipping station!')
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

    Yi1 = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row)
    Xi1 = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col)

    Xi,Yi = np.meshgrid(Xi1,Yi1)
    
    empty_grid = np.empty((num_row,num_col,))*np.nan

    for x3,y3,z3 in zip(x_origin_list,y_origin_list,z_origin_list):
        empty_grid[y3][x3] = z3



    vals = ~np.isnan(empty_grid)

    OK = OrdinaryKriging(xProj,yProj,z,variogram_model=model,verbose=False,enable_plotting=False)
    try: 
        z1,ss1 = OK.execute('grid',Xi1,Yi1,n_closest_points=10,backend='C') #n_closest_points=10

        kriging_surface = z1.reshape(num_row,num_col)

    #Calc the RMSE, MAE at the pixel loc
    #Delete at a certain point
        for loc in station_list: 
            coord_pair = projected_lat_lon[loc]

            x_orig = int((coord_pair[0] - float(bounds['minx']))/pixelHeight) #lon 
            y_orig = int((coord_pair[1] - float(bounds['miny']))/pixelWidth) #lat
            x_origin_list.append(x_orig)
            y_origin_list.append(y_orig)

            interpolated_val = kriging_surface[y_orig][x_orig] #which comes first?

            original_val = Cvar_dict[loc]
            absolute_error = abs(interpolated_val-original_val)
            absolute_error_dictionary[loc] = absolute_error

    except:
        pass

    MAE= sum(absolute_error_dictionary.values())/len(absolute_error_dictionary.values())

    overall_error = sum(error_dictionary.values())/rep
    return blocknum,MAE
