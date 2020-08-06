#coding: latin1

"""
Summary
-------
Functions to evaluate the outputs of the FWI metrics functions, such as ridge regression for relating the metrics to area burned. 

References
----------
See: 
https://en.wikipedia.org/wiki/Tikhonov_regularization
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
"""
    
#import
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import QuantileTransformer

import math
import seaborn as sns 

from scipy.spatial import distance
from scipy.spatial.distance import cdist

#functions 
def get_interpolated_val_in_fire(fire_shapefile,shapefile,latlon_dict,interpolated_surface):
    '''This is a function to get the FWI metric value inside the fire.
    We will use to calculate the max FWI metrics for a fire.
    Parameters
        fire_shapefile (str): path to the fire shapefile 
        shapefile (str): path to the study area shapefile
        latlon_dict (dict, loaded from json): dictionary of lat lon for each station 
        interpolated_surface (np_array): an array of values in the study area 
    Returns 
        ival, max_ival (either, float): maximum value in fire, either the closest point to the convex hull of the fire or 
        a sum of the points inside the fire
    '''
    lat = [] #Initialize empty lists to store data 
    lon = []

    for station_name in latlon_dict.keys(): #Loop through the list of stations 
        loc = latlon_dict[station_name]
        latitude = loc[0]
        longitude = loc[1]
        lat.append(float(latitude))
        lon.append(float(longitude))

    y = np.array(lat) #Convert to a numpy array for faster processing speed 
    x = np.array(lon)


    na_map = gpd.read_file(shapefile)
    bounds = na_map.bounds #Get the bounding box of the shapefile 
    xmax = bounds['maxx']
    xmin= bounds['minx']
    ymax = bounds['maxy']
    ymin = bounds['miny']
    pixelHeight = 10000 #We want a 10 by 10 pixel, or as close as we can get 
    pixelWidth = 10000
            
    num_col = int((xmax - xmin) / pixelHeight) #Calculate the number of rows cols to fill the bounding box at that resolution 
    num_row = int((ymax - ymin) / pixelWidth)

    source_proj = pyproj.Proj(proj='latlong', datum = 'NAD83') #We dont know but assume NAD83
    xProj, yProj = pyproj.Proj('esri:102001')(x,y) #Convert to Canada Albers Equal Area 

    yProj_extent=np.append(yProj,[bounds['maxy'],bounds['miny']]) #Add the bounding box coords to the dataset so we can extrapolate the interpolation to cover whole area
    xProj_extent=np.append(xProj,[bounds['maxx'],bounds['minx']])
    

    Yi = np.linspace(np.min(yProj_extent),np.max(yProj_extent),num_row) #Get the value for lat lon in each cell we just made 
    Xi = np.linspace(np.min(xProj_extent),np.max(xProj_extent),num_col)

    Xi,Yi = np.meshgrid(Xi,Yi)
    concat = np.array((Xi.flatten(), Yi.flatten())).T #Because we are not using the lookup file, send in X,Y order 
    send_to_list = concat.tolist()

    fire_map = gpd.read_file(fire_shapefile)
    DF = fire_map.geometry.unary_union

    meshPoints = [Point(item) for item in send_to_list]
    df = pd.DataFrame(meshPoints)
    gdf = gpd.GeoDataFrame(df, geometry=meshPoints)
    
    within_fire = gdf[gdf.geometry.within(DF)]


    if len(within_fire) == 0:
        #Get concave hull of the multipolygon
        try: 
            bounding_box = DF.convex_hull
            approx_centre_point = bounding_box.centroid.coords
        except: #theres only one polygon
            approx_centre_point = DF.centroid.coords

        reshaped_coord= np.array(approx_centre_point)
        hypot = cdist(concat,reshaped_coord)
        where_distance_small = np.argmin(hypot) #argmin returns the index of where the smallest item is 

        centre = concat[where_distance_small]

        #Get interpolated value here
        intF = interpolated_surface.flatten() 
        ival = intF[where_distance_small]
 

        return ival 
    
    else: 

        intF = interpolated_surface.flatten()
        listP = within_fire[0].tolist()
        tupArray =[x.coords for x in listP]
        xyFire = [(x[0][0],x[0][1],) for x in tupArray]

        tup_sendtolist =[tuple(x) for x in send_to_list]
        
        index = [] 
        for pair in xyFire:

        
            indices = [i for i, x in enumerate(tup_sendtolist) if x == pair]

            index.append(indices)

        ivals = []
        for i in index: 

            ivals.append(intF[i])


        max_ival = max(ivals) #get the max val inside the fire

        
        return float(max_ival)

def highest_value_first_four_days(fire_shapefile,shapefile,latlon_dict,interpolated_surface_d1,interpolated_surface_d2,interpolated_surface_d3,interpolated_surface_d4):
    '''Function to return the highest FWI values in a fire for four input arrays  
    Parameters
        fire_shapefile (str): path to the fire shapefile 
        shapefile (str): path to the study area shapefile
        latlon_dict (dict, loaded from json): dictionary of lat lon for each station 
        interpolated_surface_d1,d2,d3,d4 (np_array): an array of values in the study area, in the first four days of the fire
    Returns 
        max_val (float): maximum value for first four days since report date 
    '''

    v1 = get_interpolated_val_in_fire(fire_shapefile,shapefile,latlon_dict,interpolated_surface_d1)
    v2 = get_interpolated_val_in_fire(fire_shapefile,shapefile,latlon_dict,interpolated_surface_d2)
    v3 = get_interpolated_val_in_fire(fire_shapefile,shapefile,latlon_dict,interpolated_surface_d3)
    v4 = get_interpolated_val_in_fire(fire_shapefile,shapefile,latlon_dict,interpolated_surface_d4)
    list_vals = [v1,v2,v3,v4]
    max_val = max(list_vals)
    return max_val

def get_report_date_plus_three(fire_shapefile):
    '''Function to return the first four days of the fire
    Parameters
        fire_shapefile (str): path to the fire shapefile 
    Returns 
        fire_dates (list): list of the report date and three days after
    '''
    fire_map = gpd.read_file(fire_shapefile)
    rep_date = pd.to_datetime(fire_map['REP_DATE'].to_list()[0])
    fire_dates = [str(rep_date)[0:10]] 
    for i in range(1,4):
        next_date = rep_date+pd.DateOffset(i)
        fire_dates.append(str(next_date)[0:10])

    return fire_dates 
      
def ridge_regression(path_to_excel_spreadsheet,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,all_variables,plot_distributions,
                     plot_residual_histogram,transform): 
    '''Make a ridge regression model and print out the resulting coefficients and (if True) the histogram of residuals 
    Parameters
        path_to_excel_spreadsheet (str): path to the spreadsheet containing the FWI values for each fire and the other covariates,
        Notes: No trailing 0s in speadsheet, no space after parameter name in header 
        var1-10 (str): variable names, corresponding to the header titles 
        all_variables (bool): if True, will use all the variables, not just the FWI metrics 
        plot_distributions (bool): if True, will plot the correlation diagram for all the variables 
        plot_residual_histogram (bool): if True, will plot a histogram showing the residuals to check if normally distributed
        transform (bool): if True, it will transform the input data (i.e. the fire surface area), 
        to make it normally distributed
    Returns 
        Prints out regression coefficients, MAE, and R2 of the model 
    '''

    df = pd.read_csv(path_to_excel_spreadsheet)
    mod = RidgeCV() 
    y = np.array(df['CALC_HA']).reshape(-1, 1)
    
    if all_variables: 
        X = np.array(df[[var1,var2,var3,var4,var5,var6,var7,var8,var9,var10]])
    else: 
        X = np.array(df[[var5,var6,var7,var8,var9,var10]]) #just fwi
    
    mod.fit(X,y)
    y_pred = mod.predict(X)
    
    if plot_distributions: 
        
        if all_variables: 

            dataset = df[['CALC_HA',var1,var2,var3,var4,var5,var6,var7,var8,var9,var10]]
        else: 
            dataset = df[['CALC_HA',var1,var2,var3,var4,var5,var6,var7,var8,var9,var10]]

        _ = sns.pairplot(dataset, kind='reg', diag_kind='kde') #check for correlation

        _.fig.set_size_inches(15,15)
    
    print('Coefficients: %s'%mod.coef_)
    print('Mean squared error: %s'% mean_absolute_error(df['CALC_HA'], y_pred))
    print('Coefficient of determination: %s'% r2_score(df['CALC_HA'], y_pred))

    f,  ax0 = plt.subplots(1, 1)
    ax0.scatter(df['CALC_HA'], y_pred)
    ax0.plot([0, 150000], [0, 150000], '--k')
    ax0.set_ylabel('Target predicted')
    ax0.set_xlabel('True Target')
    ax0.set_title('Ridge Regression \n without target transformation')
    ax0.set_xlim([0, 150000])
    ax0.set_ylim([0, 150000])
    plt.show()

    if plot_residual_histogram:
        fig, ax = plt.subplots()
        residuals = y - mod.predict(X)
        
        mu = sum(residuals)/len(residuals)
        var  = sum(pow(x-mu,2) for x in residuals) / len(residuals)
        sigma  = math.sqrt(var)
        n, bins, patches = ax.hist(residuals,50,density=1,align='left')
        line = ((1 / (np.sqrt(2 * np.pi) * sigma)) *np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
        ax.plot(bins, line, '--')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        plt.show()

    if transform:
        mod2 = RidgeCV() 

        bc =QuantileTransformer(output_distribution='normal')
        y_trans_bc = bc.fit(y).transform(y)
        mod2.fit(X,y_trans_bc)

        y_pred2 = mod2.predict(X) 
        
        #Make new histogram
        fig, ax = plt.subplots()

        residuals = y - mod2.predict(X)
        
        mu = sum(residuals)/len(residuals)
        var  = sum(pow(x-mu,2) for x in residuals) / len(residuals)
        sigma  = math.sqrt(var)
        n, bins, patches = ax.hist(residuals,50,density=1,align='left')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        plt.show()
        
        print('Coefficients (T): %s'%mod.coef_)
        print('Mean squared error (T): %s'% mean_absolute_error(df['CALC_HA'], y_pred2))
        print('Coefficient of determination (T): %s'% r2_score(df['CALC_HA'], y_pred2))

        f, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
        ax0.scatter(df['CALC_HA'], y_pred)
        ax0.plot([0, 150000], [0, 150000], '--k')
        ax0.set_ylabel('Target predicted')
        ax0.set_xlabel('True Target')
        ax0.set_title('Ridge Regression \n without transformation')
        ax0.set_xlim([0, 150000])
        ax0.set_ylim([0, 150000])

        ax1.scatter(df['CALC_HA'], y_pred2)
        ax1.plot([0, 150000], [0, 150000], '--k')
        ax1.set_ylabel('Target predicted')
        ax1.set_xlabel('True Target')
        ax1.set_title('Ridge Regression \n with transformation')
        ax1.set_xlim([0, 150000])
        ax1.set_ylim([0, 150000])

        f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        plt.show()
        


def get_RMSE(absolute_error_dictionary):
    '''Calc RMSE
    Parameters
        absolute_error_dictionary (dict): dictionary output from the cross-validation procedure
    Returns
        RMSE (float)
    '''
    squared = [x**2 for x in absolute_error_dictionary.values()]
    RMSE = sum(squared)*(1/len(squared))
    return RMSE

def get_MAE(absolute_error_dictionary):
    '''Calc mean absolute error 
    Parameters
        absolute_error_dictionary (dict): dictionary output from the cross-validation procedure
    Returns
        MAE (float)
        MAE_max (float): highest absolute error at any of the stations 
    '''
    MAE = sum(absolute_error_dictionary.values()) / len(absolute_error_dictionary)
    MAE_max = max(absolute_error_dictionary.values()) 
    return MAE,MAE_max    
    
def is_it_in_zone(file_path,file_path_zones,Zone1,Zone2,Zone3,Zone4):
    '''Check if a fire is inside a zone (ecozone, intensively managed zone, etc.) 
    Parameters
        file_path (str): path to the location where the fire shapefiles are stored by year
        file_path_zones (str): path to where the shapefiles for the zones are located
        Zone1-4 (str): names of the zones, Zone1 is required, the rest are optional, takes up to 4
        Note: if you are not using the other three zones, input "None"
    Returns
        Prints out the fires plus what zone it is in
    '''
    dirname = file_path_zones
    fire_dates = []
    for year in os.listdir(file_path):
        fires = []
        if int(year) >= 1956: #Can change this if you only want to run a certain year 
            for fire_shp in os.listdir(file_path+year+'/'):
                if fire_shp.endswith('.shp'):
                    shpfile = file_path+year+'/'+fire_shp
                    fire_map1 = gpd.read_file(shpfile)
                    fireDF = gpd.GeoDataFrame(fire_map1)

                    try:
                        DF = fireDF.geometry.unary_union
                        fire_map = DF.convex_hull
                        approx_centre_point = Point(fire_map.centroid.coords)
                        

                    except: #There's only one fire polygon
                        fire_map = fireDF.geometry.unary_union
                        approx_centre_point = Point(fire_map.centroid.coords)

                    pointDF = pd.DataFrame([approx_centre_point])
                    gdf = gpd.GeoDataFrame(pointDF, geometry=[approx_centre_point])


                    shpfile_zone1 = os.path.join(dirname, Zone1+'.shp')
                    zone1_map = gpd.read_file(shpfile_zone1)
                    Zone1_gdf = gpd.GeoDataFrame(zone1_map)
                    

                    
                    shpfile_zone2 = os.path.join(dirname, Zone2+'.shp')
                    zone2_map = gpd.read_file(shpfile_zone2)
                    Zone2_gdf = gpd.GeoDataFrame(zone2_map)
                    Zone2 = None 
                        
                    
                    shpfile_zone3 = os.path.join(dirname, Zone3+'.shp')
                    zone3_map = gpd.read_file(shpfile_zone3)
                    Zone3_gdf = gpd.GeoDataFrame(zone3_map)
                    Zone3 = None
                        
                    
                    shpfile_zone4 = os.path.join(dirname, Zone4+'.shp')
                    zone4_map = gpd.read_file(shpfile_zone4)
                    Zone4_gdf = gpd.GeoDataFrame(zone4_map) 
                    Zone4 = None 
                        



                    #Find if the convex hull of the fire is INTERSECTING the zone
                    df = pd.DataFrame({'geometry': [fireDF.geometry.unary_union.convex_hull]})
                    geodf = gpd.GeoDataFrame(df)

                    rep_date = pd.to_datetime(fireDF['REP_DATE'].to_list()[0])
                    fire_id = fireDF['FIRE_ID'].to_list()[0]
                    calc_ha = fireDF['CALC_HA'].to_list()[0]
                    cause = fireDF['CAUSE'].to_list()[0]


                    if str(rep_date)[0:4] != 'None' and int(str(rep_date)[0:4]) >= 1956:
                        
                        

                    
                        if len(gpd.overlay(gdf,zone1_map,how='intersection')) > 0: 
                            if len(fireDF['REP_DATE'].to_list()) > 0:
                                #if float(calc_ha) >= 200: # Uncomment if you only want the fires > 200 ha 
                                if fire_shp[:-4] not in fires:
                                    print(fire_id + ',' + str(rep_date)[0:10]+ ',' + str(calc_ha)+','+Zone1)
                                    fires.append(fire_shp[:-4]) 
                                    

                                  
                            if len(gpd.overlay(gdf,zone2_map,how='intersection')) > 0: 
                                if len(fireDF['REP_DATE'].to_list()) > 0:
                            #if float(calc_ha) >= 200: 
                                    if fire_shp[:-4] not in fires:
                                        print(fire_id + ',' + str(rep_date)[0:10]+ ',' + str(calc_ha)+',' +Zone2)
                                        fires.append(fire_shp[:-4])

                            
                            if len(gpd.overlay(gdf,zone3_map,how='intersection')) > 0: 
                                if len(fireDF['REP_DATE'].to_list()) > 0:
                                #if float(calc_ha) >= 200: 
                                    if fire_shp[:-4] not in fires:
                                        print(fire_id + ',' + str(rep_date)[0:10]+ ',' + str(calc_ha)+',' +Zone3) 
                                        fires.append(fire_shp[:-4])
                        
        
                            if len(gpd.overlay(gdf,zone4_map,how='intersection')) > 0: 
                                if len(fireDF['REP_DATE'].to_list()) > 0:
                                #if float(calc_ha) >= 200: 
                                    if fire_shp[:-4] not in fires:
                                        print(fire_id + ',' + str(rep_date)[0:10]+ ',' + str(calc_ha)+',' +Zone4)   
                                        fires.append(fire_shp[:-4])

                        
                        else: 
                            #if float(calc_ha) >= 200: 
                            print(fire_id + ',' + str(rep_date)[0:10]+ ',' + str(calc_ha)+',' +'0')
                            fires.append(fire_shp[:-4])
                


                    fires.append(fire_shp[:-4])
                    fire_dates.append(str(rep_date)[0:10])
