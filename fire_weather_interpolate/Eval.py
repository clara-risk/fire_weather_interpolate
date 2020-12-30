#coding: utf-8

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
from sklearn.linear_model import RidgeCV,LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor

import math, statistics
import seaborn as sns 

from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy import stats

import statsmodels.api as sm

#functions 

def plot_LOOCV_error(loc_dict,error_dictionary,shapefile,var_name):
     '''Print a map of the LOOCV error (not absolute valued) so we can see the variation in error
    accross the study area 
     Parameters
         error_dictionary (dict): a dictionary of the error, produced by the cross-validate function
         shapefile (str): path to the study area shapefile
         var_name (str): variable name, for example, Start Day (Days since March 1) 
     Returns 
         Prints a map 
     '''
     print(error_dictionary) 
     lat = []
     lon = []
     error = []
     for station_name in error_dictionary.keys():

        if station_name in loc_dict.keys():

            loc = loc_dict[station_name]
            latitude = loc[0]
            longitude = loc[1]
            error_val = error_dictionary[station_name]
            lat.append(float(latitude))
            lon.append(float(longitude))
            error.append(error_val)
     y = np.array(lat)
     x = np.array(lon)
     z = np.array(error)

     source_proj = pyproj.Proj(proj='latlong', datum = 'NAD83') 
     xProj, yProj = pyproj.Proj('esri:102001')(x,y)

     fig, ax = plt.subplots(figsize= (15,15))
     crs = {'init': 'esri:102001'}
     plt.rcParams.update({'font.size': 16})
     plt.rcParams['image.cmap']='RdBu'
     na_map = gpd.read_file(shapefile)
     na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=10,alpha=0.1)
     plt.scatter(xProj,yProj,c=z,edgecolors='k')
     cbar = plt.colorbar()
     cbar.set_label(var_name)
     title = 'Spatial distribution of error for %s'%(var_name)
     fig.suptitle(title, fontsize=14)
     plt.xlabel('Longitude')
     plt.ylabel('Latitude')
     ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
     ax.ticklabel_format(useOffset=False, style='plain')
     plt.show()
        
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
        
def stratify_size(df,size_condition,all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10):
    '''This is a sub-function that will be called to stratify the fires based on size in the ridge regression (stratify) function. 
    ''' 
    if size_condition == 'all': 
        y = np.array(df['CALC_HA']).reshape(-1, 1)
        if all_variables:
            X = np.array(df[[var1,var2,var3,var4,var5,var6,var7,var8,var9,var10]])
        else: 
            X = np.array(df[[var5,var6,var7,var8,var9,var10]])
    elif size_condition == '<200':
        lessThan200 = df.loc[df['CALC_HA'] < 200]
        y = np.array(lessThan200['CALC_HA']).reshape(-1, 1)
        if all_variables:
            X = np.array(lessThan200[[var1,var2,var3,var4,var5,var6,var7,var8,var9,var10]])
        else: 
            X = np.array(lessThan200[[var5,var6,var7,var8,var9,var10]]) #just fwi
    elif size_condition == '>=200':
        greaterThan200 = df.loc[df['CALC_HA'] >= 200]
        y = np.array(greaterThan200['CALC_HA']).reshape(-1, 1)
        if all_variables:
            X = np.array(greaterThan200[[var1,var2,var3,var4,var5,var6,var7,var8,var9,var10]])
        else: 
            X = np.array(greaterThan200[[var5,var6,var7,var8,var9,var10]]) #just fwi
    else:
        print('The size condition is not in the correct format. Try again.')
        sys.exit()
    return X, y 

def stratify_dataset(df,stratify_condition,size_condition,all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10):
    '''This is a sub-function that will be called to stratify the fires based on different characteristics in the ridge regression (stratify) function. 
    ''' 
    if stratify_condition == 'none': 
        if size_condition == 'all': 
            X,y = stratify_size(df,'all',all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10) 
        elif size_condition == '<200':
            X,y = stratify_size(df,'<200',all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10) 
        elif size_condition == '>=200':
            X,y = stratify_size(df,'>=200',all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10) 
        else:
            print('The size condition is not in the correct format. Try again.')
            sys.exit()
    elif stratify_condition == 'human':
        humanDf = df.loc[df['CAUSE'] =='H']
        if size_condition == 'all': 
            X,y = stratify_size(humanDf,'all',all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10) 
        elif size_condition == '<200':
            X,y = stratify_size(humanDf,'<200',all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10) 
        elif size_condition == '>=200':
            X,y = stratify_size(humanDf,'>=200',all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10) 
        else:
            print('The size condition is not in the correct format. Try again.')
            sys.exit()
    elif stratify_condition == 'lightning':
        lightningDf = df.loc[df['CAUSE'] =='L']
        if size_condition == 'all': 
            X,y = stratify_size(lightningDf,'all',all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10) 
        elif size_condition == '<200':
            X,y = stratify_size(lightningDf,'<200',all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10) 
        elif size_condition == '>=200':
            X,y = stratify_size(lightningDf,'>=200',all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10) 
        else:
            print('The size condition is not in the correct format. Try again.')
            sys.exit()
    elif stratify_condition == '60conifer':
        coniferDf = df.loc[df['NLEAF'] >= 60]
        if size_condition == 'all': 
            X,y = stratify_size(coniferDf,'all',all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10) 
        elif size_condition == '<200':
            X,y = stratify_size(coniferDf,'<200',all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10) 
        elif size_condition == '>=200':
            X,y = stratify_size(coniferDf,'>=200',all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10) 
        else:
            print('The size condition is not in the correct format. Try again.')
            sys.exit()
    elif stratify_condition == '60conifer & lightning':
        LConiferDf = df.loc[(df['NLEAF'] >= 60) & (df['CAUSE'] == 'L')]
        if size_condition == 'all': 
            X,y = stratify_size(LConiferDf,'all',all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10) 
        elif size_condition == '<200':
            X,y = stratify_size(LConiferDf,'<200',all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10) 
        elif size_condition == '>=200':
            X,y = stratify_size(LConiferDf,'>=200',all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10) 
        else:
            print('The size condition is not in the correct format. Try again.')
            sys.exit()
    else:
        print('That is not a correct stratify condition. Try again.')

    return X,y 

def ridge_regression_stratify(path_to_excel_spreadsheet,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,all_variables,
                              plot_distributions,plot_residual_histogram,stratify_condition, size_condition,ecozone_condition): 
    '''Make a ridge regression model stratified based on different ecozones and fire characteristics
    and print out the resulting coefficients and (if True) the histogram of residuals 
    Parameters
        path_to_excel_spreadsheet (str): path to the spreadsheet containing the FWI values for each fire and the other covariates,
        Notes: No trailing 0s in speadsheet, no space after parameter name in header 
        var1-10 (str): variable names, corresponding to the header titles 
        all_variables (bool): if True, will use all the variables, not just the FWI metrics 
        plot_distributions (bool): if True, will plot the correlation diagram for all the variables 
        plot_residual_histogram (bool): if True, will plot a histogram showing the residuals to check if normally distributed
        stratify_condition (str): can be one of 'human', 'lightning', '60conifer', '60conifer & lightning','none'
        size_condition (str): can be one of 'all', '<200', '>=200'
        ecozone_condition (str): can be one of 'none','taiga','boreal1' (west), 'boreal2' (east), 'hudson'
    Returns 
        Prints out regression coefficients, MAE, and R2 of the model 
    '''

    df = pd.read_csv(path_to_excel_spreadsheet)
    mod = RidgeCV()

    if ecozone_condition == 'none': 
        X,y = stratify_dataset(df,stratify_condition,size_condition,all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10)
    elif ecozone_condition == 'taiga':
        taiga_df = df.loc[df['Ecozone'] =='Taiga']
        X,y = stratify_dataset(taiga_df,stratify_condition,size_condition,all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10)
    elif ecozone_condition == 'boreal1':
        boreal1_df = df.loc[df['Ecozone'] =='Boreal 1']
        X,y = stratify_dataset(boreal1_df,stratify_condition,size_condition,all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10)
    elif ecozone_condition == 'boreal2':
        boreal2_df = df.loc[df['Ecozone'] =='Boreal 2']
        X,y = stratify_dataset(boreal2_df,stratify_condition,size_condition,all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10)
    elif ecozone_condition == 'hudson':
        hudson_df = df.loc[df['Ecozone'] =='Hudson']
        X,y = stratify_dataset(hudson_df,stratify_condition,size_condition,all_variables,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10)
    else:
        print('That is not a valid ecozone name. Try again.')
        sys.exit() 
    
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
    print('Mean squared error: %s'% mean_absolute_error(y, y_pred))
    print('Coefficient of determination: %s'% r2_score(y, y_pred))

    f,  ax0 = plt.subplots(1, 1)
    ax0.scatter(y, y_pred)
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

def get_RMSE(absolute_error_dictionary):
    '''Calc RMSE
    Parameters
        absolute_error_dictionary (dict): dictionary output from the cross-validation procedure
    Returns
        RMSE (float)
    '''
    squared = [x**2 for x in absolute_error_dictionary.values()]
    innerRMSE = sum(squared)*(1/len(squared))
    RMSE = math.sqrt(innerRMSE) 
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

                    
                    
def plot(shapefile,maxmin,idw1_grid,idw2_grid,idew1_grid,idew2_grid,tpss_grid,rf_grid,ok_grid,varname): 
    '''Plot all the maps for the different spatial interpolation methods on one figure 
    Parameters
        shapefile (str): math to the shapefile 
        maxmin (list): extent of the image to plot, you can get it from any of the interpolation functions
        idw1_grid (etc.): the grids for the different spatial interpolation methods
        varname (str): name of the variable being shown 
    Returns
        Plots a figure with the different maps 
    '''
    plt.rcParams["font.family"] = "Times New Roman" #Set the font to Times New Roman 
    fig, ax = plt.subplots(2,4)
    
    crs = {'init': 'esri:102001'}

    na_map = gpd.read_file(shapefile)

    yProj_min = maxmin[0]
    yProj_max = maxmin[1]
    xProj_min = maxmin[3]
    xProj_max = maxmin[2]
    
    circ = PolygonPatch(na_map['geometry'][0],visible=False)
    ax[0, 0].add_patch(circ)
    ax[0, 0].imshow(idw1_grid,extent=(xProj_min,xProj_max,yProj_max,yProj_min),clip_path=circ, clip_on=True,origin='upper')
    ax[0, 0].invert_yaxis()
    ax[0,0].set_title('IDW B=1')

    circ2 = PolygonPatch(na_map['geometry'][0],visible=False)

    ax[0, 1].add_patch(circ2) 
    ax[0, 1].imshow(idw2_grid,extent=(xProj_min,xProj_max,yProj_max,yProj_min),clip_path=circ2, clip_on=True,origin='upper')
    ax[0, 1].invert_yaxis()
    ax[0,1].set_title('IDW B=2')
    

    circ3 = PolygonPatch(na_map['geometry'][0],visible=False)

    ax[0, 2].add_patch(circ3) 
    im = ax[0, 2].imshow(idew1_grid,extent=(xProj_min,xProj_max,yProj_max,yProj_min),clip_path=circ3, clip_on=True,origin='upper') 
    ax[0, 2].invert_yaxis()
    ax[0,2].set_title('IDEW B=1')

    circ4 = PolygonPatch(na_map['geometry'][0],visible=False)

    ax[0, 3].add_patch(circ4) 
    ax[0, 3].imshow(idew2_grid,extent=(xProj_min,xProj_max,yProj_max,yProj_min),clip_path=circ4, clip_on=True,origin='upper') 
    ax[0, 3].invert_yaxis()
    ax[0,3].set_title('IDEW B=2')

    circ5 = PolygonPatch(na_map['geometry'][0],visible=False)

    ax[1, 0].add_patch(circ5) 
    ax[1, 0].imshow(tpss_grid,extent=(xProj_min,xProj_max,yProj_max,yProj_min),clip_path=circ5, clip_on=True,origin='upper') 
    ax[1, 0].invert_yaxis()
    ax[1,0].set_title('TPSS')

    
    circ6 = PolygonPatch(na_map['geometry'][0],visible=False)

    ax[1, 1].add_patch(circ6) 
    ax[1, 1].imshow(rf_grid,extent=(xProj_min,xProj_max,yProj_max,yProj_min),clip_path=circ6, clip_on=True,origin='upper') 
    ax[1, 1].invert_yaxis()
    ax[1,1].set_title('RF')

    circ7 = PolygonPatch(na_map['geometry'][0],visible=False)

    ax[1, 2].add_patch(circ7) 
    ax[1, 2].imshow(ok_grid,extent=(xProj_min,xProj_max,yProj_max,yProj_min),clip_path=circ7, clip_on=True,origin='upper') 
    ax[1, 2].invert_yaxis()
    ax[1,2].set_title('OK')


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax, aspect=0.01,label=varname)


    fig.delaxes(ax[1,3])

    fig.text(0.5,0.04, "Longitude", ha="center", va="center")
    fig.text(0.05,0.5, "Latitude", ha="center", va="center", rotation=90)


    plt.show()

def select_random_station(groups,blocknum,replacement,used_stations):
    '''Select a random station from each group for bagging xval 
    Parameters
        groups (dict): dictionary of what group each station belongs to 
        blocknum (int): number of blocks 
        replacement (bool): whether or not to use replacement 
        used_stations (list): empty list if using replacement, if not, it is a list
        of already used stations 
    Returns
        stations_selected (dict): dictionary of selected stations 
    '''
    if not replacement: 
          stations_selected = {}
          used_stations = [x for y in used_stations for x in y] #merge all sublists
          for group in range(1,blocknum+1):
               try: 
                    group1 = [k for k,v in groups.items() if v == group]
                    for station in used_stations:
                         if station in group1: 
                              group1.remove(station) 
                    group1_selection = np.random.choice(group1,1)
                         
                    #print('Group selection %s is: %s'%(group,group1_selection[0]))
                    stations_selected[group] = group1_selection[0] 
               except ValueError: #No stations in that group!
                    pass
                    #print('No stations in group %s'%group)

          return stations_selected
    else: #Replace the stations
          stations_selected = {}
          for group in range(1,blocknum+1):
               try: 
                    group1 = [k for k,v in groups.items() if v == group]
                    group1_selection = np.random.choice(group1,1)
                    stations_selected[group] = group1_selection[0]
               except ValueError:
                    pass 
          return stations_selected


def linear_regression(path_to_excel_spreadsheet,plot_distributions,plot_residual_histogram,transform):
     '''Make a linear regression model and print out the resulting coefficients and (if True) the histogram of residuals 
    Parameters
        path_to_excel_spreadsheet (str): path to the spreadsheet containing the fire season start/end values,
        Notes: No trailing 0s in speadsheet, no space after parameter name in header 
        plot_distributions (bool): if True, will plot the correlation diagram for all the variables 
        plot_residual_histogram (bool): if True, will plot a histogram showing the residuals to check if normally distributed
        transform (bool): if True, it will transform the input data, to make it normally distributed
    Returns 
        Prints out regression coefficients, MAE, and R2 of the model & p-values
    '''
     df = pd.read_csv(path_to_excel_spreadsheet)
     mod = LinearRegression()
     y = np.array(df['NFDB_DATE']).reshape(-1, 1)
     X = np.array(df['AV_SEASON_DATE']).reshape(-1, 1)
     mod.fit(X,y)
     y_pred = mod.predict(X)

     if plot_distributions:
          dataset = df[['NFDB_DATE','AV_SEASON_DATE']]
          _ = sns.pairplot(dataset, kind='reg', diag_kind='kde') #check for correlation
          _.fig.set_size_inches(15,15)
    
     print('Coefficients: %s'%mod.coef_)
     print('Mean squared error: %s'% mean_absolute_error(df['NFDB_DATE'], y_pred))
     print('Coefficient of determination: %s'% r2_score(df['NFDB_DATE'], y_pred))

     #Calc the significance
     newX = np.append(np.ones((len(X),1)), X, axis=1)
     MSE = mean_absolute_error(df['NFDB_DATE'], y_pred)

     params = np.append(mod.intercept_,mod.coef_)
    
     var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
     sd_b = np.sqrt(var_b)
     ts_b = params/ sd_b
     p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]
     p_values = np.round(p_values,15)

     print('P-values: %s'%p_values)
     myDF3 = pd.DataFrame()
     myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilities"] = [params,sd_b,ts_b,p_values]
     print(myDF3)

     f,  ax0 = plt.subplots(1, 1)
     ax0.scatter(df['NFDB_DATE'], y_pred)
     ax0.plot([0, 200], [0, 200], '--k')
     ax0.set_ylabel('Target predicted')
     ax0.set_xlabel('True Target')
     ax0.set_title('Linear Regression \n without target transformation')
     ax0.set_xlim([0, 200])
     ax0.set_ylim([0, 200])
     plt.show()

     if plot_residual_histogram:
          fig, ax = plt.subplots()
          residuals = y - mod.predict(X)
          mu = sum(residuals)/len(residuals)
          var  = sum(pow(x-mu,2) for x in residuals) / len(residuals)
          sigma  = math.sqrt(var)
          n, bins, patches = ax.hist(residuals,10,density=1,align='left')
          line = ((1 / (np.sqrt(2 * np.pi) * sigma)) *np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
          ax.plot(bins, line, '--')
          ax.set_xlabel('Residuals')
          ax.set_ylabel('Frequency')
          plt.show()

     if transform:
          mod2 = LinearRegression()
          y_trans = np.log1p(y)
          mod2.fit(X,y_trans)
          y_pred2 = mod2.predict(X)
          #Make new histogram
          fig, ax = plt.subplots()
          residuals = y_trans - mod2.predict(X)
        
          mu = sum(residuals)/len(residuals)
          var  = sum(pow(x-mu,2) for x in residuals) / len(residuals)
          sigma  = math.sqrt(var)
          n, bins, patches = ax.hist(residuals,10,density=1,align='left')
          ax.set_xlabel('Residuals')
          ax.set_ylabel('Frequency')
          plt.show()
        
          print('Coefficients (T): %s'%mod.coef_)
          print('Mean squared error (T): %s'% mean_absolute_error(X, y_pred2))
          print('Coefficient of determination (T): %s'% r2_score(X, y_pred2))

          f, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
          ax0.scatter(X, y_pred)
          ax0.plot([0, 100], [0, 100], '--k')
          ax0.set_ylabel('Target predicted')
          ax0.set_xlabel('True Target')
          ax0.set_title('Linear Regression \n without transformation')
          ax0.set_xlim([0, 200])
          ax0.set_ylim([0, 200])

          ax1.scatter(X, y_pred2)
          ax1.plot([0, 200], [0, 200], '--k')
          ax1.set_ylabel('Target predicted')
          ax1.set_xlabel('True Target')
          ax1.set_title('Linear Regression \n with transformation')
          ax1.set_xlim([0, 200])
          ax1.set_ylim([0, 200])

          f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
          plt.show()
     
def check_p_value(path_to_excel_spreadsheet):
     '''Quality-control for p-value calculation'''

     df = pd.read_csv(path_to_excel_spreadsheet)
     mod = LinearRegression()
     
     y = np.array(df['NFDB_DATE']).reshape(-1, 1)
     X = np.array(df['AV_SEASON_DATE']).reshape(-1, 1)

     X2 = sm.add_constant(X)
     est = sm.OLS(y, X2)
     est2 = est.fit()
     print(est2.summary())
     
