#coding: utf-8
"""
Summary
-------
Code to make a figure showing all the interpolation methods. 
"""
    
#import
import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import os, sys, json
import time 
from descartes import PolygonPatch
import math

import get_data as GD
import idw as idw
import idew as idew
import ok as ok
import tps as tps
import fwi as fwi
import Eval as Eval
import rf as rf
import fwi as fwi

def plot_dates(shapefile,maxmin,idw2_grid,idw3_grid,idw4_grid,tpss_grid,rf_grid,input_year,varname,Cvar_dict,loc_dict):
    if max(Cvar_dict.values()) > 94:
        print('Alert!')
        print(max(Cvar_dict.values()))
    print(max(Cvar_dict.values()))
    maxval = 94
    lat = []
    lon = []
    Cvar = []
    for station_name in Cvar_dict.keys(): #DONT use list of stations, because if there's a no data we delete that in the climate dictionary step
        if station_name in loc_dict.keys():
            loc = loc_dict[station_name]
            latitude = loc[0]
            longitude = loc[1]
            cvar_val = Cvar_dict[station_name]
            lat.append(float(latitude))
            lon.append(float(longitude))
            Cvar.append(cvar_val)
    y = np.array(lat)
    x = np.array(lon)

    source_proj = pyproj.Proj(proj='latlong', datum = 'NAD83') #We dont know but assume 
    xProj, yProj = pyproj.Proj('esri:102001')(x,y)


    plt.rcParams["font.family"] = "Calibri"
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['image.cmap']='Greys' #Spectral_r
    
    fig, ax = plt.subplots(2,3)
    
    crs = {'init': 'esri:102001'}

    na_map = gpd.read_file(shapefile)

    yProj_min = maxmin[0]
    yProj_max = maxmin[1]
    xProj_min = maxmin[3]
    xProj_max = maxmin[2]
    
    circ = PolygonPatch(na_map['geometry'][0],visible=False)
    ax[0, 0].add_patch(circ)
    ax[0, 0].imshow(idw2_grid,extent=(xProj_min,xProj_max,yProj_max,yProj_min),vmin=0,vmax=maxval, clip_path=circ, clip_on=True,origin='upper')
    na_map.plot(ax = ax[0,0],facecolor="none",edgecolor='k',linewidth=1)
    ax[0, 0].invert_yaxis() #vmax=32,
    ax[0,0].set_title('IDW B=2')

    ax[0,0].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ax[0,0].ticklabel_format(useOffset=False, style='plain')
    circ2 = PolygonPatch(na_map['geometry'][0],visible=False)

    ax[0, 1].add_patch(circ2)
    ax[0, 1].imshow(idw3_grid,extent=(xProj_min,xProj_max,yProj_max,yProj_min),vmin=0,vmax=maxval, clip_path=circ2, clip_on=True,origin='upper')
    na_map.plot(ax = ax[0,1],facecolor="none",edgecolor='k',linewidth=1)
    ax[0, 1].invert_yaxis()
    ax[0,1].set_title('IDW B=3')

    ax[0,1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ax[0,1].ticklabel_format(useOffset=False, style='plain')

    circ3 = PolygonPatch(na_map['geometry'][0],visible=False)

    ax[0, 2].add_patch(circ3)
    im = ax[0, 2].imshow(idw4_grid,extent=(xProj_min,xProj_max,yProj_max,yProj_min),vmin=0,vmax=maxval, clip_path=circ3, clip_on=True,origin='upper') 
    na_map.plot(ax = ax[0,2],facecolor="none",edgecolor='k',linewidth=1)
    ax[0, 2].invert_yaxis()
    ax[0,2].set_title('IDW B=4')

    ax[0,2].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ax[0,2].ticklabel_format(useOffset=False, style='plain')
    circ4 = PolygonPatch(na_map['geometry'][0],visible=False)

    ax[1, 0].add_patch(circ4)
    ax[1, 0].imshow(tpss_grid,extent=(xProj_min,xProj_max,yProj_max,yProj_min),vmin=0,vmax=maxval, clip_path=circ4, clip_on=True,origin='upper') 
    na_map.plot(ax = ax[1, 0],facecolor="none",edgecolor='k',linewidth=1)
    ax[1, 0].invert_yaxis()
    ax[1, 0].set_title('TPSS')
    
    ax[1, 0].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ax[1, 0].ticklabel_format(useOffset=False, style='plain')
    circ5 = PolygonPatch(na_map['geometry'][0],visible=False)

    ax[1, 1].add_patch(circ5)
    ax[1, 1].imshow(rf_grid,extent=(xProj_min,xProj_max,yProj_max,yProj_min),vmin=0,vmax=maxval, clip_path=circ5, clip_on=True,origin='upper') 
    na_map.plot(ax = ax[1, 1],facecolor="none",edgecolor='k',linewidth=1)
    ax[1, 1].invert_yaxis()
    ax[1, 1].set_title('RF')

    ax[1, 1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ax[1, 1].ticklabel_format(useOffset=False, style='plain')
    
    circ7 = PolygonPatch(na_map['geometry'][0],visible=False)
    na_map.plot(ax = ax[1, 2],color='white',edgecolor='k',linewidth=1)
    ax[1, 2].imshow(rf_grid,extent=(xProj_min,xProj_max,yProj_max,yProj_min),vmin=0,vmax=maxval, clip_path=circ7, clip_on=True,origin='upper') #hack to get the right zoom
    ax[1, 2].scatter(xProj,yProj,c=Cvar,edgecolors='k',linewidth=1,vmin=0,vmax=maxval,s=15) #c=Cvar
    ax[1, 2].invert_yaxis()
    ax[1,2].set_title('Weather Stations')

    ax[1,2].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ax[1,2].ticklabel_format(useOffset=False, style='plain')


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax, aspect=0.01,label=varname)

    plt.show()
    
    
    if __name__ == "__main__":
    
    dirname = '' #Path to datasets
    file_path_daily = os.path.join(dirname, 'datasets/weather/daily_feather/')
    file_path_hourlyf = os.path.join(dirname, 'datasets/weather/hourly_feather/')
    file_path_hourly = '' #Path to csv files
    file_path_daily = ''
    shapefile = os.path.join(dirname, 'datasets/study_area/QC_ON_albers_dissolve.shp')

    file_path_elev = os.path.join(dirname,'datasets/lookup_files/elev_csv.csv')
    idx_list = GD.get_col_num_list(file_path_elev,'elev')
    input_date = '2018'
    year = '2018'


    print('getting dictionary...') 
    days_dict, latlon_station = fwi.end_date_calendar_csv(file_path_daily,year)
    print(days_dict)
    print('making maps...') #End Date (# Days since October 1) #Start Date (# Days since March 1)
    idw2_grid,maxmin = idw.IDW(latlon_station,days_dict,input_date,'End Date (# Days since September 1)',shapefile,False,2)
    idw3_grid,maxmin = idw.IDW(latlon_station,days_dict,input_date,'End Date (# Days since September 1)',shapefile,False,3)
    idw4_grid,maxmin = idw.IDW(latlon_station,days_dict,input_date,'End Date (# Days since September 1)',shapefile,False,4)

    num_stations_d = int(len(days_dict.keys())) 
    smoothing_parameterD = int(num_stations_d)-(math.sqrt(2*num_stations_d))
    print(num_stations_d)
    tpss_grid,maxmin = tps.TPS(latlon_station,days_dict,input_date,'End Date (# Days since September 1)',shapefile,False,smoothing_parameterD)
    rf_grid, maxmin = rf.random_forest_interpolator(latlon_station,days_dict,input_date,'End Date (# Days since September 1)',shapefile,False,file_path_elev,idx_list)
    plot_dates(shapefile,maxmin,idw2_grid,idw3_grid,idw4_grid,tpss_grid,rf_grid,input_date,'End Date (# Days since September 1)',days_dict,latlon_station)

