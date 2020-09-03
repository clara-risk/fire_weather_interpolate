#coding: utf-8
#import
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import os, sys, json
import time, gc
from datetime import datetime, timedelta, date
from descartes import PolygonPatch


if __name__ == "__main__":
    
    dirname = os.path.dirname(__file__) #Or insert the directory path 
    file_path_daily = os.path.join(dirname, 'datasets/weather/daily_feather/')
    file_path_hourlyf = os.path.join(dirname, 'datasets/weather/hourly_feather/')
    file_path_hourly = os.path.join(dirname, 'datasets/weather/hourly_csv/')
    shapefile = os.path.join(dirname, 'datasets/study_area/QC_ON_albers_dissolve.shp')
    lookup_file = os.path.join(dirname,'datasets/lookup_files/fire_dates.csv')

    na_map = gpd.read_file(shapefile)
    fig, ax = plt.subplots()
    na_map.plot(ax = ax,color='white',edgecolor='k',linewidth=2,zorder=2,alpha=0.5)
    
    #First ecozone 
    plt.annotate('BOREAL \nSHIELD \nEAST',xy=(1345484, 1180301),verticalalignment='center',zorder=11) 
    boreal1_shapefile = os.path.join(dirname, 'datasets/study_area/ecozone/boreal1_ecozone61.shp')
    boreal1_map = gpd.read_file(boreal1_shapefile)
    boreal1_map.plot(ax = ax,color='#2C3E50',edgecolor=None,linewidth=2,zorder=10,alpha=0.7)
    
    #Second ecozone
    plt.annotate('BOREAL \nSHIELD \nWEST',xy=(155000, 1270879),verticalalignment='center',zorder=11) 
    boreal2_shapefile = os.path.join(dirname, 'datasets/study_area/ecozone/boreal2_easternC5.shp')
    boreal2_map = gpd.read_file(boreal2_shapefile)
    boreal2_map.plot(ax = ax,color='#566573',edgecolor=None,linewidth=2,zorder=10,alpha=0.7)
    
    #Third ecozone
    taiga_shapefile = os.path.join(dirname, 'datasets/study_area/ecozone/taiga_shield.shp')
    taiga_map = gpd.read_file(taiga_shapefile)
    taiga_map.plot(ax = ax,color='#17202A',edgecolor=None,linewidth=2,zorder=10,alpha=0.7)
    plt.annotate('TAIGA \nSHIELD',xy=(1291106, 2026208),verticalalignment='center',zorder=11)
    
    #Fourth ecozone 
    hudson_shapefile = os.path.join(dirname, 'datasets/study_area/ecozone/hudson.shp')
    hudson_map = gpd.read_file(hudson_shapefile)
    hudson_map.plot(ax = ax,color='#808B96',edgecolor=None,linewidth=2,zorder=10,alpha=0.7)
    plt.plot([698761, 698761], [1950000, 1490000], 'k-', lw=1,zorder=15)
    plt.annotate('HUDSON \nPLAIN',xy=(570763, 2046091),verticalalignment='center',zorder=11)
    
    #Get rid of the lat/lon tick marks / scientific notation stuff 
    plt.ticklabel_format(useOffset=False)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.show()
