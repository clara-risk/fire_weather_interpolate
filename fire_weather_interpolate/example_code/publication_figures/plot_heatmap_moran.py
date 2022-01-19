#coding: utf-8
print('Checkpoint1 passed') 
#import
import geopandas as gpd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import warnings
import os,sys,time
import math,statistics
import json 
import pandas as pd 
import seaborn as sns


sns.set()
plt.rcParams.update({'font.size': 22})
sns.set(font_scale=1.4)
dirname = 'C:/Users/clara/Documents/Earth_Space_Revisions/Moran/'
fp = 'dist_temp.csv'
fpp='dist_pcp.csv'
fpr='dist_rh.csv'
fpw='dist_wind.csv'
t = pd.read_csv(dirname+fp)
p = pd.read_csv(dirname+fpp)
r = pd.read_csv(dirname+fpr)
w = pd.read_csv(dirname+fpw)
print(t)


cv_types = list(set(list(p['Date'])))
date = [x[0:10] for x in cv_types]
print(date)
#t = data[data['Var'] == 'Temperature']
t['Distance Class (km)'] = pd.Categorical(t['Distance Class (km)'], ["0-200", "0-500","0-800",\
                                                           "0-1000","0-1300","Overall"])
t = t[['Distance Class (km)','Date','Spatial Autocorrelation']]\
    .pivot('Distance Class (km)','Date')
print(t)

fig, axs = plt.subplots(2,2)
   
ax = sns.heatmap(t,cmap="bwr",ax=axs[0,0],\
                 xticklabels=False,cbar=False,linewidths=1)
for _, spine in ax.spines.items():
    spine.set_visible(True)
axs[0,0].set_title('Temperature')
axs[0,0].set_xlabel('')
    
#axs[0,0].set_ylabel('')

p['Distance Class (km)'] = pd.Categorical(p['Distance Class (km)'], ["0-200", "0-500","0-800",\
                                                           "0-1000","0-1300","Overall"])
p = p[['Distance Class (km)','Date','Spatial Autocorrelation']]\
    .pivot('Distance Class (km)','Date')

   
ax = sns.heatmap(p,cmap="bwr",ax=axs[1,1],\
                 xticklabels=False,cbar=False,linewidths=1)
axs[1,1].set_title('Precipitation')
axs[1,1].set_xlabel('')
axs[1,1].set_ylabel('')
#axs[1,1].set_xticklabels(cv_types, rotation=35, \
                         #fontsize=8, rotation_mode='anchor', ha='right')
date_tick = list(np.arange(0,len(date)))
axs[1,1].set(xticks=date_tick)
axs[1,1].set_xticklabels(sorted(date), rotation=35, \
                         fontsize=12, rotation_mode='anchor', ha='right')

r['Distance Class'] = pd.Categorical(r['Distance Class'], ["0-200", "0-500","0-800",\
                                                           "0-1000","0-1300","Overall"])
r = r[['Distance Class','Date','Spatial Autocorrelation']]\
    .pivot('Distance Class','Date')

   
ax = sns.heatmap(r,cmap="bwr",ax=axs[0,1],\
                 xticklabels=False,cbar=False,linewidths=1)
axs[0,1].set_title('Relative Humidity')
axs[0,1].set_xlabel('')
axs[0,1].set_ylabel('')

w['Distance Class (km)'] = pd.Categorical(w['Distance Class (km)'], ["0-200", "0-500","0-800",\
                                                           "0-1000","0-1300","Overall"])
w = w[['Distance Class (km)','Date','Spatial Autocorrelation']]\
    .pivot('Distance Class (km)','Date')

   
ax = sns.heatmap(w,cmap="bwr",ax=axs[1,0],\
                 xticklabels=False,cbar=False,linewidths=1)
axs[1,0].set_title('Wind')
axs[1,0].set_xlabel('')
#axs[1,0].set_ylabel('')
date_tick = list(np.arange(0,len(date)))
axs[1,0].set(xticks=date_tick)
axs[1,0].set_xticklabels(sorted(date), rotation=35, \
                         fontsize=12, rotation_mode='anchor', ha='right')

plt.show()

##rh = data[data['Var'] == 'RH']
##rh = rh[['Interp','CV','MAE']].pivot("Interp", "CV")


##ax2 = sns.heatmap(rh,cmap="Spectral_r",\
##                 cbar_kws={'label': 'MAE (%)'},ax=axs[0,1],\
##                  xticklabels=False)
##
##axs[0,1].set_title('Relative Humidity')
##axs[0,1].set_xlabel('')
##axs[0,1].set_ylabel('')
##
##
##ws = data[data['Var'] == 'WS']
##ws = ws[['Interp','CV','MAE']].pivot("Interp", "CV")
##
##ax2 = sns.heatmap(ws,cmap="Spectral_r",\
##                 cbar_kws={'label': 'MAE (km/h)'},ax=axs[1,0])
##
##axs[1,0].set_title('Wind Speed')
##axs[1,0].set_xlabel('')
##axs[1,0].set_ylabel('')
##axs[1,0].set_xticklabels(sorted(cv_types), rotation=35, \
##                         fontsize=8, rotation_mode='anchor', ha='right')
##
##pcp = data[data['Var'] == 'PCP']
##pcp = pcp[['Interp','CV','MAE']].pivot("Interp", "CV")
##
##ax2 = sns.heatmap(pcp,cmap="Spectral_r",\
##                 cbar_kws={'label': 'MAE (mm)'},ax=axs[1,1])
##
##axs[1,1].set_title('Precipitation')
##axs[1,1].set_xlabel('')
##axs[1,1].set_ylabel('')
##axs[1,1].set_xticklabels(sorted(cv_types), rotation=35, \
##                         fontsize=8, rotation_mode='anchor', ha='right')
##
##plt.show()
##
