#coding: utf-8

#import
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os,sys,time
import math,statistics
import pandas as pd 
import seaborn as sns


sns.set()
fp = 'cv_error.csv'
data = pd.read_csv('cv_error.csv')
print(data)


cv_types = list(set(list(data['CV'])))
t = data[data['Var'] == 'Temperature']
t = t[['Interp','CV','MAE']].pivot("Interp", "CV")
print(t)

fig, axs = plt.subplots(2,2)
   
ax = sns.heatmap(t,cmap="Spectral_r",\
                 cbar_kws={'label': 'MAE (°C)'},ax=axs[0,0],\
                 xticklabels=False)
axs[0,0].set_title('Temperature')
axs[0,0].set_xlabel('')
axs[0,0].set_ylabel('')

rh = data[data['Var'] == 'RH']
rh = rh[['Interp','CV','MAE']].pivot("Interp", "CV")

ax2 = sns.heatmap(rh,cmap="Spectral_r",\
                 cbar_kws={'label': 'MAE (%)'},ax=axs[0,1],\
                  xticklabels=False)

axs[0,1].set_title('Relative Humidity')
axs[0,1].set_xlabel('')
axs[0,1].set_ylabel('')


ws = data[data['Var'] == 'WS']
ws = ws[['Interp','CV','MAE']].pivot("Interp", "CV")

ax2 = sns.heatmap(ws,cmap="Spectral_r",\
                 cbar_kws={'label': 'MAE (km/h)'},ax=axs[1,0])

axs[1,0].set_title('Wind Speed')
axs[1,0].set_xlabel('')
axs[1,0].set_ylabel('')
axs[1,0].set_xticklabels(sorted(cv_types), rotation=35, \
                         fontsize=8, rotation_mode='anchor', ha='right')

pcp = data[data['Var'] == 'PCP']
pcp = pcp[['Interp','CV','MAE']].pivot("Interp", "CV")

ax2 = sns.heatmap(pcp,cmap="Spectral_r",\
                 cbar_kws={'label': 'MAE (mm)'},ax=axs[1,1])

axs[1,1].set_title('Precipitation')
axs[1,1].set_xlabel('')
axs[1,1].set_ylabel('')
axs[1,1].set_xticklabels(sorted(cv_types), rotation=35, \
                         fontsize=8, rotation_mode='anchor', ha='right')

plt.show()
