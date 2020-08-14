#coding: utf-8

#import
import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import os, sys, json
import time
import json, math 

import get_data as GD
import idw as idw
import idew as idew
import ok as ok
import tps as tps
import fwi as fwi
import Eval as Eval
import rf as rf 

if __name__ == "__main__":
    
    dirname = '' #Enter the directory to the file with the data. 
    file_name = 'RF.csv' #The name of the csv in the directory with the data inside. 

    #All fires 
    print('ALL FIRES--------------------------------------------------------------------------------') 
    print('All fires, just fwi') 
    Eval.ridge_regression(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',False,False,False,False) #Just FWI
    print('All fires, all predictors') 
    Eval.ridge_regression(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',True,False,False,False) #All pred
    print('All fires, just fwi, human fires')
    Eval.ridge_regression_stratify(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',False,False,False,
                                   'human', 'all','none')
    print('All fires, just fwi, lightning fires')
    Eval.ridge_regression_stratify(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',False,False,False,
                                   'lightning', 'all','none')
    print('All fires, just fwi, 60 % conifer')
    Eval.ridge_regression_stratify(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',False,False,False,
                                   '60conifer', 'all','none')

    print('All fires, just fwi, lightning fires 60 % conifer')
    Eval.ridge_regression_stratify(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',False,False,False,
                                   '60conifer & lightning', 'all','none')

    print('All fires, all predictors, human fires')
    Eval.ridge_regression_stratify(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',True,False,False,
                                   'human', 'all','none')

    print('All fires, all predictors, lightning fires')
    Eval.ridge_regression_stratify(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',True,False,False,
                                   'lightning', 'all','none')

    print('All fires, all predictors, 60 % conifer')
    Eval.ridge_regression_stratify(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',True,False,False,
                                   '60conifer', 'all','none')

    print('All fires, all predictors, lightning fires 60 % conifer')
    Eval.ridge_regression_stratify(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',True,False,False,
                                   '60conifer & lightning', 'all','none')

    #Stratify automatically by ecozone
    
    #Ecozones
    ecozone_list = ['taiga','boreal1','boreal2','hudson']
    for eco in ecozone_list: 
        print('%s --------------------------------------------------------------------------------------------'%eco) 
        print('Just %s, just fwi'%eco)
        Eval.ridge_regression_stratify(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',False,False,False,
                                       'none', 'all',eco)

        print('Just %s, all predictors'%eco)
        Eval.ridge_regression_stratify(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',True,False,False,
                                       'none', 'all',eco)
        
        print('Just %s, just fwi, human fires'%eco)
        Eval.ridge_regression_stratify(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',False,False,False,
                                       'human', 'all',eco)
        print('Just %s, just fwi, lightning fires'%eco)
        Eval.ridge_regression_stratify(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',False,False,False,
                                       'lightning', 'all',eco)
        print('Just %s, just fwi, 60 percent conifer'%eco)
        Eval.ridge_regression_stratify(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',False,False,False,
                                       '60conifer', 'all',eco)

        print('Just %s, just fwi, lightning fires 60 percent conifer'%eco)
        Eval.ridge_regression_stratify(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',False,False,False,
                                       '60conifer & lightning', 'all',eco)

        print('Just %s, all predictors, human fires'%eco)
        Eval.ridge_regression_stratify(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',True,False,False,
                                       'human', 'all',eco)

        print('Just %s, all predictors, lightning fires'%eco)
        Eval.ridge_regression_stratify(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',True,False,False,
                                       'lightning', 'all',eco)

        print('Just %s, all predictors, 60 percent conifer'%eco)
        Eval.ridge_regression_stratify(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',True,False,False,
                                       '60conifer', 'all',eco)

        print('Just %s, all predictors, lightning fires 60 percent conifer'%eco)
        Eval.ridge_regression_stratify(dirname+file_name,'ZONE','ELEV','SLOPE','NLEAF','DC','DMC','FFMC','FWI','ISI','BUI',True,False,False,
                                       '60conifer & lightning', 'all',eco)
         
