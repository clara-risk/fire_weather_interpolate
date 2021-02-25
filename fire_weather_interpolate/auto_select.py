#coding: utf-8

"""
Summary
-------
Auto-selection protocol for spatial interpolation methods using shuffle-split cross-validation. 
References
----------
"""
from idw import idw
from tps import TPS
from gpr import gpr
from rf import rf
from IDEW import IDEW

def run_comparison(interpolation_types,rep,loc_dictionary,cvar_dictionary,file_path_elev,elev_array,idx_list,**kwargs):
     '''Execute the shuffle-split cross-validation for the given interpolation types 
     Parameters
         interpolation_types (list of str): list of interpolation types to consider
     Returns 
         interpolation_best (str): returns the selected interpolation type name 
     '''
     MAE_dict = {} 
     for method in interpolation_types:
         if method not in ['IDW2','IDW3','IDW4','IDEW2','IDEW3','IDEW4','TPS','GPR','RF']:
             print('The method %s is not currently a supported interpolation type.'%(method))
             sys.exit() 
                
         else:
             if method == 'IDW2':
                 MAE = idw.shuffle_split(loc_dictionary,cvar_dictionary,shapefile,2,rep,False)
                 MAE_dict[method] = MAE
                 
             if method == 'IDW3':
                 MAE = idw.shuffle_split(loc_dictionary,cvar_dictionary,shapefile,3,rep,False)
                 MAE_dict[method] = MAE
                
             if method == 'IDW4':
                 MAE = idw.shuffle_split(loc_dictionary,cvar_dictionary,shapefile,4,rep,False)
                 MAE_dict[method] = MAE

             if method == 'IDEW2':
                 MAE = idew.shuffle_split_IDEW(loc_dictionary,cvar_dictionary,shapefile,file_path_elev,elev_array,idx_list,2,rep)
                 MAE_dict[method] = MAE

             if method == 'IDEW3':
                 MAE = idew.shuffle_split_IDEW(loc_dictionary,cvar_dictionary,shapefile,file_path_elev,elev_array,idx_list,3,rep)
                 MAE_dict[method] = MAE

             if method == 'IDEW4':
                 MAE = idew.shuffle_split_IDEW(loc_dictionary,cvar_dictionary,shapefile,file_path_elev,elev_array,idx_list,4,rep)
                 MAE_dict[method] = MAE

             if method == 'TPS':
                 if 'phi_input' in kwargs.keys(): 
                     MAE= tps.shuffle_split_tps(loc_dictionary,cvar_dictionary,shapefile,phi_input,10)
                     MAE_dict[method] = MAE
                
                 else:
                     print('Please pass phi_input keyword argument')
                     sys.exit()
                     
            if method == 'RF':
                  MAE = rf.shuffle_split_rf(loc_dictionary,cvar_dictionary,shapefile,file_path_elev,elev_array,idx_list,10)
                  MAE_dict[method] = MAE
                  
            if method == 'GPR':
                if 'kernels' in kwargs.keys():
                    MAE = gpr.shuffle_split_gpr(loc_dictionary,cvar_dictionary,shapefile,file_path_elev,elev_array,idx_list,kernels,10)
                    MAE_dict[method] = MAE
                else:
                     print('Please pass kernels keyword argument')
                     sys.exit()
                     
     best_method = min(MAE_dict, key=MAE_dict.get)
     print(best_method)
     return best_method 
