#coding: utf-8

"""
Summary
-------
Auto-selection protocol for spatial interpolation methods using shuffle-split cross-validation. 
References
----------
"""

def run_comparison(interpolation_types):
     '''Execute the shuffle-split cross-validation for the given interpolation types 
     Parameters
         interpolation_types (list of str): list of interpolation types to consider
     Returns 
         interpolation_best (str): returns the selected interpolation type name 
     '''    
