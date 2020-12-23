"""
fire_weather_interpolate
======

Summary
--------

Code for spatially interpolating weather data and also for calculating the
Canadian fire weather index (FWI) metrics. Includes code for comparison and
evaluation of accuracy (cross-validation).

get_data: contains functions to extract the weather data (and data from
lookup files, such as for slope) from the files provided with the code.

idw: has functions to interpolate using IDW interpolation and to cross-validate 
for each weather station.

tps: has functions to interpolate using thin plate (smoothing) splines and
to cross-validate. 

idew: has functions to interpolate using IDEW interpolation and to
cross-validate. 

ok: has functions to interpolate using ordinary kriging and to cross-validate.

rf: has functions to interpolate using random forest and to cross-validate. 

fwi: has functions to calculate the FWI metrics and do procedures such as
calculating the station start-up date and overwintering the drought code.

Eval: has functions for evaluating the strength of the relationship with 
area burned and also for evaluating the leave-one-out cross validation 
(i.e., RMSE calculation). 


References
----------

.. [1]Wang, X., Wotton, B. M., Cantin, A. S., Parisien, M. A., Anderson,
    K., Moore, B., & Flannigan, M. D. (2017). cffdrs: an R package for the
    Canadian Forest Fire Danger Rating System. Ecological Processes, 6(1).
    https://doi.org/10.1186/s13717-017-0070-z

.. [2] Lawson, B.D. & Armitage, O.B. (2008). Weather Guide for the Canadian
    Forest Fire Danger Rating System (pp. 1-84). Natural Resources Canada,
    Canadian Forest Service, Northern Forestry Centre.

.. [3] Wotton, B. M., & Flannigan, M. D. (1993). Length of the fire season in
    a changing climate. Forestry Chronicle, 69(2), 187–192. 
    https://doi.org/10.5558/tfc69187-2


"""

# import from the separate files
from . import get_data as GD
from . import idw as idw
from . import idew as idew
from . import ok as ok
from . import tps as tps
from . import rf as rf
from . import gpr as gpr
from . import fwi as fwi
from . import Eval as Eval  
from . import cluster_3d as c3d
from . import make_blocks as mbk
#Check import 

try:
    from fire_weather_interpolate._version import __version__
except ImportError:  
    print('The package was not installed correctly')
    __version__ = '0.0.1'

__version__ = '0.0.1'
__author__ = 'clara <clara.risk@mail.utoronto.ca>'

    
