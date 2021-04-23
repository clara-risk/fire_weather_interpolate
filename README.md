[![DOI](https://zenodo.org/badge/279318014.svg)](https://zenodo.org/badge/latestdoi/279318014)

Fire Weather Interpolate 

A series of Python functions for interpolating the Canadian Forest Fire Weather Index (FWI) variables, evaluating interpolation methods using various cross-validation methods, and calculating fire season duration. For detailed information about the functions, please see: https://clara-risk.github.io/fire_weather_interpolate/fire_weather_interpolate.html#module-Eval

References Statement 

A lot of the code is based off of the R package cffdrs: 

https://github.com/cran/cffdrs/tree/master/R

Wang, X., Wotton, B. M., Cantin, A. S., Parisien, M. A., Anderson, K., Moore, B., & Flannigan, M. D. (2017). cffdrs: an R package for the Canadian Forest Fire Danger Rating System. Ecological Processes, 6(1). https://doi.org/10.1186/s13717-017-0070-z

Other parts of the code are based off of the following publications: 

Wotton, B. M., & Flannigan, M. D. (1993). Length of the fire season in a changing climate. Forestry Chronicle, 69(2), 187–192. https://doi.org/10.5558/tfc69187-2

Lawson, B.D. & Armitage, O.B. (2008). Weather Guide for the Canadian Forest Fire Danger Rating System (pp. 1-84). Natural Resources Canada, Canadian Forest Service, Northern Forestry Centre. 

Some modules rely on other resources, which are cited at the top of the module files. 

The code relies on many packages, including: 

Scipy 

Virtanen, P., R. Gommers, T. E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, E. Burovski, P. Peterson, W. Weckesser, J. Bright, S. J. van der Walt, M. Brett, J. Wilson, K. J. Millman, N. Mayorov, A. R. J. Nelson, E. Jones, R. Kern, E. Larson, C. J. Carey, İ. Polat, Y. Feng, E. W. Moore, J. VanderPlas, D. Laxalde, J. Perktold, R. Cimrman, I. Henriksen, E. A. Quintero, C. R. Harris, A. M. Archibald, A. H. Ribeiro, F. Pedregosa, P. van Mulbregt, A. Vijaykumar, A. Pietro Bardelli, A. Rothberg, A. Hilboll, A. Kloeckner, A. Scopatz, A. Lee, A. Rokem, C. N. Woods, C. Fulton, C. Masson, C. Häggström, C. Fitzgerald, D. A. Nicholson, D. R. Hagen, D. V. Pasechnik, E. Olivetti, E. Martin, E. Wieser, F. Silva, F. Lenders, F. Wilhelm, G. Young, G. A. Price, G. L. Ingold, G. E. Allen, G. R. Lee, H. Audren, I. Probst, J. P. Dietrich, J. Silterra, J. T. Webber, J. Slavič, J. Nothman, J. Buchner, J. Kulick, J. L. Schönberger, J. V. de Miranda Cardoso, J. Reimer, J. Harrington, J. L. C. Rodríguez, J. Nunez-Iglesias, J. Kuczynski, K. Tritz, M. Thoma, M. Newville, M. Kümmerer, M. Bolingbroke, M. Tartre, M. Pak, N. J. Smith, N. Nowaczyk, N. Shebanov, O. Pavlyk, P. A. Brodtkorb, P. Lee, R. T. McGibbon, R. Feldbauer, S. Lewis, S. Tygier, S. Sievert, S. Vigna, S. Peterson, S. More, T. Pudlik, T. Oshima, T. J. Pingel, T. P. Robitaille, T. Spura, T. R. Jones, T. Cera, T. Leslie, T. Zito, T. Krauss, U. Upadhyay, Y. O. Halchenko, and Y. Vázquez-Baeza. 2020. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods 17:261–272.

Numpy 

Harris, C. R., K. J. Millman, S. J. van der Walt, R. Gommers, P. Virtanen, D. Cournapeau, E. Wieser, J. Taylor, S. Berg, N. J. Smith, R. Kern, M. Picus, S. Hoyer, M. H. van Kerkwijk, M. Brett, A. Haldane, J. F. del Río, M. Wiebe, P. Peterson, P. Gérard-Marchant, K. Sheppard, T. Reddy, W. Weckesser, H. Abbasi, C. Gohlke, and T. E. Oliphant. 2020. Array programming with NumPy.

Matplotlib

Hunter, J. D. 2007. Matplotlib: A 2D graphics environment. Computing in Science and Engineering 9:90–95.

scikit-learn

Pedregosa, F., G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and É. Duchesnay. 2011. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research 12:2825–2830.




