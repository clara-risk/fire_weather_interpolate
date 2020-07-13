# -*- coding: latin1 -*-
"""Spatial interpolation tools for calculation FWI metrics in Québec Ontario."""

import os
from setuptools import setup, find_packages, Extension

CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering :: GIS",
]

def requirements():
    with open('requirements.txt') as f:
        return f.read().strip().split('\n')
    
setup(name='fwi_interpolate',
      version='0.0.1',
      author='Clara',
      author_email='clara.risk@mail.utoronto.ca',
      description='Spatial interpolation tools for calculation FWI metrics in Québec Ontario',
      classifiers=CLASSIFIERS,
      install_requires=requirements(),
      packages=find_packages(),
      include_package_data=True,
)


