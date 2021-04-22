# -*- coding: utf-8 -*-
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

##def requirements():
##    with open('requirements.txt') as f:
##        return f.read().strip().split('\n')

setup(
    name='fire_weather_interpolate',
    url='https://github.com/clara-risk/fire_weather_interpolate',
    author='Clara Risk',
    author_email='clara.risk@mail.utoronto.ca',
    packages=['fire_weather_interpolate'],
    install_requires=['numpy', 'pandas', 'matplotlib'],
    version='0.1.1',
    license='CC0',
    description='Spatial interpolation tools for calculation FWI metrics in Québec Ontario',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    #project_urls={'Documentation': ''}
)
