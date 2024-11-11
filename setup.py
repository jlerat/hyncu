#!/usr/bin/env python
""" Setup script for hyncu"""

import os
from pathlib import Path

from setuptools import setup, Extension, find_packages

import versioneer

cmdclass = versioneer.get_cmdclass()

setup(
    name='hyncu',
    version=versioneer.get_version(),
    cmdclass = cmdclass,
    packages=find_packages(),
    install_requires=[
        'numpy>=1.8.1',
        'pandas>=0.12.1',
        'netcdf4'
    ],
    package_data = {
        #'package': [
            #'template/*',
        #]
    },

    # Metadata
    author='Julien Lerat, CSIRO Environment'
)
