# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 20:16:30 2018

@author: kalvi
"""

from setuptools import find_packages
from setuptools import setup

#REQUIRED_PACKAGES = ['some_PyPI_package>=1.0']
REQUIRED_PACKAGES = [
        'os==',
        'shutil==',
        'time==',
        'importlib==', 
        'numpy==', 
        'tensorflow==', 
        'pandas==', 
        're==', 
        'itertools==', 
        'nltk==', 
        'IPython.display==', 
        '__future__==', 
        ]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)