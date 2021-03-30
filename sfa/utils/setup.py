'''
# -*- coding:UTF-8 -*-
# Author: Meng Liu
# DoC: 2020.09.27
# email:
# Description:
# Refer from: https://github.com/qianguih/voxelnet/blob/master/utils/setup.py
'''



from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='box overlaps',
    ext_modules=cythonize('./box_overlaps.pyx')
)
