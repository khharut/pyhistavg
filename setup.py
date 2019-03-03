#!/usr/bin/env python
from setuptools import setup

setup(name='pyhistavg',
      version='1.14',
      description='Dynamic thresholds for Monitis monitors',
      url='http://www.monitis.com/',
      author='Harutyun Khachatryan, Tigran Khachikyan, Anoush Ghambaryan',
      author_email='harutyun.khachatryan@monitis.com',
      license='Monitis Inc.',
      packages=['pyhistavg'],
      install_requires=[
          'scipy>=0.13.3','statsmodels>=0.5.0','pandas>=0.13.1','numpy>=1.8.2',
          'matplotlib>=1.2.1','pystl>=1.0'
      ],
      zip_safe=False)