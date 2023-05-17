# from distutils.core import setup
from setuptools import find_packages
from setuptools import setup 

setup(name='sort',
      version='1.0',
      description='Retail Intelligence sort',
      author='nautec',
      packages=find_packages(),
      install_requires=[
        "numpy>=1.24.2",
        "filterpy>=1.4.5",
        "scikit-image>=0.17.2",
        "lap>=0.4.0"
      ],
      url='https://github.com/retail-intelligence/sort',
    )