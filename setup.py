from setuptools import setup, find_packages

setup(
    name='sort',
    version='1.0',
    author='Alex Bewley',
    packages=find_packages(),
    install_requires=['filterpy', 'lap'],
    python_requires='>=3.6'
)

