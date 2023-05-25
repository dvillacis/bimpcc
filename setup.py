import setuptools

import os
version = {}
with open(os.path.join('bimpcc', 'version.py')) as fp:
    exec(fp.read(), version)
__version__ = version['__version__']

try:
    from numpy.distutils.core import setup, Extension
except ModuleNotFoundError:
    print('Numpy is required to install this package')
    
try:
    import pyproximal
except ModuleNotFoundError:
    print('Pyproximal is required to install this package')
    
try:
    import pyoptsparse
except ModuleNotFoundError:
    print('Pyoptsparse is required to install this package')
    
try:
    import pandas
except ModuleNotFoundError:
    print('Pandas is required to install this package')
    
try:
    import PIL
except ModuleNotFoundError:
    print('Pillow is required to install this package')
    
try:
    import skimage
except ModuleNotFoundError:
    print('Scikit-Image is required to install this package')
    
# Main package setup
setup(
    name='bimpcc',
    version=__version__,
    description='Nonsmooth Bilevel Parameter Learning as MPCC',
    long_description=open('README.md').read(),
    author='David Villacís',
    author_email='david.villacis01@epn.edu.ec',
    packages=['bimpcc'],
    license='GNU GPL',
    keywords='complementarity constraints image processing parameter learning bilevel optimization',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    install_requires=['numpy', 'scipy', 'pylops', 'pyproximal','pyoptsparse','pandas','pillow','scikit-image'],
)