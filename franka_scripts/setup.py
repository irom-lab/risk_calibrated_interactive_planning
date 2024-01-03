from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import find_packages

d = generate_distutils_setup()
d['packages'] = ['src', 'rviz','data']
d['package_dir'] = {'src': 'src', 'rviz': 'rviz', 'data':'data'}

setup(**d)
