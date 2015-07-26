from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
  name = 'sharedict',
  ext_modules = cythonize([Extension("sharedict", ["cas.c", "sharedict.pyx"], libraries=["rt", "pthread"])])
)
