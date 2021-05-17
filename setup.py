from setuptools import setup
from Cython.Build import cythonize

setup(
    name='cython_LDA',
    ext_modules=cythonize("Cython_ST_LDA_functions.pyx"),
    zip_safe=False,
)
