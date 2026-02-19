from distutils.core import setup
import numpy
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("NGP", ["NGP.pyx"])]
)