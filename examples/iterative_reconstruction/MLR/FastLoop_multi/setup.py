import sys  
sys.path.insert(0, "..")  
  
from distutils.core import setup  
from distutils.extension import Extension  
from Cython.Build import cythonize  
from Cython.Distutils import build_ext
  
# ext_module = cythonize("TestOMP.pyx")  
ext_module = Extension(
						"FastLoop_multi",
            ["FastLoop_multi.pyx"],
            extra_compile_args=["-fopenmp"],
            extra_link_args=["-fopenmp"],
            )
  
setup(
    cmdclass = {'build_ext': build_ext},
		ext_modules = [ext_module], 
) 