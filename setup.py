import sys
from setuptools import setup, find_packages, Extension
from setuptools.command.sdist import sdist as _sdist
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.build_ext import build_ext as _build_ext
import numpy
from Cython.Build import cythonize

class build_py(_build_py):
    def run(self):
        _build_py.run(self)
        
class build_ext(_build_ext):
    # subclass setuptools extension builder to avoid importing numpy
    # at top level in setup.py. See http://stackoverflow.com/a/21621689/1382869
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process
        # see http://stackoverflow.com/a/21621493/1382869
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())
        
class sdist(_sdist):
    # subclass setuptools source distribution builder to ensure cython
    # generated C files are included in source distribution.
    # See http://stackoverflow.com/a/18418524/1382869
    def run(self):
        # Make sure the compiled Cython files in the distribution are up-to-date
        from Cython.Build import cythonize
        cythonize(cython_extensions)
        _sdist.run(self)

sys.path.insert(0,"pycosie")

desc = "Anaylsis tools to help with TD (and hopefully other simulations...)"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
cython_extensions = [
    Extension("cgauss_smooth", ["./pycosie/utils/cgauss_smooth.pyx"],
              include_dirs=[numpy.get_include()],
              #library_dirs=["/usr/lib","/usr/lib/x86_64-linux-gnu"],
              libraries=["m"],
              #extra_compile_args=["-march=x86-64"], 
              ),
    Extension("trapz", ["./pycosie/utils/trapz.pyx"],
              include_dirs=[numpy.get_include()],
              #library_dirs=["/usr/lib","/usr/lib/x86_64-linux-gnu"],
              libraries=["m"],
              #extra_compile_args=["-march=x86-64"], 
              )
]

setup(
    name="pycosie",
    description=desc,
    long_description=long_description,
    version="0.1.5",
    packages=find_packages(),
    cmdclass={
        'sdist': sdist,
        'build_ext': build_ext,
        'build_py': build_py
    },
    ext_modules = cythonize(cython_extensions, annotate=True), # cythonize to show xython usage and files (pyx, pxd)
    include_dirs=[numpy.get_include()], # need to include numpy in setup, and for cython compile
    author="Samir Kusmic",
    project_url={
        "Pycosie":"https://github.com/kusmic/pycosie",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ],
    license="GNU General Public License v2 (GPLv2)",
    install_requires=[
        "numpy",
        "scipy",
	    "yt==4.0.1",
        "astropy",
        "h5py",
        "Cython==3.0.0a11",
        "numba",
    ]
)
