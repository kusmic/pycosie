import sys
from setuptools import setup, find_packages

sys.path.insert(0,"pycosie")

desc = "Anaylsis tools to help with TD (and hopefully other simulations...)"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pycosie",
    description=desc,
    long_description=long_description,
    version="0.1.0",
    packages=find_packages(),
    author="Samir Kusmic",
    project_url={
        "Pycosie":"https://github.com/kusmic/pycosie",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Punlic License v2.0",
        "Operating System :: OS Independent",
    ],
    license="GNU Public License v2.0",
    install_requires=[
        "numpy",
        "scipy",
        "yt",
        "caesar",
        "astropy",
        "h5py"
    ],
    dependency_links=["https://github.com/dnarayanan/caesar/tarball/master#egg=caesar-0.2"]
)
