# pycosie
pycosie (pie cozy) is a complimentary suite of analyses software written in Python meant to be analyzed on Technicolor Dawn and quasarcosie outputs. It is the Python derivative of ARCOSIE.

### Dependencies
The dependencies include: numpy, scipy, yt, caesar, astropy, and h5py.

Most of these can be either `pip` or `conda` installed directly. For example:
```
conda install h5py numpy scipy astropy
```

For `yt`, they have specific installations available on their [website](https://yt-project.org/), but it is recommended to use the `conda` installation:
```
conda install yt -c conda-forge
```

For `caesar`, you can find the necessary files and their documentation [at their GitHub](https://github.com/dnarayanan/caesar), which includes how to install. As of our knowledge, you can use the `setup.py` available:
```
git clone https://github.com/dnarayanan/caesar.git caesar
cd caesar
python setup.py install
```

### Installation
There are a couple ways to do this. One can download this repository, open a terminal, `cd` into it and run:

```
python setup.py install
```

Otherwise, one can use pip:

```
pip install pycosie
```
