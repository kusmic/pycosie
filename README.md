# pycosie
pycosie (pie cozy) is a complimentary suite of analyses software written in Python meant to be analyzed on Technicolor Dawn and quasarcosie outputs. It is the Python support package of ARCOSIE.

### Dependencies
The dependencies include: numpy, scipy, yt, caesar, astropy, and h5py.

Most of these can be either `pip` or `conda` installed directly. For example:
```
conda install h5py numpy scipy astropy
```

For `yt==4.0.1`, they have specific installations available on their [website](https://yt-project.org/), but it is recommended to use the `pip` installation:
```
pip install yt
```

For `caesar`, you can find the necessary files and their documentation [at their GitHub](https://github.com/dnarayanan/caesar), which includes how to install. As of our knowledge, you can use the `setup.py` available:
```
git clone https://github.com/dnarayanan/caesar.git caesar
cd caesar
python setup.py install
```

Extra dependencies come from the baked-in compiling from `pyjulia`, so make sure you have these libraries available in your PATH:
```
pip install pyjulia
```
This also depends on having Julia on your machine. Please follow the Julia website to install on your machine and make sure it is in your PATH.

### Installation
There are a couple ways to do this. One can download this repository, open a terminal, `cd` into it and run:

```
python setup.py install
python setup.py build_ext --inplace
```

Otherwise, one can use pip. HOWEVER, due to how quickly this software is updated and bugs arise and get squashed, I recommend using the
first option. I HIGHLY RECOMMEND NOT TO PIP INSTALL AT THE MOMENT. It is only mentioned for completeness:

```
pip install pycosie
```
### Running

Since it has `pyjulia` implemented it needs different set up to sun. *If it is a source code file*, you can run it as such:

```
python-jl your-code.py <args>
``` 
If you are running it on a Jupyter notebook, I could not get the kernel to find `python-jl` so I suggest implementing this if you also have that issue, needs to be the first thing (**note**: this will take a while just to run this):

```
from julia.api import Julia
jl = Julia(compiled_modules=False)
```

### Latest Notes
