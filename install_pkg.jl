# run this to install the julia dependencies on this machine
# necessary for the module. To run:

# julia install_pkg.jl

using Pkg

Pkg.add("PyCall")
Pkg.add("Trapz")