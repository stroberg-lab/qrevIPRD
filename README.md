# qrevIPRD
Quasi-reversible simulation wrapper for Interacting Particle Reaction Diffusion simulations.

This package contains source code for qrevIPRD simualtions. To run, download source code and place in python path.

Runnding simulations requires several packages including ReaDDy2, a particle-based reaction diffusion simulator (https://readdy.github.io). We have included a conda environment file env_readdy.yml containing all packages needed to run qrevIPRD. If using Anaconda, we recommend creating an environment "env_readdy" by using the file env_readdy.yml. To do so, run "conda env create -f env_readdy.yml". Alternatively, the list of packages contained in env_readdy.yml can be installed manually.

Testing
A simple test case is supplied under the "test" folder.
