#!/bin/bash

# Run analysis script for calculating survival probabilities
# from quasi-reversible simulations ensembles

source activate env_readdy

python calc_survival_prob.py

python plot_survival_prob.py
