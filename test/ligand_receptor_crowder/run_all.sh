#!/bin/bash

# Will run full test simulation including analysis. A plot of the survival
# probabilities will be generated as SurvivalProbability.pdf

# Run Brownian Dynamics simulations in bound state and sample reactions w/ Monte Carlo
echo "Running simulation in the bound state..."
./run_bound_simulation_and_calc_dissoc.sh > bound.out 2>&1 
echo "Bound simulation completed."

# Run Brownian Dynamics simulations in unbound state and sample reactions w/ Monte Carlo
echo "Running set of simulations in the unbound state. This may take a few minutes..."
./run_unbound_simulation_set_and_calc_assoc.sh > unbound.out 2>&1
echo "Unbound simulation set completed."

# Post process and plot
echo "Post processing and plotting..."
./run_post_processing.py > analyze.out 2>&1
echo "Quasi-reversible test simulations completed!"
