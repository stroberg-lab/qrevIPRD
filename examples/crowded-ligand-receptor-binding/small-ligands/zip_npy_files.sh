#!/bin/bash

zip rundir/run_bulk_nL${1}_nC${2}/trajectory_${3}/unbound_output.zip rundir/run_bulk_nL${1}_nC${2}/trajectory_${3}/unbound_simulations_fine_output/*.npy
