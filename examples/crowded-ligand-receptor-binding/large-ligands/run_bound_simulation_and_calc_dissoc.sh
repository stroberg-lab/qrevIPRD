#!/bin/bash

# Set parameters
N_L=10
N_C=0
N_TRAJSTART=0
N_TRAJ=1
N_TRAJEND=$(($N_TRAJSTART+$N_TRAJ))
N_STEPS=100000
N_THREADS=1
BASERUNDIR="visualization_rundir/run_bulk_nL${N_L}_nC${N_C}"
RUNSCRIPT="run_bound_state_simulation.py"

# make base run directory and copy script
mkdir -p ./"$BASERUNDIR"

# activate the Readdy conda environment
source activate env_readdy

# run simulations in seperate subdirectories
for ((i=$N_TRAJSTART; i<$N_TRAJEND; i++))
do
    #RUNDIR="trajectory_"$i
    RUNDIR="${BASERUNDIR}/trajectory_"$i
    echo "$RUNDIR"
    mkdir -p ./"$RUNDIR"
    python $RUNSCRIPT -nL $N_L -nC $N_C -rd $RUNDIR -ns $N_STEPS -nt $N_THREADS

    # Run MC simulation on trajectories to get dissociation probability
    python calc_dissoc_prob.py -nL $N_L -nC $N_C -rd "${RUNDIR}"

done

