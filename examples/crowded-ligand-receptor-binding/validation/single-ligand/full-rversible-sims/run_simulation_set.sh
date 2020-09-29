#!/bin/bash

# Set parameters
N_L=1
N_C=0
kOn=10000
kOff=1
N_TRAJSTART=20
N_TRAJ=20
N_TRAJEND=$(($N_TRAJSTART+$N_TRAJ))
N_STEPS=10000000
N_THREADS=1
BASERUNDIR="run_bulk_nL${N_L}_nC${N_C}_kOn${kOn}_kOff${kOff}"
RUNSCRIPT="ligand_receptor_bulk.py"

# make base run directory and copy script
mkdir -p ./"$BASERUNDIR"
cp "$RUNSCRIPT" ./"$BASERUNDIR"
cd ./"$BASERUNDIR"

# activate the Readdy conda environment
source activate env_readdy

# run simulations in seperate subdirectories
for ((i=$N_TRAJSTART; i<$N_TRAJEND; i++))
do
    RUNDIR="trajectory_"$i
    echo "$RUNDIR"
    mkdir -p ./"$RUNDIR"
    python $RUNSCRIPT -nL $N_L -nC $N_C -kOn $kOn -kOff $kOff -rd $RUNDIR -ns $N_STEPS -nt $N_THREADS
done
