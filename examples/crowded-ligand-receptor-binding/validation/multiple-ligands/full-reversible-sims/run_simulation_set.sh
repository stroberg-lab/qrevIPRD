#!/bin/bash

# Set parameters
N_L=2
N_C=1
kOn="1E+4"
kOff="1E-0"
N_TRAJSTART=120
N_TRAJ=40
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

echo "`date "+%Y-%m-%d %H:%M:%S"`" # for timing purposes

# run simulations in seperate subdirectories
for ((i=$N_TRAJSTART; i<$N_TRAJEND; i++))
do
    RUNDIR="trajectory_"$i
    echo "$RUNDIR"
    mkdir -p ./"$RUNDIR"
    python $RUNSCRIPT -nL $N_L -nC $N_C -kOn $kOn -kOff $kOff -rd $RUNDIR -ns $N_STEPS -nt $N_THREADS
done

echo "`date "+%Y-%m-%d %H:%M:%S"`"
