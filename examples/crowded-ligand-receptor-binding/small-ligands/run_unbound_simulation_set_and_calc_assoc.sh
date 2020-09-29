#!/bin/bash

# Set parameters
N_L=10
N_C=${1}
TRAJNUM=1

BASEDIR="rundir/run_bulk_nL${N_L}_nC${N_C}"
#BASEDIR="visualization_rundir/run_bulk_nL${N_L}_nC${N_C}"
RUNDIR="${BASEDIR}/trajectory_$TRAJNUM"
TRAJDIR="unbound_simulations_fine_output"
OUTDIR="${RUNDIR}/${TRAJDIR}"
RUNSCRIPT="run_unbound_state_simulation.py"
NUMLINES=$(awk 'END {print NR}' ${RUNDIR}/accepted_dissociation_moves.txt)
echo "$NUMLINES"

CLEARTRAJFILES=1
SAVEXYZ=0

N_TRAJ=3000
START=0
#STEP=1
STEP=$((NUMLINES / N_TRAJ))
echo "$STEP"
if [ $STEP -lt 1 ]
then
    STEP=1
    echo "Not enough initital conditions for requested number of trajectories, setting step to 1"
fi

N_STEPS=100000
N_THREADS=1

# activate the Readdy conda environment
source activate env_readdy

# run simulations in seperate subdirectories
for ((i=0, ind=$START; i<$N_TRAJ; i++, ind+=$STEP))
do
    echo "$ind"
    TRAJFILE="${OUTDIR}/LigandDiffusion_unbound_out_bulk_index_${ind}.h5"
    SAVEFILE="${TRAJDIR}/unbound_reaction_event_density_nL_${N_L}_${ind}.npy"
    # run simulation
    python $RUNSCRIPT -nL $N_L -nC $N_C -rd "${RUNDIR}/" -od "${OUTDIR}/" -ns $N_STEPS -nt $N_THREADS -oi $ind -xyz ${SAVEXYZ}
    # run MC simulation on trajectory
    python calc_assoc_prob_v2.py -nL $N_L -nC $N_C -rd "${RUNDIR}/" -tf $TRAJFILE -sf $SAVEFILE
    if [ $ind -gt $START ] && [ $CLEARTRAJFILES == 1 ]
    then
        # delete simulation trajectory
        rm $TRAJFILE 
        echo "DELETING $TRAJFILE"
    fi    
done

