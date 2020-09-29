#!/bin/bash

# Set parameters
N_L=2
N_C=1
TRAJNUM=0
BOXSIZE=5

BASEDIR="boxsize_${BOXSIZE}_${BOXSIZE}_${BOXSIZE}/run_bulk_nL${N_L}_nC${N_C}"
RUNDIR="${BASEDIR}/trajectory_$TRAJNUM"
TRAJDIR="unbound_simulations_fine_output"
OUTDIR="${RUNDIR}/${TRAJDIR}"
RUNSCRIPT="run_unbound_state_simulation.py"
NUMLINES=$(awk 'END {print NR}' ${RUNDIR}/accepted_dissociation_moves.txt)
echo "$NUMLINES"

CLEARTRAJFILES=1

N_TRAJ=1000
START=0
STEP=6

N_STEPS=50000
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
    python $RUNSCRIPT -nL $N_L -rd "${RUNDIR}/" -od "${OUTDIR}/" -ns $N_STEPS -nt $N_THREADS -oi $ind
    # run MC simulation on trajectory
    python calc_assoc_prob_v2.py -nL $N_L -nC $N_C -rd "${RUNDIR}/" -tf $TRAJFILE -sf $SAVEFILE
    if [ $ind -gt $START ] && [ $CLEARTRAJFILES == 1 ]
    then
        # delete simulation trajectory
        rm $TRAJFILE 
        echo "DELETING $TRAJFILE"
    fi    
done

# Run MC simulation on trajectories to get dissociation probability
#python calc_assoc_prob_v2.py -nL $N_L -nC $N_C -bd $BASEDIR
