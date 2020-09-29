import numpy as np
import argparse
import time

import LSCmodel
###########################################################################################
###########################################################################################
if __name__=="__main__":
   
    #----------------------------------------------------------------------------------------
    # Parse command line arguments
    #----------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument("--nLigand", "-nL", help="set ligand number")
    parser.add_argument("--nCrowder", "-nC", help="set crowder number")
    parser.add_argument("--rundir","-rd", help="set run directory")
    parser.add_argument("--trajfile","-tf", help="set specific tajectory file to analyze")
    parser.add_argument("--savefile","-sf", help="set specific file for output as npy filetype")
    parser.add_argument("--ncores","-ncore", help="set number of processors for parallel computation")
    args = parser.parse_args()

    if args.nLigand:
        nLtag = int(args.nLigand)	# number of ligand added to box
    else:
        nLtag = 2

    if args.nCrowder:
        nC = int(args.nCrowder)	# number of crowders added to box
    else:
        nC = 1

    if args.rundir:
        rundir = args.rundir
    else:
        box_size = np.array([5.,5.,5.])
        rundir = "./boxsize_{}_{}_{}/run_bulk_nL{}_nC{}/trajectory_0/".format(int(box_size[0]),int(box_size[1]),int(box_size[2]),nLtag,nC)

    if args.trajfile:
        trajfile = args.trajfile
        print("TRAJFILE={}".format(trajfile))
    else:
        trajfile = None

    if args.savefile:
        savefile = args.savefile
        print("SAVEFILE={}".format(savefile))
    else:
        savefile = "unbound_reaction_event_density_nL_{}".format(nLtag)

    if args.ncores:
        n_cores = int(args.ncores)
    else:
        n_cores = 1

    #----------------------------------------------------------------------------------------
    # Create LSCmodel object
    #----------------------------------------------------------------------------------------
    model = LSCmodel.LSCModel(nLtag,nC)

    tstart = time.time()
    react_probs = model.calc_reaction_probs(rundir,trajfile,savefile,n_cores)

    print("Reaction Probability Computation Time = {}".format(time.time() - tstart))
