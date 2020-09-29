import numpy as np
import argparse

import LSCmodel
###########################################################################################

###########################################################################################
if __name__=="__main__":


    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--nLigand", "-nL", help="set ligand number")
    parser.add_argument("--nCrowder", "-nC", help="set crowder number")
    parser.add_argument("--rundir","-rd", help="set run directory")
    args = parser.parse_args()

    if args.nLigand:
        nL = int(args.nLigand)	# number of ligand added to box
    else:
        nL = 2
    if args.nCrowder:
        nC = int(args.nCrowder)	# number of crowders added to box
    else:
        nC = 1

    if args.rundir:
        rundir = args.rundir
    else:
        box_size = np.array([5.,5.,5.])
        rundir = "./boxsize_{}_{}_{}/run_bulk_nL{}_nC{}/trajectory_0".format(int(box_size[0]),int(box_size[1]),int(box_size[2]),nL,nC)


    trajfile = rundir + "/LigandDiffusion_out_bulk.h5"
    savefile = rundir + "/accepted_dissociation_moves.txt"

    model = LSCmodel.LSCModel(nL,nC)
    model.calc_dissociation_prob(trajfile,savefile)

