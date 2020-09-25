import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx


import os
import glob
from joblib import Parallel, delayed

import qrevIPRD.analysis as qan
import LSCmodel
import time as pytime
#----------------------------------------------------------------------------
def calc_effective_rates(tau_n,mean_n):
    kon = mean_n/tau_n
    koff = (1.-mean_n)/tau_n
    return (kon,koff)

#----------------------------------------------------------------------------
if __name__=="__main__":

    nL = 2
    nCs = [1]
    traj_number = 0


    # Set intrisic reaction rates
    kao = [1.0e+2,1.0e+3,1.0e+4]
    kdo = 1.0e-0

    models = [[] for i in kao]
    t_unbind = [[] for i in kao]
    t_bind = [[] for i in kao]
    time = [[] for i in kao]
    surv_prob = [[] for i in kao]
    extrap_surv = [[] for i in kao]
    sep_prob = [[] for i in kao]
    corr_func = [[] for i in kao]
    tau_n = [[] for i in kao]
    n_mean = [[] for i in kao]
    n_var = [[] for i in kao]
    coarsening = [[] for i in kao]
    kon_inf = [[] for i in kao]


    x = []
    for i,kaoi in enumerate(kao):
        for nCi in nCs:
            x.append((kaoi,nCi))

    target_surv_prob = 1e-4  	# where to extrapolate to with exponential function
    nsteps = None		# number of steps after which extrapolation is forced
    ntrajs = [10]		# number of trajectories to include in processing

   
    # if directory for analysis output does not exist, make it
    if not os.path.exists('analyzed_data'):
        os.makedirs('analyzed_data')

    def parfunc(x,ntraj):
        kaoi = x[0]
        nCi = x[1]
        models.append(LSCmodel.LSCModel(nL,nCi))
        dt = models[-1].dt

        basedir = "./boxsize_5_5_5/run_bulk_nL{}_nC{}/trajectory_{}".format(nL,nCi,traj_number)
        dissocdatafile = basedir+"/accepted_dissociation_moves.txt"
        unbounddata_template = basedir+"/unbound_simulations_fine_output/unbound_reaction_event_density_nL_{}_*.npy".format(nL)

        # Load reaction probability data for each timepoint for each trajectory
        unbound_data_zipfile =  "./boxsize_5_5_5/run_bulk_nL{}_nC{}".format(nL,nCi)+"/unbound_output_combined.zip"

        coarse = 10
        tstart = pytime.time()
        #out = qan.process_quasireversible_simulations(kaoi,kdo,dt,dissocdatafile,unbound_data_files,coarse) 
        out = qan.process_quasireversible_simulations(kaoi,kdo,dt,dissocdatafile,unbound_data_zipfile,coarse,
                                                      zipped=True,target_surv_prob=target_surv_prob,nsteps=nsteps,ntraj=ntraj) 
        print("Processed run nC = {}, kao = {}, ntraj = {} in {:.2f} min".format(nCi,kaoi,ntraj,(pytime.time()-tstart)/60.))
            
        # Save output to .npz file for plotting
        out['nC'] = nCi
        out['kao'] = kaoi
        out['kdo'] = kdo
        if isinstance(ntraj,tuple):
            outfilename = "./analyzed_data/analysis_out_nC_{}_kao_{}_kdo_{}_targetprob_{}_nsteps_{}_ntraj_{}_{}.npz".format(
                                                        nCi,kaoi,kdo,target_surv_prob,nsteps,ntraj[1]-ntraj[0],ntraj[0])
        else:
            outfilename = "./analyzed_data/analysis_out_nC_{}_kao_{}_kdo_{}_targetprob_{}_nsteps_{}_ntraj_{}.npz".format(
                                                        nCi,kaoi,kdo,target_surv_prob,nsteps,ntraj)
        np.savez(outfilename,out)

        return 1

    n_proc = 1
    
    for ntraji in ntrajs:
        outs = Parallel(n_jobs=n_proc)(delayed(parfunc)(xi,ntraji) for xi in x) 



