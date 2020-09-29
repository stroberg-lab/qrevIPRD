import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

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

    nL = 10
    #nCs = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    nCs = [60, 70, 80]
    traj_number = 0


    # Set intrisic reaction rates
    kao = [0.3e+1,1.0e+1,0.3e+2,1.0e+2,0.3e+3,1.0e+3]
    #kao = [0.3e+5,1.0e+5,]
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

    def parfunc(x):
        kaoi = x[0]
        nCi = x[1]
        models.append(LSCmodel.LSCModel(nL,nCi))
        dt = models[-1].dt

        basedir = "./rundir/run_bulk_nL{}_nC{}/trajectory_{}".format(nL,nCi,traj_number)
        dissocdatafile = basedir+"/accepted_dissociation_moves.txt"
        unbounddata_template = basedir+"/unbound_simulations_fine_output/unbound_reaction_event_density_nL_{}_*.npy".format(nL)

        # Load reaction probability data for each timepoint for each trajectory
        unbound_data_files = []
        for datai in glob.glob(unbounddata_template):
            unbound_data_files.append(datai)
        unbound_data_zipfile =  basedir+"/unbound_output.zip"

        coarse = 10
        target_surv_prob = 1e-4  # where to extrapolate to with exponential function
        tstart = pytime.time()
        #out = qan.process_quasireversible_simulations(kaoi,kdo,dt,dissocdatafile,unbound_data_files,coarse) 
        out = qan.process_quasireversible_simulations(kaoi,kdo,dt,dissocdatafile,unbound_data_zipfile,coarsening=coarse,zipped=True,target_surv_prob=target_surv_prob) 
        print("Processed run nC = {}, kao = {} in {:.2f} min".format(nCi,kaoi,(pytime.time()-tstart)/60.))

        # Save output to .npz file for plotting
        out['nC'] = nCi
        out['kao'] = kaoi
        out['kdo'] = kdo

        outfilename = "./analyzed_data/analysis_out_nC_{}_kao_{}_kdo_{}.npz".format(nCi,kaoi,kdo)
        
        np.savez(outfilename,out)

        return 1

    n_proc = 3
    outs = Parallel(n_jobs=n_proc)(delayed(parfunc)(xi) for xi in x) 


