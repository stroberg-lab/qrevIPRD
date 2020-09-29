import numpy as np
import matplotlib.pyplot as plt
import readdy
from scipy import interpolate
from scipy.integrate import trapz, cumtrapz, quadrature
import glob
from joblib import Parallel, delayed
import time as pytime

import qrevIPRD.analysis as qan
import qrevIPRD.potentials as pot

#----------------------------------------------------------------------------
def calculate_lma_occupancy(micro_onrate,micro_offrate,V,Vex,Vreact):
    Kon = micro_onrate * Vreact/(V-Vex)
    Koff = micro_offrate
    n_occ = 1./ (1. + Koff/Kon)
    return n_occ

#-----------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
if __name__=="__main__":

    # Define simulation paramters
    nL = 1
    nC = 0
    #V = 10.*10.*10.
    V = 5.*5.*5.
    vL = 4./3.*np.pi*pow(1.,3)
    vR = 4./3.*np.pi*pow(1.,3)
    Veff = V - nL*vL-vR
    c = nL / Veff
    sigma = 2.	# molecular size (cross section)
    D = 2.	# relative diffusion coefficient
    
    eps = 1/1.5
    m = 12.
    n = 6.
    sigmaLR = 2.0
    rmin = lambda sigma: sigma*pow(m/n, 1./(m-n))	# location of minimum of LJ potential (use for cut-shift point)
    rc = rmin(sigmaLR)
    LR_pot = lambda r: pot.lennard_jones(r,eps,m,n,sigmaLR,rc,shifted=True)

    Rreact = 2.0
    Vex = qan.calc_Vex(LR_pot,rc)
    Vreact_eff = qan.calc_eff_volume(LR_pot,Rreact)


    kdo = 1E-0

    ##############################################################################################
    # Load quasi-reversible simulation data
    traj_number = 0

    #1basedir_qr = "./db_quasi_reversible_sims/long_sims"
    basedir_qr = "./db_quasi_reversible_sims/2020_06_09"
    # Set intrisic rate constants for analysis
    kao_qr = np.logspace(0,4,20)	# values for comparison w/ reversible simulations


    models_qr = [[] for i in kao_qr]
    t_unbind_qr = [[] for i in kao_qr]
    t_bind_qr = [[] for i in kao_qr]
    time_qr = [[] for i in kao_qr]
    surv_prob_qr = [[] for i in kao_qr]
    extrap_surv_qr = [[] for i in kao_qr]
    sep_prob_qr = [[] for i in kao_qr]
    corr_func_qr = [[] for i in kao_qr]
    tau_n_qr = [[] for i in kao_qr]
    n_mean_qr = [[] for i in kao_qr]
    n_var_qr = [[] for i in kao_qr]
    coarsening_qr = [[] for i in kao_qr]
    kon_inf_qr = [[] for i in kao_qr]
    preb_qr = [[] for i in kao_qr]

    target_prob = 0.0001
    #nsteps = [10000,100000,None]
    nsteps = [None]
    ntraj = 4000
    for i,nstep in enumerate(nsteps):
        for kaoj in kao_qr:

            # Load data from .npz file
            #outfilename = "./analyzed_data/analysis_out_nC_{}_kao_{}_kdo_{}_targetprob_{}.npz".format(nCs[j],kao[i],kdo,target_prob,nsteps)
            #if nstep is not None:
            #    outfilename = basedir_qr+"/analyzed_data/analysis_out_nC_{}_kao_{}_kdo_{}_targetprob_{}_nsteps_{}.npz".format(nC,kaoj,kdo,target_prob,nstep)
            #else:
            #    outfilename = basedir_qr+"/analyzed_data/analysis_out_nC_{}_kao_{}_kdo_{}_targetprob_{}.npz".format(nC,kaoj,kdo,target_prob)
            outfilename = basedir_qr+"/analyzed_data/analysis_out_nC_{}_kao_{}_kdo_{}_targetprob_{}_nsteps_{}_ntraj_{}.npz".format(nC,kaoj,kdo,target_prob,nstep,ntraj)

            out = np.load(outfilename,allow_pickle=True)
            out = {key:out[key].item() for key in out}
            out = out['arr_0']

            t_unbind_qr[i].append(out['mfpt_unbind'])
            t_bind_qr[i].append(out['mfpt_bind'])
            time_qr[i].append(out['time'])
            surv_prob_qr[i].append(out['surv_prob'])
            extrap_surv_qr[i].append(out['extrap_surv'])
            sep_prob_qr[i].append(out['sep_prob'])
            corr_func_qr[i].append(out['corr_func'])
            tau_n_qr[i].append(out['tau_n'])
            n_mean_qr[i].append(out['n_mean'])
            n_var_qr[i].append(out['n_var'])
            coarsening_qr[i].append(out['coarsening'])
   
            # Long-time limit of association rate
            ind = out['extrap_surv'].shape[0]
            dlogSij = np.diff(np.log(out['surv_prob'][-ind:-ind+2]))
            dtij = np.diff(out['time'][-ind:-ind+2])
            kon_inf_qr[i].append(-dlogSij/dtij)

            # Escape/Rebinding probabilities
            extrap_time = out['time'][-out['extrap_surv'].shape[0]:]
            log_extrap = np.log(out['extrap_surv'])
            pfit = np.polyfit(extrap_time,log_extrap,1)
            intercept = pfit[1]
            pesc = np.exp(intercept)
            preb_qr[i].append(1. - pesc)



    ##############################################################################################
    # Process reversible data

    kao_r = [10,100,1000,10000]
    basedir_r = './full_reversible_sims/small_box_5_5_5'
    datadirs = [basedir_r+'/run_bulk_nL{}_nC{}_kOn{:.0f}_kOff1/'.format(nL,nC,kaoi) for kaoi in kao_r]

    [n_mean_r, n_var_r, surv_data, logsurv_data, event_density_data, nmean_rev] = qan.process_reversible_data_v2(datadirs)


    ##############################################################################################
    # Calculate expected occupancy for Detailed-balance reversible simulations
    micro_onrates = np.logspace(0.,5.,100)
    micro_offrate = kdo #kdo_r
    n_db = calculate_lma_occupancy(micro_onrates,micro_offrate,V,Vex,Vreact_eff)
    n_var_db = n_db * (1. - n_db)

    ##############################################################################################
    # PLOTTING
    ##############################################################################################
    # Plot mean activation and variance (point statistics)
    figPS, axPS = plt.subplots(1,2,figsize=(10,5))


    for i,nstep in enumerate(nsteps):
        #axPS[0].plot(kao_qr,n_mean_qr[i],"-o",label="Quasi-Reversible, $n_{{ext}}={}$".format(nstep))
        axPS[0].plot(kao_qr,n_mean_qr[i],"-o",label="Quasi-Reversible".format(nstep))
        axPS[1].plot(kao_qr,n_var_qr[i],"-o",label="Quasi-Reversible")

    axPS[0].plot(kao_r,n_mean_r,'s',label="Reversible")
    axPS[1].plot(kao_r,n_var_r,'s',label="Reversible")

    axPS[0].plot(micro_onrates,n_db,'--',label="Expected DB")
    axPS[1].plot(micro_onrates,n_var_db,'--',label="Expected DB")


    axPS[0].set_xlabel(r'$k_{a}^{o}$',usetex=True,fontsize=18)
    axPS[0].set_ylabel(r'$\langle n \rangle$',usetex=True,fontsize=18)
    axPS[0].legend()

    axPS[1].set_xlabel(r'$k_{a}^{o}$',usetex=True,fontsize=18)
    axPS[1].set_ylabel(r'$\sigma_{n}^2$',usetex=True,fontsize=18)

    axPS[0].set_xscale('log')
    axPS[1].set_xscale('log')

    figPS.tight_layout()

    figPS.savefig("./Figures/Mean_Variance.pdf")


    plt.show()
