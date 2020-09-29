import numpy as np
import matplotlib.pyplot as plt

params = {'text.latex.preamble' : [ r'\usepackage{mathrsfs}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)

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

    #basedir_qr = "./db_quasi_reversible_sims/long_sims"
    basedir_qr = "./db_quasi_reversible_sims/2020_06_09"
    # Set intrisic rate constants for analysis
    kao_qr = np.logspace(1,4,4)	# values for comparison w/ reversible simulations
    #kao_q = np.logspace(0,4,20)	# additional values for plotting


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
    nsteps = [None]
    for i,nstep in enumerate(nsteps):
        for kaoj in kao_qr:

            # Load data from .npz file
            #outfilename = "./analyzed_data/analysis_out_nC_{}_kao_{}_kdo_{}_targetprob_{}.npz".format(nCs[j],kao[i],kdo,target_prob,nsteps)
            #if nstep is not None:
            #    outfilename = basedir_qr+"/analyzed_data/analysis_out_nC_{}_kao_{}_kdo_{}_targetprob_{}_nsteps_{}.npz".format(nC,kaoj,kdo,target_prob,nstep)
            #else:
            #    outfilename = basedir_qr+"/analyzed_data/analysis_out_nC_{}_kao_{}_kdo_{}_targetprob_{}.npz".format(nC,kaoj,kdo,target_prob)
            outfilename = basedir_qr+"/analyzed_data/analysis_out_nC_{}_kao_{}_kdo_{}_targetprob_{}_nsteps_{}.npz".format(nC,kaoj,kdo,target_prob,nstep)

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

    #---------------------------------------------------------------------------------------------
    # Plot survival probability
    figU, axU = plt.subplots(1,2,figsize=(10,5))

    # Quasi-revesible
    colors = []
    #linestyles = [':','--','-']
    linestyles = ['-']
    for i,nstep in enumerate(nsteps):
        for j in range(len(kao_qr)):
            p = axU[0].plot(time_qr[i][j][::1],surv_prob_qr[i][j][::1],linestyle=linestyles[i],label="$k_{{a}}^{{o}}={}, n_{{ext}}={}$".format(kao_qr[j],nstep))
            colors.append(p[0].get_color())
            axU[1].plot(time_qr[i][j][::1],surv_prob_qr[i][j][::1],'-',color=colors[-1])

    # Reversible
    for (sdi,lsdi,kai,color) in zip(surv_data,logsurv_data,kao_r,colors):
        #axU[0].plot(sdi[:,0],sdi[:,1],'-.',color=color)
        axU[0].plot(lsdi[:,0]+0.5*lsdi[:,1],lsdi[:,2],'-.',color=color,label="$k_{{a}}^{{o}}={}$".format(kai))
        axU[1].plot(lsdi[:,0]+0.5*lsdi[:,1],lsdi[:,2],'-.',color=color,label="$k_{{a}}^{{o}}={}$".format(kai))

    axU[0].set_xlabel(r'Time',usetex=True,fontsize=18)
    axU[0].set_ylabel(r'$\mathscr{S}_{\mathrm{rad}}(t|\sigma)$',usetex=True,fontsize=18)
    axU[0].set_xscale('log')
    axU[0].set_yscale('log')
    axU[0].set_ylim((1e-3,1.0))
    #axU[0].legend()

    axU[1].set_xlabel(r'Time',usetex=True,fontsize=18)
    axU[1].set_ylabel(r'$\mathscr{S}_{\mathrm{rad}}(t|\sigma)$',usetex=True,fontsize=18)
    axU[1].set_xscale('log')
    axU[1].set_yscale('log')

    figU.tight_layout()

    figU.savefig("./Figures/SurvivalProbability.pdf")

    #---------------------------------------------------------------------------------------------
    # Plot seperation probability and autocorrelation function
    '''
    figS, axS = plt.subplots(1,2,figsize=(10,5))

    #plot_inds = [0,4,9,14,19]
    plot_inds = range(0,len(kao_qr),1)
    for i in plot_inds:
        axS[0].plot(time_qr[i][::coarsening],sep_prob_qr[i],'-',label="$k_{{a}}^{{o}}={}$".format(kao_qr[i]))
        axS[1].plot(time_qr[i][::coarsening],corr_func_qr[i]/n_var_qr[i],'-')

    axS[0].set_xlabel(r'Time',fontsize=18)
    axS[0].set_ylabel(r'$\mathcal{S}_{rev}(t|\star)$',fontsize=18)
    axS[0].legend()

    axS[1].set_xlabel(r'Time',fontsize=18)
    axS[1].set_ylabel(r'$C(\tau)$',fontsize=18)

    figS.tight_layout()
    '''

    plt.show()
