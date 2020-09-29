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
    nL = 2
    nC = 1
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
    # Process quasi-reversible simulation data
    traj_number = 0


    basedir_q = "./db_quasi_reversible_sims/module_weights_L05_S05/boxsize_5_5_5/run_bulk_nL{}_nC{}/trajectory_{}".format(nL,nC,traj_number)
    #basedir_q = "./db_quasi_reversible_sims/test_low_memory/boxsize_5_5_5/run_bulk_nL{}_nC{}/trajectory_{}".format(nL,nC,traj_number)

    dissoc_data_file = basedir_q + "/accepted_dissociation_moves.txt" 
    unbound_data_files_template = basedir_q + "/unbound_reaction_event_density_fine_output_nL_{}_*1000001_*.npy".format(nL) 
    #unbound_data_files_template = basedir_q + "/unbound_simulations_fine_output/unbound_reaction_event_density_nL_{}_*.npy".format(nL) 

    print(unbound_data_files_template)
    # Load reaction probability data for each timepoint for each trajectory
    unbound_data_files = []
    for datai in glob.glob(unbound_data_files_template):
        unbound_data_files.append(datai)

    # Set intrisic rate constants for analysis
    kao_qr = np.logspace(0,4,5)	# values for comparison w/ reversible simulations
    kao_q = np.logspace(0,4,20)	# additional values for plotting

    dt = 1E-4

    num_cores = 4
    def parfunc(kaoi):
        return qan.process_quasireversible_simulations(kaoi,kdo,dt,dissoc_data_file,unbound_data_files)

    out_q = Parallel(n_jobs=num_cores)(delayed(parfunc)(kaoi) for kaoi in kao_q)

    n_mean = [xi[6] for xi in out_q]
    n_var = [xi[7] for xi in out_q]
    time = [xi[1] for xi in out_q]
    surv_prob = [xi[2] for xi in out_q]
    sep_prob = [xi[3] for xi in out_q]
    corr_func = [xi[4] for xi in out_q]
    coarsening = [xi[-1] for xi in out_q][0]

    out_qr = Parallel(n_jobs=num_cores)(delayed(parfunc)(kaoi) for kaoi in kao_qr)

    n_mean_qr = [xi[6] for xi in out_qr]
    n_var_qr = [xi[7] for xi in out_qr]
    kd = [xi[0] for xi in out_qr]
    time_qr = [xi[1] for xi in out_qr]
    surv_prob_qr = [xi[2] for xi in out_qr]
    sep_prob_qr = [xi[3] for xi in out_qr]
    corr_func_qr = [xi[4] for xi in out_qr]
    coarsening_qr = [xi[-1] for xi in out_qr][0]

    print(kd)
    ##############################################################################################
    # Process reversible data

    #kao_r = [10,100,1000,10000]
    #basedir_r = './full_reversible_sims/weights_L1_S0'
    #datadirs = [basedir_r+'/run_bulk_nL{}_nC{}_kOn{:.0f}_kOff1/'.format(nL,nC,kaoi) for kaoi in kao_r]

    #[n_mean_r, n_var_r, surv_data, logsurv_data, event_density_data, nmean_rev] = qan.process_reversible_data(datadirs)

    kao_rw05_strings = ["1E+0","1E+1","1E+2","1E+3","1E+4"]
    kao_rw05 = [1E+0,1E+1,1E+2,1E+3,1E+4]
    basedir_r = './full_reversible_sims/weights_L05_S05'
    datadirs = [basedir_r+'/run_bulk_nL{}_nC{}_kOn{}_kOff1E-0/'.format(nL,nC,kaoi) for kaoi in kao_rw05_strings]

    [n_mean_rw05, n_var_rw05, surv_dataw05, logsurv_dataw05, event_density_dataw05, nmean_revw05] = qan.process_reversible_data_v2(datadirs)


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

    axPS[0].plot(kao_q,n_mean,"-o",label="Quasi-Reversible")
    axPS[1].plot(kao_q,n_var,"-o",label="Quasi-Reversible")

    #axPS[0].plot(kao_r,n_mean_r,'s',label="Reversible")
    #axPS[1].plot(kao_r,n_var_r,'s',label="Reversible")

    axPS[0].plot(kao_rw05,n_mean_rw05,'s',label="Reversible, wL=0.5")
    axPS[1].plot(kao_rw05,n_var_rw05,'s',label="Reversible, wL=0.5")

    #axPS[0].plot(micro_onrates,n_db,'--',label="Expected DB")
    #axPS[1].plot(micro_onrates,n_var_db,'--',label="Expected DB")

    #axPS[0].plot(kao_r,nmean_rev,'>')

    axPS[0].set_xlabel(r'$k_{a}^{o}$',fontsize=18)
    axPS[0].set_ylabel(r'$\langle n \rangle$',fontsize=18)
    axPS[0].legend()

    axPS[1].set_xlabel(r'$k_{a}^{o}$',fontsize=18)
    axPS[1].set_ylabel(r'$\sigma_{n}^2$',fontsize=18)

    axPS[0].set_xscale('log')
    axPS[1].set_xscale('log')

    figPS.tight_layout()

    figPS.savefig("./Figures/Mean_Variance.eps")

    #---------------------------------------------------------------------------------------------
    # Plot survival probability
    figU, axU = plt.subplots(1,2,figsize=(10,5))

    #plot_inds = [0,5,10,15,19] 
    plot_inds = range(0,len(kao_qr),1)
    colors = []
    for i in plot_inds:
        p = axU[0].plot(time_qr[i],surv_prob_qr[i],'-',label="$k_{{a}}^{{o}}={}$".format(kao_qr[i]))
        colors.append(p[0].get_color())
        axU[1].plot(time_qr[i],surv_prob_qr[i],'-',color=colors[-1])

    #for (sdi,lsdi,kai,color) in zip(surv_data,logsurv_data,kao_r,colors):
    #    axU[0].plot(sdi[:,0],sdi[:,1],'-.',color=color)
        #axU[1].plot(sdi[:,0],sdi[:,1],'--',color=color,label="$k_{{a}}^{{o}}={}$".format(kai))
    #    axU[1].plot(lsdi[:,0],lsdi[:,1],'-.',label="$k_{{a}}^{{o}}={}$".format(kai))

    for (sdi,lsdi,kai,color) in zip(surv_dataw05,logsurv_dataw05,kao_rw05,colors):
        axU[0].plot(sdi[:,0]+0.5*sdi[:,1],sdi[:,2],'--',color=color)
        #axU[1].plot(sdi[:,0],sdi[:,1],'--',color=color,label="$k_{{a}}^{{o}}={}$".format(kai))
        axU[1].plot(lsdi[:,0]+0.5*lsdi[:,1],lsdi[:,2],'--',label="$k_{{a}}^{{o}}={}, wL=0.5$".format(kai))

    axU[0].set_xlabel(r'Time',fontsize=18)
    axU[0].set_ylabel(r'$S_{rad}(t|\sigma)$',fontsize=18)
    axU[0].legend()

    axU[1].set_xlabel(r'Time',fontsize=18)
    axU[1].set_ylabel(r'$S_{rad}(t|\sigma)$',fontsize=18)
    axU[1].set_xscale('log')
    #axU[1].set_yscale('log')

    figU.tight_layout()

    figU.savefig("./Figures/SurvivalProbability.eps")

    #---------------------------------------------------------------------------------------------
    # Plot seperation probability and autocorrelation function
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


    plt.show()
