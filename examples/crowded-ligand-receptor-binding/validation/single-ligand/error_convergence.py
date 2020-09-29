import numpy as np
import matplotlib.pyplot as plt
import readdy
from scipy import interpolate
from scipy.integrate import simps
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
    basedir_qr = "./db_quasi_reversible_sims/2020_06_09"

    # Set intrisic rate constants for analysis
    kao_qr = np.logspace(1,4,4)	# values for comparison w/ reversible simulations

    
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
    nstep = None
    ntrajs = [100,1000,2000,4000]


    for i,kaoj in enumerate(kao_qr):
        for j,ntraj in enumerate(ntrajs):

            # Load data from .npz file
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
    datadirs = [basedir_r+'/run_bulk_nL{}_nC{}_kOn{}_kOff1/'.format(nL,nC,kaoi) for kaoi in kao_r]

    #[n_mean_r, n_var_r, surv_data, logsurv_data, event_density_data, nmean_rev] = qan.process_reversible_data_v2(datadirs)

    ntraj_r = [1, 5, 10, 20, 30, 40]
    out_r = []
    for ntraji in ntraj_r:
        out_r.append(qan.process_reversible_data_v2(datadirs,ntraj=ntraji))

    n_mean_r = [x[0] for x in out_r]
    n_var_r = [x[1] for x in out_r]
    surv_data_r = [x[2] for x in out_r]
    logsurv_data_r = [x[3] for x in out_r]
    event_density_r = [x[4] for x in out_r]
    nmean_r = [x[5] for x in out_r]

    ##############################################################################################
    # Calculate expected occupancy for Detailed-balance reversible simulations
    micro_onrates = np.logspace(0.,5.,100)
    micro_offrate = kdo #kdo_r
    n_db = calculate_lma_occupancy(micro_onrates,micro_offrate,V,Vex,Vreact_eff)
    n_var_db = n_db * (1. - n_db)

    ##############################################################################################
    # Calculate error in mean occupancy and variance of occupancy
    err_mean = []
    err_var = []
    for i, (means_qri,vars_qri) in enumerate(zip(n_mean_qr,n_var_qr)):
        err_meani = [np.sqrt((means_qri[j] - n_mean_r[-1][i])**2.)/n_mean_r[-1][i] for j in range(len(means_qri))]
        err_mean.append(err_meani)

        err_vari = [np.sqrt((vars_qri[j] - n_var_r[-1][i])**2.)/n_var_r[-1][i] for j in range(len(vars_qri))]
        err_var.append(err_vari)

    ##############################################################################################
    # Calculate error in integral of survival probability (mfpt of binding)
    err_surv = []
    for i, survi in enumerate(surv_prob_qr):
        ti_r = logsurv_data_r[-1][i][:,0]+0.5*logsurv_data_r[-1][i][:,1]
        si_r = logsurv_data_r[-1][i][:,2]
        f_surv_r_i = simps(si_r,ti_r)

        f_surv_qr_i = t_bind_qr[i]
        err_surv_i = [np.sqrt((f_surv_qr_ij - f_surv_r_i)**2.)/f_surv_r_i for f_surv_qr_ij in f_surv_qr_i]
        err_surv.append(err_surv_i)

    ##############################################################################################
    # Calculate error in mean occupancy and variance of occupancy relative to finest q-rev
    err_mean_qr = []
    err_var_qr = []
    for i, (means_qri,vars_qri) in enumerate(zip(n_mean_qr,n_var_qr)):
        err_meani = [np.sqrt((means_qrij - means_qri[-1])**2.)/means_qri[-1] for means_qrij in means_qri[:-1]]
        err_mean_qr.append(err_meani)

        err_vari = [np.sqrt((vars_qrij - vars_qri[-1])**2.)/vars_qri[-1] for vars_qrij in vars_qri[:-1]]
        err_var_qr.append(err_vari)

    ##############################################################################################
    # Calculate error in integral of survival probability (mfpt of binding) relative to finest q-rev
    err_surv_qr = []
    for i, survi in enumerate(surv_prob_qr):
        ti_r = logsurv_data_r[-1][i][:,0]+0.5*logsurv_data_r[-1][i][:,1]
        si_r = logsurv_data_r[-1][i][:,2]
        f_surv_r_i = simps(si_r,ti_r)

        f_surv_qr_i = t_bind_qr[i]
        err_surv_i = [np.sqrt((f_surv_qr_ij - f_surv_qr_i[-1])**2.)/f_surv_qr_i[-1] for f_surv_qr_ij in f_surv_qr_i[:-1]]
        err_surv_qr.append(err_surv_i)

    ##############################################################################################
    # Calculate error in mean occupancy and variance of occupancy relative to finest reversible sims
    err_mean_r = []
    err_var_r = []
    for i, means_ri in enumerate(n_mean_r):	# loop over ntraj vals for convergence study
        err_meani = []
        for j, means_rij in enumerate(means_ri):	# loop over kao vals
            err_meanij = np.sqrt((means_rij - n_mean_r[-1][j])**2.)/n_mean_r[-1][j]
            err_meani.append(err_meanij)
        err_mean_r.append(err_meani)


    ##############################################################################################
    # Calculate error in integral of survival probability (mfpt of binding) relative to reversible
    err_surv_r = []
    for i, lsurvi in enumerate(logsurv_data_r):
        err_surv_i = []
        for j, lsurvij in enumerate(lsurvi):
            tij_r = lsurvij[:,0]+0.5*lsurvij[:,1]
            sij_r = lsurvij[:,2]
            f_surv_r_ij = simps(sij_r,tij_r)

            tfine_r = logsurv_data_r[-1][j][:,0]+0.5*logsurv_data_r[-1][j][:,1]
            sfine_r = logsurv_data_r[-1][j][:,2]
            f_surv_r_fine = simps(sfine_r,tfine_r)


            err_surv_ij = np.sqrt((f_surv_r_ij - f_surv_r_fine)**2.)/f_surv_r_fine
            err_surv_i.append(err_surv_ij)
    
        err_surv_r.append(err_surv_i)


    ##############################################################################################
    # PLOTTING
    ##############################################################################################
    # Plot error in mean activation and survival prob relative to finest sims for quasi-rev simulations
    figqERR, axqERR = plt.subplots(1,2,figsize=(10,5))
    
    for i,kao_qri in enumerate(kao_qr):
        axqERR[0].plot(ntrajs[:-1],err_mean_qr[i],"-o",label="$k_{{a}}^{{o}} = {}$".format(kao_qri))
        axqERR[1].plot(ntrajs[:-1],err_surv_qr[i],"-o")


    axqERR[0].set_xlabel(r'$N_{traj}$',fontsize=18)
    axqERR[0].set_ylabel(r'Error($\langle n \rangle$)',fontsize=18)
    axqERR[0].legend()

    axqERR[1].set_xlabel(r'$N_{traj}$',fontsize=18)
    axqERR[1].set_ylabel(r'Error($\int S(t)dt$)',fontsize=18)

    axqERR[0].set_xscale('log')
    axqERR[1].set_xscale('log')
    axqERR[0].set_yscale('log')
    axqERR[1].set_yscale('log')

    figqERR.tight_layout()

    figqERR.savefig("./Figures/Error_Mean_Variance.eps")


    #-----------------------------------------------------------------------------------------------
    # Plot error in mean activation and survival prob relative to finest sims for reversible simulations
    figrERR, axrERR = plt.subplots(1,2,figsize=(10,5))
    for i,kao_qri in enumerate(kao_qr):
        axrERR[0].plot(ntraj_r[:-1],[ej[i] for ej in err_mean_r[:-1]],"-o",label="$k_{{a}}^{{o}} = {}$".format(kao_qri))
        axrERR[1].plot(ntraj_r[:-1],[ej[i] for ej in err_surv_r[:-1]],"-o")


    axrERR[0].set_xlabel(r'$N_{traj}$',fontsize=18)
    axrERR[0].set_ylabel(r'Error($\langle n \rangle$)',fontsize=18)
    axrERR[0].legend()

    axrERR[1].set_xlabel(r'$N_{traj}$',fontsize=18)
    axrERR[1].set_ylabel(r'Error($\int S(t)dt$)',fontsize=18)

    axrERR[0].set_xscale('log')
    axrERR[1].set_xscale('log')
    axrERR[0].set_yscale('log')
    axrERR[1].set_yscale('log')

    figrERR.tight_layout()

    figrERR.savefig("./Figures/Error_Mean_Variance.eps")

    #-----------------------------------------------------------------------------------------------
    # Plot error in mean activation and survival prob
    figERR, axERR = plt.subplots(1,2,figsize=(10,5))
    
    for i,kao_qri in enumerate(kao_qr):
        axERR[0].plot(ntrajs,err_mean[i],"-o",label="$k_{{a}}^{{o}} = {}$".format(kao_qri))
        axERR[1].plot(ntrajs,err_surv[i],"-o")

    axERR[0].set_xlabel(r'$N_{traj}$',fontsize=18)
    axERR[0].set_ylabel(r'Error($\langle n \rangle$)',fontsize=18)
    axERR[0].legend()

    axERR[1].set_xlabel(r'$N_{traj}$',fontsize=18)
    axERR[1].set_ylabel(r'Error($\int S(t)dt$)',fontsize=18)

    axERR[0].set_xscale('log')
    axERR[1].set_xscale('log')
    axERR[0].set_yscale('log')
    axERR[1].set_yscale('log')

    figERR.tight_layout()

    figERR.savefig("./Figures/Error_Mean_Variance.eps")



    plt.show()
