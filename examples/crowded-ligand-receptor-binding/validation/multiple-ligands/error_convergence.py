import numpy as np

import matplotlib as mpl
mpl.rcParams['text.usetex'] =True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
 
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
    # Load quasi-reversible simulation data
    basedir_qr = "./db_quasi_reversible_sims/2020_06_10_mark2"

    # Set intrisic rate constants for analysis
    kao_qr = np.logspace(0,4,5)	# values for comparison w/ reversible simulations

    
    time_qr = [[] for i in kao_qr]
    surv_prob_qr = [[] for i in kao_qr]
    n_mean_qr = [[] for i in kao_qr]
    t_bind_qr = [[] for i in kao_qr]

    ntraj_finest = 94000
    true_means_qr = []
    true_time_qr = []
    true_surv_prob_qr = []
    true_t_bind_qr = []

    target_prob = 0.0001
    nstep = None
    ntrajs = [100,1000,10000,30000]

    for i,kaoj in enumerate(kao_qr):
        for j,ntraj in enumerate(ntrajs):

            # Load data from .npz file
            outfilenames = basedir_qr+"/analyzed_data/analysis_out_nC_{}_kao_{}_kdo_{}_targetprob_{}_nsteps_{}_ntraj_{}_*.npz".format(nC,kaoj,kdo,target_prob,nstep,ntraj)

            time_qrij = []
            surv_prob_qrij = []
            n_mean_qrij = []
            t_bind_qrij = []

            for outfilename in glob.glob(outfilenames):
                out = np.load(outfilename,allow_pickle=True)
                out = {key:out[key].item() for key in out}
                out = out['arr_0']

                time_qrij.append(out['time'])
                surv_prob_qrij.append(out['surv_prob'])
                n_mean_qrij.append(out['n_mean'])
                t_bind_qrij.append(out['mfpt_bind'])
   

            time_qr[i].append(time_qrij)
            surv_prob_qr[i].append(surv_prob_qrij)
            n_mean_qr[i].append(n_mean_qrij)
            t_bind_qr[i].append(t_bind_qrij)
   
        # Get true mean from finest simulation set
        outfilename = basedir_qr+"/analyzed_data/analysis_out_nC_{}_kao_{}_kdo_{}_targetprob_{}_nsteps_{}_ntraj_{}.npz".format(nC,kaoj,kdo,target_prob,nstep,ntraj_finest)
        out = np.load(outfilename,allow_pickle=True)
        out = {key:out[key].item() for key in out}
        out = out['arr_0']

        true_time_qr.append(out['time'])
        true_surv_prob_qr.append(out['surv_prob'])
        true_means_qr.append(out['n_mean'])
        true_t_bind_qr.append(out['mfpt_bind'])


    ##############################################################################################
    # Process reversible data

    kao_r_strings = ["1E+0","1E+1","1E+2","1E+3","1E+4"]
    kao_r = [1E+0,1E+1,1E+2,1E+3,1E+4]
    basedir_r = './full_reversible_sims/weights_L05_S05'
    datadirs = [basedir_r+'/run_bulk_nL{}_nC{}_kOn{}_kOff1E-0/'.format(nL,nC,kaoi) for kaoi in kao_r_strings]

    #[n_mean_r, n_var_r, surv_data, logsurv_data, event_density_data, nmean_rev] = qan.process_reversible_data_v2(datadirs)

    traj_length = 1e7		# number of timesteps for each trajectory

    #ntraj_r = [1, 5, 10, 20, 40, 80, 160]
    ntraj_r = [1, 10, 30, 80]
    number_trajs = [10, 10, 5, 3] # number of traj sets in ensemble for each ntraj
    ntraj_tuples = [[(i*ntraji, (i+1)*ntraji) for i in range(number_trajsi)] for ntraji,number_trajsi in zip(ntraj_r,number_trajs)]

    true_ntraj_r = 240

    n_mean_r = [ [ [] for ntraji in ntraj_r] for kaoi in kao_r]
    logsurv_data_r = [ [ [] for ntraji in ntraj_r] for kaoi in kao_r]

    for i,(ntraji,tuplesi) in enumerate(zip(ntraj_r,ntraj_tuples)):

        for tuplesij in tuplesi:
            out_rij = qan.process_reversible_data_v2(datadirs,ntraj=tuplesij) 	# output for ntraj_i and tuple_set_j for all kao vals
            
            for k in range(len(kao_r)):
                n_mean_r[k][i].append(out_rij[0][k])
                logsurv_data_r[k][i].append(out_rij[3][k])


    true_out_r = qan.process_reversible_data_v2(datadirs,ntraj=true_ntraj_r)
    true_n_mean_r = true_out_r[0]
    true_logsurv_data_r = true_out_r[3]

    ##############################################################################################
    # Calculate expected occupancy for Detailed-balance reversible simulations
    micro_onrates = np.logspace(0.,5.,100)
    micro_offrate = kdo #kdo_r
    n_db = calculate_lma_occupancy(micro_onrates,micro_offrate,V,Vex,Vreact_eff)
    n_var_db = n_db * (1. - n_db)
    
    ##############################################################################################
    # Calculate error in mean occupancy and variance of occupancy for qr sims relative to finest r sim
    err_mean = []
    for i, means_qri in enumerate(n_mean_qr):
        err_mean_qri = []
        for j, means_qrij in enumerate(means_qri):

            errs_mean_ij = np.array([np.sqrt((means_qrijk - true_n_mean_r[i])**2.)/true_n_mean_r[i] for means_qrijk in means_qrij])
            mean_error_of_means_ij = np.mean(errs_mean_ij)

            err_mean_qri.append(mean_error_of_means_ij)
        err_mean.append(err_mean_qri)

    
    ##############################################################################################
    # Calculate error in integral of survival probability (mfpt of binding) for qr sims relative to finest r sim
    err_surv = []
    for i, survi in enumerate(surv_prob_qr):

        ti_r = true_logsurv_data_r[i][:,0]+0.5*true_logsurv_data_r[i][:,1]
        si_r = true_logsurv_data_r[i][:,2]
        f_surv_r_i = simps(si_r,ti_r)

        err_surv_i = []
        for j, survij in enumerate(survi):
            f_surv_qr_ij = t_bind_qr[i][j]

            err_surv_ij = [np.sqrt((f_surv_qr_ijk - f_surv_r_i)**2.)/f_surv_r_i for f_surv_qr_ijk in f_surv_qr_ij]
            mean_error_of_integrated_surv_ij = np.mean(err_surv_ij)
            
            err_surv_i.append(mean_error_of_integrated_surv_ij)

        err_surv.append(err_surv_i)
    
    ##############################################################################################
    # Calculate error in mean occupancy relative to finest q-rev
    err_mean_qr = []
    for i, means_qri in enumerate(n_mean_qr):
        err_mean_qri = []
        for j, means_qrij in enumerate(means_qri):

            errs_mean_ij = np.array([np.sqrt((means_qrijk - true_means_qr[i])**2.)/true_means_qr[i] for means_qrijk in means_qrij])
            mean_error_of_means_ij = np.mean(errs_mean_ij)

            err_mean_qri.append(mean_error_of_means_ij)

        err_mean_qr.append(err_mean_qri)
    
    ##############################################################################################
    # Calculate error in integral of survival probability (mfpt of binding) relative to finest q-rev
    err_surv_qr = []
    for i, survi in enumerate(surv_prob_qr):
        err_surv_i = []
        f_true_qr_i = true_t_bind_qr[i]
        for j, survij in enumerate(survi):
            f_surv_qr_ij = t_bind_qr[i][j]

            err_surv_ij = [np.sqrt((f_surv_qr_ijk - f_true_qr_i)**2.)/f_true_qr_i for f_surv_qr_ijk in f_surv_qr_ij]
            mean_error_of_integrated_surv_ij = np.mean(err_surv_ij)
            
            err_surv_i.append(mean_error_of_integrated_surv_ij)

        err_surv_qr.append(err_surv_i)
    
    ##############################################################################################
    # Calculate error in mean occupancy and variance of occupancy relative to finest reversible sims
    err_mean_r = []
    for i, means_ri in enumerate(n_mean_r):	# loop over kao vals for convergence study
        err_mean_ri = []
        for j, means_rij in enumerate(means_ri):	# loop over ntraj vals
            err_mean_ij = np.array([np.sqrt((means_rijk - true_n_mean_r[i])**2.)/true_n_mean_r[i] for means_rijk in means_rij])

            mean_error_of_means_ij = np.mean(err_mean_ij)
            err_mean_ri.append(mean_error_of_means_ij)

        err_mean_r.append(err_mean_ri)

    ##############################################################################################
    # Calculate error in integral of survival probability (mfpt of binding) relative to reversible
    err_surv_r = []
    for i, lsurvi in enumerate(logsurv_data_r):	#loop over kao vals
        err_surv_i = []
        true_t_r = true_logsurv_data_r[i][:,0]+0.5*true_logsurv_data_r[i][:,1]
        true_s_r = true_logsurv_data_r[i][:,2]
        true_f_surv_r = simps(true_s_r,true_t_r)

        for j, lsurvij in enumerate(lsurvi): # loop over ntraj vals
            err_surv_ij = []
            for lsurvijk in lsurvij: 
                tijk_r = lsurvijk[:,0]+0.5*lsurvijk[:,1]
                sijk_r = lsurvijk[:,2]
                f_surv_r_ijk = simps(sijk_r,tijk_r)

                err_surv_ij.append(np.sqrt((f_surv_r_ijk - true_f_surv_r)**2.)/true_f_surv_r)

            err_surv_i.append(np.mean(err_surv_ij))
    
        err_surv_r.append(err_surv_i)

    ##############################################################################################
    # PLOTTING
    ##############################################################################################
    # Plot error in mean activation and survival prob relative to finest sims for quasi-rev simulations
    figqERR, axqERR = plt.subplots(1,2,figsize=(10,5))
    
    for i,kao_qri in enumerate(kao_qr):
        axqERR[0].plot(ntrajs,err_mean_qr[i],"-o",label="$k_{{a}}^{{o}} = {}$".format(kao_qri))
        axqERR[1].plot(ntrajs,err_surv_qr[i],"-o")


    axqERR[0].set_xlabel(r'$N_{\text{traj}}$',fontsize=18)
    axqERR[0].set_ylabel(r'$E_{\langle n \rangle}^{\text{qr}}$',fontsize=18)
    axqERR[0].legend()

    axqERR[1].set_xlabel(r'$N_{\text{traj}}$',fontsize=18)
    axqERR[1].set_ylabel(r'$E_{I_{S}}^{\text{qr}}$',fontsize=18)

    axqERR[0].set_xscale('log')
    axqERR[1].set_xscale('log')
    axqERR[0].set_yscale('log')
    axqERR[1].set_yscale('log')

    figqERR.tight_layout()

    figqERR.savefig("./Figures/Error_Mean_Variance_qr.pdf")

    
    #-----------------------------------------------------------------------------------------------
    # Plot error in mean activation and survival prob relative to finest sims for reversible simulations
    figrERR, axrERR = plt.subplots(1,2,figsize=(10,5))
    for i,kao_qri in enumerate(kao_qr):
        axrERR[0].plot([ni*traj_length for ni in ntraj_r],err_mean_r[i],"-o",label="$k_{{a}}^{{o}} = {}$".format(kao_qri))
        axrERR[1].plot([ni*traj_length for ni in ntraj_r],err_surv_r[i],"-o")

    axrERR[0].set_xlabel(r'$N_{\text{ts}}$',fontsize=18)
    axrERR[0].set_ylabel(r'$E_{\langle n \rangle}^{\text{r}}$',fontsize=18)
    axrERR[0].legend()

    axrERR[1].set_xlabel(r'$N_{\text{ts}}$',fontsize=18)
    axrERR[1].set_ylabel(r'$E_{I_{S}}^{\text{r}}$',fontsize=18)


    axrERR[0].set_xscale('log')
    axrERR[1].set_xscale('log')
    axrERR[0].set_yscale('log')
    axrERR[1].set_yscale('log')

    figrERR.tight_layout()

    figrERR.savefig("./Figures/Error_Mean_Variance_r.pdf")
    
    #-----------------------------------------------------------------------------------------------
    # Plot error in mean activation and survival prob
    figERR, axERR = plt.subplots(1,2,figsize=(10,5))
    
    for i,kao_qri in enumerate(kao_qr):
        axERR[0].plot(ntrajs,err_mean[i],"-o",label="$k_{{a}}^{{o}} = {}$".format(kao_qri))
        axERR[1].plot(ntrajs,err_surv[i],"-o")

    axERR[0].set_xlabel(r'$N_{\text{traj}}$',fontsize=18)
    axERR[0].set_ylabel(r'$E_{\langle n \rangle}^{\text{qr-r}}$',fontsize=18)
    axERR[0].legend()

    axERR[1].set_xlabel(r'$N_{\text{traj}}$',fontsize=18)
    axERR[1].set_ylabel(r'$E_{I_{S}}^{\text{qr-r}}$',fontsize=18)


    axERR[0].set_xscale('log')
    axERR[1].set_xscale('log')
    axERR[0].set_yscale('log')
    axERR[1].set_yscale('log')

    figERR.tight_layout()

    figERR.savefig("./Figures/Error_Mean_Variance_qr_r.pdf")


    
    plt.show()
