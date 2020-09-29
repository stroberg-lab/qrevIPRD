import numpy as np
import matplotlib.pyplot as plt

params = {'text.usetex': True, 'text.latex.preamble' : [ r'\usepackage{mathrsfs}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)

import readdy
from scipy import interpolate
from scipy.integrate import trapz, cumtrapz, quadrature
from scipy.optimize import curve_fit

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

############################################################################################
#----------------------------------------------------------------------------
# Define model for fitting
def exp_model(t,theta0,theta1):
    return theta0 * np.exp(theta1 * t)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# Fit long-time limit of survival data to model
def fit_survival_prob_nonlinear(time_data,surv_data):
    ##### Extrapolate survival prob w/ nonlinear regression model #####

    # Set initial guess for parameters based on log-transform linear regression
    lin_fit = np.polyfit(time_data,np.log(surv_data),1)

    exponent_estimate = lin_fit[0]
    prefactor_estimate = pow(10.,lin_fit[1])

    # Fit model to data
    p0 = [prefactor_estimate, exponent_estimate]
    popt, pcov = curve_fit(exp_model,time_data,surv_data,p0)

    return (popt, pcov)


#----------------------------------------------------------------------------
def fit_and_extrapolate_surv_prob(tdata,sdata,target_surv_prob=1e-4):

    popt_ij, pcov_ij = fit_survival_prob_nonlinear(tdata,sdata)

    # Extrapolate survival data using nonlinear fit

    final_time = tdata[-1] + (np.log(target_surv_prob) - np.log(sdata[-1]))/popt_ij[1]
    #print("Final time after extrapolation = {}".format(final_time))

    dt = tdata[1]-tdata[0]
    extrap_time_ij = np.linspace(tdata[-1]+dt,final_time,10000)
    extrap_surv_ij = exp_model(extrap_time_ij,popt_ij[0],popt_ij[1])

    return popt_ij, extrap_time_ij, extrap_surv_ij

############################################################################################

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
    traj_number = 0

    basedir_qr = "."

    # Set intrisic rate constants for analysis
    kao_qr = np.logspace(2,3,2)	# values for comparison w/ reversible simulations


    time_qr = [[] for i in kao_qr]
    surv_prob_qr = [[] for i in kao_qr]

    target_prob = 0.0001
    nsteps = [None]
    ntraj = 64000
    for i,nstep in enumerate(nsteps):
        for kaoj in kao_qr:

            # Load data from .npz file
            outfilename = basedir_qr+"/analyzed_data/analysis_out_nC_{}_kao_{}_kdo_{}_targetprob_{}_nsteps_{}_ntraj_{}.npz".format(nC,kaoj,kdo,target_prob,nstep,ntraj)

            out = np.load(outfilename,allow_pickle=True)
            out = {key:out[key].item() for key in out}
            out = out['arr_0']

            survij = out['surv_prob']
            timeij = out['time']
            extrij = out['extrap_surv']
 
            # Remove previous extrapolated data, if present
            if survij.shape[0]>extrij.shape[0]:
                surv_prob_qr[i].append(survij[0:-extrij.shape[0]])
                time_qr[i].append(timeij[0:-extrij.shape[0]])
            else:
                time_qr[i].append(timeij)
                surv_prob_qr[i].append(survij)


    ##############################################################################################

    ##############################################################################################
    # Calculate fits for different parameter sets

    nonlinear_fit = [[] for i in kao_qr]
    nonlinear_extrap_surv = [[] for i in kao_qr]
    nonlinear_extrap_time = [[] for i in kao_qr]
    delta = [[] for i in kao_qr]
    target_surv_prob = 1e-4
    data_cutoff = 10000
    npoints =  1000
    for i,nstep in enumerate(nsteps):
        for j,kaoj in enumerate(kao_qr):

            print(i,j,len(time_qr),len(time_qr[i]))
            timeij = time_qr[i][j]		# time data points excluding extrapolated points
            survij = surv_prob_qr[i][j]		# surv prob points excluding extrapolated points
 
            timeij_fit = timeij[data_cutoff-npoints:data_cutoff]          
            survij_fit = survij[data_cutoff-npoints:data_cutoff]          
 
            popt_ij, extrap_time_ij, extrap_surv_ij = fit_and_extrapolate_surv_prob(timeij_fit,survij_fit,target_surv_prob)
            nonlinear_fit[i].append(popt_ij)

            # Evaluate fit model at same time points as data for comparison
            time_eval = timeij[data_cutoff:]
            surv_eval = exp_model(time_eval,popt_ij[0],popt_ij[1]) 

            nonlinear_extrap_time[i].append(time_eval)
            nonlinear_extrap_surv[i].append(surv_eval)

            delta[i].append(surv_eval - survij[data_cutoff:])
            #delta[i].append(abs(surv_eval - survij[data_cutoff:])/survij[data_cutoff:])

    ##############################################################################################


    ##############################################################################################
    # PLOTTING
    ##############################################################################################

    #---------------------------------------------------------------------------------------------
    # Plot survival probability
    figU, axU = plt.subplots(1,2,figsize=(10,5))

    # Quasi-revesible
    colors = []
    linestyles = ['-']
    thinning = 1
    for i,nstep in enumerate(nsteps):
        for j in range(len(kao_qr)):

            survi = surv_prob_qr[i][j]
            timei = time_qr[i][j]

            nonlin_extr_survi = nonlinear_extrap_surv[i][j]
            nonlin_extr_timei = nonlinear_extrap_time[i][j]

            p = axU[0].plot(timei,survi,linestyle=linestyles[i],label="$k_{{a}}^{{o}}={}$".format(kao_qr[j]))
            colors.append(p[0].get_color())

            axU[0].plot(nonlin_extr_timei,nonlin_extr_survi,'--',color='black')
            axU[0].plot(nonlin_extr_timei[0],nonlin_extr_survi[0],'*',color='black')

            axU[1].plot(nonlin_extr_timei, delta[i][j],linewidth=2,color=colors[-1])


    axU[0].set_xlabel(r'Time',usetex=True,fontsize=18)
    axU[0].set_ylabel(r'$\mathscr{S}_{\mathrm{rad}}(t|\sigma)$',usetex=True,fontsize=18)
    axU[0].set_xscale('log')
    axU[0].set_yscale('log')
    axU[0].set_ylim((1e-4,1.0))
    axU[0].legend()

    axU[1].set_xlabel(r'Time',usetex=True,fontsize=18)
    axU[1].set_ylabel(r'$\mathscr{S}_{\mathrm{rad}}(t|\sigma)^{\text{extrap}} - \mathscr{S}_{\mathrm{rad}}(t|\sigma)$',usetex=True,fontsize=18)
    #axU[1].set_xscale('log')
    #axU[1].set_yscale('log')

    figU.tight_layout()

    figU.savefig("./Figures/ExtrapolationValidation.pdf")



    plt.show()
