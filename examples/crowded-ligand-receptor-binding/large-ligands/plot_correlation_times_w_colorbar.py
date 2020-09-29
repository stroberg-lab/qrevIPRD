import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

params = {'text.usetex' : True, 'text.latex.preamble' : [ r'\usepackage{mathrsfs}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)

from joblib import Parallel, delayed

import qrevIPRD.analysis as qan

import extrapolate_surv_prob_nonlinear as extrapNL

#----------------------------------------------------------------------------
def calc_effective_rates(tau_n,mean_n):
    kon = mean_n/tau_n
    koff = (1.-mean_n)/tau_n
    return (kon,koff)

#----------------------------------------------------------------------------
def calc_correlation_time(time,surv_prob,mfpt_unbind,kdo,coarsening=10):

    dissoc_prob = 1./(kdo*mfpt_unbind)

    # Calculate mean first passage time for a binding to occur
    mfpt_bind = qan.trap_integrate(surv_prob,time)

    # Calculate point statistics for binary switching process
    n_mean = mfpt_unbind/(mfpt_bind + mfpt_unbind)
    n_var = n_mean * (1. - n_mean)

    # Calculate separation probability for bound pair
    sep_prob = qan.calc_separation_prob(surv_prob[::coarsening],time[::coarsening],kdo,dissoc_prob)

    # Calculate correlation function from separation probability
    pstar_star = 1. - sep_prob
    p0 = n_mean
    n_varC = p0 * (pstar_star[0] - p0)
    corr_func = p0 * (pstar_star - p0)

    # Calculate correlation time directly from definition in Kaizu 2014 supp. equation S8
    tau_n = 1./n_var * qan.trap_integrate(corr_func,time[::coarsening])

    return tau_n
#----------------------------------------------------------------------------
if __name__=="__main__":

    nL = 10
    nCs = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    traj_number = 0


    # Set intrisic reaction rates
    kao = [0.3e+1,1.0e+1,0.3e+2,1.0e+2,0.3e+3,1.0e+3,0.3e+4]
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
    preb = [[] for i in kao]


    for i in range(len(kao)):
        for j in range(len(nCs)):

            # Load data from .npz file
            outfilename = "./analyzed_data/analysis_out_nC_{}_kao_{}_kdo_{}.npz".format(nCs[j],kao[i],kdo)
            out = np.load(outfilename,allow_pickle=True)
            out = {key:out[key].item() for key in out}
            out = out['arr_0']

            t_unbind[i].append(out['mfpt_unbind'])
            t_bind[i].append(out['mfpt_bind'])
            time[i].append(out['time'])
            surv_prob[i].append(out['surv_prob'])
            extrap_surv[i].append(out['extrap_surv'])
            sep_prob[i].append(out['sep_prob'])
            corr_func[i].append(out['corr_func'])
            tau_n[i].append(out['tau_n'])
            n_mean[i].append(out['n_mean'])
            n_var[i].append(out['n_var'])
            coarsening[i].append(out['coarsening'])
   
           

    ##############################################################################################
    # Calculate volume fraction of crowding
    sigmaC = 1.
    sigmaL = 1.
    sigmaR = 1.
    vC = 1./6.*np.pi*pow(sigmaC,3.)
    vL = 1./6.*np.pi*pow(sigmaL,3.)
    vR = 1./6.*np.pi*pow(sigmaR,3.)
    nR = 1.
    V = 5.*5.*5.
    phi = [(nCi*vC + nL*vL + nR*vR)/V for nCi in nCs]
    c = [nL/((1.-phii)*V) for phii in phi]


    ##############################################################################################
    # Calculate fits and correlation times for different parameter sets

    nonlinear_fit = [[] for i in kao]
    nonlinear_extrap_surv = [[] for i in kao]
    nonlinear_extrap_time = [[] for i in kao]
    target_surv_prob = 1e-4
    coarsening = 10

    #------------------------------------------
    def parfunc(i,j):

        print("Processing data for kao = {}, nC = {}".format(kao[i],nCs[j]))

        timeij = time[i][j]
        survij = surv_prob[i][j]
        extrij = extrap_surv[i][j]
 
        popt_ij, extrap_time_ij, extrap_surv_ij = extrapNL.fit_and_extrapolate_surv_prob(timeij,survij,extrij,target_surv_prob)
        nonlinear_fit[i].append(popt_ij)
        nonlinear_extrap_time[i].append(extrap_time_ij)
        nonlinear_extrap_surv[i].append(extrap_surv_ij)
 
        # Build new survival prob w/ new extrapolation
        if survij.shape[0] > extrij.shape[0]: 
            surv_dataij = survij[0:-extrij.shape[0]]
            time_dataij = timeij[0:-extrij.shape[0]]
 
            surv_new = np.hstack((surv_dataij,extrap_surv_ij))
            time_new = np.hstack((time_dataij,extrap_time_ij))
        else:
            surv_new = survij
            time_new = timeij
 
        ## Calculate correltion time for extrapolated survival data
        return calc_correlation_time(time_new,surv_new,t_unbind[i][j],kdo,coarsening)
    #------------------------------------------

    nproc = 4
    tau_out = Parallel(n_jobs=nproc)(delayed(parfunc)(i,j) for i in range(len(kao)) for j in range(len(nCs)))

    for i in range(len(kao)):
        for j in range(len(nCs)):
            tau_n[i][j] = tau_out[i*len(nCs) + j]
 
 
    ##############################################################################################
    # PLOTTING
    ##############################################################################################


    #---------------------------------------------------------------------------------------------
    # Plot correlation time and relative error
    #---------------------------------------------------------------------------------------------
    figTn, axTn = plt.subplots(1,2,figsize=(10,4))

    myCmap = plt.get_cmap('cool')
    cNorm = colors.LogNorm(vmin=kao[0],vmax=kao[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=myCmap)


    for i,kaoi in enumerate(kao):
        colorVal = scalarMap.to_rgba(kaoi)
        p = axTn[0].plot(phi,tau_n[i]/tau_n[i][0],'s-',color=colorVal,label=r"$k_{{a}}^{{o}}={}$".format(kaoi))
        rel_err = [2.*tau_nij/n_varij for tau_nij,n_varij in zip(tau_n[i],n_var[i])]
        axTn[1].plot(phi,rel_err,'s-',color=colorVal,label=r"$k_{{a}}^{{o}}={}$".format(kaoi))

    axTn[0].set_xlabel(r'$\phi$',fontsize=18)
    axTn[0].set_ylabel(r'$\tau_{n}/\tau_{n}^{\text{o}}$',fontsize=18)
    #axTn[0].set_yscale('log')

    #axTn[0].legend()

    axTn[1].set_xlabel(r'$\phi$',fontsize=18)
    axTn[1].set_ylabel(r'$\frac{2\tau_n}{\sigma_n^2}$',fontsize=18)
    axTn[1].set_yscale('log')

    #figTn.tight_layout()

    figTn.subplots_adjust(right=0.8,bottom=0.2,wspace=0.45,hspace=0.4)
    cbar_ax = figTn.add_axes([0.85, 0.2, 0.02, 0.65])	#[left, bottom, width, height]
    cbar = figTn.colorbar(scalarMap, cax=cbar_ax)
    cbar.set_label(r'$k_{\text{a}}^{\text{o}}$',fontsize=18)

    figTn.savefig("./Figures/correlation_times_w_colorbar.pdf")

    plt.show()
