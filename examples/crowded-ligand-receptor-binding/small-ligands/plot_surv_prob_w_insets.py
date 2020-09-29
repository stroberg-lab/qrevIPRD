import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import ticker

params = {"text.usetex": True,'text.latex.preamble' : [ r'\usepackage{mathrsfs}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)

import glob
from joblib import Parallel, delayed

import qrevIPRD.analysis as qan
import LSCmodel
import time as pytime

import extrapolate_surv_prob_nonlinear as extrapNL

#----------------------------------------------------------------------------
def calc_effective_rates(tau_n,mean_n):
    kon = mean_n/tau_n
    koff = (1.-mean_n)/tau_n
    return (kon,koff)

#----------------------------------------------------------------------------
if __name__=="__main__":

    nL = 10
    nCs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    #nCs = [0, 20, 40, 60, 80]
    traj_number = 0


    # Set intrisic reaction rates
    kao = [0.3e+1,1.0e+1,0.3e+2,1.0e+2,0.3e+3,1.0e+3,0.3e+4]#,1.0e+4]
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
   
            # Long-time limit of association rate
            ind = out['extrap_surv'].shape[0]
            dlogSij = np.diff(np.log(out['surv_prob'][-ind:-ind+2]))
            dtij = np.diff(out['time'][-ind:-ind+2])
            kon_inf[i].append(-dlogSij/dtij)

            # Escape/Rebinding probabilities
            extrap_time = out['time'][-out['extrap_surv'].shape[0]:]
            log_extrap = np.log(out['extrap_surv'])
            pfit = np.polyfit(extrap_time,log_extrap,1)
            intercept = pfit[1]
            pesc = np.exp(intercept)
            preb[i].append(1. - pesc)
           

    ##############################################################################################
    # Calculate volume fraction of crowding
    sigmaC = 1.
    sigmaL = 0.2
    sigmaR = 1.
    vC = 1./6.*np.pi*pow(sigmaC,3.)
    vL = 1./6.*np.pi*pow(sigmaL,3.)
    vR = 1./6.*np.pi*pow(sigmaR,3.)
    nR = 1.
    V = 5.*5.*5.
    phi = [(nCi*vC + nL*vL + nR*vR)/V for nCi in nCs]
    c = [nL/((1.-phii)*V) for phii in phi]

    ##############################################################################################
    # Calculate fits for different parameter sets

    nonlinear_fit = [[] for i in kao]
    nonlinear_extrap_surv = [[] for i in kao]
    nonlinear_extrap_time = [[] for i in kao]
    target_surv_prob = 1e-4

    for i in range(len(kao)):
        for j in range(len(nCs)):

            timeij = time[i][j]
            survij = surv_prob[i][j]
            extrij = extrap_surv[i][j]

            popt_ij, extrap_time_ij, extrap_surv_ij = extrapNL.fit_and_extrapolate_surv_prob(timeij,survij,extrij,target_surv_prob)
            nonlinear_fit[i].append(popt_ij)
            nonlinear_extrap_time[i].append(extrap_time_ij)
            nonlinear_extrap_surv[i].append(extrap_surv_ij)


    ##############################################################################################
    # PLOTTING
    ##############################################################################################
    #---------------------------------------------------------------------------------------------
    # Plot survival probability with insets
    #---------------------------------------------------------------------------------------------

    #kao_inds = [3,4,5]
    kao_inds = [3,5,6]
    figS, axS = plt.subplots(1,len(kao_inds),figsize=(5*len(kao_inds),4))

    # Create an inset in the lower right corner (loc=4) with borderpad=1, i.e.
    # 10 points padding (as 10pt is the default fontsize) to the parent axes
    axins = []
    for axi in axS:
        axins.append(inset_axes(axi, width="100%", height="100%", 
                                bbox_to_anchor=(.15, .05, .4, .4),
                                bbox_transform=axi.transAxes, loc=3, borderpad=1))
    # Turn ticklabels of insets off
    #for axi in axins:
    #    axi.tick_params(labelleft=False, labelbottom=False)

    print("k_a vals: {}".format([kao[i] for i in kao_inds]))

    myCmap = plt.get_cmap('jet')
    #cNorm = colors.Normalize(vmin=phi[0],vmax=phi[-1])
    cNorm = colors.Normalize(vmin=0.0,vmax=0.4)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=myCmap)

    for j,kao_ind in enumerate(kao_inds):
        axSj = axS[j]
        xmax = 10.
        ymin = 1.
        thinning = 10
        for i,nCi in enumerate(nCs):
            timei = time[kao_ind][i]
            survi = surv_prob[kao_ind][i]
            coarsei = coarsening[kao_ind][i]
            extri = extrap_surv[kao_ind][i]

            colorVal = scalarMap.to_rgba(phi[i])

            if survi.shape[0]>extri.shape[0]:
                surv_datai  = survi[0:-extri.shape[0]:thinning]
                time_datai  = timei[0:-extri.shape[0]:thinning]
                time_extrapi = timei[-extri.shape[0]::thinning]
                extr_datai = extri[::thinning]

                nonlin_extr_survi = nonlinear_extrap_surv[kao_ind][i]
                nonlin_extr_timei = nonlinear_extrap_time[kao_ind][i]

                axSj.plot(time_datai,surv_datai,'-',color=colorVal,label="$n_{{C}}={}$".format(nCi))
                axSj.plot(nonlin_extr_timei,nonlin_extr_survi,'-',color=colorVal)
                #axSj.plot(nonlin_extr_timei,nonlin_extr_survi,'--',color=colorVal)
                #axSj.plot(time_extrapi,extr_datai,'--',color=colorVal)
                #axSj.plot(time_extrapi,extr_datai,'-',color=colorVal)

                axins[j].plot(time_datai,surv_datai,'-',color=colorVal,label="$n_{{C}}={}$".format(nCi))
                axins[j].plot(nonlin_extr_timei,nonlin_extr_survi,'-',color=colorVal)
                #axins[j].plot(nonlin_extr_timei,nonlin_extr_survi,'--',color=colorVal)
                #axins[j].plot(time_extrapi,extr_datai,'--',color=colorVal)
                #axins[j].plot(time_extrapi,extr_datai,'-',color=colorVal)

                ymin = min(ymin,extr_datai[np.argmax(time_extrapi>xmax)])

            else: 
                axSj.plot(timei[::thinning],survi[::thinning],'-',color=colorVal,label="$n_{{C}}={}$".format(nCi))
                axins[j].plot(timei[::thinning],survi[::thinning],'-',color=colorVal,label="$n_{{C}}={}$".format(nCi))
                ymin = min(ymin,survi[np.argmax(timei>xmax)])


        axSj.set_xlabel(r'Time',fontsize=18)
        axSj.set_ylabel(r'$\mathscr{S}_{\mathrm{rad}}(t|\sigma)$',fontsize=18)
        axSj.set_xscale('log')
        axSj.set_yscale('log')
        axSj.set_xlim((1e-4,xmax))
        #axSj.set_ylim((ymin,1.))
        axSj.set_ylim((1e-2,1.))
        #axSj.legend()

        axSj.set_title(r'$k_{{\text{{a}}}}^{{\text{{o}}}}={}$'.format(kao[kao_ind]))

        # Inset axis
        axins[j].set_yscale('log')
        axins[j].set_ylim((1e-4,1.))


    figS.subplots_adjust(right=0.8,bottom=0.2,wspace=0.45,hspace=0.4)
    cbar_ax = figS.add_axes([0.85, 0.2, 0.02, 0.65])	#[left, bottom, width, height]
    cbar = figS.colorbar(scalarMap, cax=cbar_ax)
    cbar.set_label(r'$\phi$',fontsize=18)

    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()

    #---------------------------------------------------------------------------------------------
    # Save figures
    #---------------------------------------------------------------------------------------------

    savefigs = 1
    if savefigs:
        figS.savefig("./Figures/SurvivalProb_inset_kdo_{}_kao_{}.pdf".format(kdo,kao))

    plt.show()
