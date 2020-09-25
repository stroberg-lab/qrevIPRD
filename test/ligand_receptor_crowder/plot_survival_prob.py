import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
if __name__=="__main__":

    # Define simulation paramters
    nL = 2
    nC = 1

    kdo = 1E-0

    ##############################################################################################
    # Load quasi-reversible simulation data
    traj_number = 0

    basedir_qr = "."

    # Set intrisic rate constants for analysis
    kao_qr = np.logspace(2,4,3)	 


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
    ntraj = 10
    for i,nstep in enumerate(nsteps):
        for kaoj in kao_qr:

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




    ##############################################################################################
    #---------------------------------------------------------------------------------------------
    # Plot survival probability
    figU, axU = plt.subplots(1,2,figsize=(10,5))

    # Quasi-revesible
    colors = []
    linestyles = ['-']
    for i,nstep in enumerate(nsteps):
        for j in range(len(kao_qr)):
            p = axU[0].plot(time_qr[i][j][::1],surv_prob_qr[i][j][::1],linestyle=linestyles[i],label="$k_{{a}}^{{o}}={}$".format(kao_qr[j]))
            colors.append(p[0].get_color())
            axU[1].plot(time_qr[i][j][::1],surv_prob_qr[i][j][::1],'-',color=colors[-1])


    axU[0].set_xlabel(r'Time',fontsize=18)
    axU[0].set_ylabel(r'$S_{rad}(t|\sigma)$',fontsize=18)

    axU[1].set_xlabel(r'Time',fontsize=18)
    axU[1].set_ylabel(r'$S_{rad}(t|\sigma)$',fontsize=18)


    axU[0].set_xscale('log')
    axU[0].set_yscale('log')
    axU[0].set_ylim((1e-3,1.0))
    axU[0].legend()

    axU[1].set_xscale('log')

    figU.tight_layout()

    figU.savefig("./SurvivalProbability.pdf")
