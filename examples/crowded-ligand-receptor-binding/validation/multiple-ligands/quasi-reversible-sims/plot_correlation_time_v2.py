import numpy as np
import matplotlib.pyplot as plt
import readdy
from joblib import Parallel, delayed

from plot_survival_prob_v2 import *

#----------------------------------------------------------------------------
def calc_correlation_time(react_prob,time,dissoc_prob,kao,kdo):

    dt = time[1]-time[0]
    surv_prob = calc_survival_prob(react_prob,time,kao)
    sep_prob = calc_separation_prob(surv_prob,time,kdo,dissoc_prob)

    pstar_star = 1. - sep_prob
    p0 = pstar_star[-1]
    n_var = p0 * pstar_star[0] - p0**2.
    corr_func = p0 * (pstar_star - p0)

    # Calculate correlation time directly from definition in Kaizu 2014 supp. equation S8
    tau_n = 1./n_var * trap_integrate(corr_func,time)

    return tau_n

#----------------------------------------------------------------------------
def calc_tau_approx(ka,kd,kD,c):
    return (ka + kD) / ((ka*c + kd)*kD)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
if __name__=="__main__":

    # Set data files for processing
    nLtags = [1]#, 50, 70]
    nL = 1
    traj_number = 0

    # Set intrinsic reaction rates
    kao = np.logspace(-1,5,10) #1.0e-0
    kdo = 1.0e+1

    tau_n = []
    react_probs = []
    times = []
    dissoc_probs = []
    #dts = [0.1,0.1,0.1]
    dts = [10.*1e-4 for x in nLtags]
    for nLtag,dt in zip(nLtags,dts):
        unbounddatafile_base = "./boxsize_10_10_10/run_bulk_nL{}/trajectory_{}/unbound_reaction_event_density_nL_{}_*.npy".format(nLtag,traj_number,nL)
        dissocdatafile = "./boxsize_10_10_10/run_bulk_nL{}/trajectory_{}/accepted_dissociation_moves.txt".format(nLtag,traj_number)

        # Parse filename of unbound data file
        for filename in glob.glob(unbounddatafile_base):
            unbounddatafile = filename
            split_filename = filename.split(".")[1]
            split_filename = split_filename.split("_")
            tstart = int(split_filename[-5])
            tstop = int(split_filename[-3])
            sample_freq = int(split_filename[-1])

        time_indices = range(tstart,tstop,sample_freq)

        # Load reaction probability for each timepoint
        react_prob_data = np.load(unbounddatafile)

        # Read dissociation prob from accepted moves file header
        with open(dissocdatafile, 'r') as f:
            header = f.readline()
        split_header = header.split()
        dissoc_prob = float(split_header[6])
        dissoc_probs.append(dissoc_prob)

        time_sim = dt * np.array(range(react_prob_data.shape[1]))
        times.append(time_sim)
        react_probs.append(react_prob_data[1,:])

         # Define correlation time calculation function for parallel processing
        def parfunc_calctau(kaoi):
            return calc_correlation_time(react_prob_data,time_sim,dissoc_prob,kaoi,kdo)

        num_cores = 4
        tau_ni = Parallel(n_jobs=num_cores)(delayed(parfunc_calctau)(kaoi) for kaoi in kao)

        tau_n.append(tau_ni)

    V = 10.*10.*10.
    rL = 1.0
    vL = 4./3.*np.pi*(2.*rL)**3.
    phi = [nLtagi*vL/V for nLtagi in nLtags]

    c = nL/V
    Do = 2.
    sigma = 2.
    kDo = 4.*np.pi*sigma*Do

    # Analytical approximation
    tau_approx_nocrowd = [calc_tau_approx(kaoi,kdo,kDo,c) for kaoi in kao]
    tau_an = []
    for nLi,dissoc_probi in zip(nLtags,dissoc_probs):
        ceff = c*(1.-nLi*vL/V)
        kd = kdo*dissoc_probi
        print(ceff,kd,kao[0]*ceff,kao[-1]*ceff,kDo)
        
        tau_an.append([calc_tau_approx(kaoi,kd,kDo,ceff) for kaoi in kao])
    # --------------------------------------------------------------- #
    # Plotting

    fig, ax = plt.subplots(1,2,figsize=(10,5))
    
    for taui,phii,tau_ani in zip(tau_n,phi,tau_an):
        ax[0].plot(kao,taui,label="$\phi={:.2f}$".format(phii))
        color = ax[0].lines[-1].get_color()
        ax[0].plot(kao,tau_ani,'--',color=color)

    ax[0].plot(kao,tau_approx_nocrowd,'--',label="$\phi=0$ limit")

    ax[0].set_xlabel(r"$k_{a}^{o}$",fontsize=18)
    ax[0].set_ylabel(r"$\tau_{n}$",fontsize=18)
    ax[0].legend()
    ax[0].set_xscale("log")

    for timei,react_probi in zip(times,react_probs):
        print(react_probi.shape)
        ax[1].plot(timei,react_probi)

    ax[1].set_xlabel(r"$Time$",fontsize=18)
    ax[1].set_ylabel(r"$p_{a}(t|\sigma)$",fontsize=18)


    fig.tight_layout()

    #fig.savefig("Figures/taucVska_kdo_{}_nL_{}.eps".format(kdo,nL))

    plt.show()
