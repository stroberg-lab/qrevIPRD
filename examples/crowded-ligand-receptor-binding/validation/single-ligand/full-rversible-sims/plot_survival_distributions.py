import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
####################################################################################
if __name__=="__main__":

    # Load survival histogram data from files
    basedir = "./"
    nL = 1
    nS = 1
    nCvals = [0]#,100,200,300,400,600]
    kOnvals = [10,100,1000,10000]
    kOffvals = [1]

    paramvals = [(nCi,kOni,kOffi) for nCi in nCvals for kOni in kOnvals for kOffi in kOffvals]

    logrithmic_bins = False
    if logrithmic_bins == True:
        datafiles = [basedir + "run_bulk_nL{}_nC{}_kOn{}_kOff{}/survival_hist_logrithmic.txt".format(nL,nCi,kOni,kOffi) for (nCi,kOni,kOffi) in paramvals]
    else:
        datafiles = [basedir + "run_bulk_nL{}_nC{}_kOn{}_kOff{}/survival_hist.txt".format(nL,nCi,kOni,kOffi) for (nCi,kOni,kOffi) in paramvals]

    box_vol = 5.*5.*5.
    particle_vol = 1.*np.pi*4./3.
    Survivaldata = []
    event_density_data = []
    for dfi in datafiles:
        event_density = np.loadtxt(dfi)
        surv = np.ones(event_density.shape)
        if logrithmic_bins == False:
            #event_density[:,1] = event_density[:,1]/np.sum(event_density[:,1])
            event_density[:,1] = event_density[:,1]/np.trapz(event_density[:,1],event_density[:,0])
            surv[:,0] = event_density[:,0]
            #surv[:,1] = 1. - np.cumsum(event_density[:,1])
            surv[1:,1] = 1. - cumtrapz(event_density[:,1],event_density[:,0])
            print(event_density[-1,0])
            print(event_density[:,1])

        else:
            log_bin_mids = event_density[:,0]
            logdx = np.diff(log_bin_mids)[0]
            #log_bin_edges = log_bin_mids - 0.5*np.diff(log_bin_mids)
            log_bin_edges = log_bin_mids - 0.5*logdx
            #log_bin_edges = np.append(log_bin_edges,log_bin_mids[-1] + 0.5*(log_bin_mids[-1]-log_bin_mids[-2]))
            log_bin_edges = np.append(log_bin_edges,log_bin_mids[-1] + 0.5*logdx)
            bin_edges = pow(log_bin_edges,10.)
            bin_widths = np.diff(bin_edges)
            normalization_constant = np.sum(event_density[:,1]*bin_widths)
            event_density[:,1] = event_density[:,1]/normalization_constant
            surv[:,0] = event_density[:,0]
            #surv[:,1] = 1. - np.cumsum(event_density[:,1])
            surv[1:,1] = 1. - cumtrapz(event_density[:,1],event_density[:,0])

        event_density_data.append(event_density)
        Survivaldata.append(surv)

    fig, ax = plt.subplots(1,1)

    for (nCi,kOni,kOffi),Sdatai,edi in zip(paramvals,Survivaldata,event_density_data):
        volume_fraction = (nCi+nL+nS)*particle_vol/box_vol

        if nCi==0:
            linestyle = "-*"
        else:
            linestyle = "-o"

        # Normalize distributions to sum to 1
        normed_survival_prob = Sdatai[:,1]/np.sum(Sdatai[:,1])

        if logrithmic_bins==True:
            ax.plot(pow(10.,Sdatai[:,0]),normed_survival_prob,linestyle,label=r"$k_{{On}}={}, k_{{On}}={}, n_{{C}}={}$".format(kOni,kOffi,nCi))
        else:
            ax.plot(Sdatai[:,0],normed_survival_prob,linestyle,label=r"$k_{{On}}={},k_{{Off}}={}, n_{{C}}={}$".format(kOni,kOffi,nCi))
            ax.plot(edi[:,0],edi[:,1],'.')

    ax.set_xlabel("Survival Time",fontsize=18)
    ax.set_ylabel("Probability",fontsize=18)

    if logrithmic_bins==True:
        ax.set_xscale('log')
    ax.set_yscale('log')

    ax.legend()

    fig.tight_layout()

    savefig = False
    if savefig==True:
        if logrithmic_bins==True:
            plt.savefig("./Figures/survival_distributions_logrithmic_nC0and80.eps")
        else:
            plt.savefig("./Figures/survival_distributions.eps")

    plt.show()
