import numpy as np
import matplotlib.pyplot as plt

####################################################################################
if __name__=="__main__":

    # Load survival histogram data from files
    basedir = "./"
    nL = 2
    nS = 1
    nCvals = [1]#,100,200,300,400,600]
    #kOnvals = [10,100,1000,10000]
    kOnvals = ["1E+0","1E+1","1E+2","1E+3","1E+4"]
    kOffvals = ["1E-0"]

    paramvals = [(nCi,kOni,kOffi) for nCi in nCvals for kOni in kOnvals for kOffi in kOffvals]

    logrithmic_bins = True
    if logrithmic_bins == True:
        datafiles = [basedir + "run_bulk_nL{}_nC{}_kOn{}_kOff{}/bound_hist_logrithmic.txt".format(nL,nCi,kOni,kOffi) for (nCi,kOni,kOffi) in paramvals]
    else:
        datafiles = [basedir + "run_bulk_nL{}_nC{}_kOn{}_kOff{}/bound_hist.txt".format(nL,nCi,kOni,kOffi) for (nCi,kOni,kOffi) in paramvals]

    box_vol = 5.*5.*5.
    particle_vol = pow(0.5,3.)*np.pi*4./3.
    Survivaldata = []

    for dfi in datafiles:
        data = np.loadtxt(dfi)
        Survivaldata.append(data)

    fig, ax = plt.subplots(1,1)

    for (nCi,kOni,kOffi),Sdatai in zip(paramvals,Survivaldata):
        volume_fraction = (nCi+nL+nS)*particle_vol/box_vol

        if nCi==0:
            linestyle = "-*"
        else:
            linestyle = "-o"

        # Normalize distributions to sum to 1
        normed_survival_prob = Sdatai[:,1]/np.sum(Sdatai[:,1])

        if logrithmic_bins==True:
            bin_mids = pow(10.,Sdatai[:,0])
            #ax.plot(pow(10.,Sdatai[:,0]),normed_survival_prob,linestyle,label=r"$k_{{On}}={}, n_{{C}}={}$".format(kOni,nCi))
        else:
            bin_mids = Sdatai[:,0]

        ax.plot(bin_mids,normed_survival_prob,linestyle,label=r"$k_{{On}}={}, n_{{C}}={}$".format(kOni,nCi))

        mfpt_unbind = np.sum(bin_mids * normed_survival_prob)
        print("Unbind rate constant for kOn={} = {}".format(float(kOni),1./mfpt_unbind))
    

    ax.set_xlabel("Bound Time",fontsize=18)
    ax.set_ylabel("Probability",fontsize=18)

    if logrithmic_bins==True:
        ax.set_xscale('log')
    ax.set_yscale('log')

    ax.legend()

    fig.tight_layout()

    savefig = True
    if savefig==True:
        if logrithmic_bins==True:
            plt.savefig("./Figures/bound_distributions_logrithmic.eps")
        else:
            plt.savefig("./Figures/bound_distributions.eps")

    plt.show()
