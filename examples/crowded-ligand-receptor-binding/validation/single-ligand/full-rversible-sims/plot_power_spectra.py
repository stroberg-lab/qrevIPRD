import numpy as np
import matplotlib.pyplot as plt

####################################################################################
if __name__=="__main__":

    # Load power spectrum data from files
    basedir = "./"
    nL = 1
    nS = 1
    nCvals = [0]#,100,200,300,400,600]
    kOnvals = [10,100,1000]
    kOffvals = [1]

    paramvals = [(nCi,kOni,kOffi) for nCi in nCvals for kOni in kOnvals for kOffi in kOffvals]


    datafiles = [basedir + "run_bulk_nL{}_nC{}_kOn{}_kOff{}/power_spectrum.txt".format(nL,nCi,kOni,kOffi) for (nCi,kOni,kOffi) in paramvals]

    box_vol = 10.*10.*10.
    particle_vol = pow(2,3.)*np.pi*4./3.
    Pdata = []

    for dfi in datafiles:
        data = np.loadtxt(dfi)
        Pdata.append(data)

    fig, ax = plt.subplots(1,1)

    for (nCi,kOni,kOffi),Pdatai in zip(paramvals,Pdata):
        volume_fraction = (nCi+nL+nS)*particle_vol/box_vol
        ax.plot(Pdatai[:,0],Pdatai[:,1],label=r"$k_{{On}}={},k_{{Off}}={}, n_{{C}} = {}$".format(kOni,kOffi,nCi))

    ax.set_xlabel("Frequency",fontsize=18)
    ax.set_ylabel("Power Spectral Density",fontsize=18)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.legend(loc=1)

    fig.tight_layout()

    plt.savefig("./Figures/noise_power_spectra.eps")

    plt.show()
