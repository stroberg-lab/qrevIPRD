import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

if __name__=="__main__":

    # Specify data files for plotting
    kon = 100
    nCvals = [100, 200, 300, 400, 600]
    basedir = "./"
    datafiles = [basedir + "run_bulk_nL5_nC{}_kOn{}/fluctuation_vs_tau_data.txt".format(nCi,kon) for nCi in nCvals]

    timestep = 1e-4
    box_vol = 10.*10.*10.
    particle_vol = 4./3.*np.pi*pow(0.5,3)

    vol_fraction = [nCi*particle_vol/box_vol for nCi in nCvals]

    # Read data from files
    fluc_data = [np.loadtxt(di) for di in datafiles]

    # Plot error vs measurement window time
    fig_var,ax_var = plt.subplots(1,1,figsize=(6,4))

    for phi,fdi in zip(vol_fraction,fluc_data):
        timewindow = fdi[:,0] * timestep
        fluctuations = fdi[:,1]
        ax_var.plot(timewindow,fluctuations,linewidth=2,label=r"$\phi={:.2f}$".format(phi))

    # get legend to display as 10^x
    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
    fmt = mticker.FuncFormatter(g)


    ax_var.set_xlabel(r'$\tau_{{meas}}$',fontsize=20)
    ax_var.set_ylabel(r'$\left(\frac{\delta n}{\bar{n}}\right)^2$',fontsize=20)
    ax_var.legend(fontsize=12,loc=1)

    ax_var.set_xscale('log')
    ax_var.locator_params(axis='y',nbins=5)
    ax_var.tick_params(labelsize=14)

    fig_var.tight_layout()

    plt.show()
