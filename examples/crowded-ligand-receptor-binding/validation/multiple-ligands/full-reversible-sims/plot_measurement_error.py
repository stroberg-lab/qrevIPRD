import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import readdy

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def calc_mean_var_activation(receptor_state,T_chain_vals,true_mean):

    # Histogram data
    n_chain_vals = [int(receptor_state.shape[0]/Ti) for Ti in T_chain_vals]

    hists = []
    bins = []
    for n_chain_i in n_chain_vals:
        normalized_active_hist, bin_edges = histogram_activation_data(receptor_state,int(n_chain_i),normalized=True)
        bin_mids = np.array([bin_edges[i] + (bin_edges[i+1]-bin_edges[i])/2. for i in range(len(bin_edges)-1)])
        hists.append(normalized_active_hist)
        bins.append(bin_mids)

    mean_act_list = [np.sum(xi*hi) for (hi,xi) in zip(hists,bins)]
    var_act_list = [np.sum((xi-true_mean)*(xi-true_mean)*hi) for (hi,xi) in zip(hists,bins)]
    mean_error = [mean_act_i - true_mean for mean_act_i in mean_act_list]

    return [mean_act_list, var_act_list, mean_error, hists, bins]

#-----------------------------------------------------------------------------------
def histogram_activation_data(receptor_state,n_chains,normalized=True,nbins=25):
    # Calculate length of subsampling for timeseries data
    T_chain = int(receptor_state.shape[0]/(n_chains+1))

    # Calculate occupany for each subsampled timeseries
    occupancy = []
    for i in range(n_chains):
        rec_state_chain = receptor_state[i*T_chain:(i+1)*T_chain]
        total_active = int(np.sum(rec_state_chain))
        occupancy.append(total_active/T_chain)

    # Bin occupancy data
    active_hist, bin_edges = np.histogram(occupancy,nbins,density=False)
    if normalized:
        active_hist = active_hist/np.sum(active_hist)

    return active_hist, bin_edges
#####################################################################################
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#####################################################################################
if __name__=="__main__":

    nL= 2
    nC = 1
    kOn = "1E+4"
    kOff = "1E-0"
    rundir = './run_bulk_nL{}_nC{}_kOn{}_kOff{}/'.format(nL,nC,kOn,kOff)

    #ntrajs = [240]
    ntraj_stride = 80
    ntraj_count = 3
    ntrajs = [(i*ntraj_stride,(i+1)*ntraj_stride) for i in range(ntraj_count)]

    periodic_directions = [0,1,2]
    timestep = 1e-4

    for ntraj in ntrajs:
        if isinstance(ntraj,tuple):
            basedirs = [rundir + 'trajectory_{}/'.format(i) for i in range(ntraj[0],ntraj[1])]
        else:
            basedirs = [rundir + 'trajectory_{}/'.format(i) for i in range(0,ntraj)]

        # Combine trajectories into single dataset
        activeS_data = []
        for i,basedir in enumerate(basedirs):
            traj_file = basedir + 'LR_out_bulk.h5'
            print(traj_file, flush=True)

            traj = readdy.Trajectory(traj_file)

            time_sim, counts = traj.read_observable_number_of_particles()

            time = time_sim * timestep 

            dataS = 1. - counts[:,1]
            activeS_data.append(dataS)
        print(len(activeS_data), flush=True) 
        if len(activeS_data)>1:
            combined_data = np.hstack(activeS_data)
        else:
            combined_data = np.array(activeS_data)

        # Calculate estimates of mean and variance
        T_chain_vals = np.logspace(2,7,24)	# set of different tau_measurement to use for analysis

        true_mean = np.mean(combined_data)
        point_var = np.var(combined_data)
        print(true_mean,point_var)

        mean_act_list, var_act_list, mean_error_list, hists, bins = calc_mean_var_activation(combined_data,T_chain_vals,true_mean)

        flucs_normed = [vi/true_mean**2. for vi in var_act_list]

        # Save fluctuations to file for plotting
        header = "point statistics: mean {} var {}".format(true_mean,point_var)
        print(header)

        if isinstance(ntraj,tuple):
            ntraj_tag = "ntraj_{}_{}".format(ntraj[1]-ntraj[0],ntraj[0])
        else:
            ntraj_tag = "ntraj_{}".format(ntraj)

        outfilename = rundir + "fluctuation_vs_tau_data_" + ntraj_tag + ".txt"
        print(outfilename)
        outputdata = np.vstack((T_chain_vals,flucs_normed)).transpose()
        np.savetxt(outfilename,outputdata,header=header)

    #####################################################################################
    # PLOTTING
    #####################################################################################
    # Plot error vs measurement window time
    fig_var,ax_var = plt.subplots(1,1,figsize=(6,4))

    # get legend to display as 10^x
    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
    fmt = mticker.FuncFormatter(g)

    ax_var.plot(T_chain_vals[::-1]*timestep,flucs_normed[::-1],linewidth=2)

    ax_var.set_xlabel(r'$\tau_{{meas}}$',fontsize=20)
    ax_var.set_ylabel(r'$\left(\frac{\delta n}{\bar{n}}\right)^2$',fontsize=20)
    ax_var.legend(fontsize=12,loc=1)

    ax_var.set_xscale('log')
    ax_var.set_yscale('log')
    #ax_var.locator_params(axis='y',nbins=5)
    ax_var.tick_params(labelsize=14)

    fig_var.tight_layout()

    # Plot selected histograms
    fig_hist,ax_hist = plt.subplots(1,1)

    plot_inds = [2,int(len(hists)/3.),int(2.*len(hists)/3.),len(hists)-1]

    for i,(histi,xi,Ti) in enumerate(zip(hists,bins,T_chain_vals)):
        if i in plot_inds:
            dx = xi[1]-xi[0]
            pi = histi/dx
            ax_hist.bar(xi,pi,width=0.8*dx,align='center',alpha=0.5,label=r'$\tau_{{meas}} = {:.2f}$'.format(Ti*timestep))

    ax_hist.legend()
    ax_hist.set_xlabel('Active Ratio',fontsize=18)
    ax_hist.set_ylabel(r'$p(Active)$',fontsize=18)

    fig_hist.tight_layout()

    # Save figures
    save_figures = False
    if save_figures:
        fig_var.savefig("./Figures/measurement_error_vs_tau_nC{}_kOn{}.pdf".format(nC,kOn))
        fig_hist.savefig("./Figures/activation_histograms_nC{}_kOn{}.pdf".format(nC,kOn))


    plt.show()
