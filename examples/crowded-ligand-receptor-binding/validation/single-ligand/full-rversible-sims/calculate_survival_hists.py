import numpy as np
import matplotlib.pyplot as plt
import readdy
from joblib import Parallel,delayed

#----------------------------------------------------------------------------------------
#def calc_survival_hist(traj,bins=10,bound_bins=10,logrithmic_bins=False):
def calc_survival_times(traj_file):


    print(traj_file)
    traj = readdy.Trajectory(traj_file)

    periodic_directions = [0,1,2]
    timestep = 1e-4

    time_sim, counts = traj.read_observable_number_of_particles()

    time = time_sim * timestep 

    counts = counts.astype(np.int8)	# copy number of inactive sensor at each timestep
    diffS = np.diff(counts[:,1])

    act_reacts_inds = np.where(diffS==-1)	# timesteps of S->SA reactions
    deact_reacts_inds = np.where(diffS==1)	# timesteps of SA->S reactions

    act_times = time[act_reacts_inds]
    deact_times = time[deact_reacts_inds]
    if counts[0,1]==0: # Sensor is initially in bound state
        act_times = np.insert(act_times,0,0.)

    # Calculate survival times (i.e. unbound durations)
    if act_times.shape[0]==deact_times.shape[0]:
        survival_times = act_times[1:] - deact_times[:-1]
    else:
        survival_times = act_times[1:] - deact_times


    # Calculate suruvival time of complex (i.e. bound durations)
    if act_times.shape[0]==deact_times.shape[0]:
        bound_times = deact_times - act_times
    else:
        bound_times = deact_times - act_times[:-1]

    return (survival_times, bound_times)

#----------------------------------------------------------------------------------------
def calc_survival_hist(trajectory_files,bins=10,bound_bins=10,logrithmic_bins=False,num_cores=1):
    
    #survival_bound_times = [calc_survival_times(traji) for traji in trajectory_files]
    
    survival_bound_times = Parallel(n_jobs=num_cores)(delayed(calc_survival_times)(traji) for traji in trajectory_files)
    survival_times = np.hstack([xi[0] for xi in survival_bound_times])
    bound_times = np.hstack([xi[1] for xi in survival_bound_times])

    print(survival_times.shape)
    # Histogram data
    if logrithmic_bins:
        survival_hist, bins = np.histogram(np.log10(survival_times),bins)
        bound_hist, bound_bins = np.histogram(np.log10(bound_times),bound_bins)

        bins = pow(10.,bins)
        bound_bins = pow(10.,bound_bins)
    else:
        survival_hist, bins = np.histogram(survival_times,bins)
        bound_hist, bound_bins = np.histogram(bound_times,bound_bins)


    return (survival_hist, bins), (bound_hist, bound_bins)
#----------------------------------------------------------------------------------------
################################################################################
if __name__=="__main__":

    nL = 1
    nC = 0
    kOn = 10000
    kOff = 1
    rundir = './run_bulk_nL{}_nC{}_kOn{}_kOff{}/'.format(nL,nC,kOn,kOff)
    ntrajs = [30,40]

    for ntraj in ntrajs:
        basedirs = [rundir + 'trajectory_{}/'.format(i) for i in range(0,ntraj)]
        traj_files = [basedir + 'LR_out_bulk.h5' for basedir in basedirs]

        logrithmic_bins = True

        nbins = 200
        nbound_bins = 100

        num_cores = 4
        (survival_hist, bins), (bound_hist,bound_bins) = calc_survival_hist(traj_files,nbins,nbound_bins,logrithmic_bins,num_cores)

        bin_mids = (bins[:-1] + bins[1:]) / 2
        bound_bin_mids = (bound_bins[:-1] + bound_bins[1:]) / 2

        bin_lhs = bins[:-1]
        bin_widths = np.diff(bins)
        bound_bin_lhs = bound_bins[:-1]
        bound_bin_widths = np.diff(bound_bins)

        # Save histogram to file
        savedata = 1
        if savedata==True:
            if logrithmic_bins:
                outfilename = rundir + 'survival_hist_logrithmic_ntraj_{}.txt'.format(ntraj)
            else:
                outfilename = rundir + 'survival_hist_ntraj_{}.txt'.format(ntraj)
            outputdata = np.vstack((bin_lhs,bin_widths,survival_hist)).transpose()
            np.savetxt(outfilename,outputdata)

            if logrithmic_bins:
                outfilename_bound = rundir + 'bound_hist_logrithmic_ntraj_{}.txt'.format(ntraj)
            else:
                outfilename_bound = rundir + 'bound_hist_ntraj_{}.txt'.format(ntraj)
            outputdata_bound = np.vstack((bound_bin_lhs,bound_bin_widths,bound_hist)).transpose()
            np.savetxt(outfilename_bound,outputdata_bound)


    ################################################################################
    # Plot Survival histogram
    plot_survival_hist = True
    if plot_survival_hist == True:
        figHist, axHist = plt.subplots(1,1)
        axHist.bar(bin_lhs+0.5*bin_widths,np.log10(survival_hist),align='center',width=bin_widths)


    # Plot Activation-Inactivation timeseries
    plot_act = False
    if plot_act == True:
        fig, ax = plt.subplots(2,1,figsize=(8,4))

        ax[0].plot(time,counts[:,0],label="L")
        ax[1].plot(time,counts[:,1],label="S")
        ax[1].plot(time,counts[:,2],label="C")

        ax[0].set_xlabel("Time",fontsize=18)    
        ax[0].set_ylabel("Count",fontsize=18)    
        ax[1].set_xlabel("Time",fontsize=18)    
        ax[1].set_ylabel("Count",fontsize=18)    
        ax[0].legend()
        ax[1].legend()


    plt.show()
