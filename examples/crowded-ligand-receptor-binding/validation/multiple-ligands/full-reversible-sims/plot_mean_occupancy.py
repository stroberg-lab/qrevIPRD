import numpy as np
import matplotlib.pyplot as plt
import readdy
import glob

#####################################################################################
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#####################################################################################
if __name__=="__main__":

    # Specify data directories for analysis
    kon = 100
    nCvals = [100, 200, 300, 400, 600]
    basedir = "./"
    datadirs = [basedir + "run_bulk_nL5_nC{}_kOn{}/".format(nCi,kon) for nCi in nCvals]

    box_vol = 10.*10.*10.
    particle_vol = 4./3.*np.pi*pow(0.5,3)

    vol_fraction = [nCi*particle_vol/box_vol for nCi in nCvals]

    # Loop over data and calculate mean of combined trajectories
    mean_occupancy = []
    for di in datadirs:
        activeS_data = []
        for subdir in glob.iglob(di+"trajectory_*"):
            traj_file = subdir + '/LR_out_bulk.h5'
            print(traj_file)
            try:
                traj = readdy.Trajectory(traj_file)
            except:
                print("An error occurred reading the data file {}".format(traj_file))

            time_sim, counts = traj.read_observable_number_of_particles()

            dataS = 1. - counts[:,1]
            activeS_data.append(dataS)

        combined_data = np.hstack(activeS_data)
        mean_occupancy.append(np.mean(combined_data))

    # Plot mean values vs crowder volume fraction

    fig, ax = plt.subplots(1,2,figsize=(8,4))

    ax[0].plot(vol_fraction,mean_occupancy,'-o')

    ax[0].set_xlabel(r"Volume Fraction",fontsize=18)
    ax[0].set_ylabel(r"Mean Occupancy",fontsize=18)

    normalized_occupancy = [moi/mean_occupancy[0] for moi in mean_occupancy]
    phi = np.linspace(0.,max(vol_fraction),100)
    theoretical_occupancy = 1./(1.-phi)

    ax[1].plot(vol_fraction,normalized_occupancy,'-o')
    ax[1].plot(phi,theoretical_occupancy,'--',label=r"$\frac{1}{1-\phi}$")

    ax[1].set_xlabel(r"Volume Fraction",fontsize=18)
    ax[1].set_ylabel(r"Normalized Mean Occupancy",fontsize=18)

    ax[1].legend()

    fig.tight_layout()

    fig.savefig("./Figures/mean_occupancy_vs_crowding_kon_{}.eps".format(kon))

    plt.show()
