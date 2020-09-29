import numpy as np
import matplotlib.pyplot as plt
import readdy
#----------------------------------------------------------------------------------------

if __name__=="__main__":

    basedir = './run_bulk_nL2_nC1_kOn1E+1_kOff1E-0/trajectory_0/'
    traj_file = basedir + 'LR_out_bulk.h5'

    traj = readdy.Trajectory(traj_file)

    periodic_directions = [0,1,2]
    timestep = 1e-4

    time_sim, counts = traj.read_observable_number_of_particles()

    time = time_sim * timestep 

    counts = counts.astype(np.int8)	# copy number of inactive sensor at each timestep


    fig, ax = plt.subplots(1,1,figsize=(8,3))

    #ax.plot(time[20000:22000]-time[20000],1.-counts[20000:22000,1],label="S")
    ax.plot(time,1.-counts[:,1],label="SL complex")

    ax.set_xlabel("Time",fontsize=18)    
    ax.set_ylabel("Receptor Occupacy",fontsize=18)    

    fig.tight_layout()

    fig.savefig("./Figures/occupancy_timeseries.eps")

    plt.show()
