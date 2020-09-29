import numpy as np
import matplotlib.pyplot as plt
import readdy
from matplotlib.pyplot import psd
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
if __name__=="__main__":

    nL = 1
    nC = 0
    kOn = 10
    kOff = 1
    rundir = './run_bulk_nL{}_nC{}_kOn{}_kOff{}/'.format(nL,nC,kOn,kOff)
    basedirs = [rundir + 'trajectory_{}/'.format(i) for i in range(0,10)]

    activeS_data = []
    for i,basedir in enumerate(basedirs):
        traj_file = basedir + 'LR_out_bulk.h5'
        print(traj_file)
        traj = readdy.Trajectory(traj_file)


        periodic_directions = [0,1,2]
        timestep = 1e-4

        time_sim, counts = traj.read_observable_number_of_particles()

        time = time_sim * timestep 

        dataS = 1. - counts[:,1]
        activeS_data.append(dataS)

    combined_data = np.hstack(activeS_data)

    fig, ax = plt.subplots(1,1)
    pxx, freqs = ax.psd(combined_data,Fs=1e4,detrend='none',NFFT=pow(2,14))

    ax.set_xscale('log')

    # Save histogram to file
    outfilename = rundir + 'power_spectrum.txt'
    outputdata = np.vstack((freqs,pxx)).transpose()
    np.savetxt(outfilename,outputdata)

    plt.show()
