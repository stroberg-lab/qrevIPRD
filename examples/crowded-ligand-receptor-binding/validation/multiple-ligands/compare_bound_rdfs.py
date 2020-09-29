import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------
def parse_rdf_data_file(rdf_filename):
    # Parse header
    with open(rdf_filename, 'r') as f:
        header = f.readline()
    split_header = header.split()
    rmin_q = float(split_header[2])
    rmax_q = float(split_header[4])
    nrbins_q = float(split_header[6])
    tmin_q = float(split_header[8])
    tmax_q = float(split_header[10])
    ntbins_q = float(split_header[12])

    r_edges = np.linspace(rmin_q,rmax_q,nrbins_q+1)
    t_edges = np.linspace(tmin_q,tmax_q,ntbins_q+1)


    # Load 2d histogram
    hist2d = np.loadtxt(rdf_filename,skiprows=1)

    return hist2d, r_edges, t_edges
###########################################################################################
if __name__=="__main__":

    # Load quasi-reversible rdf data
    nL = 2
    nC = 1

    traj_number_q = 0
    rundir_q = "./db_quasi_reversible_sims/module_weights_L05_S05/boxsize_5_5_5/run_bulk_nL{}_nC{}/trajectory_{}/".format(nL,nC,traj_number_q)
    bound_rdf_file_q = rundir_q + "bound_rdf_time_2dhist.txt"

    hist2d_q, r_edges_q, t_edges_q = parse_rdf_data_file(bound_rdf_file_q)
    rbin_centers_q = r_edges_q[:-1] + 0.5*np.diff(r_edges_q)

    # Normalize rdf
    hist2d_q *= 1./np.sum(hist2d_q)


    # Load reversible simulation rdf data
    kOn = "1E+0"
    kOff = "1E-0"
    rundir_r = "./full_reversible_sims/weights_L05_S05/run_bulk_nL{}_nC{}_kOn{}_kOff{}/".format(nL,nC,kOn,kOff)
    bound_rdf_file_r = rundir_r + "bound_rdf_time_2dhist.txt"

    hist2d_r, r_edges_r, t_edges_r = parse_rdf_data_file(bound_rdf_file_r)
    rbin_centers_r = r_edges_r[:-1] + 0.5*np.diff(r_edges_r)
    tbin_centers_r = t_edges_r[:-1] + 0.5*np.diff(t_edges_r)

    # Normalize rdfs for each time bin
    for i in range(hist2d_r.shape[1]):
        print(i,np.sum(hist2d_r[:,i]))
        hist2d_r[:,i] *= 1./np.sum(hist2d_r[:,i])


    print(t_edges_r)
    early_times = range(0,1)
    early_rdf_r = np.mean(hist2d_r[:,early_times],axis=1)

    mean_times = range(40,50)
    mean_rdf_r = np.mean(hist2d_r[:,mean_times],axis=1)
    
    ####################################################################
    # Plot
    figRDF, axRDF = plt.subplots(1,1)

    axRDF.plot(rbin_centers_q,hist2d_q,linewidth=3,label="Quasi-Reversible")

    axRDF.plot(rbin_centers_r,early_rdf_r,"-o",label="Reversible, Initial")
    axRDF.plot(rbin_centers_r,mean_rdf_r,"-s",label="Reversible, Mean")

    sigma_L = 1.0
    sigma_S = 1.0
    sigma_SL = pow(pow(sigma_L,3.)+pow(sigma_S,3.),1./3)
    sigma_SLL = pow(sigma_L*sigma_SL,1./2.)
    axRDF.axvline(2.*sigma_SLL,linestyle="--")

    axRDF.set_xlabel(r'$r$',fontsize=18)
    axRDF.set_ylabel(r'$g(r)$',fontsize=18)

    axRDF.legend()

    figRDF.tight_layout()

    figRDF.savefig("./Figures/bound_rdfs.eps")

    plt.show()
