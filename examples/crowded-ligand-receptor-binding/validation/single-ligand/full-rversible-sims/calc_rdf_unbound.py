import numpy as np
import readdy
import matplotlib.pyplot as plt
import glob
from joblib import Parallel, delayed

###########################################################################################
#-----------------------------------------------------------------
def dist(x1,x2,periodic_box=None):
    dx = x2 - x1
    if periodic_box is not None:
        dx = np.where(dx > periodic_box * 0.5,dx-periodic_box,dx)
        dx = np.where(dx <= -periodic_box * 0.5,dx+periodic_box,dx)
    return np.linalg.norm(dx,axis=1)

#-----------------------------------------------------------------
def calc_LR_distance(traj,nL,timerange=None):

    timestep, types, ids, positions = traj.read_observable_particles()
    
    if timerange==None:
        timerange = range(timestep.shape[0])

    d = np.zeros((len(timerange),nL))
    for i in timerange:
        idsi = np.array([traj.species_name(j) for j in types[i]])
        indsL = np.char.equal(idsi,"L")
        indsR = np.char.equal(idsi,"S")
        posLi = positions[i][indsL]
        posRi = positions[i][indsR]

        d[i,:] = dist(posLi,posRi,traj.box_size)

    return d

#-----------------------------------------------------------------
def calc_rdf(traj,nL,nbins,rrange,timerange=None):

    d = calc_LR_distance(traj,nL,timerange)

    bins = np.linspace(rmin,rmax,nbins+1)
    dr = bins[1]-bins[0]
    bin_centers = bins[0:-1] + dr*0.5

    density_ideal = nL / (traj.box_size[0]*traj.box_size[1]*traj.box_size[2])
    weights = 1./(4.*np.pi*bin_centers**2.*dr * density_ideal)

    rdfs = np.array([np.histogram(di,bins=bins)[0]*weights for di in d])

    return rdfs, bins

###########################################################################################
if __name__=="__main__":

    # Load trajectory data
    nLtag = 1
    traj_number = 0
    rundir = "./run_bulk_nL{}_nC0_kOn100_kOff100/trajectory_{}/".format(nLtag,traj_number)

    # Set number of ligands vs inert crowders
    nL = nLtag

    #config_indices = range(0,50000,1000)
    trajfile = rundir+'LR_out_bulk.h5'

    # Build bins for radial distribution function of ligands as function of time
    traj = readdy.Trajectory(trajfile)
    nbins = 100
    rmin = 1.0
    rmax = traj.box_size[0] * 0.5 # (this can be improved upon)
    bin_edges = np.linspace(rmin,rmax,nbins+1)


    timerange = range(0,10000)
    rdfs, bin_edges = calc_rdf(traj,nL,nbins,(rmin,rmax),timerange)

    print(rdfs.shape)

    # Save RDF
    header = "rmin {} rmax {} nbins {}".format(rmin,rmax,nbins)
    outfile = rundir + "rdf.txt"
    np.savetxt(outfile,rdfs,header=header)

    # Plot RDF 
    bin_centers = bin_edges[0:-1] + 0.5*(bin_edges[1]-bin_edges[0])

    early_rdf = np.mean(rdfs[0:1000,:],axis=0)
    mid_rdf = np.mean(rdfs[100:200,:],axis=0)
    mean_rdf = np.mean(rdfs[1000:,:],axis=0)
  
    figRDF, axRDF = plt.subplots(1,1)
 
    axRDF.plot(bin_centers,mean_rdf,'*-')
    axRDF.plot(bin_centers,early_rdf,'o-')
    #axRDF.plot(bin_centers,mid_rdf,'x-')
    axRDF.axvline(1,linestyle="--")

    axRDF.set_xlabel(r'$r$',fontsize=18)
    axRDF.set_ylabel(r'$g(r)$',fontsize=18)

    figRDF.tight_layout()

    
    plt.show()
