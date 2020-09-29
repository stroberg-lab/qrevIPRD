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
def get_bound_unbound_intervals(traj):

    timestep, counts = traj.read_observable_number_of_particles()

    counts = counts.astype(np.int8)	# copy number of each species [L,S,C] at each observation time (1 timestep)
    diffS = np.diff(counts[:,1])	# times at which copy number of unbound S changes

    act_reacts_inds = np.where(diffS==-1)	# timesteps of S->SL reactions
    deact_reacts_inds = np.where(diffS==1)	# timesteps of SL->S reactions

    act_times = timestep[act_reacts_inds]
    deact_times = timestep[deact_reacts_inds]


    if counts[0,1]==0: # Sensor is initially in bound state
        act_times = np.insert(act_times,0,0.)

    # Calculate survival times (i.e. unbound durations)
    if act_times.shape[0]==deact_times.shape[0]:
        unbound_intervals = [(deact_ti,act_ti) for (deact_ti,act_ti) in zip(deact_times[:-1],act_times[1:])]
    else:
        unbound_intervals = [(deact_ti,act_ti) for (deact_ti,act_ti) in zip(deact_times,act_times[1:])]


    # Calculate suruvival time of complex (i.e. bound durations)
    if act_times.shape[0]==deact_times.shape[0]:
        bound_intervals = [(act_ti,deact_ti) for (deact_ti,act_ti) in zip(deact_times,act_times)]
    else:
        bound_intervals = [(act_ti,deact_ti) for (deact_ti,act_ti) in zip(deact_times,act_times[:-1])]

    return bound_intervals, unbound_intervals
#-----------------------------------------------------------------
def calc_AB_distance(traj,typeA,typeB,nA,nB,timerange=None):

    timestep, types, ids, positions = traj.read_observable_particles()
    
    if timerange==None:
        timerange = range(timestep.shape[0])

    elif len(timerange)==2:
       tstart = timerange[0]
       tstop = timerange[1]
       timeinds = np.where(np.logical_and(timestep>tstart,timestep<tstop))
       timerange = timeinds[0]


    #d = np.zeros((len(timerange),nL))
    d = []
    for i in timerange:
        idsi = np.array([traj.species_name(j) for j in types[i]])
        indsA = np.char.equal(idsi,typeA)
        indsB = np.char.equal(idsi,typeB)
        posAi = positions[i][indsA]
        posBi = positions[i][indsB]
        #d[i,:] = dist(posLi,posRi,traj.box_size)
	#di = dist(posLi,posRi,traj.box_size)
        #print(di)
        d.append(dist(posAi,posBi,traj.box_size))

    return np.array(d), timerange-timerange[0]
#-----------------------------------------------------------------
def calc_LS_distance(traj,nL,timerange=None):
    return calc_AB_distance(traj,"L","S",nL,1,timerange)

#-----------------------------------------------------------------
def calc_LSL_distance(traj,nL,timerange=None):
    return calc_AB_distance(traj,"L","SL",nL,1,timerange)

#-----------------------------------------------------------------
def calc_rdf(traj,nL,nbins,rrange,timerange=None):

    d = calc_LS_distance(traj,nL,timerange)

    rmin = rrange[0]
    rmax = rrange[1]
    bins = np.linspace(rmin,rmax,nbins+1)
    dr = bins[1]-bins[0]
    bin_centers = bins[0:-1] + dr*0.5

    density_ideal = nL / (traj.box_size[0]*traj.box_size[1]*traj.box_size[2])
    weights = 1./(4.*np.pi*bin_centers**2.*dr * density_ideal)

    rdfs = np.array([np.histogram(di,bins=bins)[0]*weights for di in d])

    return rdfs, bins

#-----------------------------------------------------------------
def calc_bound_rdf(trajfiles,nL,rbins,tbins,rrange):

    r = []
    t = []
    for trajfile in trajfiles:
        print("Processing trajectory: {}".format(trajfile))
        traji = readdy.Trajectory(trajfile)
        density_ideal = nL / (traji.box_size[0]*traji.box_size[1]*traji.box_size[2])
        try:
            bound_intervals, unbound_intervals = get_bound_unbound_intervals(traji)

            def parfunc(bi):
                ri,ti = calc_LSL_distance(traji,nL,bi)
                return (ri,ti)

            n_proc = 4
            out = Parallel(n_jobs=n_proc)(delayed(parfunc)(bi) for bi in bound_intervals)

            [r.append(xi[0]) for xi in out]
            [t.append(xi[1]) for xi in out]

        except IndexError as e:
            print(e)
    r = np.vstack(r)
    t = np.hstack(t)


    r = np.array(r).flatten()
    t = np.array(t).flatten()

    trange = (0.,np.max(t))


    rmin = rrange[0]
    rmax = rrange[1]
    rbins = np.linspace(rmin,rmax,rbins+1)
    dr = rbins[1]-rbins[0]
    rbin_centers = rbins[0:-1] + dr*0.5

    rweights = 1./(4.*np.pi*rbin_centers**2.*dr * density_ideal)


    hist2d, r_edges, t_edges = np.histogram2d(r,t,bins=[rbins,tbins],range=[rrange,trange])

    for j in range(hist2d.shape[1]):
        #hist2d[:,j] *= rweights / (np.sum(hist2d[:,j]))
        hist2d[:,j] *= rweights
   
    return hist2d, r_edges, t_edges 
     
###########################################################################################
if __name__=="__main__":

    # Load trajectory data
    nLtag = 2
    nC = 1
    kOn = "1E+2"
    kOff = "1E-0"
    traj_numbers = range(0,10)
    rundir = "./run_bulk_nL{}_nC{}_kOn{}_kOff{}/".format(nLtag,nC,kOn,kOff)

    # Set number of free ligands
    nL = nLtag - 1

    trajfiles = [rundir+'trajectory_{}/LR_out_bulk.h5'.format(traji) for traji in traj_numbers]

    # Build bins for radial distribution function of ligands as function of time
    traj0 = readdy.Trajectory(trajfiles[0])
    rbins = 100
    rmin = 1.7
    rmax = traj0.box_size[0] * 0.5 # (this can be improved upon)
    bin_edges = np.linspace(rmin,rmax,rbins+1)
    rrange = (rmin,rmax)

    tbins = 100

    hist2d, r_edges, t_edges = calc_bound_rdf(trajfiles,nL,rbins,tbins,rrange)

    print(hist2d.shape)

    # Normalize rdf at each time
    for j in range(hist2d.shape[1]):
        print(j,np.sum(hist2d[:,j]))
        #hist2d[:,j] *= 1./np.sum(hist2d[:,j])


    # Plot 2d histogram as heatmap
    fig2d,ax2d = plt.subplots(1,1)

    ax2d.imshow(hist2d.transpose(),origin='lower')


    # Plot rdf at selected times
    early_times = range(0,2)
    early_rdf = np.mean(hist2d[:,early_times],axis=1)
    early_rdf = early_rdf/np.sum(early_rdf)

    mid_times = range(2,10)
    mid_rdf = np.mean(hist2d[:,mid_times],axis=1)
    mid_rdf = mid_rdf/np.sum(mid_rdf)

    mean_times = range(20,50)
    mean_rdf = np.mean(hist2d[:,mean_times],axis=1)
    mean_rdf = mean_rdf/np.sum(mean_rdf)

    rbin_centers = r_edges[:-1] + 0.5*np.diff(r_edges)

    figRDF, axRDF = plt.subplots(1,1)
 
    axRDF.plot(rbin_centers,early_rdf,'o-')
    axRDF.plot(rbin_centers,mid_rdf,'*-')
    axRDF.plot(rbin_centers,mean_rdf,'-')

    sigma_L = 1.0
    sigma_S = 1.0
    sigma_SL = pow(pow(sigma_L,3.)+pow(sigma_S,3.),1./3)
    sigma_SLL = pow(sigma_L*sigma_SL,1./2.)
    axRDF.axvline(2.*sigma_SLL,linestyle="--")

    axRDF.set_xlabel(r'$r$',fontsize=18)
    axRDF.set_ylabel(r'$g(r)$',fontsize=18)

    figRDF.tight_layout()

    # Save RDF
    header = "rmin {} rmax {} nrbins {} tmin {} tmax {} ntbins {}".format(
		r_edges[0],r_edges[-1],len(r_edges)-1,t_edges[0],t_edges[-1],len(t_edges)-1)
    outfile = rundir + "bound_rdf_time_2dhist.txt"
    np.savetxt(outfile,hist2d,header=header)
    
    plt.show()
