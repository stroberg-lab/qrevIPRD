"""
Defines functions for simulating reactions using qrevIPRD.

Created by
Wylie Stroberg 2020-04-28
"""

import numpy as np
import scipy.integrate
import random
import glob
from joblib import Parallel, delayed
#from tqdm import tqdm
import readdy
import qrevIPRD.util  as util

###########################################################################################
#### Fission Events
#-----------------------------------------------------------------
def calc_cumulative_proposal_distribution(fission_radii,pot):
    integrand = lambda r: 4.*np.pi*r**2.*np.exp(-pot(r))
    lower_bound = np.argwhere(integrand(fission_radii)>0.)[0][0]-0	# manual adjustment of lower bound for accuracy of integral!
    out = [scipy.integrate.quadrature(integrand,fission_radii[lower_bound],ri,tol=1.49e-8,maxiter=400) for ri in fission_radii[lower_bound:]]
    c = np.zeros(fission_radii.shape)
    c[lower_bound:] = np.array([xi[0] for xi in out])
    return c/c[-1]

#-----------------------------------------------------------------
def draw_product_distance(fission_radii,cumq_distribution):
    u = random.random()
    ind = np.argwhere(u<cumq_distribution)[0]
    return fission_radii[ind]

#-----------------------------------------------------------------
def draw_new_product_positions(pos,n,fission_radii,cumq_distribution,weightL=0.5,weightR=0.5,periodic_box=None):
    orientations = util.gen_rand_vecs(n)
    rs = np.array([draw_product_distance(fission_radii,cumq_distribution) for i in range(n)])
    rL = (1.-weightL) * rs
    rR = (1.-weightR) * rs
    Lpos = pos + rL*orientations
    Rpos = pos + -rR*orientations
    # Wrap trial positions in periodic box if necessary
    if periodic_box is not None:
        Lpos = np.where(Lpos > 0.5*periodic_box, Lpos-periodic_box,Lpos)
        Lpos = np.where(Lpos <= -0.5*periodic_box, Lpos+periodic_box,Lpos)
        Rpos = np.where(Rpos > 0.5*periodic_box, Rpos-periodic_box,Rpos)
        Rpos = np.where(Rpos <= -0.5*periodic_box, Rpos+periodic_box,Rpos)

    return (Lpos,Rpos)


#-----------------------------------------------------------------
def sample_dissociation_events(traj,potdict,rneighbor,R_react,weight_L,weight_S,n_samples, n_trials):

    timestep, types, ids, positions = traj.read_observable_particles()

    # Define distance function for periodic box
    dist = lambda x1,x2: util.dist(x1,x2,traj.box_size)


    lj_LL = potdict['lj_LL']
    lj_SL = potdict['lj_SL']
    lj_SLL = potdict['lj_SLL']
    lj_CL = potdict['lj_CL']
    lj_CC = potdict['lj_CC']
    lj_CS = potdict['lj_CS']
    lj_SLC = potdict['lj_SLC']


    npoints = 10000	# points for cumulative fission radii distribution
    fission_radii = np.linspace(1e-4,R_react,npoints)
    cumq = calc_cumulative_proposal_distribution(fission_radii,lj_SL)

    draw_product_pos = lambda pos: draw_new_product_positions(pos,n_trials,fission_radii,cumq,weightL=weight_L,weightR=weight_S,periodic_box=traj.box_size)
    
    # Loop over timesteps in chunks
    Naccepted = 0
    Ntotal = 0
    stored_accepted_moves = []
    rvals = []

    sample_start = int(len(timestep)/5)		# timepoint to start analysis
    sample_step = max(1,int((len(timestep)-sample_start)/(n_samples/n_trials - 1)))	# timesteps between samples
    for i in range(sample_start,len(timestep),sample_step):
        idsi = np.array([traj.species_name(j) for j in types[i]])
        indsL = np.char.equal(idsi,"L")
        indsC = np.char.equal(idsi,"C")
        indsSL = np.char.equal(idsi,"SL")
        posLi = positions[i][indsL]
        posCi = positions[i][indsC]
        posSLi = positions[i][indsSL]

        # Calculate energy of bound state and build neighbor list
        rSL_L = dist(posLi,posSLi)
        neighborsL = np.nonzero(rSL_L<rneighbor)[0]

        rSL_C = dist(posCi,posSLi)
        neighborsC = np.nonzero(rSL_C<rneighbor)[0]

        Ebound = 0
        if rSL_L[neighborsL].size!=0:
            Ebound += np.sum(lj_SLL(rSL_L[neighborsL]))
        if rSL_C[neighborsC].size!=0:
            Ebound += np.sum(lj_SLC(rSL_C[neighborsC]))

        # Choose positions for products
        Ltestpos, Stestpos = draw_product_pos(posSLi)
        rvals.append(dist(Ltestpos,Stestpos))

        # Loop over N trials
        Etest = np.zeros((n_trials,1))
        for j in range(n_trials):

            Enew = 0.	# Energy of proposed unbound state, excluding L-S test particle interactions (DB scheme)
            if rSL_L[neighborsL].size!=0:
                # Calc energy of test particles with other neighboring ligands
                rLi = dist(posLi[neighborsL,:],Ltestpos[j])
                rSi = dist(posLi[neighborsL,:],Stestpos[j])
                EtestL_L = np.sum(lj_LL(rLi))	# energy between test ligand and other ligands
                EtestS_L = np.sum(lj_SL(rSi))	# energy between test sensor and other ligands
                Etest_L = EtestL_L + EtestS_L
                Enew += Etest_L

            if rSL_C[neighborsC].size!=0:
                # Calc energy of test particles with other neighboring crowders
                rLi = dist(posCi[neighborsC,:],Ltestpos[j])
                rSi = dist(posCi[neighborsC,:],Stestpos[j])
                EtestL_C = np.sum(lj_CL(rLi))	# energy between test ligand and crowders
                EtestS_C = np.sum(lj_CS(rSi))	# energy between test sensor and crowders
                Etest_C = EtestL_C + EtestS_C
                Enew += Etest_C

            # Calc change in internal energy of dissociation for each test fission event
            dE = Enew - Ebound

            # For each test accept or reject move based on Metropolis Criteria
            accept = 0
            if dE<0:
                accept = 1
            elif np.exp(-dE) > np.random.uniform():
                accept = 1
                
            # Tally accepted vs rejected dissociation attempts
            Naccepted = Naccepted + accept
            Ntotal = Ntotal + 1

            # Store timestep, energy change, and test particle position for each successful dissociation
            if accept:
                stored_accepted_moves.append([i, dE, Ltestpos[j][0],  Ltestpos[j][1], Ltestpos[j][2], Stestpos[j][0],  Stestpos[j][1], Stestpos[j][2]])

    Paccept = Naccepted/Ntotal

    stored_accepted_moves = np.array(stored_accepted_moves)

    total_time = len(timestep)-sample_start


    return stored_accepted_moves, Naccepted, Ntotal, Paccept, total_time, rvals

#----------------------------------------------------------------
def calculate_unbinding_moves(trajfile,potdict,rneighbor,R_react,weight_L,weight_S,**kwargs):

    if 'n_samples' in kwargs:
        n_samples = kwargs['n_samples']
    else:
        n_samples = 10000

    if 'n_trials' in kwargs:
        n_trials = kwargs['n_trials']
    else:
        n_trials = 1

    if 'savefile' in kwargs:
        savefile = kwargs['savefile']
    else:
        savefile = "accepted_dissociation_moves.txt"

    if 'tau_mol' in kwargs:
        tau_mol = kwargs['tau_mol']
    else:
        tau_mol = None

    if 'micro_model' in kwargs:
        if kwargs['micro_model']=="dimer":
            sample_diss = lambda traj: sample_dissociation_events_dimer(traj,potdict,rneighbor,R_react,weight_L,weight_S,n_samples,n_trials)
        elif kwargs['micro_model']=="sphere":
            sample_diss = lambda traj: sample_dissociation_events(traj,potdict,rneighbor,R_react,weight_L,weight_S,n_samples,n_trials)
        else:
            print("ERROR: micro_model must be either dimer or sphere, if defined!")

    else:
        sample_diss = lambda traj: sample_dissociation_events(traj,potdict,rneighbor,R_react,weight_L,weight_S,n_samples,n_trials)

    traj = readdy.Trajectory(trajfile)
    #stored_accepted_moves, Naccepted, Ntotal, Paccept, total_time, rvals = sample_dissociation_events(traj,potdict,rneighbor,R_react,weight_L,weight_S,n_samples,n_trials)
    stored_accepted_moves, Naccepted, Ntotal, Paccept, total_time, rvals = sample_diss(traj)

    print("**********************************************")
    print("Accepted/Total trials: {}/{}".format(Naccepted,Ntotal))
    print("Acceptance Probability: {:.2f}".format(Paccept))
    if tau_mol is not None:
        print("Sampling Time/Molecular Timescale: {:.1f} independent samples".format(total_time/tau_mol))
    print("**********************************************")

    # Write accepted dissociation configurations to seperate files to use as initital conditions
    header = "R_react {} weight_L {} weight_S {} Pacc {}".format(R_react,weight_L,weight_S,Paccept)
    np.savetxt(savefile, stored_accepted_moves, header=header)

    return


###########################################################################################
#### Fussion Events
###########################################################################################
#-----------------------------------------------------------------
def doi_reaction_model(r,r_react):
    """ Implemenration of the Doi reaction model. Returns True for r <= r_react.
    """
    return np.where(r<=r_react,1,0)

#-----------------------------------------------------------------
def calc_accept(trajfile,potdict,r_react,weight_L,nL,nLtag):

    print("Processing trajectory: {}".format(trajfile))

    # Load trajectory
    try:
        traj = readdy.Trajectory(trajfile)
    except (OSError, ValueError) as e:
        print("OSError while loading trajectory for index {}".format(config_index))
        return None

    timestep, types, ids, positions = traj.read_observable_particles()
    # Define distance function for periodic box
    dist = lambda x1,x2,boxsize: util.dist(x1,x2,boxsize)
 
    lj_LL = potdict['lj_LL']
    lj_SL = potdict['lj_SL']
    lj_SLL = potdict['lj_SLL']
    lj_CL = potdict['lj_CL']
    lj_CC = potdict['lj_CC']
    lj_CS = potdict['lj_CS']
    lj_SLC = potdict['lj_SLC']

    tstart = 0
    tstop = timestep.shape[0]
    sample_freq = 1
    time_indices = range(tstart,tstop,sample_freq)
    react_prob = np.zeros((len(time_indices),))

    # Choose which ligands to label as ligands vs crowders, if applicable
    ids0 = np.array([traj.species_name(j) for j in types[0]])
    if nL!=nLtag:	# NEEDS FIXING

        indsAllL = np.char.equal(ids0,"L")
        indsR = np.char.equal(ids0,"R")

        posAllL = positions[0][indsAllL]
        posR = positions[0][indsR]


        dLR = util.dist(posAllL,posR,traj.box_size)
        closestLigand = np.argmin(dLR)

        choose_prob = np.ones(indsAllL.shape)
        choose_prob[indsR] = 0
        choose_prob[closestLigand] = 0
        if nLtag != 1:
            indsLbulk = np.random.choice(indsAllL.shape[0],size=nL-1,replace=False,p=choose_prob/np.sum(choose_prob))
   
        indsL = np.full(indsAllL.shape,False)
        if nLtag != 1:
            indsL[indsLbulk] = True
        indsL[closestLigand] = True

        indsC = np.full(indsAllL.shape,True)
        indsC[indsL] = False
        indsC[indsR] = False
    else:
        indsL = np.char.equal(ids0,"L")
        indsC = np.char.equal(ids0,"C")
        indsS = np.char.equal(ids0,"S")
        posLi = positions[0][indsL]
        posCi = positions[0][indsC]
        posSi = positions[0][indsS]


    # Loop over timepoints with step size sample_freq
    for ti,i in enumerate(time_indices):
        posLi = positions[i][indsL]
        posSi = positions[i][indsS]
        posCi = positions[i][indsC]

        # Get distances from receptor to all ligands along trajectory (w/ periodic boundaries)
        dLS = util.dist(posLi,posSi,traj.box_size)

        # Get distances from receptor to all crowders along trajectory (w/ periodic boundaries)
        dSC = util.dist(posCi,posSi,traj.box_size)

        # Calculate internal energy of receptor (in unbound state)
        Ereceptor = np.sum(lj_SL(dLS)) + np.sum(lj_CS(dSC))

        # Calculate reaction propensity
        react_propensity = doi_reaction_model(dLS,r_react)
        react_candidates = np.nonzero(react_propensity)[0]
        
        # Test binding event
        accept = 0
        p_no_react_i = 1.

        #for j in react_candidates:
        if len(react_candidates)>0:
            # Choose randomly from reaction candidates for test reaction
            j = random.choice(react_candidates)

            # Choose position for fussion reaction product accounting for periodic boundaries
            vecStoLi = util.wrapped_vector(posLi[j]-posSi,traj.box_size)
            posComplex = util.wrapped_vector(posSi + weight_L * vecStoLi,traj.box_size)

            # Calculate needed intermolecular distances
            dLLj = util.dist(posLi[j],posLi[np.arange(len(posLi))!=j],traj.box_size)		# Ligand_j to other ligands
            dcrowderLj = util.dist(posLi[j],posCi,traj.box_size)					# Ligand_j to crowders
            dComplexL = util.dist(posComplex,posLi[np.arange(len(posLi))!=j],traj.box_size)	# Proposed complex to other ligands
            dComplexCrowder = util.dist(posComplex,posCi,traj.box_size)				# Proposed complex to crowders

            # Interaction energy between ligand_j and other ligands
            EligandjL = lj_LL(dLLj)

            # Interaction energy between ligand_j and crowders
            Eligandjcrowder = lj_CL(dcrowderLj)

            # Total interaction energy of ligand_j excluding interaction w/ receptor
            Eligandj = np.sum(EligandjL) + np.sum(Eligandjcrowder)

            # Interaction energy between ligand_j and receptor
            EligandjS = lj_SL(dLS[j])

            # Interaction energy of test particle with other ligands
            EComplexL = lj_SLL(dComplexL)

            # Interaction energy of test particle with crowders
            EComplexCrowder = lj_SLC(dComplexCrowder)

            # Total interaction energy of test particle
            EComplex = np.sum(EComplexL) + np.sum(EComplexCrowder)

            # Energy change for fussion reaction
            dE = EComplex - (Eligandj + Ereceptor)

            # Add back energy of fusion educt internal energy since accounted for by proposal density
            dE = dE + EligandjS

            if dE>0:
                p_no_react_i *= 1.-np.exp(-dE)
            else:
                p_no_react_i *= 0
            

        react_prob[ti] = 1. - p_no_react_i

    return react_prob

#-----------------------------------------------------------------
def calculate_reaction_probs(rundir,potdict,r_react,weight_L,nL,nLtag,**kwargs):#n_cores=1,savefile):

    # Parse kwargs
    if 'n_cores' in kwargs:
        n_cores = kwargs['n_cores']
    else:
        n_cores = 1

    if 'savefile' in kwargs:
        savefile = kwargs['savefile']
    else:
        savefile = "unbound_reaction_event_density_nL_{}".format(nL)

    if 'trajfile' in kwargs:
        trajfile = kwargs['trajfile']
    else:
        trajfile = None

    if 'micro_model' in kwargs:
        if kwargs['micro_model']=="dimer":
            parfunc_calc_accept = lambda trajfile: calc_accept_dimer(trajfile,potdict,r_react,weight_L,nL,nLtag)
        elif kwargs['micro_model']=="sphere":
            parfunc_calc_accept = lambda trajfile: calc_accept(trajfile,potdict,r_react,weight_L,nL,nLtag)
        else:
            print("ERROR: micro_model must be either dimer or sphere, if defined!")

    else:
        parfunc_calc_accept = lambda trajfile: calc_accept(trajfile,potdict,r_react,weight_L,nL,nLtag)

    # Loop over unbound simulations
    if trajfile is None:
        trajfilebase = rundir+'unbound_simulations_fine_output/LigandDiffusion_unbound_out_bulk_index_*.h5'
        config_indices = []
        trajfiles = []
        total_errors = 0
        for filepath in glob.iglob(trajfilebase):
            trajfiles.append(filepath)

    else:
        trajfiles = [trajfile]

    traj0 = readdy.Trajectory(trajfiles[0])
    timestep, types, ids, positions = traj0.read_observable_particles()
    tstart = 0
    tstop = timestep.shape[0]
    sample_freq = 1
    time_indices = range(tstart,tstop,sample_freq)

    #parfunc_calc_accept = lambda trajfile: calc_accept(trajfile,potdict,r_react,weight_L,nL,nLtag)

    react_probs = Parallel(n_jobs=n_cores)(delayed(parfunc_calc_accept)(traji) for traji in trajfiles)

    react_probs = [x for x in react_probs if x is not None]

    react_probs = np.array(react_probs)

    # Save reaction probability
    header = "tstart {} tstop {} sample_freq {} nL {}".format(tstart,tstop,sample_freq,nL)
    outfile = rundir + savefile
    np.save(outfile,react_probs)
    print("Reaction Probabilities saved to: {}".format(outfile))

    return react_probs


###########################################################################################
#### Tangent Dimer Fission Events
###########################################################################################

#-----------------------------------------------------------------
def draw_new_product_positions_dimer(pos,n,fission_radii,cumq_distribution,weightL=0.5,weightR=0.5,periodic_box=None):
    
    Rpos_bound = pos[0]		# position of receptor in bound dimer state
    Lpos_bound = pos[1]		# position of ligand in bound dimer state

    # Get unit vector in direction from receptor to ligand in bound state
    r_bound = np.linalg.norm(Lpos_bound - Rpos_bound)
    orientation = (Lpos_bound - Rpos_bound)/r_bound

    # Get midpoint between ligand and receptor centers in bound state
    #midpoint = Rpos_bound + 0.5*d_bound*orientation

    # Get separation distance for fission products
    rs = draw_product_distance(fission_radii,cumq_distribution)

    # Change of separation upon fission
    dr = np.linalg.norm(rs-r_bound)

    # Set new positions of fission products
    drL = weightR * dr		# shift distance of ligand upon fission (zero when weightL=1, weightR=0)
    drR = weightL * dr		# shift distance of receptor upon fission
    Lpos = Lpos_bound + drL*orientation
    Rpos = Rpos_bound + -drR*orientation

    # Wrap trial positions in periodic box if necessary
    if periodic_box is not None:
        Lpos = np.where(Lpos > 0.5*periodic_box, Lpos-periodic_box,Lpos)
        Lpos = np.where(Lpos <= -0.5*periodic_box, Lpos+periodic_box,Lpos)
        Rpos = np.where(Rpos > 0.5*periodic_box, Rpos-periodic_box,Rpos)
        Rpos = np.where(Rpos <= -0.5*periodic_box, Rpos+periodic_box,Rpos)

    return (Lpos,Rpos)

#-----------------------------------------------------------------
def sample_dissociation_events_dimer(traj,potdict,rneighbor,R_react,weight_L,weight_S,n_samples, n_trials):

    timestep, types, ids, positions = traj.read_observable_particles()

    # Define distance function for periodic box
    dist = lambda x1,x2: util.dist(x1,x2,traj.box_size)


    lj_LL = potdict['lj_LL']
    lj_SL = potdict['lj_SL']
    #lj_SLL = potdict['lj_SLL']
    lj_CL = potdict['lj_CL']
    lj_CC = potdict['lj_CC']
    lj_CS = potdict['lj_CS']
    #lj_SLC = potdict['lj_SLC']


    npoints = 10000	# points for cumulative fission radii distribution
    fission_radii = np.linspace(1e-4,R_react,npoints)
    cumq = calc_cumulative_proposal_distribution(fission_radii,lj_SL)

    draw_product_pos = lambda pos: draw_new_product_positions_dimer(pos,n_trials,fission_radii,cumq,weightL=weight_L,weightR=weight_S,periodic_box=traj.box_size)
    
    # Loop over timesteps with sample stepsize/frequency
    Naccepted = 0
    Ntotal = 0
    stored_accepted_moves = []
    rvals = []

    sample_start = int(len(timestep)/5)							# timepoint to start analysis
    sample_step = max(1,int((len(timestep)-sample_start)/(n_samples/n_trials - 1)))	# timesteps between samples
    for i in range(sample_start,len(timestep),sample_step):
        # Get indices of each molecule type at timestep i
        idsi = np.array([traj.species_name(j) for j in types[i]])
        indsL = np.char.equal(idsi,"L")		# unbound ligands
        indsC = np.char.equal(idsi,"C")		# crowders
        indsS = np.char.equal(idsi,"Sb")	# bound receptor
        indsLb = np.char.equal(idsi,"Lb")	# bound ligand

        # Get positions of each molecule type at timestep i
        posLi = positions[i][indsL]
        posCi = positions[i][indsC]
        posSi = positions[i][indsS]
        posLbi = positions[i][indsLb]

        posSLi = [posSi,posLbi]		# list of positions of bound complex molecules

        # Calculate energy of bound state and build neighbor list
        rS_L = dist(posLi,posSi)
        neighborsS_L = np.nonzero(rS_L<rneighbor)[0]

        rLb_L = dist(posLi,posLbi)
        neighborsLb_L = np.nonzero(rLb_L<rneighbor)[0]

        rS_C = dist(posCi,posSi)
        neighborsS_C = np.nonzero(rS_C<rneighbor)[0]

        rLb_C = dist(posCi,posLbi)
        neighborsLb_C = np.nonzero(rLb_C<rneighbor)[0]

        rS_Lb = dist(posLbi,posSi)


        Ebound = 0
        if rS_L[neighborsS_L].size!=0:
            Ebound += np.sum(lj_SL(rS_L[neighborsS_L]))		# energy of bound receptor w/ free ligands
        if rS_C[neighborsS_C].size!=0:
            Ebound += np.sum(lj_CS(rS_C[neighborsS_C]))		# energy of bound receptor w/ crowders

        if rLb_L[neighborsLb_L].size!=0:
            Ebound += np.sum(lj_LL(rLb_L[neighborsLb_L]))	# energy of bound ligand w/ free ligands
        if rLb_C[neighborsLb_C].size!=0:
            Ebound += np.sum(lj_CL(rLb_C[neighborsLb_C]))	# energy of bound ligand w/ crowders

        Ebound += lj_SL(rS_Lb)					# energy of bound ligand w/ bound receptor

        # Choose positions for products
        Ltestpos, Stestpos = draw_product_pos(posSLi)
        rvals.append(dist(Ltestpos,Stestpos))

        # Loop over N trials (n_trials should always be 1 in this case)
        Etest = np.zeros((n_trials,1))
        for j in range(n_trials):

            Enew = 0.	# Energy of proposed unbound state, excluding L-S test particle interactions (DB scheme)
            # Calc energy of test particles with other neighboring ligands
            if rS_L[neighborsS_L].size!=0:
                rSi = dist(posLi[neighborsS_L,:],Stestpos[j])
                EtestS_L = np.sum(lj_SL(rSi))	# energy between test sensor and other ligands
                Enew += EtestS_L
            if rLb_L[neighborsLb_L].size!=0:
                rLi = dist(posLi[neighborsLb_L,:],Ltestpos[j])
                EtestL_L = np.sum(lj_LL(rLi))	# energy between test ligand and other ligands
                Enew += EtestL_L

            # Calc energy of test particles with other neighboring crowders
            if rS_C[neighborsS_C].size!=0:
                rSi = dist(posCi[neighborsS_C,:],Stestpos[j])
                EtestS_C = np.sum(lj_CS(rSi))	# energy between test sensor and crowders
                Enew += EtestS_C
            if rLb_C[neighborsLb_C].size!=0:
                rLi = dist(posCi[neighborsLb_C,:],Ltestpos[j])
                EtestL_C = np.sum(lj_CL(rLi))	# energy between test ligand and crowders
                Enew += EtestL_C

            # Calc change in internal energy of dissociation for each test fission event
            dE = Enew - Ebound

            # For each test accept or reject move based on Metropolis Criteria
            accept = 0
            if dE<0:
                accept = 1
            elif np.exp(-dE) > np.random.uniform():
                accept = 1
                
            # Tally accepted vs rejected dissociation attempts
            Naccepted = Naccepted + accept
            Ntotal = Ntotal + 1

            # Store timestep, energy change, and test particle position for each successful dissociation
            if accept:
                stored_accepted_moves.append([i, dE, Ltestpos[j][0],  Ltestpos[j][1], Ltestpos[j][2], Stestpos[j][0],  Stestpos[j][1], Stestpos[j][2]])

    Paccept = Naccepted/Ntotal

    stored_accepted_moves = np.array(stored_accepted_moves)

    total_time = len(timestep)-sample_start


    return stored_accepted_moves, Naccepted, Ntotal, Paccept, total_time, rvals

###########################################################################################
#### Tangent Dimer Fussion Events
###########################################################################################
#-----------------------------------------------------------------
def calc_accept_dimer(trajfile,potdict,r_react,weight_L,nL,nLtag):

    print("Processing trajectory: {}".format(trajfile))

    # Load trajectory
    try:
        traj = readdy.Trajectory(trajfile)
    except (OSError, ValueError) as e:
        print("OSError while loading trajectory for index {}".format(config_index))
        return None

    timestep, types, ids, positions = traj.read_observable_particles()
    # Define distance function for periodic box
    dist = lambda x1,x2,boxsize: util.dist(x1,x2,boxsize)
 
    lj_LL = potdict['lj_LL']
    lj_SL = potdict['lj_SL']
    #lj_SLL = potdict['lj_SLL']
    lj_CL = potdict['lj_CL']
    lj_CC = potdict['lj_CC']
    lj_CS = potdict['lj_CS']
    #lj_SLC = potdict['lj_SLC']

    tstart = 0
    tstop = timestep.shape[0]
    sample_freq = 1
    time_indices = range(tstart,tstop,sample_freq)
    react_prob = np.zeros((len(time_indices),))

    # Choose which ligands to label as ligands vs crowders, if applicable
    ids0 = np.array([traj.species_name(j) for j in types[0]])
    if nL!=nLtag:	# NEEDS FIXING

        indsAllL = np.char.equal(ids0,"L")
        indsR = np.char.equal(ids0,"R")

        posAllL = positions[0][indsAllL]
        posR = positions[0][indsR]


        dLR = util.dist(posAllL,posR,traj.box_size)
        closestLigand = np.argmin(dLR)

        choose_prob = np.ones(indsAllL.shape)
        choose_prob[indsR] = 0
        choose_prob[closestLigand] = 0
        if nLtag != 1:
            indsLbulk = np.random.choice(indsAllL.shape[0],size=nL-1,replace=False,p=choose_prob/np.sum(choose_prob))
   
        indsL = np.full(indsAllL.shape,False)
        if nLtag != 1:
            indsL[indsLbulk] = True
        indsL[closestLigand] = True

        indsC = np.full(indsAllL.shape,True)
        indsC[indsL] = False
        indsC[indsR] = False
    else:
        indsL = np.char.equal(ids0,"L")
        indsC = np.char.equal(ids0,"C")
        indsS = np.char.equal(ids0,"S")
        posLi = positions[0][indsL]
        posCi = positions[0][indsC]
        posSi = positions[0][indsS]


    # Loop over timepoints with step size sample_freq
    for ti,i in enumerate(time_indices):
        posLi = positions[i][indsL]
        posSi = positions[i][indsS]
        posCi = positions[i][indsC]

        # Get distances from receptor to all ligands along trajectory (w/ periodic boundaries)
        dLS = util.dist(posLi,posSi,traj.box_size)

        # Get distances from receptor to all crowders along trajectory (w/ periodic boundaries)
        dSC = util.dist(posCi,posSi,traj.box_size)

        # Calculate internal energy of receptor (in unbound state)
        Ereceptor = np.sum(lj_SL(dLS)) + np.sum(lj_CS(dSC))

        # Calculate reaction propensity
        react_propensity = doi_reaction_model(dLS,r_react)
        react_candidates = np.nonzero(react_propensity)[0]
        
        # Test binding event
        accept = 0
        p_no_react_i = 1.

        #for j in react_candidates:
        if len(react_candidates)>0:
            # Choose randomly from reaction candidates for test reaction
            j = random.choice(react_candidates)

            # Choose position for fussion reaction product accounting for periodic boundaries
            vecStoLi = util.wrapped_vector(posLi[j]-posSi,traj.box_size)
            rS_Li = np.linalg.norm(vecStoLi)
            dr = r_react - rS_Li

            drS = -weight_L * dr * vecStoLi/rS_Li
            drL = (1.-weight_L) * dr * vecStoLi/rS_Li

            posSb = util.wrapped_vector(posSi + drS,traj.box_size)
            posLb = util.wrapped_vector(posLi[j] + drL,traj.box_size)

            # Calculate needed intermolecular distances
            dLLj = util.dist(posLi[j],posLi[np.arange(len(posLi))!=j],traj.box_size)			# Ligand_j to other ligands
            dcrowderLj = util.dist(posLi[j],posCi,traj.box_size)					# Ligand_j to crowders
            dSbL = util.dist(posSb,posLi[np.arange(len(posLi))!=j],traj.box_size)		# Proposed bound receptor to other ligands
            dSbCrowder = util.dist(posSb,posCi,traj.box_size)					# Proposed bound receptor to crowders
            dLbL = util.dist(posLb,posLi[np.arange(len(posLi))!=j],traj.box_size)		# Proposed bound ligand to other ligands
            dLbCrowder = util.dist(posLb,posCi,traj.box_size)					# Proposed bound ligand to crowders


            ## Calculate energy of ligand_j before binding test move
            # Interaction energy between ligand_j and other ligands
            EligandjL = lj_LL(dLLj)

            # Interaction energy between ligand_j and crowders
            Eligandjcrowder = lj_CL(dcrowderLj)

            # Total interaction energy of ligand_j excluding interaction w/ receptor
            Eligandj = np.sum(EligandjL) + np.sum(Eligandjcrowder)

            # Interaction energy between ligand_j and receptor
            EligandjS = lj_SL(dLS[j])

            ## Calculate energy of test ligand after binding
            # Interaction energy of test particle with other ligands
            ELbL = lj_LL(dLbL)

            # Interaction energy of test particle with crowders
            ELbCrowder = lj_CL(dLbCrowder)

            # Total interaction energy of test ligand particle
            ELb = np.sum(ELbL) + np.sum(ELbCrowder)

            ## Calculate energy of test receptor after binding
            # Interaction energy of test particle with other ligands
            ESbL = lj_SL(dSbL)

            # Interaction energy of test particle with crowders
            ESbCrowder = lj_CS(dSbCrowder)

            # Total interaction energy of test receptor particle
            ESb = np.sum(ESbL) + np.sum(ESbCrowder)

            # Interaction energy between test ligand and test receptor (in the complex)
            ESbLb = lj_SL(r_react)

            # Total energy for the complex
            EComplex = ESb + ELb + ESbLb

            # Energy change for fussion reaction
            dE = EComplex - (Eligandj + Ereceptor)

            # Add back energy of fusion educt internal energy since accounted for by proposal density
            dE = dE + EligandjS

            if dE>0:
                p_no_react_i *= 1.-np.exp(-dE)
            else:
                p_no_react_i *= 0
            

        react_prob[ti] = 1. - p_no_react_i

    return react_prob


