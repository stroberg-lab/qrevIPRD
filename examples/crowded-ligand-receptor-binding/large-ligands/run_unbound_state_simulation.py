import readdy
import numpy as np
import os
import argparse
from random import sample

import LSCmodel
print(readdy.__version__)
#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
if __name__=="__main__":

    #### Parse command line arguments ####
    parser = argparse.ArgumentParser()

    parser.add_argument("--nLigand", "-nL", help="set ligand number")
    parser.add_argument("--nCrowder", "-nC", help="set crowder number")
    parser.add_argument("--rundir","-rd", help="set run directory")
    parser.add_argument("--outdir","-od", help="set output directory")
    parser.add_argument("--nSteps","-ns", help="set number of simulation steps")
    parser.add_argument("--nThreads","-nt", help="set number of threads for simulation")
    parser.add_argument("--orientation","-oi", help="index of orientation to use as initial condiation")
    parser.add_argument("--nTraj", "-ntraj", help="set trajectory number from dissociation simulations")
    parser.add_argument("--savexyz", "-xyz", help="save trajectory in .xyz format in addition to Readdy format")

    args = parser.parse_args()

    nL = int(args.nLigand)	# number of crowders/ligand (same size) to add to box
    config_index = int(args.orientation)	# index in orientation file to use as initial condition

    if args.nCrowder:
        nC = int(args.nCrowder)	# number of ligand to add to box
    else:
        nC = 0

    if args.nSteps:
        n_steps = int(args.nSteps)
    else:
        n_steps = 1000000

    if args.nThreads:
        n_threads = int(args.nThreads)
    else:
        n_threads = 1

    if args.nTraj:
        traj_number = int(args.nTraj)
    else:
        traj_number = 0

    if args.rundir:
        rundir = args.rundir
    else:
        rundir = "./boxsize_10_10_10/run_bulk_nL{}_nC{}/trajectory_{}/".format(nL,nC,traj_number)

    if not os.path.exists(rundir):
        os.makedirs(rundir)

    if args.outdir:
        outdir = args.outdir
    else:
        outdir = rundir + "unbound_simulations/"

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outfile = "{}/LigandDiffusion_unbound_out_bulk_index_{}.h5".format(outdir,config_index)

    if args.savexyz:
        save_xyz = bool(args.savexyz)
    else:
        save_xyz = False

    #-----------------------------------------------------------------------------------
    #### Load trajectory and L-R configuration files ####
    #-----------------------------------------------------------------------------------
    # Load trajectory data
    traj = readdy.Trajectory(rundir+'LigandDiffusion_out_bulk.h5')
    timestep, types, ids, positions = traj.read_observable_particles()

    # Load LR config file
    LRconfigfile = rundir + "accepted_dissociation_moves.txt"
    with open(LRconfigfile, 'r') as f:
        header = f.readline()
    split_header = header.split()


    # Get particle positions for initial condition
    orientation_data = np.loadtxt(LRconfigfile,skiprows=config_index+1,max_rows=1)
    config_time_ind = int(orientation_data[0])
    dE = orientation_data[1]
    pos_dissocL = orientation_data[2:5].reshape((1,3))
    pos_dissocS = orientation_data[5:].reshape((1,3))


    idsi = np.array([traj.species_name(j) for j in types[config_time_ind]])

    indsL = np.char.equal(idsi,"L")
    posLi = positions[config_time_ind][indsL]
    indsC = np.char.equal(idsi,"C")
    posCi = positions[config_time_ind][indsC] 

    #-----------------------------------------------------------------------------------
    #### Set up a reaction-diffusion system ####
    #-----------------------------------------------------------------------------------

    # Create LSCModel object to get parameters
    model = LSCmodel.LSCModel(nL,nC)

    # Define simulation box
    system = readdy.ReactionDiffusionSystem(model.boxsize, unit_system=None)
    system.periodic_boundary_conditions = [True, True, True] # non-periodic in z-direction

    # Define species types for simulation
    system.add_species("C", model.D_C)	# Crowding agent with diffusion constant D_C
    system.add_species("L", model.D_L)	# Ligand
    system.add_species("S",model.D_S)	# Ligand-receptor complex

    # Define pairwise potentials
    set_cutshifted_lj = lambda type1,type2, sigma: system.potentials.add_lennard_jones(type1, type2, 
                            m=int(model.m), n=int(model.n), epsilon=model.eps, sigma=sigma, cutoff=model.rmin(sigma), shift=True)

    set_cutshifted_lj("L","L",model.sigma_L)
    set_cutshifted_lj("L","S",model.sigma_S_L)
    set_cutshifted_lj("L","C",model.sigma_C_L)
    set_cutshifted_lj("C","C",model.sigma_C)
    set_cutshifted_lj("C","S",model.sigma_C_S)
    set_cutshifted_lj("S","S",model.sigma_S)

    #### Create a simulation ####
    kernel = "CPU"
    if n_threads==1:
        kernel = "SingleCPU"
    simulation = system.simulation(kernel=kernel)
    simulation.kernel_configuration.n_threads = n_threads

    simulation.output_file = outfile
    simulation.reaction_handler = "Gillespie"

    # Add particles to simulation  
    simulation.add_particles("L", positions = posLi)
    simulation.add_particles("L", positions = pos_dissocL)
    simulation.add_particles("S", positions = pos_dissocS)
    simulation.add_particles("C", positions = posCi)

    # Observables
    simulation.record_trajectory(stride=100,chunk_size=1000)
   
    simulation.observe.particles(stride=1,callback=None)

    simulation.observe.pressure(stride=1000,physical_particles=["L"],callback=None)

    # Clear existing files from previous simulations (Careful!)
    if os.path.exists(simulation.output_file):
        os.remove(simulation.output_file)


    #-----------------------------------------------------------------------------------
    #### Run simulation ####
    #-----------------------------------------------------------------------------------
    simulation.run(n_steps=n_steps, timestep=model.dt)

    # Save trajectory
    trajectory = readdy.Trajectory(outfile)
    if save_xyz == True:
        if nL!=0:
            if nC!=0:
                trajectory.convert_to_xyz(particle_radii={'L': 0.5*model.sigma_L,'C' : 0.5*model.sigma_C,'S' : 0.5*model.sigma_S})
            else:
                trajectory.convert_to_xyz(particle_radii={'L': 0.5*model.sigma_L,'S' : 0.5*model.sigma_S})
        else:
            trajectory.convert_to_xyz(particle_radii={'S' : 0.5*model.sigma_S})
