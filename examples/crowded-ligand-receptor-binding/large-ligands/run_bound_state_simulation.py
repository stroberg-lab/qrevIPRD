import readdy
import numpy as np
import os
import argparse
from random import sample

import LSCmodel

#print(readdy.__version__)
#----------------------------------------------------------------------------------------------------------------
def generate_random_sphere_packing(Np,box_size):
    # Add particles to simulation
    positions_L = np.random.uniform(size=(Np,3)) * (box_size.magnitude) - box_size.magnitude * 0.5

    # calc distance between particle centers
    subs = positions_L[:,None] - positions_L
    dist2 = np.einsum('ijk,ijk->ij',subs,subs)
    buff = 1.15*sigmaHS
    dmin2 = (sigmaHS+buff)**2.

    for i in range(Np):
        for j in range(i+1,Np):
            if dist2[i,j] < dmin2:
                min_d2i = dist2[i,j]
                try_count = 0
                while min_d2i < dmin2:
                    # test new postision
                    vinew = np.random.uniform(size=(1,3)) * (box_size.magnitude) - box_size.magnitude * 0.5
                    d2inew = np.square(np.linalg.norm(positions_L - vinew,axis=1))
                    min_d2i = np.min(d2inew)
                    try_count = try_count+1
                    if try_count % 100 == 0:
                        print ("Try count for particle {} = {}".format(i,try_count))
                positions_L[i,:] = vinew
                dist2[i,:] = d2inew

    for i in range(dist2.shape[0]):
        try:
            mini = np.min(dist2[i,i+1:])
            if mini < dmin2:
                print(mini)
        except ValueError:
            pass

    return positions_L
#----------------------------------------------------------------------------------------------------------------
def generate_sphere_packing_cubic_random_fill(Nparticle,box_size,lattice_const,avoided_particle_list=[]):
    # Determine number of lattice points in each direction
    xDim = int(box_size[0] / lattice_const)
    yDim = int(box_size[1] / lattice_const)
    zDim = int(box_size[2] / lattice_const)

    total_site = xDim*yDim*zDim
    print(total_site)
    if total_site<Nparticle:
        print("Not enough lattice sites to accomodate all particles!")
    offset = lattice_const/2.
    coordinates = []

    # Generate set of cubic lattice points in box
    for x in range(0,xDim):
        for y in range(0,yDim):
            for z in range(0,zDim):
                xCoord = x * lattice_const + offset
                yCoord = y * lattice_const + offset
                zCoord = z * lattice_const + offset
                overlap = False
                for pi in avoided_particle_list:
                    if not overlap:
                        posi = pi[0]
                        rpi = pi[3]
                        dist2 = (xCoord-posi[0])**2. +  (yCoord-posi[1])**2. + (zCoord-posi[2])**2.
                        if dist2 < center_particle_radius**2.:
                            overlap = True
                if not overlap:
                    coordinates.append([xCoord,yCoord,zCoord])
                else:
                    total_site = total_site - 1

    if total_site<Nparticle:
        print("Not enough lattice sites to accomodate all particles!")

    # Randomly fill Nparticle lattice sites
    particle_list = sample(coordinates,Nparticle)

    particle_L = np.array(particle_list)

    # Shift positions to center in simulation box
    for i,bsi in enumerate(box_size):
        shift = 0.5*bsi
        particle_L[:,i] = particle_L[:,i] - shift

    return particle_L

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
if __name__=="__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--nLigand", "-nL", help="set ligand number")
    parser.add_argument("--nCrowder", "-nC", help="set crowder number")
    parser.add_argument("--rundir","-rd", help="set run directory")
    parser.add_argument("--nSteps","-ns", help="set number of simulation steps")
    parser.add_argument("--nThreads","-nt", help="set number of threads for simulation")

    args = parser.parse_args()

    n_Ltotal = int(args.nLigand)	# number of total ligands including bound ligand
    n_L = int(args.nLigand)-1	# number of ligand to add to box
    n_SL = 1			# number of bound complexes to add to box

    if args.nCrowder:
        n_C = int(args.nCrowder)	# number of ligand to add to box
    else:
        n_C = 0

    if args.rundir:
        rundir = args.rundir
    else:
        rundir = "./"

    if not os.path.exists(rundir):
        os.makedirs(rundir)
    outfile = "{}/LigandDiffusion_out_bulk.h5".format(args.rundir)

    if args.nSteps:
        n_steps = int(args.nSteps)
    else:
        n_steps = 1000000

    if args.nThreads:
        n_threads = int(args.nThreads)
    else:
        n_threads = 1


    #-------------------------------------------------------------------------------------------------------
    #### Set up a reaction-diffusion system and simulate ####
    #-------------------------------------------------------------------------------------------------------
    # Create LSCModel object to get parameters
    model = LSCmodel.LSCModel(n_Ltotal,n_C)

    # Define simulation box
    system = readdy.ReactionDiffusionSystem(model.boxsize, unit_system=None)
    system.periodic_boundary_conditions = [True, True, True] # non-periodic in z-direction

    # Define species types for simulation
    system.add_species("C", model.D_C)	# Crowding agent with diffusion constant D_C
    system.add_species("L", model.D_L)	# Ligand
    system.add_species("SL",model.D_SL)	# Ligand-receptor complex

    # Define pairwise potentials
    set_cutshifted_lj = lambda type1,type2, sigma: system.potentials.add_lennard_jones(type1, type2, 
                            m=int(model.m), n=int(model.n), epsilon=model.eps, sigma=sigma, cutoff=model.rmin(sigma), shift=True)

    set_cutshifted_lj("L","L",model.sigma_L)
    set_cutshifted_lj("L","SL",model.sigma_SL_L)
    set_cutshifted_lj("L","C",model.sigma_C_L)
    set_cutshifted_lj("C","C",model.sigma_C)
    set_cutshifted_lj("C","SL",model.sigma_C_SL)
    set_cutshifted_lj("SL","SL",model.sigma_SL)

    
    #### Create a simulation ####
    kernel = "CPU"
    if n_threads==1:
        kernel = "SingleCPU"
    simulation = system.simulation(kernel=kernel)
    simulation.kernel_configuration.n_threads = n_threads

    simulation.output_file = outfile
    simulation.reaction_handler = "Gillespie"


    #### Place particles in simulations box, avoiding overlaps ####
    buff = 0.05#*model.sigma_L
    lattice_const = 0.85*model.sigma_L*(1. + buff)

    Ntotal = n_C+n_L+n_SL

    #available_positions = generate_random_sphere_packing(Ntotal,system.box_size,dmin)
    available_positions = generate_sphere_packing_cubic_random_fill(Ntotal,system.box_size,lattice_const)
    positions_C = available_positions[0:n_C,:]
    positions_L = available_positions[n_C:n_C+n_L,:]
    positions_SL = np.array(available_positions[n_C+n_L,:]).reshape(1,3)
    
    simulation.add_particles("C", positions = positions_C)
    simulation.add_particles("L", positions = positions_L)
    simulation.add_particles("SL", positions = positions_SL)


    # Observables
    simulation.record_trajectory(stride=100,chunk_size=1000)
    
    simulation.observe.particles(stride=100,callback=None)

    if n_L!=0:
        simulation.observe.rdf(stride=1000,
                           bin_borders = np.linspace(1.,5.,40),
                           types_count_from=["SL"],
                           types_count_to=["L"],
                           particle_to_density=1./system.box_volume,
                           callback=None)

        simulation.observe.pressure(stride=1000,physical_particles=["L"],callback=None)

    # Clear existing files from previous simulations (Careful!)
    if os.path.exists(simulation.output_file):
        os.remove(simulation.output_file)

    #### Run simulation ####
    simulation.run(n_steps=n_steps, timestep=model.dt)

    # Save trajectory
    trajectory = readdy.Trajectory(outfile)
    if n_L!=0:
        if n_C!=0:
            trajectory.convert_to_xyz(particle_radii={'L': 0.5*model.sigma_L,'C' : 0.5*model.sigma_C,'SL' : 0.5*model.sigma_SL})
        else:
            trajectory.convert_to_xyz(particle_radii={'L': 0.5*model.sigma_L,'SL' : 0.5*model.sigma_SL})
    else:
        trajectory.convert_to_xyz(particle_radii={'SL' : 0.5*model.sigma_SL})


