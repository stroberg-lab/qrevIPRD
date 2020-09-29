import readdy
import numpy as np
import os
import argparse
from random import sample
print(readdy.__version__)

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
def generate_random_sphere_packing(Np,box_size,dmin):
    # Add particles to simulation
    positions = np.random.uniform(size=(Np,3)) * (box_size.magnitude) - box_size.magnitude * 0.5

    # calc distance between particle centers
    subs = positions[:,None] - positions
    dist2 = np.einsum('ijk,ijk->ij',subs,subs)
    dmin2 = dmin*dmin
    for i in range(Np):
        for j in range(i+1,Np):
            if dist2[i,j] < dmin2:
                min_d2i = dist2[i,j]
                try_count = 0
                while min_d2i < dmin2:
                    # test new postision
                    vinew = np.random.uniform(size=(1,3)) * (box_size.magnitude) - box_size.magnitude * 0.5
                    d2inew = np.square(np.linalg.norm(positions - vinew,axis=1))
                    min_d2i = np.min(d2inew)
                    try_count = try_count+1
                    if try_count % 100 == 0:
                        print ("Try count for particle {} = {}".format(i,try_count))
                positions[i,:] = vinew
                dist2[i,:] = d2inew

    for i in range(dist2.shape[0]):
        try:
            mini = np.min(dist2[i,i+1:])
            if mini < dmin2:
                print(mini)
        except ValueError:
            pass

    return positions
#----------------------------------------------------------------------------------------------------------------
def generate_sphere_packing_cubic_random_fill(Nparticle,box_size,lattice_const):
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

                coordinates.append([xCoord,yCoord,zCoord])
    
    # Randomly fill Nparticle lattice sites
    particle_list = sample(coordinates,Nparticle)

    particles = np.array(particle_list)

    # Shift positions to center in simulation box
    for i,bsi in enumerate(box_size):
        shift = 0.5*bsi
        particles[:,i] = particles[:,i] - shift

    print(np.min(particles[:,0]),np.max(particles[:,0]))
    print(np.min(particles[:,1]),np.max(particles[:,1]))
    print(np.min(particles[:,2]),np.max(particles[:,2]))
    return particles

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
if __name__=="__main__":

    #-------------------------------------------------------------------------------------------------------
    #### Parse command line arguments ####
    #-------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument("--nLigand", "-nL", help="set ligand number")
    parser.add_argument("--nCrowder", "-nC", help="set crowder number")
    parser.add_argument("--nSensor", "-nS", help="set sensor/receptor number")
    parser.add_argument("--kOff", "-kOff", help="set microscopic off rate for reaction")
    parser.add_argument("--kOn", "-kOn", help="set microscopic on rate for reaction")
    parser.add_argument("--rundir","-rd", help="set run directory")
    parser.add_argument("--nSteps","-ns", help="set number of simulation steps")
    parser.add_argument("--nThreads","-nt", help="set number of threads for simulation")

    args = parser.parse_args()

    
    outfile = "./{}/LR_out_bulk.h5".format(args.rundir)

    n_L = int(args.nLigand)
    n_C = int(args.nCrowder)
    if args.nSensor:
        n_S = int(args.nSensor)
    else:
        n_S = 1

    if args.kOff:
        kOff = float(args.kOff)
    else:
        kOff = 100.

    if args.kOn:
        kOn = float(args.kOn)
    else:
        kOn = 100.

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

    # Define simulation box
    box_x = 5.
    box_y = 5.
    box_z = 5.
    #system = readdy.ReactionDiffusionSystem([box_x,box_y,box_z], temperature=300.*readdy.units.kelvin)
    system = readdy.ReactionDiffusionSystem([box_x,box_y,box_z], unit_system=None)
    system.periodic_boundary_conditions = [True, True, True] # non-periodic in z-direction

    # Define species types for simulation
    #system.add_species("C", 1.0)	# Crowding agent with diffusion constant 1.0
    system.add_species("L", 2.0)	# Ligand
    system.add_species("S", 0.0)	# Receptor
    system.add_species("SL",0.0)	# Ligand-receptor complex

    # Define reactions
    react_radius = 2.0
    #system.reactions.add("complex_formation: L +(2) S -> SL", rate=kOn, weight1=1., weight2=0.)
    #system.reactions.add("complex_dissociation: SL -> L +(2) S", rate=kOff, weight1=1., weight2=0.)
    system.reactions.add_fusion("complex_formation", "L", "S", "SL", kOn, react_radius, weight1=1., weight2=0.)
    system.reactions.add_fission("complex_dissociation", "SL", "L", "S", kOff, react_radius, weight1=1., weight2=0.)

    # Particle sizes (hard sphere)
    r_L = 1.0
    #r_C = 1.0
    r_S = 1.0
    #r_complex = pow(pow(r_L+r_C,3.)+pow(r_R+r_C,3.),1./3.)-r_C		# conserves excluded volume
    r_complex = pow(pow(r_L,3.)+pow(r_S,3.),1./3.)			# conserves ligand-receptor volume

    # Define pairwise potentials
    T_reduced = 1.5
    eps = 1./T_reduced
    sigmaHS = 1.	# pseudo-hard-sphere diameter
    m = 12.
    n = 6.
    #rmin = sigmaHS*pow(m/n, 1./(m-n))
    rmin = lambda sigma: sigma*pow(m/n, 1./(m-n))	# location of minimum of LJ potential (use for cut-shift point)
    #set_cutshifted_lj = lambda type1,type2: system.potentials.add_lennard_jones(type1, type2, 
    #                        m=int(m), n=int(n), epsilon=eps, sigma=sigmaHS, cutoff=rmin, shift=True)
    set_cutshifted_lj = lambda type1,type2, sigma: system.potentials.add_lennard_jones(type1, type2, 
                            m=int(m), n=int(n), epsilon=eps, sigma=sigma, cutoff=rmin(sigma), shift=True)
    #set_cutshifted_lj("C","C")
    #set_cutshifted_lj("C","L")
    #set_cutshifted_lj("C","S")
    #set_cutshifted_lj("C","SA")

    #set_cutshifted_lj("L","L")
    #set_cutshifted_lj("L","S")
    #set_cutshifted_lj("L","SA")

    sigma_L = 2.*r_L	# effective diameter for ligand
    sigma_S = 2.*r_S	# effective diameter for receptor
    sigma_SL = 2.*r_complex
    sigma_mix = lambda sig1,sig2: pow(sig1*sig2,0.5)	# geometric mean for interaction distance

    set_cutshifted_lj("L","L",sigma_L)
    set_cutshifted_lj("L","S",sigma_mix(sigma_L,sigma_S))
    set_cutshifted_lj("L","SL",sigma_mix(sigma_L,sigma_SL))


    #### Create a simulation ####
    if n_threads==1:
        simulation = system.simulation(kernel="SingleCPU")
    else:
        simulation = system.simulation(kernel="CPU")
        simulation.kernel_configuration.n_threads = n_threads

    simulation.output_file = outfile
    #simulation.reaction_handler = "Gillespie"
    simulation.reaction_handler = "DetailedBalance"


    #### Place particles in simulations box, avoiding overlaps ####
    buff = 0.05*sigmaHS
    dmin = sigmaHS+buff
    lattice_const = sigmaHS + buff

    Ntotal = n_C+n_L+n_S

    #available_positions = generate_random_sphere_packing(Ntotal,system.box_size,dmin)
    #available_positions = generate_sphere_packing_cubic_random_fill(Ntotal,system.box_size,lattice_const)
    #positions_C = available_positions[0:n_C,:]
    #positions_L = available_positions[n_C:n_C+n_L,:]
    #positions_S = np.zeros((1,3))
    #positions_S[0,:] = available_positions[n_C+n_L,:]
    positions_SL = np.zeros((1,3))
    
    #simulation.add_particles("C", positions = positions_C)
    #simulation.add_particles("L", positions = positions_L)
    #simulation.add_particles("S", positions = positions_S)
    simulation.add_particles("SL", positions = positions_SL)


    #### Define observables for simulation ####
    simulation.record_trajectory(stride=1000)
    #simulation.observe.number_of_particles(stride=1, types=["L","S","C"])
    simulation.observe.number_of_particles(stride=1, types=["L","S"])
    
    simulation.observe.particles(stride=100,callback=None)
    #simulation.observe.reaction_counts(stride=100)

    #simulation.observe.reactions(stride=100)

    if os.path.exists(simulation.output_file):
        os.remove(simulation.output_file)


    #-------------------------------------------------------------------------------------------------------
    #### Run simulation ####
    #-------------------------------------------------------------------------------------------------------
    simulation.run(n_steps=n_steps, timestep=1e-4)

    #### Save trajectory ####
    trajectory = readdy.Trajectory(outfile)
    #trajectory.convert_to_xyz(particle_radii={'C': 1.,'L': 1.,'S': 1.,'SA': 1.})
    trajectory.convert_to_xyz(particle_radii={'L': 1.,'S': 1.,'SL': 1.})


