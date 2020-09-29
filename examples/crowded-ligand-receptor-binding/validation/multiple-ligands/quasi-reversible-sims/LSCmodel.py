"""
Defines main model class for defining models for simulation using qrevIPRD.

Created by
Wylie Stroberg 2020-04-28
"""
import numpy as np
from qrevIPRD.potentials import lennard_jones
from qrevIPRD.reactions import calculate_reaction_probs, calculate_unbinding_moves

#############################################################################################################
class LSCModel:
    """ Class for defining models with lignand, a sensor, and crowders (L,S,C) for simulation using qrev. """

    def __init__(self,nL,nC,nLtag=None):
        self.nL = nL
        self.nC = nC
        if nLtag is None:
            self.nLtag = nL

        # Define simulation space
        self.boxsize = [5., 5., 5.]

        # Define particle properties

        self.r_L = 1.0
        self.r_C = 1.0
        self.r_S = 1.0
        self.r_SL = pow(pow(self.r_L,3.)+pow(self.r_S,3.),1./3.)			# conserves ligand-receptor volume

        self.kbT_over_sixPiEta = 1.0		# proportionality constant for stokes radius and diffusion coefficient
        self.D_L = self.kbT_over_sixPiEta/self.r_L		# diffusion coefficient of ligand
        self.D_C = self.kbT_over_sixPiEta/self.r_C		# diffusion coefficient of crowder
        self.D_S = self.kbT_over_sixPiEta/self.r_S		# diffusion coefficient of sensor
        self.D_SL = self.kbT_over_sixPiEta/self.r_SL

        self.dt = 1E-4
        bound_sim_sample_freq = 100.
        self.tau_mol = self.r_L**2./self.D_L / (self.dt*bound_sim_sample_freq)		# molecular timescale

        # Define test particle interactions
        T_reduced = 1.5
        self.eps = 1./T_reduced
        self.m = 12.
        self.n = 6.
        self.rmin = lambda sigma: sigma*pow(self.m/self.n, 1./(self.m-self.n))	# location of minimum of LJ potential (use for cut-shift point)

        
        self.sigma_C = 2.*self.r_C		# effective diamteter for crowder
        self.sigma_L = 2.*self.r_L		# effective diameter for ligand
        self.sigma_S = 2.*self.r_S		# effective diameter for receptor
        self.sigma_SL = 2.*self.r_SL		# effective diameter for receptor-ligand complex
        self.sigma_mix = lambda sig1,sig2: pow(sig1*sig2,0.5)	# geometric mean for interaction distance
        
        self.sigma_S_L = self.sigma_mix(self.sigma_S, self.sigma_L)
        self.sigma_SL_L = self.sigma_mix(self.sigma_SL, self.sigma_L)
        self.sigma_C_L = self.sigma_mix(self.sigma_C, self.sigma_L)
        self.sigma_C_S = self.sigma_mix(self.sigma_C, self.sigma_S)
        self.sigma_C_SL = self.sigma_mix(self.sigma_C, self.sigma_SL)

        self.rcut = max(self.rmin(self.sigma_L),self.rmin(self.sigma_C),
                        self.rmin(self.sigma_S),self.rmin(self.sigma_SL))	# max distance for potential calcs

        # Define dictionary of potentials for MC simulations 
        def lj_template(rij,sigma):
            return lennard_jones(rij, self.eps, self.m, self.n, sigma, self.rmin(sigma), shifted=True)
        
        self.potdict = {}
        
        self.potdict['lj_LL'] = lambda rij: lj_template(rij, self.sigma_L)
        self.potdict['lj_SL'] = lambda rij: lj_template(rij, self.sigma_S_L)
        self.potdict['lj_SLL'] = lambda rij: lj_template(rij, self.sigma_SL_L)
        self.potdict['lj_CL'] = lambda rij: lj_template(rij, self.sigma_C_L)
        self.potdict['lj_CC'] = lambda rij: lj_template(rij, self.sigma_C)
        self.potdict['lj_CS'] = lambda rij: lj_template(rij, self.sigma_C_S)
        self.potdict['lj_SLC'] = lambda rij: lj_template(rij, self.sigma_C_SL)
        
        
        self.rneighbor = self.rcut + max(self.r_L,self.r_C)
        
        # Define microscopic reaction parameters
        self.r_react = self.r_L + self.r_S	# distance for Doi reaction model
        self.weight_L = 0.5			# weight for placing complex between S and L reactants
        self.weight_S = 1. - self.weight_L	# weight for placing complex between S and L reactants

        # Set number of samples for dissociation MC sampling
        self.n_samples = 10000	# number of sample configurations
        self.n_trials = 1	# number of proposed position for each configuration

        return

    #--------------------------------------------------------------------
    def calc_dissociation_prob(self,trajfile,savefile):
        return calculate_unbinding_moves(trajfile, self.potdict, self.rneighbor, self.r_react, self.weight_L, self.weight_S, 
                                         savefile=savefile, tau_mol=self.tau_mol, n_samples=self.n_samples, n_trials=self.n_trials)

    #--------------------------------------------------------------------
    def calc_reaction_probs(self,rundir,trajfile,savefile,n_cores=1):
        return calculate_reaction_probs(rundir, self.potdict, self.r_react, self.weight_L, self.nL, self.nLtag,
                                        n_cores=n_cores, savefile=savefile, trajfile=trajfile)
    
    
