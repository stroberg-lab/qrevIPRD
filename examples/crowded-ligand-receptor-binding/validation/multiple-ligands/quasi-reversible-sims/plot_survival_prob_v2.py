import numpy as np
import matplotlib.pyplot as plt
import readdy
from scipy import interpolate
import glob
import time as timepy
#----------------------------------------------------------------------------
def laplace_transform_trap(s,f,t):
    dt = np.diff(t)
    fe = f*np.exp(-s*t)
    F = np.sum(0.5*dt*(fe[0:-1]+fe[1:]))
    return F

#----------------------------------------------------------------------------
def trap_integrate(f,t):
    dt = np.diff(t)
    F = np.sum(dt*0.5*(f[0:-1]+f[1:]))
    return F

#----------------------------------------------------------------------------
def calc_survival_prob(react_prob,time,kao):
    # Calculate survival probability for contact pair
    surv_prob_ij = np.zeros(react_prob.shape)
    start = timepy.time()
    for i in range(react_prob.shape[0]):

        log_surv = np.zeros((react_prob.shape[1],))
        dt = np.diff(time)
        
        dlog_surv = -kao * dt * 0.5 * (react_prob[i,0:-1] + react_prob[i,1:])
        log_surv[1:] = np.cumsum(dlog_surv)

        surv_prob_ij[i,:] = np.exp(log_surv)
    surv_prob = np.mean(surv_prob_ij,axis=0)

    return surv_prob

#----------------------------------------------------------------------------
def calc_survival_prob_v3(react_prob,time,kao):
    # Calculate survival probability for contact pair
    surv_prob_ij = np.zeros(react_prob.shape)
    start = timepy.time()
    for i in range(react_prob.shape[0]):

        log_surv = np.zeros((react_prob.shape[1],))
        dt = np.diff(time)
        #preact = (1.-np.exp(-kao*dt)) * 0.5 * (react_prob[i,0:-1] + react_prob[i,1:])
        preact = (1.-np.exp(-kao*dt)) * react_prob[i,1:]
        #preact = (1.-np.exp(-kao*dt)) * react_prob[i,0:-1]
        dlog_surv = np.log( 1. - preact)
        log_surv[1:] = np.cumsum(dlog_surv)

        surv_prob_ij[i,:] = np.exp(log_surv)
    surv_prob = np.mean(surv_prob_ij,axis=0)

    return surv_prob

#----------------------------------------------------------------------------
def calc_separation_prob(surv_prob,time,kdo,dissoc_prob):
    # Calculate separation probability for bound pair
    kd = kdo*dissoc_prob
    sep_prob = np.zeros(surv_prob.shape)
    dt = np.diff(time)
    surv_prob_func = interpolate.interp1d(time,surv_prob)
    for i in range(1,sep_prob.shape[0]):
        fj = kd * (1.-sep_prob[:i]) * surv_prob_func(time[i]-time[:i])
        sep_prob[i] = 1./(1. + 0.5*dt[i-1]*kd*surv_prob_func(time[0])) * (0.5*dt[i-1]*(kd*surv_prob_func(time[0]) + fj[-1]) + np.sum(0.5*dt[0:i-1]*(fj[0:-1]+fj[1:])) )

    return sep_prob
#----------------------------------------------------------------------------
def kabs_laplace(s,sigma,D):
    return 4.*np.pi*sigma*D*(1. + sigma*np.sqrt(s/D))/s

def krad_laplace(s,sigma,D,ka):
    kabs = kabs_laplace(s,sigma,D)
    return ka*kabs/(ka + s*kabs)

def Srad_eq_laplace(s,sigma,D,ka,c):
    return 1./(s*(1.+c*krad_laplace(s,sigma,D,ka)))

def Srad_sig_laplace(s,sigma,D,ka,c):
    return (1.-s*Srad_eq_laplace(s,sigma,D,ka,c))/(c*ka)

def Srev_star_laplace(s,sigma,D,ka,c,kd):
    cKeq = c*ka/kd
    Srad_eq = Srad_eq_laplace(s,sigma,D,ka,c)
    return 1./s * (1. - cKeq/(1. + cKeq - s*Srad_eq))

def Cn_laplace(s,sigma,D,ka,c,n_mean):
    n_var = n_mean * (1. - n_mean)
    Srad_eq = Srad_eq_laplace(s,sigma,D,ka,c)
    return n_var * n_mean * Srad_eq / (1. - (1.-n_mean) * s * Srad_eq)

def Cn_approx2_laplace(s,sigma,D,ka,c,n_mean):
    tau = sigma*np.sqrt(s/D)
    tau_c = 1./(ka*c + kd)
    kD = 4.*np.pi*sigma*D
    tau_c_prime = tau_c * (1. + ka/(kD*(1.+tau)))
    n_var = n_mean * (1. - n_mean)
    return n_var * tau_c_prime / (s*tau_c_prime + 1.)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
if __name__=="__main__":

    nLtag = 2
    nL = nLtag
    nC = 1
    traj_number = 0
    box_size = 5
    #basedir = "./boxsize_10_10_10/run_bulk_nL{}/trajectory_{}".format(nLtag,traj_number,nL)
    basedir = "./boxsize_{}_{}_{}/run_bulk_nL{}_nC{}/trajectory_{}".format(box_size,box_size,box_size,nLtag,nC,traj_number)
    dissocdatafile = basedir+"/accepted_dissociation_moves.txt"

    #unbounddatafile = basedir+"/unbound_reaction_event_density_nL_1_tstart_0_tstop_10001_freq_1.npy"
    #unbounddatafile = basedir+"/unbound_reaction_event_density_nL_1_tstart_0_tstop_100001_freq_1.npy"
    unbounddata_template = basedir+"/unbound_simulations_fine_output/unbound_reaction_event_density_nL_{}_*.npy".format(nLtag)


    # Load reaction probability data for each timepoint for each trajectory
    unbound_data_files = []
    print(unbounddata_template)
    for datai in glob.glob(unbounddata_template):
        unbound_data_files.append(datai)
    
    # Read dissociation prob from accepted moves file header
    with open(dissocdatafile, 'r') as f:
        header = f.readline()
    split_header = header.split()
    dissoc_prob = float(split_header[6])

    # Set intrinsic reaction rates
    kao = 1.0e+2
    kdo = 1.0e+0

    # set discrete time step and length of time for calculations
    simulation_timestep = 1e-4
    simulation_sample_rate = 1.	# simulation particle positions sampled every x timesteps
    dt = simulation_timestep * simulation_sample_rate

    #print(react_prob.shape)
    #time_sim = dt * np.array(range(react_prob.shape[1]))
    #time = time_sim


    # Calculate equilibrium constant and mean occupancy
    #V = 10.*10.*10.
    V = box_size*box_size*box_size
    vL = 4./3.*np.pi*pow(2.,3)
    vR = 4./3.*np.pi*pow(2.,3)
    Veff = V - nL*vL-vR
    c = nL / Veff
    Kd = kdo*dissoc_prob/kao

    sigma = 2.	# molecular size (cross section)
    D = 2.	# relative diffusion coefficient

    # Calculate survival time distribution for a bound complex
    kd = kdo * dissoc_prob				# effective dissociation rate constant w/ crowding
    complex_surv_prob = lambda t: np.exp(-kd * t)	# poisson distribution for zero events in time t
    mfpt_unbind = 1./kd					# mean first passage time for complex to unbind
    print(dissoc_prob)

    # Calculate survival probability for contact pair
    #surv_prob = calc_survival_prob(react_prob,time,kao)
    #surv_prob = calc_survival_prob_v3(react_prob,time,kao)
    surv_probs = []
    for datai in unbound_data_files:
        react_probi = np.load(datai)
        time = dt * np.array(range(react_probi.shape[1]))
        surv_probs.append(calc_survival_prob_v3(react_probi,time,kao))

    surv_probs = np.array(surv_probs)
    print(surv_probs.shape)
    surv_prob = np.mean(surv_probs,axis=0)
    print(surv_prob.shape)
        


    # Calculate mean first passage time for a binding to occur
    mfpt_bind = trap_integrate(surv_prob,time)

    # Calculate point statistics for binary switching process
    n_mean = mfpt_unbind/(mfpt_bind + mfpt_unbind)
    n_var = n_mean * (1. - n_mean)

    ka = kao#kd*n_mean/(c*(1-n_mean))
    print(n_mean/(1.-n_mean), c*(1./Kd),ka,c/(c+Kd))

    # Calculate separation probability for bound pair
    sep_prob = calc_separation_prob(surv_prob,time,kdo,dissoc_prob)

    # Calculate correlation function from separation probability
    pstar_star = 1. - sep_prob
    p0 = n_mean
    n_varC = p0 * (pstar_star[0] - p0)
    corr_func = p0 * (pstar_star - p0)

    # Calculate Laplace transform of survival prob
    s = np.linspace(1e-3,10,10000)
    Fsurv = np.array([laplace_transform_trap(si,surv_prob,time) for si in s])
    Fsurv_analytic = np.array([Srad_sig_laplace(si,sigma,D,ka,c) for si in s])

    # Calculate Laplace transform of seperation prob
    Fsep = 1./s * kd * Fsurv/(1. + kd*Fsurv)
    Fsep_analytic = 1./s * kd * Fsurv_analytic/(1. + kd*Fsurv_analytic)
    Fsep_analytic_v2 = Srev_star_laplace(s,sigma,D,ka,c,kd)

    # Calculate Laplace transform of correlation function
    ps_star_star = 1./s - Fsep
    ps_star_star_analytic = 1./s - Fsep_analytic

    Cs = p0 * (ps_star_star - p0/s)
    Cs1 = p0 * (ps_star_star_analytic - p0/s)
    Cs_analytic = Cn_laplace(s,sigma,D,ka,c,n_mean)

    # Calculate power spectrum as fourier transform of correlation function
    P = np.fft.fft(corr_func)
    freq = np.fft.fftfreq(time.shape[-1])
    tau = P[0].real/(2.*n_var)

    # Calculate correlation time directly from definition in Kaizu 2014 supp. equation S8
    tau_n = 1./n_var * trap_integrate(corr_func,time)
    tau_analytic_cn = 1./n_var * Cs_analytic[0]

    kD = 4.*np.pi*sigma*D
    tau_analytic = (ka+kD)/((kd+ka*c)*kD)
    kon = ka*kD/(ka+kD)
    koff = kd*kD/(ka+kD)
    print(ka*c,kon*c,koff)
    print("***********************************")
    print("Effective Free Volume Fraction = {:.2f}".format(Veff/V))
    print("Effective Concentration = {:.6f}".format(c))
    print("Mean First Passage Times: \n\t Binding = {:.4f} \n\t Unbinding = {:.4f}".format(mfpt_bind,mfpt_unbind))
    print("Mean occupancy and variance = {:.4f}, {:.4f}".format(n_mean,n_var))
    print("p_*|* [0...-1] = {:.4f}...{:.4f}".format(pstar_star[0],pstar_star[-1]))
    print(("Correlation time: \n\t Simulation = {}"
                             "\n\t From theoretical Cn(t) = {}"
                             "\n\t Analytical = {}").format(tau_n, tau_analytic_cn, tau_analytic))
    print("***********************************")

    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    # Plot reaction probability and survival probability vs time
    figR, axR = plt.subplots(1,3,figsize=(12,5))
    react_prob0 = np.load(unbound_data_files[0])
    axR[0].plot(time,react_prob0[0,:],'-')

    axR[0].set_xlabel(r'Time',fontsize=18)
    axR[0].set_ylabel(r'$R(t|\sigma)$',fontsize=18)
    #axR[0].set_xscale("log")
    #axR[0].set_yscale("log")

    axR[1].plot(time,surv_prob,'-')

    axR[1].set_xlabel(r'Time',fontsize=18)
    axR[1].set_ylabel(r'$\mathcal{S}_{rad}(t|\sigma)$',fontsize=18)
    #axR[1].set_xscale("log")
    #axR[1].set_yscale("log")

    axR[2].plot(time,np.log(surv_prob)/c,'-')

    axR[2].set_xlabel(r'Time',fontsize=18)
    axR[2].set_ylabel(r'$\frac{\ln(\mathcal{S}_{rad}(t|\sigma))}{c}$',fontsize=18)


    figR.tight_layout()

    #---------------------------------------------------------------------------------------------
    # Plot seperation probability and autocorrelation function
    figS, axS = plt.subplots(1,2,figsize=(10,5))
 
    axS[0].plot(time,sep_prob,'-')
    axS[1].plot(time,corr_func/n_var,'-')

    axS[0].set_xlabel(r'Time',fontsize=18)
    axS[0].set_ylabel(r'$\mathcal{S}_{rev}(t|\star)$',fontsize=18)

    axS[1].set_xlabel(r'Time',fontsize=18)
    axS[1].set_ylabel(r'$C(\tau)$',fontsize=18)

    figS.tight_layout()

    #---------------------------------------------------------------------------------------------
    # Plot laplace transform of survival probability
    figF, axF = plt.subplots(1,3,figsize=(14,5))
 
    axF[0].plot(s,Fsurv,'-',label="simulation")
    axF[0].plot(s,Fsurv_analytic,'-.',label="approx")

    axF[0].set_xlabel(r's',fontsize=18)
    axF[0].set_ylabel(r'$\hat{\mathcal{S}}_{rad}(s|\sigma)$',fontsize=18)
    axF[0].legend()
    axF[0].set_xlim((0,3))


    # Plot laplace transform of seperation probability
    axF[1].plot(s,Fsep,'-',label="simulation")
    axF[1].plot(s,Fsep_analytic,'-.',label="approx")
    axF[1].plot(s,Fsep_analytic_v2,'--',label="approx v2")

    axF[1].set_xlabel(r's',fontsize=18)
    axF[1].set_ylabel(r'$\hat{\mathcal{S}}_{rev}(s|\star)$',fontsize=18)
    axF[1].legend()
    axF[1].set_xlim((0,1))


    # Plot laplace transform of correlation function
    axF[2].plot(s,Cs,'-',label="simulation")
    axF[2].plot(s,Cs1,'--',label="approx, analytic surv prob")
    axF[2].plot(s,Cs_analytic,'-.',label="approx")

    axF[2].set_xlabel(r's',fontsize=18)
    axF[2].set_ylabel(r'$\hat{C}_{n}(s)$',fontsize=18)
    axF[2].legend()
    axF[2].set_xlim((0,1))


    figF.tight_layout()

    #---------------------------------------------------------------------------------------------
    # Plot power specturm
    figP, axP = plt.subplots(1,1)

    axP.plot(freq,P.real)

    axP.set_xlabel(r'$\omega$',fontsize=18)
    axP.set_ylabel(r'$P(\omega)$',fontsize=18)

    figF.tight_layout()

    #figF.savefig("./Figures/correlation_laplace_nLtag_{}_traj_{}_nL_{}.eps".format(nLtag,traj_number,nL))

    #---------------------------------------------------------------------------------------------
    # Plot heat map of reaction probabilities for each trajectory
    if 0:
        figMat, axMat = plt.subplots(1,1)
        matrixplot = axMat.imshow(react_prob[::1,:100:1],aspect='auto')
        figMat.colorbar(matrixplot)
        plt.show()


    plt.show()
