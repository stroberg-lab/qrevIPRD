import numpy as np
import matplotlib.pyplot as plt
import readdy
from scipy import interpolate
from scipy.integrate import trapz, cumtrapz, quadrature
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
def extrapolated_reaction_probability(react_prob,datatime,newtime):
    fss = np.mean(react_prob[:,-1000:])
    extrap_react_prob = interpolate.interp1d(datatime,react_prob,kind='linear',axis=-1,bounds_error=False,fill_value=(0.,fss))
    return extrap_react_prob(newtime)

#----------------------------------------------------------------------------
def resampled_reaction_probability(react_prob,copy_length,n_copies):
    sample_copy = react_prob[:,-copy_length:]

    extrap_react_prob = np.zeros((sample_copy.shape[0],n_copies*sample_copy.shape[1]))
    for i in range(n_copies):
        extrap_react_prob[:,i*copy_length:(i+1)*copy_length] = np.random.permutation(sample_copy)

    extrap_react_prob = np.array(extrap_react_prob)#np.tile(sample_copy,n_copies)
    new_react_prob = np.hstack((react_prob,extrap_react_prob))

    return new_react_prob

#----------------------------------------------------------------------------
def calc_survival_prob_v3(react_prob,time,kao):
    # Calculate survival probability for contact pair

    log_surv = np.zeros(react_prob.shape)
    dt = np.diff(time)[0]
    preact = (1.-np.exp(-kao*dt)) * react_prob[:,1:]	# Use if diffusion step is taken before reaction step

    dlog_surv = np.log( 1. - preact)
    log_surv[:,1:] = np.cumsum(dlog_surv,axis=1)
    surv_prob_ij = np.exp(log_surv)

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
        sep_prob[i] = ( 1./(1. + 0.5*dt[i-1]*kd*surv_prob_func(time[0])) 
              * (0.5*dt[i-1]*(kd*surv_prob_func(time[0]) + fj[-1]) + np.sum(0.5*dt[0:i-1]*(fj[0:-1]+fj[1:])) ) )

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
# Define model for fitting of exponential function
def exp_model(t,theta0,theta1):
    return theta0 * np.exp(theta1 * t)

#----------------------------------------------------------------------------
# Fit long-time limit of survival data to model
def fit_survival_prob_nonlinear(t,surv,npoints=40000):
    ##### Extrapolate survival prob w/ nonlinear regression model #####

    # Choose range for fitting
    surv_data = surv[-npoints:]
    time_data = t[-npoints:]

    # Set initial guess for parameters based on log-transform linear regression
    lin_fit = np.polyfit(time_data,np.log(surv_data),1)

    exponent_estimate = lin_fit[0]
    prefactor_estimate = pow(10.,lin_fit[1])

    # Fit model to data
    p0 = [prefactor_estimate, exponent_estimate]
    popt, pcov = curve_fit(exp_model,time_data,surv_data,p0)

    return (popt, pcov)


#----------------------------------------------------------------------------
def fit_and_extrapolate_surv_prob(tdata,sdata,extrij,target_surv_prob=1e-4,npoints=40000):

    popt_ij, pcov_ij = fit_survival_prob_nonlinear(tdata,sdata,npoints)

    # Extrapolate survival data using nonlinear fit

    final_time = tdata[-1] + (np.log(target_surv_prob) - np.log(sdata[-1]))/popt_ij[1]

    dt = tdata[1]-tdata[0]
    extrap_time_ij = np.linspace(tdata[-1]+dt,final_time,10000)
    extrap_surv_ij = exp_model(extrap_time_ij,popt_ij[0],popt_ij[1])

    return popt_ij, extrap_time_ij, extrap_surv_ij

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def process_quasireversible_simulations(kao,kdo,dt,dissocdatafile,unbound_data_files,coarsening=1,**kwargs):

    # Parse kwargs
    if 'zipped' in kwargs:
        zipped = kwargs['zipped']
    else:
        zipped = False

    if 'target_surv_prob' in kwargs:
        target_surv_prob = kwargs['target_surv_prob']
    else:
        target_surv_prob = 1.0

    if 'nsteps' in kwargs:
        nsteps = kwargs['nsteps']
    else:
        nsteps = None

    if 'ntraj' in kwargs: # max number of trajectories to process
        ntraj = kwargs['ntraj']
    else:
        ntraj = None

    if 'npoints' in kwargs: # number of data points to use in extrapolation fit for survival prob
        npoints = kwargs['npoints']
    else:
        npoints = 40000

    # Read dissociation prob from accepted moves file header
    with open(dissocdatafile, 'r') as f:
        header = f.readline()
    split_header = header.split()
    dissoc_prob = float(split_header[8])

    
    # Calculate survival time distribution for a bound complex
    kd = kdo * dissoc_prob				# effective dissociation rate constant w/ crowding
    mfpt_unbind = 1./kd					# mean first passage time for complex to unbind

    # Calculate survival probability for contact pair

    # Load zipped .npz dict-like object of arrays
    if zipped:
       unbound_data_files = np.load(unbound_data_files)

    # Choose subset of trajectories to process, if necessary
    if ntraj is not None:
        if isinstance(ntraj,tuple):
            data_range = range(ntraj[0],ntraj[1])
        else:
            data_range = range(ntraj)
    else:
        data_range = range(len(unbound_data_files))

    counter = 0
    for i,datai in enumerate(unbound_data_files):
        if i in data_range:
            try:
                if zipped:
                    react_probi = np.array(unbound_data_files[datai])
                else:
                    react_probi = np.load(datai)
                if nsteps is not None:
                    react_probi = react_probi[:,:nsteps]
    
                datatime = dt * np.array(range(react_probi.shape[1]))
    
                if counter == 0:
                    surv_prob = calc_survival_prob_v3(react_probi,datatime,kao)
                else:
                    surv_prob += calc_survival_prob_v3(react_probi,datatime,kao)
                counter += 1
            except Exception as e:
                print("Unable to load: {}".format(datai))
                print(e)
        

    # Calculate mean survival probability
    surv_prob = surv_prob/counter

    # Extrapolate if surv_prob is greater than low val
    if surv_prob[-1] > target_surv_prob:
        '''
        fit_window = [-10000,-1] 
        pfit = np.polyfit(datatime[fit_window[0]:fit_window[-1]:10],np.log(surv_prob[fit_window[0]:fit_window[-1]:10]),1) 
        
        final_time = datatime[-1] + (np.log(target_surv_prob) - np.log(surv_prob[-1]))/pfit[0]
        print("Final time after extrapolation = {}".format(final_time))

        extrap_time = np.linspace(datatime[-1]+dt,final_time,100000)
        log_extrap_surv = np.polyval(pfit,extrap_time)

        surv_prob = np.hstack((surv_prob,np.exp(log_extrap_surv)))
        time = np.hstack((datatime,extrap_time))
        extrap_surv = np.exp(log_extrap_surv)
        '''
        extrap_time, extrap_surv = fit_and_extrapolate_surv_prob(datatime,surv_prob,target_surv_prob,npoints)

        surv_prob = np.hstack((surv_prob,extrap_surv))
        time = np.hstack((datatime,extrap_time))
        
    else:
        extrap_surv = surv_prob
        time = datatime

    # Calculate mean first passage time for a binding to occur
    mfpt_bind = trap_integrate(surv_prob,time)


    # Calculate point statistics for binary switching process
    n_mean = mfpt_unbind/(mfpt_bind + mfpt_unbind)
    n_var = n_mean * (1. - n_mean)

    # Calculate separation probability for bound pair
    sep_prob = calc_separation_prob(surv_prob[::coarsening],time[::coarsening],kdo,dissoc_prob)

    # Calculate correlation function from separation probability
    pstar_star = 1. - sep_prob
    p0 = n_mean
    n_varC = p0 * (pstar_star[0] - p0)
    corr_func = p0 * (pstar_star - p0)

    # Calculate correlation time directly from definition in Kaizu 2014 supp. equation S8
    tau_n = 1./n_var * trap_integrate(corr_func,time[::coarsening])

    out_dict = {'time':time,
                'mfpt_bind':mfpt_bind,
                'mfpt_unbind':mfpt_unbind,
                'surv_prob':surv_prob,
                'sep_prob':sep_prob,
                'corr_func':corr_func,
                'tau_n':tau_n,
                'n_mean':n_mean,
                'n_var':n_var,
                'extrap_surv':extrap_surv,
                'coarsening':coarsening}

    #return (kd, time, surv_prob, sep_prob, corr_func, tau_n, n_mean, n_var, coarsening)
    return out_dict

#----------------------------------------------------------------------------
def process_reversible_data_v2(datadirs,**kwargs):

    if 'ntraj' in kwargs:
        ntraj = kwargs['ntraj']
    else:
        ntraj = None

    n_mean_r = []
    n_var_r = []
    surv_data = []
    logsurv_data = []
    event_density_data = []

    for di in datadirs:
        # Read dissociation prob from accepted moves file header
        if ntraj is None:
            with open(di+'fluctuation_vs_tau_data.txt', 'r') as f:
                header = f.readline()
                split_header = header.split()
                n_mean_r.append(float(split_header[4]))
                n_var_r.append(float(split_header[6]))
            fluct_data = np.loadtxt(di+'fluctuation_vs_tau_data.txt',skiprows=1)

            event_density = np.loadtxt(di+'survival_hist.txt')
            log_event_density = np.loadtxt(di+'survival_hist_logrithmic.txt')

        if isinstance(ntraj,tuple):
            ntraj_tag = "ntraj_{}_{}".format(ntraj[1]-ntraj[0],ntraj[0])
            with open(di+'fluctuation_vs_tau_data_'+ntraj_tag+'.txt', 'r') as f:
                header = f.readline()
                split_header = header.split()
                n_mean_r.append(float(split_header[4]))
                n_var_r.append(float(split_header[6]))
            fluct_data = np.loadtxt(di+'fluctuation_vs_tau_data_'+ntraj_tag+'.txt',skiprows=1)

            event_density = np.loadtxt(di+'survival_hist_'+ntraj_tag+'.txt')
            log_event_density = np.loadtxt(di+'survival_hist_logrithmic_'+ntraj_tag+'.txt'.format(ntraj))

        else:
            with open(di+'fluctuation_vs_tau_data_ntraj_{}.txt'.format(ntraj), 'r') as f:
                header = f.readline()
                split_header = header.split()
                n_mean_r.append(float(split_header[4]))
                n_var_r.append(float(split_header[6]))
            fluct_data = np.loadtxt(di+'fluctuation_vs_tau_data_ntraj_{}.txt'.format(ntraj),skiprows=1)

            event_density = np.loadtxt(di+'survival_hist_ntraj_{}.txt'.format(ntraj))
            log_event_density = np.loadtxt(di+'survival_hist_logrithmic_ntraj_{}.txt'.format(ntraj))

        # Normalize distributions to sum to 1
        event_density[:,2] = event_density[:,2]/np.sum(event_density[:,2])

        surv = np.ones(event_density.shape)
        surv[:,0:2] = event_density[:,0:2]

        surv[:,2] = 1. - np.cumsum(event_density[:,2])

        surv_data.append(surv)
        event_density_data.append(event_density)

        logbinsurv = np.ones(log_event_density.shape)

        logbinsurv[:,0:2] = log_event_density[:,0:2]

        log_event_density[:,2] = log_event_density[:,2]/np.sum(log_event_density[:,2])

        logbinsurv[:,2] = 1. - np.cumsum(log_event_density[:,2])
        logsurv_data.append(logbinsurv)


    nmean_rev = []
    for si,lsi in zip(surv_data,logsurv_data):
        mfpt_bind_l = np.sum(lsi[:,2]*lsi[:,1])
        mfpt_bind = np.sum(si[:,2]*si[:,1])
        mfpt_unbind = 1.
        nmean_rev.append(mfpt_unbind/(mfpt_bind_l + mfpt_unbind))

    return [n_mean_r, n_var_r, surv_data, logsurv_data, event_density_data, nmean_rev]

#----------------------------------------------------------------------------
def calculate_lma_occupancy(micro_onrate,micro_offrate,V,Vex,Vreact):
    Kon = micro_onrate * Vreact/(V-Vex)
    Koff = micro_offrate
    n_occ = 1./ (1. + Koff/Kon)
    return n_occ

#-----------------------------------------------------------------
def calc_eff_volume(pot,R):
    integrand = lambda r: 4.*np.pi*r**2.*np.exp(-pot(r))
    testr = np.linspace(1.e-4,R,10000)
    lower_bound = np.argwhere(integrand(testr)>0.)[0][0]-100
    V_eff, err = quadrature(integrand,testr[lower_bound],R,tol=1.49e-8)
    return V_eff

#-----------------------------------------------------------------
def calc_Vex(pot,rcut):
    Vint = 4./3.*np.pi*pow(rcut,3.)
    Vint_eff = calc_eff_volume(pot,rcut)
    return Vint - Vint_eff

