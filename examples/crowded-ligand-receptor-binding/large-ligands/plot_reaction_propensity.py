import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

params = {"text.usetex": True,'text.latex.preamble' : [ r'\usepackage{mathrsfs}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)

import glob
from joblib import Parallel, delayed

from scipy.signal import savgol_filter
from statsmodels.tsa.stattools import acf
from scipy.integrate import simps

import qrevIPRD.analysis as qan
import LSCmodel
import time as pytime
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
if __name__=="__main__":

    nL = 10
    nCs = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    #nCs = [0, 40, 80]
    traj_number = 0

    dt = 1e-5

    zipped = True

    react_event_density = []
    mean_excess_prob = []
    mean_excess_energy = []
    excess_dissoc_prob = []

    def parfunc(nCi):
    #for nCi in nCs:

        print("Processing data from nC = {}".format(nCi))
        basedir = "./rundir/run_bulk_nL{}_nC{}/trajectory_{}".format(nL,nCi,traj_number)
        dissocdatafile = basedir+"/accepted_dissociation_moves.txt"
        unbounddata_template = basedir+"/unbound_simulations_fine_output/unbound_reaction_event_density_nL_{}_*.npy".format(nL)
        unbound_data_zipfile = basedir+"/unbound_output.zip"
 
        # Read dissociation prob from accepted moves file header
        with open(dissocdatafile, 'r') as f:
            header = f.readline()
        split_header = header.split()
        excess_dissoc_prob = float(split_header[8])


        # Load reaction probability data for each timepoint for each trajectory
        if zipped:
            unbound_data_files = np.load(unbound_data_zipfile)
        else:
            unbound_data_files = []
            for datai in glob.glob(unbounddata_template):
                unbound_data_files.append(datai)

        # Compute mean reaction propensity
        mean_ex = []
        mean_dE = []
        in_reaction_zone = []
        count = 0
        for i,datai in enumerate(unbound_data_files):
            if zipped:
                react_ed_i = np.array(unbound_data_files[datai])
            else:
                react_ed_i = np.load(datai)


            if react_ed_i.dtype=='float64':
                possible_event_i = np.where(react_ed_i>0.,1.,0.)
                delta_i = possible_event_i[0] - react_ed_i

                if count==0:
                    react_ed = react_ed_i
                    possible_event = possible_event_i[0]
                    delta = delta_i
                else:
                    react_ed = react_ed + react_ed_i
                    possible_event = possible_event + possible_event_i[0]
                    delta = delta + delta_i


                count += 1
                excess_prob = react_ed_i[possible_event_i.astype(bool)]

                in_reaction_zone.append(react_ed_i[possible_event_i.astype(bool)].shape[0]/react_ed_i.shape[1])
                mean_ex.append(np.mean(excess_prob))
                dE = np.log(excess_prob)
                mean_dE.append(np.mean(dE))

            else:
                print(datai)

        react_ed *= 1./count
        possible_event *= 1./count
        delta *= 1./count

        return (react_ed[0,:], np.mean(np.array(mean_ex)), np.mean(np.array(mean_dE)), excess_dissoc_prob, np.mean(np.array(in_reaction_zone)), possible_event, delta )

    n_proc = 8
    out = Parallel(n_jobs=n_proc)(delayed(parfunc)(nCi) for nCi in nCs)

    react_event_density = [x[0] for x in out] 
    mean_excess_prob = [x[1] for x in out] 
    mean_excess_energy = [x[2] for x in out] 
    excess_dissoc_prob = [x[3] for x in out] 
    in_RZ_prob = [x[4] for x in out] 
    possible_event = [x[5] for x in out] 
    delta = [x[6] for x in out] 

    # Long-time limits of reaction propensities
    afcs = []
    print("Long-time limits of reaction propensity:")
    for nCi, redi, pexi in zip(nCs,react_event_density,mean_excess_prob):
        acfi = acf(redi[:100000],unbiased=True,fft=True)
        taui = 1./acfi[0] * simps(acfi,dx=dt)


        print("\t nC = {}, <R> = {:.4f}, tau_R = {:.5f}, <e^-dE> = {:.2}".format(nCi,np.mean(redi[-10000::1]),taui,pexi))


    # Histogram the relative reaction probabilities
    figH, axH = plt.subplots(1,1)

    for deltai,pos_evi in zip(delta,possible_event):
        #print(deltai.flatten().shape)
        
        hist = axH.hist(deltai[0,-50000:]/pos_evi[-50000:],bins=1000,density=True,alpha=0.5) 

    axH.set_ylim((0.,4000))
    #----------------------------------------------------------
    # Calculate Volume Fraction
    sigmaC = 1.
    sigmaL = 1.
    sigmaR = 1.
    vC = 1./6.*np.pi*pow(sigmaC,3.)
    vL = 1./6.*np.pi*pow(sigmaL,3.)
    vR = 1./6.*np.pi*pow(sigmaR,3.)
    nR = 1.
    V = 5.*5.*5.
    phi = [(nCi*vC + nL*vL + nR*vR)/V for nCi in nCs]

    #----------------------------------------------------------
    # Write data to files for easy plotting
    outfile_react_prob = "excess_reaction_probs.txt"
    header_react_prob = "phi pex_assoc pex_dissoc"
    data_react_prob = np.vstack((np.array(phi),np.array(mean_excess_prob),np.array(excess_dissoc_prob))).T
    np.savetxt(outfile_react_prob,data_react_prob,header=header_react_prob)

    outfile_contact_prob = "contact_prob.txt"
    header_contact_prob = "phi p_in_react_zone"
    data_contact_prob = np.vstack((np.array(phi),np.array(in_RZ_prob))).T
    np.savetxt(outfile_contact_prob,data_contact_prob,header=header_contact_prob)

    #----------------------------------------------------------
    # Plotting

    fig, ax = plt.subplots(1,1,figsize=(6,5))
    fig2, ax2 = plt.subplots(1,2,figsize=(10,5))

    # Create an inset in the lower right corner (loc=4) with borderpad=1, i.e.
    # 10 points padding (as 10pt is the default fontsize) to the parent axes
    axins = inset_axes(ax, width="100%", height="100%", 
                                bbox_to_anchor=(.45, .25, .4, .4),
                                bbox_transform=ax.transAxes, loc=3, borderpad=1)
    # Turn ticklabels of insets off
    #for axi in axins:
    #    axi.tick_params(labelleft=False, labelbottom=False)

    for i,(phii,redi,pos_evi,deltai) in enumerate(zip(phi,react_event_density,possible_event,delta)):

        tfinal = 100000
        data = redi[:tfinal]
        time = dt*np.array(range(data.shape[0]))
        #data = savgol_filter(data, 101, 3) # window size 101, polynomial order 3
        ax.plot(time,data,label="$\phi={:.2f}$".format(phii))
        shift = 0.025
        axins.plot(time[-int(0.5*time.shape[0]+1):],data[-int(0.5*data.shape[0]+1):]+shift*i)

        ax2[0].plot(time,pos_evi[:tfinal],label="$\phi={:.2f}$".format(phii))

        #ax[2].plot(time,(pos_evi[:30000] - data)/pos_evi[:30000],label="$\phi={:.2f}$".format(phii))
        ax2[1].plot(time,deltai[0,:tfinal]/pos_evi[:tfinal],label="$\phi={:.2f}$".format(phii))

    ax.set_xlabel(r'Time',fontsize=18)
    ax.set_ylabel(r'$\langle f_{i}(t|\sigma)\rangle$',fontsize=18)
    ax.legend(loc=9)

    #ax.set_xscale('log')
    #ax.set_yscale('log')


    ax2[0].set_xlabel(r'Time',fontsize=18)
    ax2[0].set_ylabel(r'Possible Reaction',fontsize=18)
    ax2[0].legend()

    ax2[0].set_xscale('log')
    ax2[0].set_yscale('log')

    ax2[1].set_xlabel(r'Time',fontsize=18)
    ax2[1].set_ylabel(r'Possible Reaction$-R(t|\sigma)/$Possible Reaction',fontsize=14)
    ax2[1].legend()  
    ax2[1].set_yscale('log')

    fig.tight_layout()
    fig2.tight_layout()


    #-----------------------------------------------------------------
    figEx, axEx = plt.subplots(1,1)

    extended_phi = [0] + phi
    extended_dissoc_prob = [1] + excess_dissoc_prob
    extended_excess_prob = [1] + mean_excess_prob

    p1 = axEx.plot(extended_phi,extended_dissoc_prob,'s--',fillstyle='none')
    p2 = axEx.plot(extended_phi,extended_excess_prob,'o--',fillstyle='none')

    axEx.plot(phi,excess_dissoc_prob,'s-',color=p1[-1].get_color(),label=r'Dissociation')
    axEx.plot(phi,mean_excess_prob,'o-',color=p2[-1].get_color(),label=r'Association')

    

    axEx.set_xlabel(r'$\phi$',fontsize=18)
    axEx.set_ylabel(r'$\langle e^{-\beta \Delta E_{ex}}\rangle$',fontsize=18)
    axEx.legend()

    figEx.tight_layout()

    #-----------------------------------------------------------------
    figRZ, axRZ = plt.subplots(1,1)

    axRZ.plot(phi,in_RZ_prob,'o-')

    axRZ.set_xlabel(r'$\phi$',fontsize=18)
    axRZ.set_ylabel(r'$p_{in\ react\ zone}$',fontsize=18)
    axRZ.legend()

    figRZ.tight_layout()

    #----------------------------------------------------------------
    # Save figures
    fig.savefig('./Figures/Event_density.pdf')
    figEx.savefig('./Figures/ExcessTransitionRates.pdf')


    plt.show()
   
    
