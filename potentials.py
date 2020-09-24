"""
Defines intermolecular potentials for qrevIPRD

Created by
Wylie Stroberg 2020-04-28
"""
import numpy as np

###########################################################################################

#-----------------------------------------------------------------
def screened_electrostatic(rij,C,kappa,D,sigma,n,rc):
    if rij < rc:
        return C/rij * np.exp(-kappa*rij) + D * pow(sigma/rij,n)
    else:
        return 0.

#-----------------------------------------------------------------
def lj(rij,eps,m,n,sigma):
    sigr = sigma/rij
    sigrm = pow(sigr,m)
    sigrn = pow(sigr,n)
        
    rmin =pow(m/n, 1./(m-n)) * sigma
    k = -eps / ( pow(sigma/rmin,m) - pow(sigma/rmin,n) )

    V_LJ = k*(sigrm - sigrn)

    return V_LJ

#-----------------------------------------------------------------
def lennard_jones(rij,eps,m,n,sigma,rc,shifted=True):
    if np.ndim(rij)==0:
        V_LJ = 0.
        if rij<rc:
            V_LJ = lj(rij,eps,m,n,sigma)
            if shifted:
                V_LJ_rc = lj(rc,eps,m,n,sigma)
                V_LJ = V_LJ - V_LJ_rc

    else:
        if shifted:
            V_LJ_rc = lj(rc,eps,m,n,sigma)
            V_LJ = np.where(rij>rc,0.,lj(rij,eps,m,n,sigma)-V_LJ_rc)

        else:
            V_LJ = np.where(rij>rc,0.,lj(rij,eps,m,n,sigma))

    return V_LJ


