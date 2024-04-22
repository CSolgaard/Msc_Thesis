from numba import njit
# from scipy.linalg import expm, sqrtm
import numpy as np
import copy
# import scienceplots
from pathlib import Path 
import pandas as pd
import time
# from scipy import interpolate
from dataclasses import dataclass

import warnings
# To ignore all warnings
warnings.filterwarnings("ignore")


def RTS_smoother(rts, *args, **kwargs):
    """
     rtssmoother - Perform Kalman smoothing of a navigation profile according
     to the method of Rauch, Tung and Striebel, as presented in Brown and
     Hwang, section 6.2. This algorithm expects input from a forward run of
     an error state Kalman filter with closed loop corrections.
    
     -------------------------------------------------------------------------
    
     Input:
       rts             MATLAB structure with the following sub-structures:
    
       .profile        nx(p+1) array representing the navigation profile from
                       the Kalman filter. The first column is assumed to be
                       time stamps. The remaining p parameters are the
                       navigation states, whoseerrors are modelled in the
                       Kalman filter.
       .x_posterior    nxp array representing the estimated errors (state
                       vector) at each epoch after the measurement update. The
                       state vector before the update will always be zero,
                       since the Kalman filter algorithm is assumed to use a
                       closed loop implementation.
       .P_prior        nx(p*p) array representing the error covariance
                       matrices prior to the measurement update. Every nx(p*p)
                       vector is "reshaped" into a pxp matrix using MATLABs
                       reshape command.*
       .P_posterior    nx(p*p) array representing the error covariance
                       matrices after the measurement update. Every nx(p*p)
                       vector is "reshaped" into a pxp matrix using MATLABs
                       reshape command.*
       .Transition     nx(p*p) array representing the transition matrices at
                       each epoch. Notice that the transition matrix is used
                       for propagating the state estimates at the previous
                       epoch to the current epoch. Every pxp matrix is
                       re-constructed from the corresponding 1x(p*p) vector
                       using using MATLABs reshape command.*
    
       *The original matrices can be recovered using MATLABs reshape command,
       as for example:
    
                           P = reshape(rts.P_prior(epoch,:),[p,p])


        optional inputs: 
        echo:         string: True or False, for displaying progress bar in terminal 
                      default is "True"

                        
     Output:
       rts             MATLAB structure with the following sub-structures
                       added or modified:
    
       .profile        nx(p+1) array containing the smoothed navigation
                       profile
       .profile_std    nxp array with standard deviation on state parameters
    
     -------------------------------------------------------------------------
     Author: Christian Solgaard (DTU) 18/10-2023 based on the work done by
     Tim Jensen (DTU) 13/10/2023
    """



    # Respond to variable input
    lecho = True

    # Loop through variable input
    for i in range(len(args)):
        if args[i].lower() == 'echo':
            if args[i + 1].lower() == 'off':
                lecho = False

    # Respond to input
    no_epochs = len(rts['time'])
    npar = rts['x_posterior'].shape[1]

    # Allocate space
    rts['profile_std'] = np.full((no_epochs, npar), np.nan)
    rts['P'] = np.full((no_epochs, npar * npar), np.nan)

    # Define Scaling Parameters --------------------------------------------
    S = np.ones([npar])
    Sinv = np.ones([npar])

    # Get index of diagonal elements
    id_diag = np.empty(npar, dtype=int)
    for i in range(npar):
        id_diag[i] = i + (i * npar)

    # Extract the relevant columns from rts["profile"]
    profile_values = copy.deepcopy(rts["P_prior"][:, id_diag])

    # Find the minimum and maximum values in each column
    min_values = np.nanmin(profile_values, axis=0)
    max_values = np.nanmax(profile_values, axis=0)

    # Determine the order of magnitude
    ordmin = np.log10(np.sqrt(min_values))
    ordmax = np.log10(np.sqrt(max_values))
    ordmag = np.round(0.5 * (ordmin + ordmax))

    # Change some scaling factors
    for i in range(npar): 
        S[i] = 10**(-ordmag[i])
        Sinv[i] = 10**(ordmag[i])
    # S = S*10**(-ordmag)
    # Sinv = Sinv*10**ordmag

    # Form scaling matrices
    S = np.diag(S)
    Sinv = np.diag(Sinv)

    # Initialise progress bar
    if lecho:
        blank = '                    '
        dots = '....................'
        rewind = '\b' * 21

        # Initialise progress bar
        tic = time.time()
        print(f"RTS smoother ({no_epochs} epochs):")
        print(f"0%    [ {blank}]", end='')

        # Initialise counters
        progress_inc = no_epochs / 20
        progress_mark = 0
        progress_epoch = no_epochs

    # Perform RTS smoothing
    xk_smooth = rts['x_posterior'][-1, :].reshape(-1, 1)
    Pk_smooth = np.reshape(rts['P_posterior'][-1, :], (npar, npar), order='F')

    rts['P'][-1, :] = Pk_smooth.flatten(order='F')
    rts['profile_std'][-1, :] = np.sqrt(np.diag(Pk_smooth))

    xk_smooth = np.dot(S, xk_smooth)
    Pk_smooth = np.dot(np.dot(S, Pk_smooth), S)

    # Perform backward sweep
    for epoch in range(no_epochs - 2, -1, -1):
        xkp1_smooth = xk_smooth
        Pkp1_smooth = Pk_smooth

        xk_plus = rts['x_posterior'][epoch, :].reshape(-1, 1)
        Pk_plus = np.reshape(rts['P_posterior'][epoch, :], (npar, npar), order='F')
        Pkp1_minus = np.reshape(rts['P_prior'][epoch + 1, :], (npar, npar), order='F')

        xk_plus = np.dot(S, xk_plus)
        Pk_plus = np.dot(np.dot(S, Pk_plus), S)
        Pkp1_minus = np.dot(np.dot(S, Pkp1_minus), S)

        Phi = np.reshape(rts['Transition'][epoch + 1, :], (npar, npar), order='F')
        Phi = np.dot(np.dot(S, Phi), np.linalg.inv(S))

        A = np.dot(np.dot(Pk_plus, Phi.T), np.linalg.inv(Pkp1_minus))
        xk_smooth = xk_plus + np.dot(A, xkp1_smooth)
        Pk_smooth = Pk_plus + np.dot(np.dot(A, Pkp1_smooth - Pkp1_minus), A.T)

        delta_x = xk_smooth - xk_plus
        delta_x = np.dot(Sinv, delta_x)
        rts['profile'][epoch, :] = rts['profile'][epoch, :] + delta_x.flatten()

        rts['P'][epoch, :] = np.dot(np.dot(Sinv, Pk_smooth), Sinv).flatten(order='F')
        rts['profile_std'][epoch, :] = np.sqrt(np.diag(Pk_smooth))

        # Update Progress Bar
        if lecho and (progress_epoch - epoch) > progress_inc:
            progress_mark += 1
            progress_percent = 100 * progress_mark / 20
            progress_epoch = epoch
            bspace = ' ' * (5 - len(str(progress_percent)))
            print(f"{rewind}{progress_percent:.0f}%    [ {dots[:progress_mark]}{blank[:20 - progress_mark]}]", end='')

    # Complete progress bar
    if lecho:
        print(f"{rewind}100%  [ {dots} ] done")
        toc = time.time() - tic
        print(f"Time elapsed: {toc:.2f} seconds")

    return rts