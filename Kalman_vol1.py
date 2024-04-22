# from scipy.linalg import expm
# import numpy as np

# def valLoan(F, PSD, tau): 
#     """
#     Implementation of Van Loan method for forming transition matrix and
#     system noise matrix from system matrix and power spectral density matrix
    
#     -------------------------------------------------------------------------
    
#     Input:
#       F       NxN array representing system matrx
#       PSD     NxN array representing matrix of power spectral densities
#       tau     scalar representing propagation time interval [s]
    
#     Output:
#       Phi     NxN array representing transition matrix
#       Q       NxN array representing system noise matrix
    
#     -------------------------------------------------------------------------
#     Author: Christian Solgaard (DTU), implemented on the work done by: 
#     Tim Jensen (DTU) 21/05/2018
#     """

#     # Get dimension 
#     nvar = len(F)

#     # Form index values
#     id1 = nvar
#     id2 = nvar * 2

#     # Derive Transition and System Noise Matrices
#     # Form A matrix using Brown and Hwang (3.9.22)
#     A = np.zeros([6,6])
#     A[0:3, 0:3] = -F * tau
#     A[0:3, 3:] = PSD * abs(tau)
#     A[3:, 0:3] = np.zeros([nvar, nvar])
#     A[3:, 3:] = F.T * tau

#     B = expm(A)    

#     # The lower-right partition is the transpose of the transition matrix
#     Phi = B[id1:id2, id1:id2].T

#     # Compute Q using Brown and Hwang (3.9.25)
#     Q = Phi * B[0:nvar, id1:id2]
    
#     return Phi, Q






# def rts_smoother(*args): 
#     """
#      rtssmoother - Perform Kalman smoothing of a navigation profile according
#      to the method of Rauch, Tung and Striebel, as presented in Brown and
#      Hwang, section 6.2. This algorithm expects input from a forward run of
#      an error state Kalman filter with closed loop corrections.
    
#      -------------------------------------------------------------------------
    
#      Input:
#        rts             MATLAB structure with the following sub-structures:
    
#        .profile        nx(p+1) array representing the navigation profile from
#                        the Kalman filter. The first column is assumed to be
#                        time stamps. The remaining p parameters are the
#                        navigation states, whoseerrors are modelled in the
#                        Kalman filter.
#        .x_posterior    nxp array representing the estimated errors (state
#                        vector) at each epoch after the measurement update. The
#                        state vector before the update will always be zero,
#                        since the Kalman filter algorithm is assumed to use a
#                        closed loop implementation.
#        .P_prior        nx(p*p) array representing the error covariance
#                        matrices prior to the measurement update. Every nx(p*p)
#                        vector is "reshaped" into a pxp matrix using MATLABs
#                        reshape command.*
#        .P_posterior    nx(p*p) array representing the error covariance
#                        matrices after the measurement update. Every nx(p*p)
#                        vector is "reshaped" into a pxp matrix using MATLABs
#                        reshape command.*
#        .Transition     nx(p*p) array representing the transition matrices at
#                        each epoch. Notice that the transition matrix is used
#                        for propagating the state estimates at the previous
#                        epoch to the current epoch. Every pxp matrix is
#                        re-constructed from the corresponding 1x(p*p) vector
#                        using using MATLABs reshape command.*
    
#        *The original matrices can be recovered using MATLABs reshape command,
#        as for example:
    
#                            P = reshape(rts.P_prior(epoch,:),[p,p])


#         optional inputs: 
#         echo:         string: True or False, for displaying progress bar in terminal 
#                       default is "True"

                        
#      Output:
#        rts             MATLAB structure with the following sub-structures
#                        added or modified:
    
#        .profile        nx(p+1) array containing the smoothed navigation
#                        profile
#        .profile_std    nxp array with standard deviation on state parameters
    
#      -------------------------------------------------------------------------
#      Author: Christian Solgaard (DTU) 18/10-2023 based on the work done by
#      Tim Jensen (DTU) 13/10/2023
#     """
#     # Respond to varible input 

#     #initialize logicals 
#     lecho = True

#     # Check input parameters
#     if len(args) < 2: 
#         rts = args
#     else:
#         rts, echo = args
#         if echo == False: 
#             lecho = False

    
#     # Respond to input ------------------------------------------------------
#     # get number of epochs
#     no_epochs = len(rts["time"])

#     # get number of states
#     npar = rts["x_posterior"].shape[1]
    
#     # Allocate space
#     rts["q_n_b"] = np.zeros([no_epochs, 4])
#     rts["profile_std"] = np.zeros([no_epochs, npar])
#     rts["P"] = np.zeros([no_epochs, npar**2])

#     # Define Scaling Parameters --------------------------------------------
#     S = np.ones([1, npar])
#     Sinv = np.ones([1, npar])

#     id_diag = np.zeros([npar, 1])
#     for i in range(0, npar):
#         id_diag[i] = i+(i-1)*npar
      
#     # Determine order of magnitude 
#     ordmin = np.log10(np.sqrt(min(rts["P_prior"][:, id_diag])))
#     ordmax = np.log10(np.sqrt(max(rts["P_prior"][:, id_diag])))
#     ordmag = round(.5 * (ordmin + ordmax))

#     # Change some scaling factors
#     for i in range(0, npar): 
#         S[i] = 10**(-ordmag[i])
#         Sinv[i] = 10**(ordmag[i])

#     # Form scaling matrices
#     S = np.diag(S)
#     Sinv = np.diag(Sinv)


#     # Perform RTS moothing ------------------------------------------------


#     # initialize system 
#     xk_smooth = rts["x_posterior"][-1, :].T
#     Pk_smooth = rts["P_posterior"][-1, :].reshape(npar, npar)

#     # initialise solution 
#     rts["P"][-1, :] = Pk_smooth.ravel()
#     rts["profile_std"][-1,:] = np.sqrt(np.diag(Pk_smooth))

#     # Re-scale state vector and error covariance
#     xk_smooth = S * xk_smooth
#     Pk_smooth = S * Pk_smooth * S

#     # Perform backward sweep 
#     for epoch in range(no_epochs - 1, 0, -1): 
#         # ------------------------------------------------------------------
#         #Extract Parameters Needed for This Iteratoon 
#         # ------------------------------------------------------------------

#         # Make Current estimates previous estimates
#         xkp1_smooth = xk_smooth
#         Pkp1_smooth = Pk_smooth
        
#         # Extract state vector estimates for current epoch
#         xk_plus = rts["x_posterior"][epoch, :].T

#         # Extract error covariance matrices before and after measurement update
#         Pk_plus = rts["P_posterior"][epoch, :].reshape(npar, npar)
#         Pkp1_minus = rts["P_prior"][epoch+1, :].reshape(npar, npar)

#         # Re-scale state vector and error covariance
#         xk_plus = S @ xk_plus
#         Pk_plus = S @ Pk_plus @ S
#         Pkp1_minus = S @ Pkp1_minus @ S

#         # Extract transition matrix from current to "previous" epoch
#         Phi = rts["Transition"][epoch+1, :].reshape(npar, npar)

#         # Re-scale transition matrix
#         Phi = S @ Phi @ Sinv

#         # ------------------------------------------------------------------
#         # Perform Smoothing of state Vector
#         # ------------------------------------------------------------------

#         # Compute smoothing gain from Brown and Hwang eq. 6.2.2
#         Phi_transpose = np.transpose(Phi)
#         A = np.dot(np.dot(Pk_plus, Phi_transpose), np.linalg.inv(Pkp1_minus))

#         # Compute smoothed parameters using Brown and Hwang eq. 6.2.2 and 6.2.3
#         xk_smooth = xk_plus + A @ xkp1_smooth
#         Pk_smooth = Pk_plus + A @ ( Pkp1_smooth - Pkp1_minus ) @ A.T

#         # ------------------------------------------------------------------
#         # Apply Correction to Navigation Profile
#         # ------------------------------------------------------------------

#         # compute residual 
#         delta_x = xk_smooth - xk_plus
#         delta_x = Sinv @ delta_x

#         # Correct Navigation profile

#         rts["profile"][epoch, :] = rts["profile"][epoch, :] - delta_x[:].T

#         # ------------------------------------------------------------------
#         # Store uncertainty Estiamtes
#         # ------------------------------------------------------------------

#         rts["P"][epoch, :] = (Sinv @ Pk_smooth @ Sinv).reshape(npar, npar)
#         rts["profile_std"][epoch, :] = Sinv @ np.sqrt(np.diag(Pk_smooth))

#     return rts

# def RC_filter(data, stage, ftc, dt): 
#     """
#      RC-filter from Dag Solheim
    
#      -------------------------------------------------------------------------
    
#      Input:
#        data    Vector of data series
#        stage   Number of iterations, 1 iteration = forwar+backward run
#        ftc     Filter time constant [s]
#        dt      Sample interval [s]
    
#      Output:
#        fdata   Vector of filtered data series
    
#      -------------------------------------------------------------------------
#      Author: Christian Solgaard (DTU) 23/10-2023, based on the implementation 
#      done by: Tim Jensen (DTU) 29/05/2020
#     """

#     # Set some parameters
#     edge = 200 

#     # Derive data length 
#     nmax = len(data)

#     ## Prolong Data Series with 'edge' constant readings in Each End ---------
    
#     # derive some indices
#     n1 = edge
#     n2 = nmax + edge
#     n3 = nmax + 2*edge

#     # Allocate space 
#     p = np.zeros([n3, 1])

#     # insert values
#     p[0:n1] = p[0:n1]+data[0]
#     p[edge+1 : edge+nmax] = data
#     p[n2:n3] = p[n2:n3]+data[-1]

#     # Update data length
#     nmax = n3

#     ## Perform filtering ------------------------------------------------------

#     # Derive some parameters
#     a = dt/(2*ftc)
#     b = (1.0-a)/(1+a)
#     c =  a/(1+a)

#     for m in range(0,stage): 

#       # Forward run ------------------------------------------------------------
#       # Initialize parameters 
#       a = p[0]
#       d = a
      
#       # Apply filtering
#       for i in range(1, nmax): 
#          e = b*a+c*(p[i]+d)
#          d = p[i]
#          p[i] = e
#          a = e

#       # Reverse run ------------------------------------------------------------
      
#       # initialize parameters
#       a = p[nmax]
#       d = a

#       for i in range(nmax - 1, 0, -1):
#           e = b * a + c * (p[i] + d)
#           d = p[i]
#           p[i] = e
#           a = e

      
#     # Extract original part and restore mean value
#     fdata = p[n1:n2]

#     return fdata


