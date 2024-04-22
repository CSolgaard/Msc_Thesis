# -----------------------Kalman Filters - Direct Strapdown vs Quantum --------------
#  
#
# 
# ---------------------------------------------------------------------------------
# Author: Christian Solgaard (DTU) 01/02-2024 
# ---------------------------------------------------------------------------------




from numba import njit
from scipy.linalg import expm, sqrtm
import numpy as np
import copy
import scienceplots
from pathlib import Path 
import pandas as pd
import time
from scipy import interpolate
from dataclasses import dataclass
import Direct_Strapdown as ds
import pickle
# import RTS 
# from src import RTS as RTS
import RTS as RTS
import matplotlib.pyplot as plt
plt.style.use(['science', 'grid'])
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, TransformedBbox, BboxPatch, BboxConnector)
from matplotlib.transforms import Bbox
# plt.style.use(['science', 'grid'])

params = {'axes.labelsize': 'x-large', 'axes.titlesize':'xx-large','xtick.labelsize':'large', 
          'ytick.labelsize':'large', 'legend.fontsize': 'x-large','mathtext.fontset':'stix', 'font.family':'STIXGeneral'}
plt.rcParams.update(params)

import warnings
# To ignore all warnings
warnings.filterwarnings("ignore")
# -----------------------------------------------------------------------------------------------

def SOW2HOD(SOW): 
    """ 
    Procedure to convert Seconds Of Week (SOW) to Hour Of Day (HOD) 
    --------------------------------------------------------------
    
    Input: 
        SOW:    array [N x 1]

    Output: 
        HOD:   array [N x 1] 

    ---------------------------------------------------------------
    Author: Christian Solgaard (DTU - Master student) 09/11-2023
    """
    remainder = (SOW/3600)[0]  % 24
    relative_h = SOW/3600 - (SOW/3600)[0]
    HOD = remainder + relative_h                # HOD -> Hour of Day 
    return HOD

def HOD2SOD(HOD): 
    """ 
    Procedure to convert Hour Of Day (HOD) to Secound Of Day (SOD) 
    --------------------------------------------------------------
    
    Input: 
        HOD:    array [N x 1]

    Output: 
        SOD:   array [N x 1] 

    ---------------------------------------------------------------
    Author: Christian Solgaard (DTU - Master student) 12/12-2023
    """
    SOD = HOD*60*60
    return SOD

def SOW2SOD(SOW): 
    """ 
    Procedure to convert Seconds Of Week (SOW) to Seconds Of Day (SOD) 
    --------------------------------------------------------------
    
    Input: 
        SOW:    array [N x 1]

    Output: 
        SOD:   array [N x 1] 

    ---------------------------------------------------------------
    Author: Christian Solgaard (DTU - Master student) 12/12-2023
    """
    HOD = SOW2HOD(SOW)
    SOD = HOD2SOD(HOD)
    return SOD


@njit
def approximate_expm(A, order):
    """
    Approximate the matrix exponential of A using a Taylor series expansion.
    
    Parameters:
    A (array-like): Input matrix.
    order (int): Order of the Taylor series expansion (higher order for more accuracy).
    
    Returns:
    expm_A (numpy.ndarray): Approximation of the matrix exponential.

    -------------------------------------------------------------------------
    Author: Christian Solgaard (DTU) 
    """
    expm_A = np.eye(A.shape[0])  # Initialize with the identity matrix
    A_power = np.eye(A.shape[0])  # Initialize with the identity matrix

    for i in range(1, order):
        A_power = np.dot(A_power, A) / i
        expm_A += A_power

    return expm_A

@njit
def valLoan(F, PSD, tau): 
    """
    Implementation of Van Loan method for forming transition matrix and
    system noise matrix from system matrix and power spectral density matrix
    
    -------------------------------------------------------------------------
    
    Input:
      F       NxN array representing system matrx
      PSD     NxN array representing matrix of power spectral densities
      tau     scalar representing propagation time interval [s]
    
    Output:
      Phi     NxN array representing transition matrix
      Q       NxN array representing system noise matrix
    
    -------------------------------------------------------------------------
    Author: Christian Solgaard (DTU), implemented on the work done by: 
    Tim Jensen (DTU) 21/05/2018
    """

    # Get dimension 
    nvar = len(F)

    # Form index values
    id1 = nvar
    id2 = nvar * 2

    # Derive Transition and System Noise Matrices
    # Form A matrix using Brown and Hwang (3.9.22)
    idx = F.shape

    # Derive Transition and System Noise Matrices
    # Form A matrix using Brown and Hwang (3.9.22)
    A = np.zeros((idx[0]*2,idx[0]*2))
    A[0:idx[0], 0:idx[0]] = -F * tau
    A[0:idx[0], idx[0]:] = PSD * np.abs(tau)
    A[idx[0]:, 0:idx[0]] = np.zeros((nvar, nvar))
    A[idx[0]:, idx[0]:] = F.T * tau

    # B = expm(A)
    B = approximate_expm(A, 20)    

    # The lower-right partition is the transpose of the transition matrix
    Phi = B[id1:id2, id1:id2].T

    # Compute Q using Brown and Hwang (3.9.25)
    # Q = Phi * B[0:nvar, id1:id2]
    Q = Phi @ B[0:nvar, id1:id2]
    
    return Phi, Q


def Mahalanobis_dist(dz, H, P, R):

    H = H.reshape(1, H.shape[1])
    dz = np.array([dz])
    Mahala_dist = np.sqrt(dz.T * 1/((H@P@H.T).flatten() + R) * dz)
    return Mahala_dist


def getGravTIE(time, vel_scalar, dg):
    """
    Filter gravitational time-variable signal based on specified conditions.

    Parameters:
    - time (array-like): Array of time values.
    - vel_scalar (array-like): Array of scalar velocity values.
    - dg (array-like): Array of gravitational values.

    Returns:
    - dict: A dictionary containing filtered time, velocity, and gravitational values.
      - 'time' (array-like): Time values satisfying filtering conditions.
      - 'vel' (array-like): Scalar velocity values satisfying filtering conditions.
      - 'dg' (array-like): Gravitational values satisfying filtering conditions.

    Filtering Conditions:
    - Velocity values must be less than 0.01.
    - Gravitational values must be between -100 and 100.

    -------------------------------------------------------------------------
    Author: Christian Solgaard (DTU) 18/12-2023
    """

    idx = (vel_scalar < 0.01) & (dg < 100) & (dg > -100)
    
    Gravtie = {}
    Gravtie["time"] = time[idx]
    id_ = 0
    for i in range(len(Gravtie["time"])-1):
        if Gravtie["time"][i+1] - Gravtie["time"][i] > 100: 
            id_ = i
            break 
    
    Gravtie["dg"] = np.ones(Gravtie["time"].shape)
    # Gravtie["dg"][:id_+1] = np.mean((dg[idx])[:id_+1])
    # Gravtie["dg"][id_+1:] = np.mean((dg[idx])[id_+1:])
    Gravtie["dg"][:id_] = np.mean((dg[idx])[:id_])
    Gravtie["dg"][id_:] = np.mean((dg[idx])[id_:])


    Gravtie["vel"] = vel_scalar[idx]
    # Gravtie["dg"] = dg[idx]

    Gravtie["std"] = np.ones(Gravtie["dg"].shape)*5

    return Gravtie



def KalmanFilterTIE(obs, Gravtie, varible):
    """
    Function to perform Kalman filtering using the GIRAFE absolute measurements as a bias state

    4.th order GAUSS-MARKOV statespace model, (4 state varibles.)
    
    Parameters:
    - obs: Dictionary containing time and gravity disturbance observations from DS
    - girafe: Dictionary containing time and gravity disturbance observations from GIRAFE

    Returns:
    - kf: Kalman filter estimates
    - rts: Variables passed to the smoothing routine
    """
    # Number of state variables
    ns = 4
    ns2 = ns ** 2

    # Define common timeseries. 
    kf = {}
    time = np.concatenate((obs["time"], Gravtie["time"]), axis=None)
    time = np.array(sorted(np.unique(time)))
    kf["time"] = time

    # Add a fictive time stamp in end of each timeline 
    obs["time"] = np.concatenate((obs["time"], kf["time"][-1] + 100), axis=None)
    Gravtie["time"] = np.concatenate((Gravtie["time"], kf["time"][-1] + 100), axis=None)

    # Kalman Filter Estimates
    no_updates = len(kf['time'])

    # kf = {'profile': np.zeros((no_updates, ns)),
    #       'profile_std': np.zeros((no_updates, ns))}
    
    kf["profile"] = np.zeros((no_updates, ns))
    kf["profile_std"] = np.zeros((no_updates, ns))

    # Allocate space for variables passed to smoothing routine
    rts = {'time': np.zeros(no_updates - 1),
           'profile': np.zeros((no_updates - 1, ns)),
           'x_posterior': np.zeros((no_updates - 1, ns)),   # State vectors after measurement update
           'P_prior': np.zeros((no_updates - 1, ns2)),      # System error covariance prior to measurement update
           'P_posterior': np.zeros((no_updates - 1, ns2)),  # System error covariance after measurement update
           'Transition': np.zeros((no_updates - 1, ns2))}   # State transition matrix


    # Design Kalman Filter
    dg_obs_std = 10  # Standard deviation of measurement [mGal]

    dg_sigma = 100    # Gravity anomaly standard deviation [mGal]
    dg_beta = 1 / 6100  # Gravity anomaly correlation parameter [1/m]

    # Convert to seconds
    beta_d = 100 * dg_beta

    # Accelerometer sensor bias (Ask Tim!!!)
    system_acc = 0.01

    # Initialise Kalman Filter
    dg_ini = 0          # Initial estimate of gravity disturbance [mGal]
    dg_std = 100        # Standard deviation on gravity disturbance [mGal]
    bias_std = 30       # Standard deviation on accelerometer bias [mGal]

    dg = dg_ini
    dg1 = 0
    dg2 = 0
    zbias = 0

    # Form initial (error) state vector x
    x = np.zeros([ns,1])

    # Form initial error Covarance Matrix P
    P = np.diag([dg_std**2, 1e-4, 1e-4, bias_std**2])

    kf['profile'][0, :] = [dg, dg1, dg2, zbias]
    kf['profile_std'][0, :] = np.sqrt(np.diag(P))

    # -------------------- Perform Kalman Filtering ----------------------------
    # initialise entries 
    n_obs = 1
    n_tie = 0

    # Looping through data
    for n_epoch in range(1, no_updates):

        # Derive components of PSD of 3rd order Gauss-Markov model
        PSD = np.zeros((4, 4))
        PSD[2, 2] = (16 / 3) * (beta_d**5) * (dg_sigma**2)
        PSD[3, 3] = system_acc**2

        # Form system matrix
        F = np.zeros((4, 4))
        F[0, 1] = 1
        F[1, 2] = 1
        F[2, 0] = -beta_d**3
        F[2, 1] = -3 * beta_d**2
        F[2, 2] = -3 * beta_d


        # ------------------------ Perform Forward Propagation ---------------------------
        #  Form propagation time interval
        tau = kf['time'][n_epoch] - kf['time'][n_epoch - 1]

        # Form Q and Phi matrices using method of van Loan
        # Phi, Q = KF_lowpass.valLoan(F, PSD, tau)
        Phi, Q = valLoan(F, PSD, tau)

        # Propagate error covariance using Groves (3.15) P
        P = np.dot(np.dot(Phi, P), Phi.T) + Q

        # Store transition matrix and a priori covariance
        rts['time'][n_epoch - 1] = kf['time'][n_epoch]
        rts['Transition'][n_epoch - 1, :] = Phi.flatten(order='F')
        rts['P_prior'][n_epoch - 1, :] = P.flatten(order='F')


        # --------------------- Update DS (IMAR) measurement ----------------------------
        if kf["time"][n_epoch] == obs["time"][n_obs]: 
            
            # Extract Observation 
            dg_obs = obs['fdg'][n_obs, varible]-40 # Extract estimate of dg [mGal]

            # From Measurement Error Covariance Matrix
            R = dg_obs_std**2

            # Form measurement model relating measurement to state vector
            H = np.array([[1, 0, 0, 1]])

            # Form measurement innovation using Groves (14.103)
            dz = dg_obs - (dg + zbias)

            # Form Kalman gain matrix using Groves (3.21)
            K = np.dot(np.dot(P, H.T), np.linalg.inv(np.dot(np.dot(H, P), H.T) + R))

            # Update state vector using Groves (3.24)
            x = x + K * dz

            # Update Error Covariance Groves (3.58)
            P = np.dot(np.dot(np.eye(4) - np.dot(K, H), P), (np.eye(4) - np.dot(K, H)).T) + np.dot(np.dot(K, R), K.T)

            # Increase Index for IMAR measurement 
            n_obs = n_obs + 1

        # --------------------- Update TIE/Bias measurement ----------------------------
        if kf["time"][n_epoch] == Gravtie["time"][n_tie]:

            # Extract Observation 
            dg_obs = Gravtie['dg'][n_tie]   # Extract estimate of dg [mGal]

            # From Measurement Error Covariance Matrix
            # R = Gravtie["std"][n_tie]**2
            R = Gravtie["std"][n_tie]**2      # Testing
            
            # Form measurement model relating measurement to state vector
            H = np.array([[1, 0, 0, 0]])

            # Form measurement innovation using Groves (14.103)
            dz = dg_obs - dg

            # Form Kalman gain matrix using Groves (3.21)
            K = np.dot(np.dot(P, H.T), np.linalg.inv(np.dot(np.dot(H, P), H.T) + R))

            # Update state vector using Groves (3.24)
            x = x + K * dz

            # Update Error Covariance Groves (3.58)
            P = np.dot(np.dot(np.eye(4) - np.dot(K, H), P), (np.eye(4) - np.dot(K, H)).T) + np.dot(np.dot(K, R), K.T)

            # Increase Index for IMAR measurement 
            n_tie = n_tie + 1

        # -------------- Store A-Posteriori State Vector and Covariance ---------------
        rts['x_posterior'][n_epoch - 1, :] = x.flatten()
        rts['P_posterior'][n_epoch - 1, :] = P.flatten(order='F')


        # -------------------- Update State vector x ----------------------------------
        dg = dg + x[0]              # Downward Gravity Disturbance
        dg1 = dg1 + x[1]            # ... Not used for now
        dg2 = dg2 + x[2]            # ... Not used for now 
        zbias = zbias + x[3]        # Bias estimate
        x = np.zeros([ns,1])        # reset state vector

        # ------------------- Store profile -------------------------------------------
        kf['time'][n_epoch] = kf['time'][n_epoch]
        kf['profile'][n_epoch, :] = np.array([dg, dg1, dg2, zbias]).flatten()
        rts['profile'][n_epoch - 1, :] = kf['profile'][n_epoch, :]
        kf['profile_std'][n_epoch, :] = np.sqrt(np.diag(P))

    return kf, rts



# ------------------------- Running scrips local -------------------------------
def vel_scalar(vn, ve, vd): 
    """
    Calculate scalar velocities from north, east, and down components.

    Parameters:
    - vn: Array of north components of velocity
    - ve: Array of east components of velocity
    - vd: Array of down components of velocity

    Returns:
    - Array of scalar velocities
    """
    velocity_scalar = np.sqrt(vn**2 + ve**2 + vd**2)
    return velocity_scalar

def rewritePDF2Tuple(PDF, index_start, index_end):       # Findes også i DS

    # Slice the DataFrame to remove the specified range from the start and end of each column
    PDF = PDF.iloc[index_start:index_end, :]

    # Reset the index to start from 0 again
    PDF = PDF.reset_index(drop=True)

    temp = {}
    for key, value in PDF.items():
        # temp[key] = value[index_start:index_end]
        temp[key] = value
    return temp

def Createdict(): 
    BF = {}
    BF["stages"] = 3
    BF["ftc"] = 150
    BF["srate"] = 1
    return BF 


def plot_ref_line(rts_final, DS):
    import pyproj
    from pyproj import Transformer
    survey = {}
    survey["dg"] = rts_final["profile"][:,0]
    survey["time"] = rts_final["time"]

    pipeline = "+ellps=GRS80 +proj=pipeline +step +proj=utm +zone=30"
    transform_object = Transformer.from_pipeline(pipeline)
    geodetic_corr = [DS["lon"], DS["lat"], DS["h"]]
    UTM_corr = transform_object.transform(*geodetic_corr)


    min = 9700
    max = 20200

    Ref = {}                                        # Determined using visual inspection
    Ref["all_lines"] = np.arange(min,max+1)
    Ref["Line_1"] = np.arange(min+350,min+1950+1)
    Ref["Line_2"] = np.arange(min+2250,min+4200+1)
    Ref["Line_3"] = np.arange(min+4490,min+6050+1)
    Ref["Line_4"] = np.arange(min+6400,min+8300+1)
    Ref["Line_5"] = np.arange(min+8650,max-250+1)

    # Calc distance to refererence point. 
    def dist2ref(x1, y1, refx2, refy2): 
        dist = np.sqrt((x1 - refx2)**2 + (y1 - refy2)**2)
        return dist

    ref_x = UTM_corr[0][[min+300]]
    ref_y = UTM_corr[1][[min+300]]
    
    print(ref_x)
    print(ref_y)
    print(DS["lon"][min+300])
    print(DS["lat"][min+300])
    
    @dataclass
    class Verification_line: 
        name: str
        index: np.array
        dist: np.array
        dg: np.array
        time: np.array
        # h: np.array
        # roll: np.array
        # pitch: np.array
        # yaw: np.array
    ref_line_dist = np.arange(10,140,.5)
    Line = ["Line_1", "Line_2", "Line_3", "Line_4", "Line_5"]
    RMS_line = {}
    RMS_line["dist"] = ref_line_dist
    for i in range(len(Line)): 
        dist_line = dist2ref(UTM_corr[0][Ref[Line[i]]], 
                            UTM_corr[1][Ref[Line[i]]], ref_x, ref_y)
        RMS_line[Line[i]] = ds.interpolate_DS(dist_line*1e-3, rts_final["profile"][Ref[Line[i]], 0]-50, ref_line_dist, "linear", "extrapolate")

        Line[i] = Verification_line(Line[i], Ref[Line[i]], 
                                    dist_line, rts_final["profile"][Ref[Line[i]], 0], 
                                    rts_final["time"][Ref[Line[i]]])
        
    RMS_line["combined"] = np.array([RMS_line["Line_1"], RMS_line["Line_2"], RMS_line["Line_3"], RMS_line["Line_4"], RMS_line["Line_5"]]).T
    RMS_line["mean_line"] = np.sum(RMS_line["combined"], axis=1)/len(RMS_line["combined"][0,:])

    # res_ref = np.array([RMS_line["Line_1"] - RMS_line["mean_line"], [RMS_line["Line_2"] - RMS_line["mean_line"], [RMS_line["Line_3"] - RMS_line["mean_line"], [RMS_line["Line_4"] - RMS_line["mean_line"], [RMS_line["Line_5"] - RMS_line["mean_line"] ])
    # Create an array with the differences
    res_ref = np.array([
        RMS_line["Line_1"] - RMS_line["mean_line"],
        RMS_line["Line_2"] - RMS_line["mean_line"],
        RMS_line["Line_3"] - RMS_line["mean_line"],
        RMS_line["Line_4"] - RMS_line["mean_line"],
        RMS_line["Line_5"] - RMS_line["mean_line"]
    ])

    # Concatenate along the second axis (axis=1) to get a single vector
    res_ref_concatenated = np.concatenate(res_ref)

    def rms(x): 
        rms_ = np.sqrt(np.sum(x**2)/len(x))
        return rms_

    def LOO_summary_stat(RMS_line):     # Calculate Summary statistics using Leave One Out.
        for i in range(1, 6): 
            excluded_line = f"Line_{i}"
            included_lines = [f"Line_{j}" for j in range(1, 6) if j != i]
            mean_line = np.mean([RMS_line[line] for line in included_lines], axis=0)

            res = RMS_line[excluded_line] - mean_line

            # Calculate Summary Statistics 
            mean = np.mean(res)
            min = np.min(res)
            max = np.max(res)
            STD = np.std(res)
            rms_ = rms(res)
            rmse = rms_/np.sqrt(2)

            if i == 1: 
                print("Excluded Profile:   Mean:   Min:    Max:    STD:    RMS:    RMSE")
            print(f"Profile {i}         {np.round(mean,2)}    {np.round(min,2)}   {np.round(max,2)}    {np.round(STD,2)}    {np.round(rms_,2)}    {np.round(rmse,2)}")
        return
    LOO_summary_stat(RMS_line)
    
    rms_ = rms(res_ref_concatenated)/np.sqrt(2)

    # fig = plt.figure(figsize=(10,5))
    # for i in range(5): 
    #     plt.plot(Line[i].dist*1e-3, Line[i].dg-50, label =  f"Line$_{i+1}$", linewidth=3)

    # plt.text(0.75, 0.08, f'RMSE: {rms_:.2f} [mGal]', transform=plt.gca().transAxes,
    #         verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # # plt.grid()
    # plt.plot(RMS_line["dist"], RMS_line["mean_line"], 'b--', linewidth=1, label="Mean reference")
    # plt.legend(loc="lower right", fontsize="12")
    # plt.xlabel("Distance from reference Point [km]")
    # plt.ylabel("Gravity Disturbance [mGal]")
    # plt.title("Flight 115, Biscay Bay Reference Line: Attitude @ 300Hz\nKalman and RTS")
    # plt.xlim(5, 140)
    # plt.ylim(-60, 60)
    # save_path = "../Figures/Figures4meeting/Biscay_lines_nav@300Hz_final_Kalman_RTS.png"
    # plt.savefig(save_path)
    # plt.show()


    fig, ax=plt.subplots(figsize=(10,5))
    for i in range(5): 
        ax.plot(Line[i].dist*1e-3, Line[i].dg-50, label =  f"Profile {i+1}", linewidth=3, zorder=0)

    ax.text(0.75, 0.08, f'RMSE: {rms_:.2f} [mGal]', transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.plot(RMS_line["dist"], RMS_line["mean_line"], 'b--', linewidth=1, label="Mean Reference", zorder=0)
    ax.legend(loc="lower right", fontsize="12")
    ax.set_xlabel("Distance From Reference Point [km]")
    ax.set_ylabel("Gravity Disturbance [mGal]")
    ax.set_xlim(5, 140)

    # axins = inset_axes(ax, "100%", "100%", bbox_to_anchor=[.2, .55, .4, .4], bbox_transform=ax.transAxes, borderpad=0)
    insetPosition = Bbox.from_bounds(0.3, 0.55, 0.3, 0.3)  # Define position of inset axes
    axins = fig.add_axes(insetPosition)

    for i in range(5): 
        axins.plot(Line[i].dist*1e-3, Line[i].dg-50, linewidth=2)
    axins.set(xlim=(24,36), ylim=(-27.5,-12.5))
    axins.plot(RMS_line["dist"], RMS_line["mean_line"], 'b--', linewidth=1)

    def my_mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
        rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)
        pp = BboxPatch(rect, fill=False, **kwargs)
        parent_axes.add_patch(pp)
        p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
        inset_axes.add_patch(p1)
        p1.set_clip_on(False)
        p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
        inset_axes.add_patch(p2)
        p2.set_clip_on(False)

        return pp, p1, p2
    
    my_mark_inset(ax, axins, loc1a=2, loc1b=2, loc2a=4, loc2b=4, fc="none", ec="0.5")





    cutoff = 1000
    plt.figure(figsize=(12,6))
    plt.plot(survey["time"][cutoff:-cutoff], survey["dg"][cutoff:-cutoff],'--', lw=1.5, label="Full Survey")
    plt.plot(survey["time"][min+350:min+1950+1], survey["dg"][min+350:min+1950+1], label="Profile 1", lw=3)
    plt.plot(survey["time"][min+2250:min+4200+1], survey["dg"][min+2250:min+4200+1], label="Profile 2", lw=3)
    plt.plot(survey["time"][min+4490:min+6050+1], survey["dg"][min+4490:min+6050+1], label="Profile 3", lw=3)
    plt.plot(survey["time"][min+6400:min+8300+1], survey["dg"][min+6400:min+8300+1], label="Profile 4", lw=3)
    plt.plot(survey["time"][min+8650:max-250+1], survey["dg"][min+8650:max-250+1], label="Profile 5", lw=3)

    plt.legend()
    # for i in range(5):
    #     plt.plot(survey["time"][Ref1[Line[i]]], survey["dg"][Ref1[Line[i]]], lw=3, label =  f"Line$_{i+1}$")
    plt.xlabel("Time")
    plt.ylabel("Gravity Disturbance [mGal]")
    plt.show()


def case2(): 
    print("Running Kalman Lowpass Filter with Tie/Bias states")

    # BF = {}
    # BF["stages"] = 3
    # BF["ftc"] = 150
    # BF["srate"] = 1 
    BF = Createdict()

    dg = ds.but2_v2(DS["dg"][450:-450, 2], BF["stages"], BF["ftc"], BF["srate"])

    Gravtie = getGravTIE(DS["time"][450:-450], DS["vel_scalar"][450:-450], dg)
    kf, rts = KalmanFilterTIE(DS, Gravtie, 2)

    rts_test = {}
    cutoff = 1000
    # for key, value in rts.items():
    #     if key == "time":
    #         rts_test[key] = value[cutoff:-cutoff]
    #     else: 
    #         rts_test[key] = value[cutoff:-cutoff, :]
            
    # rts_final = RTS.RTS_smoother(copy.deepcopy(rts_test))
    # rts_final = RTS.RTS_smoother(rts_test)
    rts_final = RTS.RTS_smoother(rts)


    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,6))

    ax1.plot(ds.SOW2HOD(rts_final["time"][cutoff:-cutoff]), rts_final["profile"][cutoff:-cutoff,0], label="Kalman Tie/Bias + RTS", linewidth=2)
    ax2.plot(ds.SOW2HOD(rts_final["time"][cutoff:-cutoff]), rts_final["profile"][cutoff:-cutoff,3], label="Bias Estimate", linewidth=2)
    ax1.plot(ds.SOW2HOD(Gravtie["time"][1:-1]), Gravtie["dg"][:-1], '.', label="Tie Values")
    ax1.legend(loc="lower left")
    ax2.legend(loc = "lower left")
    # plt.grid()
    # ax2.grid()
    ax2.set_xlabel("Time Hour of day [h]")
    ax1.set_ylabel("Gravity Disturbance [mgal]")
    ax2.set_ylabel("Bias [mGal]")

    # Plot Reference lines (Verification)
    plot_ref_line(rts_final, DS)
    plt.show()


if __name__ == "__main__":
    import Direct_Strapdown as ds

    file_path = "../Results/DS_300Hz_150s_3_stages.pkl"

    print(f"Loading DS results from: {file_path}")
    with open(file_path, 'rb') as file: 
        DS = pickle.load(file)
    
    DS["vel_scalar"] = vel_scalar(DS["vn"], DS["ve"], DS["vd"])
    
    case2()
    
