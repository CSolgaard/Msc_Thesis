# -----------------------Kalman Filters - Direct Strapdown vs Quantum --------------
#  
#Take raw Direct strapdown results, as observation and raw GIRAFE as measurement update.
# 
# ---------------------------------------------------------------------------------
# Author: Christian Solgaard (DTU) 21/02-2024 
# ---------------------------------------------------------------------------------


from numba import jit, njit
import Direct_Strapdown as ds
from scipy.linalg import expm, sqrtm
import numpy as np
import copy
import scienceplots
from pathlib import Path 
import pandas as pd
import time
from scipy import interpolate
from dataclasses import dataclass
import pickle
import RTS as RTS
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, TransformedBbox, BboxPatch, BboxConnector)
from matplotlib.transforms import Bbox
plt.style.use(['science', 'grid'])

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


def SOD2HOD(SOD): 
    """ 
    Procedure to convert Seconds Of Day (SOD) to Hour Of Day (HOD) 
    --------------------------------------------------------------
    
    Input: 
        SOD:    array [N x 1]

    Output: 
        HOD:   array [N x 1] 

    ---------------------------------------------------------------
    Author: Christian Solgaard (DTU - Master student) 30/01-2023
    """
    HOD = SOD / (60*60)
    return HOD


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



def KalmanFilterGIRAFE(obs, DS_girafe):
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
    # Redefine the timestamps for girafe. 
    # DS_girafe["time"]= DS_girafe["time"]       # +2 hour UTC time - 18 GPS clock offset
    obs["time"] = SOW2SOD(obs["time"])          # Convert IMU/GNSS time to seconds of day instead of week 

    # Define common timeseries. 
    kf = {}
    time = np.concatenate((obs["time"], DS_girafe["time"]), axis=None)
    time = np.array(sorted(np.unique(time)))
    kf["time"] = time

    # Add a fictive time stamp in end of each timeline 
    obs["time"] = np.concatenate((obs["time"], kf["time"][-1] + 100), axis=None)
    DS_girafe["time"] = np.concatenate((DS_girafe["time"], kf["time"][-1] + 100), axis=None)

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
    dg_obs_std = 205  # Standard deviation of Observation (IMAR) [mGal]
    girafe_std = 200    # Standard deviation of measurement (GIRAFE) [mGal]

    dg_sigma = 100    # Gravity anomaly standard deviation [mGal]
    dg_beta = 1 / 6100  # Gravity anomaly correlation parameter [1/m]

    # Convert to seconds
    beta_d = 100 * dg_beta

    # Accelerometer sensor bias 
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
    n_girafe = 0

    turning = {}
    turning["T1"] = np.arange(50, 650)
    turning["T2"] = np.arange(1950, 2600)
    turning["T3"] = np.arange(4150, 4800)
    turning["T4"] = np.arange(6050, 6700)
    turning["T5"] = np.arange(8250, 9000)

    # Create a mask to identify indices to keep
    turning_idx = np.ones(len(DS_girafe["time"]), dtype=bool)
    for indices in turning.values():
        turning_idx[indices] = 0

    plt.figure()
    plt.plot(DS_girafe["time"], turning_idx)
    

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

        def process_girafe_measurement(n_girafe, P, x):
            dg_obs = DS_girafe['dg'][n_girafe]
            R = girafe_std**2
            H = np.array([[1, 0, 0, 0]])
            dz = dg_obs - dg
            K = np.dot(np.dot(P, H.T), np.linalg.inv(np.dot(np.dot(H, P), H.T) + R))
            x = x + K * dz
            P = np.dot(np.dot(np.eye(4) - np.dot(K, H), P), (np.eye(4) - np.dot(K, H)).T) + np.dot(np.dot(K, R), K.T)
            return n_girafe + 1, P, x

        def process_ds_measurement(n_obs, P, x):
            dg_obs = obs['dg'][n_obs, 2] - 40
            R = dg_obs_std**2
            H = np.array([[1, 0, 0, 1]])
            dz = dg_obs - (dg + zbias)
            K = np.dot(np.dot(P, H.T), np.linalg.inv(np.dot(np.dot(H, P), H.T) + R))
            x = x + K * dz
            P = np.dot(np.dot(np.eye(4) - np.dot(K, H), P), (np.eye(4) - np.dot(K, H)).T) + np.dot(np.dot(K, R), K.T)
            return n_obs + 1, P, x

        if turning_idx[n_girafe] == 0: 
            girafe_std = girafe_std*10      # Force method to avoid using GIRAFE in unstable conditions. 

        cond_same_time = (kf["time"][n_epoch] == DS_girafe["time"][n_girafe]) and (kf["time"][n_epoch] == obs["time"][n_obs])
        if cond_same_time:
            cond = dg_obs_std > girafe_std
            if cond:
                n_girafe, P, x = process_girafe_measurement(n_girafe, P, x)
                n_obs += 1
            else:
                n_obs, P, x = process_ds_measurement(n_obs, P, x)
                n_girafe += 1
        else:
            if kf["time"][n_epoch] == obs["time"][n_obs]:
                n_obs, P, x = process_ds_measurement(n_obs, P, x)
            if kf["time"][n_epoch] == DS_girafe["time"][n_girafe]:
                n_girafe, P, x = process_girafe_measurement(n_girafe, P, x)

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
        
        girafe_std = 200    # Reset for next Iteration 
    print(n_obs)
    return kf, rts


def plot_ref_line_GIRAFE(rts_final):
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
    Ref["Line_1"] = np.arange(min+450,min+1850+1)
    Ref["Line_2"] = np.arange(min+2350,min+4100+1)
    Ref["Line_3"] = np.arange(min+4490,min+5850+1)
    Ref["Line_4"] = np.arange(min+6400,min+8300+1)
    Ref["Line_5"] = np.arange(min+8650,max-350+1)


    # Calc distance to refererence point. 
    def dist2ref(x1, y1, refx2, refy2): 
        dist = np.sqrt((x1 - refx2)**2 + (y1 - refy2)**2)
        return dist

    ref_x = UTM_corr[0][[min+300]]
    ref_y = UTM_corr[1][[min+300]]

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
        RMS_line[Line[i]] = ds.interpolate_DS(dist_line*1e-3, rts_final["profile"][Ref[Line[i]], 0], ref_line_dist, "linear", "extrapolate")

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
    
    rms_ = rms(res_ref_concatenated)/np.sqrt(2)
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
            print(f"{excluded_line}          {np.round(mean,2)}    {np.round(min,2)}   {np.round(max,2)}    {np.round(STD,2)}    {np.round(rms_,2)}    {np.round(rmse,2)}")
        return
    LOO_summary_stat(RMS_line)
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
    # # plt.title("Flight 115, Biscay Bay Reference Line: Attitude @ 300Hz\nKalman and RTS")
    # plt.xlim(5, 140)
    # # plt.ylim(-60, 60)
    # # save_path = "../Figures/Figures4meeting/Biscay_lines_nav@300Hz_final_Kalman_RTS.png"
    # # plt.savefig(save_path)
    # # plt.show()

    fig, ax=plt.subplots(figsize=(10,5))
    for i in range(5): 
        ax.plot(Line[i].dist*1e-3, Line[i].dg, label =  f"Profile {i+1}", linewidth=3, zorder=0)

    ax.text(0.75, 0.08, f'RMSE: {rms_:.2f} [mGal]', transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.plot(RMS_line["dist"], RMS_line["mean_line"], 'b--', linewidth=1, label="Mean reference", zorder=0)
    ax.legend(loc="lower right", fontsize="12")
    ax.set_xlabel("Distance From Reference Point [km]")
    ax.set_ylabel("Gravity Disturbance [mGal]")
    ax.set_xlim(5, 140)

    # axins = inset_axes(ax, "100%", "100%", bbox_to_anchor=[.2, .55, .4, .4], bbox_transform=ax.transAxes, borderpad=0)
    insetPosition = Bbox.from_bounds(0.3, 0.55, 0.3, 0.3)  # Define position of inset axes
    axins = fig.add_axes(insetPosition)

    for i in range(5): 
        axins.plot(Line[i].dist*1e-3, Line[i].dg, linewidth=2)
    axins.set(xlim=(24,36), ylim=(-17.5,-2.5))
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
    save_path = "../../Figures/Figures4meeting/TEST_NICE_FIGURE.eps"
    # plt.show()
    plt.savefig(save_path)

    plt.figure(figsize=(12,6))
    plt.plot(survey["time"], survey["dg"],'--', lw=1.5, label="Full Survey")
    plt.plot(survey["time"][min+450:min+1850+1], survey["dg"][min+450:min+1850+1], label="Profile 1", lw=3)
    plt.plot(survey["time"][min+2350:min+4100+1], survey["dg"][min+2350:min+4100+1], label="Profile 2", lw=3)
    plt.plot(survey["time"][min+4490:min+5850+1], survey["dg"][min+4490:min+5850+1], label="Profile 3", lw=3)
    plt.plot(survey["time"][min+6400:min+8300+1], survey["dg"][min+6400:min+8300+1], label="Profile 4", lw=3)
    plt.plot(survey["time"][min+8650:max-350+1], survey["dg"][min+8650:max-350+1], label="Profile 5", lw=3)

    plt.legend()
    # for i in range(5):
    #     plt.plot(survey["time"][Ref1[Line[i]]], survey["dg"][Ref1[Line[i]]], lw=3, label =  f"Line$_{i+1}$")
    plt.xlabel("Time")
    plt.ylabel("Gravity Disturbance [mGal]")
    plt.show()


def rewritePDF2Tuple(PDF, index_start, index_end):       # Findes også i DS
    
    if isinstance(PDF, dict):  # Check if PDF is already a dictionary
        return {key: value[index_start:index_end] for key, value in PDF.items()}

    # Slice the DataFrame to remove the specified range from the start and end of each column
    PDF = PDF.iloc[index_start:index_end, :]

    # Reset the index to start from 0 again
    PDF = PDF.reset_index(drop=True)

    temp = {}
    for key, value in PDF.items():
        # temp[key] = value[index_start:index_end]
        temp[key] = value
    return temp


def rewritePDF2Tuple(PDF, index_start, index_end):       # Findes også i DS
    
    if isinstance(PDF, dict):  # Check if PDF is already a dictionary
        return {key: value[index_start:index_end] for key, value in PDF.items()}

    # Slice the DataFrame to remove the specified range from the start and end of each column
    PDF = PDF.iloc[index_start:index_end, :]

    # Reset the index to start from 0 again
    PDF = PDF.reset_index(drop=True)

    temp = {}
    for key, value in PDF.items():
        # temp[key] = value[index_start:index_end]
        temp[key] = value
    return temp


def run_kalman(DS_girafe, DS): 
    DS_girafe = rewritePDF2Tuple(DS_girafe, 200, -200)

    # DS_girafe = Cutoff_Turning(DS_girafe)
    kf, rts = KalmanFilterGIRAFE(DS, DS_girafe)

    rts_final = RTS.RTS_smoother(rts)

    cutoff = 1
    plt.figure(figsize=(12,6))
    plt.plot(rts_final["time"][cutoff:-cutoff], rts_final["profile"][cutoff:-cutoff,0], label="RTS Girafe + IMAR", linewidth=2)
    # plt.plot(girafe["time"][500:-200-1], girafe["dg"][500:-200], label="GIRAFE 130s", linewidth=2)
    plt.legend(loc="upper left")
    # plt.ylim(12300, 12600)
    plt.grid()
    plt.xlabel("Time, Secound of day [SOD]")
    plt.ylabel("Gravity Disturbance [mgal]")
    plt.grid()
    # save_path = "../../Figures/Figures4meeting/Biscay_Legacy_code_300Hz_Kalman_lowpass_full.png"
    # plt.savefig(save_path)
 
    # Plot Reference lines (Verification)
    # plot_ref_line(rts_final)

    turning = {}
    turning["T1"] = np.arange(50, 650)
    turning["T2"] = np.arange(1950, 2600)
    turning["T3"] = np.arange(4150, 4800)
    turning["T4"] = np.arange(6050, 6700)
    turning["T5"] = np.arange(8250, 9000)

    # Create a mask to identify indices to keep
    turning_idx = np.ones(len(DS_girafe["time"]))*70
    for indices in turning.values():
        turning_idx[indices] = 0
    boundaries = np.diff(turning_idx).nonzero()[0] + 1

    plt.figure(figsize=(12,6))
    plt.plot(kf["time"][cutoff:-cutoff], kf["profile"][cutoff:-cutoff,0], label="Kalman Girafe + IMAR", linewidth=2)
    
    plt.plot(rts_final["time"][cutoff:-cutoff], rts_final["profile"][cutoff:-cutoff,0], label="RTS Girafe + IMAR", linewidth=2)
    # plt.plot(DS_girafe["time"], turning_idx, label="Turning Index", lw=2)
    for i, (start, end) in enumerate(zip(boundaries[::2], boundaries[1::2])):
        plt.fill_between(DS_girafe["time"][start:end], -80, 80, color='grey', alpha=0.3, label="Turning Index" if i == 0 else None)

    # plt.plot(DS_girafe["time"][500:-200-1], DS_girafe["dg"][500:-200], label="GIRAFE 130s", linewidth=2)
    plt.legend(loc="upper left")
    plt.ylim(-80,80)
    plt.grid()
    plt.xlabel("Time, Secound of day [SOD]")
    plt.ylabel("Gravity Disturbance [mgal]")
    plt.grid()

    plot_ref_line_GIRAFE(rts_final)


    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,6))

    ax1.plot(rts_final["time"][cutoff:-cutoff], rts_final["profile"][cutoff:-cutoff,0], label="RTS GIRAFE + IMAR", linewidth=2)
    ax2.plot(rts_final["time"][cutoff:-cutoff], rts_final["profile"][cutoff:-cutoff,3], label="Bias", linewidth=2)
    ax1.legend(loc="upper left")
    ax2.legend(loc = "upper left")
    # plt.grid()
    # ax2.grid()
    ax2.set_xlabel("Time, Secound of day [SOD]")
    ax1.set_ylabel("Gravity Disturbance [mgal]")
    ax2.set_ylabel("Bias [mGal]")
    plt.show()

    pkl_file_path = '../Results/RTS_IMAR_GIRAFE_RAW.pkl'

    # Save the solution dictionary to a pickle file
    with open(pkl_file_path, 'wb') as pkl_file:
        pickle.dump(rts_final, pkl_file)

    print(f'Data saved to {pkl_file_path}')



def count_false_beginning_end(index):
    # Count False values at the beginning
    count_beginning = 0
    for val in index:
        if val == False:
            count_beginning += 1
        else:
            break  # Break the loop when encountering the first True value
    
    # Count False values at the end
    count_end = 0
    for val in reversed(index):
        if val == False:
            count_end += 1
        else:
            break  # Break the loop when encountering the first True value
    
    return count_beginning, count_end

def workAround(): 
    keys = ["sfreq", "srate"]
    for i in range(len(keys)): 
        DS_girafe[keys[i]] = DS_girafe[keys[i]]*np.ones(DS_girafe["time"].shape)


if __name__ == "__main__":

    # ---------------------- Data Load + raw Visualize ------------------------
    file_path_DS = "../Results/DS_300Hz_150s_3_stages.pkl"

    print(f"Loading DS results from: {file_path_DS}")
    with open(file_path_DS, 'rb') as file: 
        DS = pickle.load(file)
    
    file_path_girafe = "../Results/dg_GIRAFE_biscay_ref_profile_2019.pkl"
    print(f"Loading DS results from: {file_path_girafe}")
    with open(file_path_girafe, 'rb') as file: 
        DS_girafe = pickle.load(file)
    DS_girafe["time"] = DS_girafe["time"] #- 7182 
    
    workAround()

    # turning = {}
    # turning["T1"] = np.arange(150, 500)
    # turning["T2"] = np.arange(2050, 2400)
    # turning["T3"] = np.arange(4250, 4650)
    # turning["T4"] = np.arange(6200, 6500)
    # turning["T5"] = np.arange(8350, 8850)

    # # Create a mask to identify indices to keep
    # turning_idx = np.ones(len(DS_girafe["time"]), dtype=bool)
    # for indices in turning.values():
    #     turning_idx[indices] = False

    # plt.figure()
    # plt.plot(DS_girafe["time"], DS_girafe["dg"], label="Full")
    # plt.plot(DS_girafe["time"][turning_idx], DS_girafe["dg"][turning_idx], label="Cut")
    # plt.legend()
    # plt.show()

    # file1 = Path("..", "..", "data", "airgravi2019_etalon", "girafe", "GIRAFE_Raw_Calibration profile.txt")
    # names = ["time", "az"]
    # girafe_temp = pd.read_csv(file1, header=0, delimiter="\s+", names=names)
    # girafe_temp.time = girafe_temp.time - 7182                   # Time difference to GNSS time
    # DS_girafe, DS, beginning_count = Calc_dg(girafe_temp, DS)
    beginning_count = 8890
    run_kalman(DS_girafe, DS)
    