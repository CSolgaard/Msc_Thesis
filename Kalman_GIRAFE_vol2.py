# -----------------------Kalman Filters - Direct Strapdown vs Quantum --------------
#  
#
# 
# ---------------------------------------------------------------------------------
# Author: Christian Solgaard (DTU) 13/02-2024 
# ---------------------------------------------------------------------------------




from numba import jit, njit
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
# import RTS 
# from src import RTS as RTS
import RTS as RTS
import matplotlib.pyplot as plt
plt.style.use(['science', 'grid'])
# plt.style.use(['science', 'grid'])

params = {'axes.labelsize': 'x-large', 'axes.titlesize':'xx-large','xtick.labelsize':'large', 
          'ytick.labelsize':'large', 'legend.fontsize': 'x-large','mathtext.fontset':'stix', 'font.family':'STIXGeneral'}
plt.rcParams.update(params)

import warnings
# To ignore all warnings
warnings.filterwarnings("ignore")
# -----------------------------------------------------------------------------------------------

@jit
def RC_filter(data, stage, ftc, dt):
    """
     RC-filter from Dag Solheim
     -------------------------------------------------------------------------
    
     Input:
       data    Vector of data series
       stage   Number of iterations, 1 iteration = forwar+backward run
       ftc     Filter time constant [s]
       dt      Sample interval [s]
    
     Output:
       fdata   Vector of filtered data series
    
     -------------------------------------------------------------------------
     Author: Christian Solgaard (DTU) 23/10-2023, based on the implementation 
     done by: Tim Jensen (DTU) 29/05/2020
    """

    
    # Set some parameters
    edge = 200

    # Derive data length
    nmax = len(data)

    # Prolong Data Series with 'edge' constant readings in Each End
    n1 = edge
    n2 = nmax + edge + 1
    n3 = nmax + 2 * edge

    # Allocate space
    p = np.empty(n3)
    p[:n1] = data[0]
    p[edge:n2-1] = data
    p[n2:] = data[-1]

    # Update data length
    nmax = n3

    # Perform Filtering
    a = dt / (2 * ftc)
    b = (1.0 - a) / (1 + a)
    c = a / (1 + a)

    for m in range(stage):

        # Forward run
        a, d = p[0], p[0]

        for i in range(1, nmax):
            e = b * a + c * (p[i] + d)
            d, p[i] = p[i], e
            a = e

        # Reverse run
        a, d = p[-1], p[-1]

        for i in range(nmax - 2, -1, -1):
            e = b * a + c * (p[i] + d)
            d, p[i] = p[i], e
            a = e

    # Extract original part and restore mean value
    fdata = p[n1 + 1 : n2 - 1]

    return fdata

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
    # DS_girafe["time_girafe"]= DS_girafe["time_girafe"]       # +2 hour UTC time - 18 GPS clock offset
    obs["time"] = SOW2SOD(obs["time"])          # Convert IMU/GNSS time to seconds of day instead of week 

    # Define common timeseries. 
    kf = {}
    time = np.concatenate((obs["time"], DS_girafe["time_girafe"]), axis=None)
    time = np.array(sorted(np.unique(time)))
    kf["time"] = time

    # Add a fictive time stamp in end of each timeline 
    obs["time"] = np.concatenate((obs["time"], kf["time"][-1] + 100), axis=None)
    DS_girafe["time_girafe"] = np.concatenate((DS_girafe["time_girafe"], kf["time"][-1] + 100), axis=None)

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
    dg_obs_std = 166  # Standard deviation of Observation (IMAR) [mGal]
    girafe_std = 200    # Standard deviation of measurement (GIRAFE) [mGal]

    dg_sigma = 100    # Gravity anomaly standard deviation [mGal]
    dg_beta = 1 / 5000  # Gravity anomaly correlation parameter [1/m]

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
    n_girafe = 0

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
            dg_obs = obs['dg'][n_obs, 2]-40   # Extract estimate of dg [mGal]

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

        # --------------------- Update GIRAFE measurement ----------------------------
        if kf["time"][n_epoch] == DS_girafe["time_girafe"][n_girafe]: 
            
            # Extract Observation 
            dg_obs = DS_girafe['girafe_dg'][n_girafe]   # Extract estimate of dg [mGal]

            # From Measurement Error Covariance Matrix
            R = girafe_std**2

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
            n_girafe = n_girafe + 1

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


def plot_ref_line_GIRAFE(rts_final):
    import pyproj
    from pyproj import Transformer
    survey = {}
    survey["dg"] = rts_final["profile"][:,0]
    survey["time"] = rts_final["time"]

    rts_corr = {}
    rts_corr["lat"] = ds.interpolate_DS(DS["time"][1:], DS["lat"], rts_final["time"], "linear", "extrapolate")
    rts_corr["lon"] = ds.interpolate_DS(DS["time"][1:], DS["lon"], rts_final["time"], "linear", "extrapolate")
    rts_corr["h"] = ds.interpolate_DS(DS["time"][1:], DS["h"], rts_final["time"], "linear", "extrapolate")


    pipeline = "+ellps=GRS80 +proj=pipeline +step +proj=utm +zone=30"
    transform_object = Transformer.from_pipeline(pipeline)
    geodetic_corr = [rts_corr["lon"], rts_corr["lat"], rts_corr["h"]]
    UTM_corr = transform_object.transform(*geodetic_corr)

    min = 9700 - beginning_count
    max = 30000 - beginning_count

    Ref = {}                                        # Determined using visual inspection
    Ref["all_lines"] = np.arange(min,max+1)
    Ref["Line_1"] = np.arange(min+850,min+3400+1)
    Ref["Line_2"] = np.arange(min+4200,min+7400+1)
    Ref["Line_3"] = np.arange(min+8300,min+10800+1)
    Ref["Line_4"] = np.arange(min+11900,min+15000+1)
    Ref["Line_5"] = np.arange(min+15900,max-1700+1)

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
        rms_ = np.sum(x**2)/len(x)
        return rms_

    rms_ = rms(res_ref_concatenated)/np.sqrt(2)

    fig = plt.figure(figsize=(10,5))
    for i in range(5): 
        plt.plot(Line[i].dist*1e-3, Line[i].dg-50, label =  f"Line$_{i+1}$", linewidth=3)

    plt.text(0.75, 0.08, f'RMS: {rms_:.2f} [mGal]', transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # plt.grid()
    plt.plot(RMS_line["dist"], RMS_line["mean_line"], 'b--', linewidth=1, label="Mean reference")
    plt.legend(loc="lower right", fontsize="12")
    plt.xlabel("Distance from reference Point [km]")
    plt.ylabel("Gravity Disturbance [mGal]")
    plt.title("Flight 115, Biscay Bay Reference Line: Attitude @ 300Hz\nKalman and RTS")
    plt.xlim(5, 140)
    # plt.ylim(-60, 60)
    # save_path = "../Figures/Figures4meeting/Biscay_lines_nav@300Hz_final_Kalman_RTS.png"
    # plt.savefig(save_path)
    # plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(survey["time"], survey["dg"],'--', lw=1.5, label="Full Survey")
    plt.plot(survey["time"][min+850:min+3400+1], survey["dg"][min+850:min+3400+1], label="Line 1", lw=3)
    plt.plot(survey["time"][min+4200:min+7400+1], survey["dg"][min+4200:min+7400+1], label="Line 2", lw=3)
    plt.plot(survey["time"][min+8300:min+10800+1], survey["dg"][min+8300:min+10800+1], label="Line 3", lw=3)
    plt.plot(survey["time"][min+11900:min+15000+1], survey["dg"][min+11900:min+15000+1], label="Line 4", lw=3)
    plt.plot(survey["time"][min+15900:max-1700+1], survey["dg"][min+15900:max-1700+1], label="Line 5", lw=3)

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



def run_kalman(DS_girafe, DS): 
    DS_girafe = rewritePDF2Tuple(DS_girafe, 700, -200)

    # DS_girafe = Cutoff_Turning(DS_girafe)
    kf, rts = KalmanFilterGIRAFE(DS, DS_girafe)

    rts_final = RTS.RTS_smoother(rts)

    cutoff = 1
    plt.figure(figsize=(12,6))
    plt.plot(rts_final["time"][cutoff:-cutoff], rts_final["profile"][cutoff:-cutoff,0], label="RTS Girafe 130s + IMAR", linewidth=2)
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
    plot_ref_line_GIRAFE(rts_final)


    plt.figure(figsize=(12,6))
    plt.plot(kf["time"][cutoff:-cutoff], kf["profile"][cutoff:-cutoff,0], label="Kalman Girafe 130s + IMAR", linewidth=2)

    plt.plot(rts_final["time"][cutoff:-cutoff], rts_final["profile"][cutoff:-cutoff,0], label="RTS Girafe 130s + IMAR", linewidth=2)
    # plt.plot(DS_girafe["time_girafe"][500:-200-1], DS_girafe["girafe_dg"][500:-200], label="GIRAFE 130s", linewidth=2)
    plt.legend(loc="upper left")
    # plt.ylim(12300, 12600)
    plt.grid()
    plt.xlabel("Time, Secound of day [SOD]")
    plt.ylabel("Gravity Disturbance [mgal]")
    plt.grid()

    plt.show()



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


# def Cutoff_Turning(girafe): 
#     turning = {}
#     turning["T1"] = np.arange(150, 500)
#     turning["T2"] = np.arange(2050, 2400)
#     turning["T3"] = np.arange(4250, 4650)
#     turning["T4"] = np.arange(6200, 6500)
#     turning["T5"] = np.arange(8350, 8850)

#     # Create a mask to identify indices to keep
#     mask = np.ones(len(girafe["time"]), dtype=bool)
#     for indices in turning.values():
#         mask[indices] = False

#     # Apply the mask to girafe dictionary
#     girafe["time"] = girafe["time"][mask]
#     girafe["dg"] = girafe["dg"][mask]

#     return girafe


def Calc_girafe_dg(girafe, DS):
    girafe = {}
    for key, value in girafe_temp.items():
        girafe[key] = value

    DS["time_girafe"] = SOW2SOD(DS["time"])

    girafe["faz"] = RC_filter(girafe["az"].values, 2, 2, .1)

    # ------------------ Interpolate needed information from DS solution --------------------
    # Invert => interpolate raw girafe az to GNSS epoch => use results from DS. 

    DS["girafe_az"] = ds.interpolate_DS(girafe["time"][1:], girafe["faz"], SOW2SOD(DS["time"]), "linear", False, "extrapolate")

    start = girafe["time"][0]
    end = girafe["time"].values[-1]
    index = (DS["time_girafe"] >= start) & (DS["time_girafe"] <= end)
    DS["girafe_az"][~index] = 9.8

    beginning_count, end_count = count_false_beginning_end(index)
    
    plt.figure(figsize=(12,6))
    plt.plot(SOD2HOD(DS["time_girafe"]), DS["girafe_az"], label="GIRAFE Measurement @ 1Hz GNSS Epoch")
    plt.plot(SOD2HOD(DS["time_girafe"][~index]), DS["girafe_az"][~index],'.', label="Placeholder Values from interpolation")
    plt.ylabel(r"Vertical Acceleration [m/s$^2$]")
    plt.xlabel("Time Hour of Day [HOD]")
    plt.legend()

    import copy
    DS_girafe = {}
    names = ["lat", "lon", "h", "girafe_az", "time", "fdg", "gps_acc", "tacc", "gamma_down", "time_girafe"]
    for i in range(len(names)): 
        DS_girafe[names[i]] = DS[names[i]][index].copy()

    # ----------------------- Derive Gravity Disturbance --------------------
    DS_girafe["girafe_dg"] = (DS_girafe["gps_acc"][:,2] - DS_girafe["girafe_az"]*(-1) + DS_girafe["tacc"][:,2] - DS_girafe["gamma_down"].flatten())*10**5          # Calculate Gravity Disturbance

    DS_girafe["girafe_fdg"] = ds.but2_v2(DS_girafe["girafe_dg"], 3, 227, 1)

    return DS_girafe, DS, beginning_count


if __name__ == "__main__":
    import Direct_Strapdown as ds

    # ---------------------- Data Load + raw Visualize ------------------------
    file_path = "../Results/DS_300Hz_150s_3_stages.pkl"

    print(f"Loading DS results from: {file_path}")
    with open(file_path, 'rb') as file: 
        DS = pickle.load(file)
    
    file1 = Path("..", "..", "data", "airgravi2019_etalon", "girafe", "GIRAFE_Raw_Calibration profile.txt")
    names = ["time", "az"]
    girafe_temp = pd.read_csv(file1, header=0, delimiter="\s+", names=names)
    girafe_temp.time = girafe_temp.time - 7182                   # Time difference to GNSS time
    DS_girafe, DS, beginning_count = Calc_girafe_dg(girafe_temp, DS)
 
    run_kalman(DS_girafe, DS)
    print(beginning_count)
    


