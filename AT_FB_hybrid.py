# ----------------Hybridization of GIRAFE and Classical Gravimeter ---------------
#
# Hybridization method, based on the ONERA's method described in: 
# "Absolute marine gravimetry with matter-wave interferometry"
#           [Y. Bidel et al. 2018] 
# 
# ---------------------------------------------------------------------------------
# Author: Christian Solgaard (DTU) 18/01-2024 
# ---------------------------------------------------------------------------------

import numpy as np
import copy
import scienceplots
import pickle
import matplotlib.pyplot as plt
# plt.style.use(['science', 'grid', 'no-latex'])
plt.style.use(['science', 'grid'])

from dataclasses import dataclass, asdict 
import GIRAFE_src as src
from tqdm import tqdm
from numba import jit

params = {'axes.labelsize': 'x-large', 'axes.titlesize':'xx-large','xtick.labelsize':'large', 
          'ytick.labelsize':'large', 'legend.fontsize': 'x-large','mathtext.fontset':'stix', 'font.family':'STIXGeneral'}
plt.rcParams.update(params)

from pathlib import Path
import pandas as pd
import os
import warnings

# To ignore all warnings
warnings.filterwarnings("ignore")

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
    HOD = SOD.values/(60*60)
    return HOD



def Load_data(file_path):
    
    names = ["time", "P2", "T_", "alpha", "az0", "az1", "ax0", "ax1", "ay0", "ay1", "temp", "aatcont"]
    girafe = pd.read_csv(file_path, header=0, delimiter="\t+", names=names)

    titles = ["Placeholder",
            "Probability transition measured after the atom interferometer sequence", 
            "Half-interrogation time used in atom interferometer", 
            "Radiofrequency ramp applied to Raman laser frequency",
            "Acceleration measured by the classical accelerometer along z convolute \nwith the transfer function of the atom accelerometer",
            "Acceleration measured by the classical accelerometer along z averaged \nover one measurement cycle (100 ms)",
            "Acceleration measured by the classical accelerometer along z convolute \nwith the transfer function of the atom accelerometer",
            "Acceleration measured by the classical accelerometer along z averaged \nover one measurement cycle (100 ms)",
            "Acceleration measured by the classical accelerometer along y convolute \nwith the transfer function of the atom accelerometer",
            "Acceleration measured by the classical accelerometer along y averaged \nover one measurement cycle (100 ms)",
            "Temperature of the classical accelerometer z",
            "Acceleration deduced from our hybridization algorithm \n(mean acceleration over one measurement cycle)"]
    ylabels = ["Placeholder",
            "P2 []", 
            "T [s]", 
            r"alpha [Hz/s]", 
            r"az0 [$10^{-5} \mathrm{~m} / \mathrm{s}^5$]", 
            r"az1 [$10^{-5} \mathrm{~m} / \mathrm{s}^5$]",
            r"ax0 [$10^{-5} \mathrm{~m} / \mathrm{s}^5$]", 
            r"ax1 [$10^{-5} \mathrm{~m} / \mathrm{s}^5$]", 
            r"ay0 [$10^{-5} \mathrm{~m} / \mathrm{s}^5$]", 
            r"ay1 [$10^{-5} \mathrm{~m} / \mathrm{s}^5$]", 
            r"temp [$C^{\circ}$]", 
            r"aatcont [$10^{-5} \mathrm{~m} / \mathrm{s}^5$]"]


    # # ------------------------- Plot of RAW GIRAFE data -------------------
    # for i in range(len(names)):
    #     if names[i] in ("az0","az1"):
    #         plt.figure(figsize=(10,5))
    #         plt.plot(girafe.time, girafe[names[i]])
    #         plt.xlabel("Time Seconds of Day [SOD], UTC ref: 00:00")
    #         plt.ylabel(ylabels[i])
    #         plt.title(titles[i])
    #         plt.show()
    return girafe, names

# Initialize some parameters from Y. Bidel. 
Pm = 0.486
C = 0.26
keff = 16105753.52

def Convert_units_2_mGal(): 
    # Convert Units to [mGal]
    girafe["T2"] = girafe.T_**2
    for i in range(len(names)): 
        if names[i] in ("az0", "az1"): 
            girafe[names[i]] = (girafe[names[i]]/100000)        #+9.8            # Conversion to [mGal] and voltage offset ()
        elif names[i] in ("ax0", "ax1", "ay0", "ay1", "aatcont"): 
            girafe[names[i]] = girafe[names[i]]/100000                  # Conversion to [mGal]
    # girafe.time = girafe.time - 18

def plot_time_stamps():
    # Check time stamps: 
    dt = np.diff(girafe.time)
    plt.figure(figsize=(8,4))
    plt.plot(dt)
    plt.ylim(0.098, 0.102)
    plt.ylabel('Time Increment [s]')
    plt.xlabel("Index")
    output_name = "../Results/Figures/Timestamp_check_ONERA_Hybrid.eps"
    plt.savefig(output_name)
    # plt.show()

def plot_T(): 
    plt.figure(figsize=(8,4))
    plt.plot(girafe.time, girafe.T_*1000)
    # plt.ylim(0.098, 0.102)
    plt.ylabel('T Half Interrogation Time [ms]')
    plt.xlabel("Time [s]")


def plot_az():
    ## Plot accelerations along z 
    plt.figure(figsize=(12,6))
    plt.plot(SOD2HOD(girafe.time), girafe.az0, label=r"az0 - Classicl Convoluted with h$_{at}$", lw=.8, linestyle="-.")
    plt.plot(SOD2HOD(girafe.time), girafe.az1, label="az1 - Classical 100 ms average", lw=.8, linestyle="-.")
    plt.xlabel("Time Hour of Day [HOD]")
    plt.ylabel(r"Acceleration [m/s$^2$]")
    plt.legend()
    output_name = "../Results/Figures/az0Conv_az1.eps"
    plt.savefig(output_name)
    # plt.show()


## Estimate Probablity Offset and Contrast Pm and C
@jit
def Calc_Pm_and_C(df):
    
    #window size
    wlen = 100

    # allocate memory
    df["C"] = np.zeros_like(df.time, dtype=float)
    df["Pm"] = np.zeros_like(df.time, dtype=float)
    df["alpha2"] = np.zeros_like(df.time, dtype=float)

    # Loop through the data with tqdm
    for n_epoch in tqdm(range(wlen, len(df.time) - wlen), colour="green"):
        # Extract observations
        y = df.P2[n_epoch - wlen:n_epoch + wlen]

        # Determine Range
        ymax = np.nanmax(y)
        ymin = np.nanmin(y)
        yrange = ymax - ymin
        # yrange = np.nanmax(y) - np.nanmin(y)

        # Derive contract and offset
        df.C[n_epoch] = yrange
        df.Pm[n_epoch] = ymax - yrange/2

        # Correct Radiofrequency fringe alpha
        df.alpha2[n_epoch] = (abs(df.alpha[n_epoch]) - keff * df.aatcont[n_epoch - 1]) * df.T2[n_epoch]

    df.alpha = df.alpha-df.alpha2

    df.C[:wlen] = df.C[wlen+1]
    df.C[n_epoch:] = (df.C[:-wlen]).iloc[-1]

    df.Pm[:wlen] = df.Pm[wlen+1]
    df.Pm[n_epoch:] = (df.Pm[:-wlen]).iloc[-1]

    # plt.figure(figsize=(10,5))
    # plt.plot(df.time, df.P2)

    return df, n_epoch
# girafe, n_epoch = Calc_Pm_and_C(girafe)

def plot_last_estimate_Pm_C():

    # # window size
    wlen = 100

    # Extract observations
    y = girafe['P2'][n_epoch - wlen:n_epoch + wlen]
    alpha_abs = np.abs(girafe['alpha'][n_epoch - wlen:n_epoch + wlen])

    # Plot scatter plot
    plt.figure(figsize=(6,6))
    plt.plot(alpha_abs, y, '.')

    # Plot horizontal line with dashed red style
    plt.plot([np.min(alpha_abs), np.max(alpha_abs)], [girafe['Pm'][n_epoch], girafe['Pm'][n_epoch]], '--r', label=r'Offset, P$_{m}$= $\bar{\mu}_{P2_n}$', lw=2)
    plt.plot(np.min(alpha_abs) * np.ones(2), [girafe['Pm'][n_epoch]-0.02, girafe['Pm'][n_epoch]+0.02], '--r', lw=2)
    plt.plot(np.max(alpha_abs) * np.ones(2), [girafe['Pm'][n_epoch]-0.02, girafe['Pm'][n_epoch]+0.02], '--r', lw=2)

    # Plot dashed blue lines for range
    range_ = (np.min(alpha_abs) + np.max(alpha_abs))/2
    # plt.plot(range_ * np.ones(2), [girafe['Pm'][n_epoch] - girafe['C'][n_epoch]/2, girafe['Pm'][n_epoch] + girafe['C'][n_epoch]/2], '--b', label=r'Contrast, C = min($P2_n$) - max($P2_n$))', lw=2)
    plt.plot(range_ * np.ones(2), [girafe['Pm'][n_epoch] - girafe['C'][n_epoch]/2, girafe['Pm'][n_epoch] + girafe['C'][n_epoch]/2], '--b', label=r'Contrast, C = Range', lw=2)
    plt.plot([range_-0.014*10**7, range_+0.014*10**7], (girafe['Pm'][n_epoch] - girafe['C'][n_epoch]/2)*np.ones(2), '--b', lw=2)
    plt.plot([range_-0.014*10**7, range_+0.014*10**7], (girafe['Pm'][n_epoch] + girafe['C'][n_epoch]/2)*np.ones(2), '--b', lw=2)


    # Set labels and grid
    plt.ylabel(r'P2 []')
    plt.xlabel(r'$\vert \alpha \vert$ [Hz/s]')

    # Show plot
    legend = plt.legend(loc="upper left", bbox_to_anchor=(.6, 1))
    output_name = "../Results/Figures/Pm_C_last_epoch.eps"
    plt.savefig(output_name)

    # plt.show()


    fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

    ax[0].plot(SOD2HOD(girafe.time), girafe.C, ".-", markersize=.5, lw=.5, label="C")
    ax[0].set_ylabel("Contrast C")
    ax[0].text(10.7, .53, r"$\mathbf{a)}$", fontsize=14)
    ax[0].axhline(.26, label="Value from Yannic", color="red", lw=2)
    ax[0].legend()

    ax[1].plot(SOD2HOD(girafe.time), girafe.Pm, ".-", markersize=.5, lw=.5, label="Pm")
    # ax[1].set_ylim(0.25, 0.5)
    ax[1].set_ylabel("Offset Pm")
    ax[1].set_xlabel("Time Hour of Day [h]")
    ax[1].text(10.7, .45, r"$\mathbf{b)}$", fontsize=14)
    ax[1].axhline(.486, label="Value from Yannic", color="red", lw=2)
    ax[1].legend()
    plt.tight_layout()
    output_name = "../Results/Figures/Pm_C.pdf"
    plt.savefig(output_name)
    output_name = "../Results/Figures/Pm_C.png"
    plt.savefig(output_name)


## ------------- Hybridization -----------------
def hybridization():
    # Allocate memory 
    girafe["aatcont2"] = np.zeros_like(girafe.time, dtype=float)
    girafe["s"] = np.zeros_like(girafe.time, dtype=float)
    girafe["n"] = np.zeros_like(girafe.time, dtype=float)

    term_arg = 2*(girafe.Pm-girafe.P2)/girafe.C
    # Ensure the argument is within the valid range [-1, 1]
    flag = ((term_arg > 1) | (term_arg < -1))           # Flag constrained events for later analysis
    term_arg = np.clip(term_arg, -1, 1)                 # Constrain term to range [-1, 1] for cosine 

    term1 = np.arccos(term_arg)

    n1 = np.round(((girafe.az0 - 2 * np.pi * girafe.alpha / keff) * keff * girafe.T2 - term1) / 2 / np.pi)
    n2 = np.round(((girafe.az0 - 2 * np.pi * girafe.alpha / keff) * keff * girafe.T2 + term1) / 2 / np.pi)

    # plt.figure(figsize=(10,5))
    # plt.plot(girafe.time, girafe.az0, label="az0")
    # plt.legend()

    # plt.figure(figsize=(10,5))
    # plt.plot(girafe.time, girafe.alpha, label="alpha")
    # plt.legend()
    
    # plt.figure(figsize=(10,5))
    # plt.plot(girafe.time, girafe.T2, label="T2")
    # plt.legend()
    
    # plt.figure(figsize=(10,5))
    # plt.plot(girafe.time, term1, label="term1")
    # plt.legend()

    # print(f"Value of keff: {keff}")


    val1 = (term1 + 2 * np.pi * n1) / keff / girafe.T2 + 2 * np.pi * girafe.alpha / keff
    val2 = (-term1 + 2 * np.pi * n2) / keff / girafe.T2 + 2 * np.pi * girafe.alpha / keff

    dif1 = np.abs(val1 - girafe.az0)
    dif2 = np.abs(val2 - girafe.az0)

    condition_mask = dif1 < dif2

    # Use the mask to set values for girafe.n, girafe.s, and girafe.aatcont2
    girafe.n[condition_mask] = n1[condition_mask]
    girafe.s[condition_mask] = 1
    girafe.aatcont2[condition_mask] = val1[condition_mask]

    # Use the inverse of the mask to set values for the other condition
    girafe.n[~condition_mask] = n2[~condition_mask]
    girafe.s[~condition_mask] = -1
    girafe.aatcont2[~condition_mask] = val2[~condition_mask]

    plt.figure(figsize=(10,6))
    plt.plot(girafe.time, girafe.aatcont2)
    

    girafe_a_cont = girafe.aatcont2*(-1) + girafe.az0 - girafe.az1
    # girafe_a_cont = girafe.az0 - girafe.az1

    # girafe["girafe_a_cont"] = girafe.aatcont2
    girafe["girafe_a_cont"] = girafe_a_cont


def plot_s_n():
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    ax[0].plot(SOD2HOD(girafe.time), girafe.n, ".", markersize=.5)
    ax[0].set_ylabel("Integer n [.]")
    ax[0].text(10.7, 1500, r"$\mathbf{a)}$", fontsize=14)

    ax[1].plot(SOD2HOD(girafe.time), girafe.s, ".", markersize=.5)
    ax[1].set_ylabel("Sign s [.]")
    ax[1].set_xlabel("Time Hour of Day [h]")
    ax[1].text(10.7, .7, r"$\mathbf{b)}$", fontsize=14)
    plt.tight_layout()
    output_name = "../Results/Figures/s_n.pdf"
    plt.savefig(output_name)
    output_name = "../Results/Figures/s_n.png"
    plt.savefig(output_name)
    # plt.show()


def plot_compare_result():
    
    fONERA = src.but2_v2(girafe.aatcont.values, 3, 174, .1)
    fcsol = src.but2_v2(girafe.girafe_a_cont.values, 3, 174, .1)


    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.plot(SOD2HOD(girafe.time), girafe.aatcont, label = "ONERA", lw=.8, linestyle="-.")
    plt.plot(SOD2HOD(girafe.time), girafe.girafe_a_cont, label = "csol", lw=.8, linestyle="-.")
    
    # plt.plot(SOD2HOD(girafe.time), fONERA, label = "ONERA", lw=.8, linestyle="-.")
    # plt.plot(SOD2HOD(girafe.time), fcsol, label = "csol", lw=.8, linestyle="-.")

    # plt.plot(SOD2HOD(girafe.time), fONERA, label = "ONERA", lw=2, linestyle="-.")
    # plt.plot(SOD2HOD(girafe.time), fcsol, label = "csol", lw=2, linestyle="-.")


    plt.ylabel(r"Downward acceleration [m/s$^2$]")
    plt.legend()
    plt.text(10.7, 13.5, r"$\mathbf{a)}$", fontsize=14)

    # plt.show()
    plt.subplot(2,1,2)
    plt.plot(SOD2HOD(girafe.time), (girafe.aatcont - girafe.girafe_a_cont)*10**5, label="Difference (ONERA - csol)", lw=.8, linestyle="-.")
    # plt.plot(SOD2HOD(girafe.time), (fONERA - fcsol)*10**5, label="Difference (ONERA - csol)", lw=.8, linestyle="-.")
    # plt.plot(SOD2HOD(girafe.time), (fONERA - fcsol)*10**5, label="Difference (ONERA - csol)", lw=2, linestyle="-.")

    plt.xlabel("Time Hour of Day [h]")
    plt.ylabel(r"Downward acceleration [mGal]")
    plt.text(10.7, -100, r"$\mathbf{b)}$", fontsize=14)
    plt.legend(loc="lower right")
    plt.tight_layout()

    output_name = "../Results/Figures/Compare_ONERA_with_ownhybridization_.eps"
    plt.savefig(output_name)
    output_name = "../Results/Figures/Compare_ONERA_with_ownhybridization_.png"
    plt.savefig(output_name)



def correct_bias_cl(): 
    # Correction to the vertical component of the FB gravimeter, in order to alligen with AT 


    girafe.ax0 = girafe.ax0 + 0.01247 # 1247/1000    # Convert til m/s^2
    girafe.ax1 = girafe.ax1 + 0.01247 # 1247/1000
    girafe.ay0 = girafe.ay0 - 0.00286 # 286/1000
    girafe.ay1 = girafe.ay1 - 0.00286 # 286/1000

    epsilonX = -5.1e-4
    epsilonY = -7.4e-5 

    girafe.az0 = girafe.az0 - (epsilonX*girafe.ax0 + epsilonY*girafe.ay0)
    girafe.az1 = girafe.az1 - (epsilonX*girafe.ax1 + epsilonY*girafe.ay1)

    # girafe["az_corr0"] = (5.1e-4) * girafe.ax0 + (7.4e-5)*girafe.ay0
    # girafe.az0 = girafe.az_corr0

    # girafe["az_corr1"] = (5.1e-4) * girafe.ax1 + (7.4e-5)*girafe.ay1
    # girafe.az1 = girafe.az_corr1

def correct_bias_at():
    for i in range(len(girafe.time)): 
        if girafe.T_[i]*1000 == 20: 
            girafe.girafe_a_cont[i] = girafe.girafe_a_cont[i]+.59/1000000
        elif girafe.T_[i]*1000 == 10: 
            girafe.girafe_a_cont[i] = girafe.girafe_a_cont[i]+.825/1000000
        elif girafe.T_[i]*1000 == 5: 
            girafe.girafe_a_cont[i] = girafe.girafe_a_cont[i]+3.365/1000000
        elif girafe.T_[i]*1000 == 2.5: 
            girafe.girafe_a_cont[i] = girafe.girafe_a_cont[i]+11.68/1000000 


if __name__ == "__main__":
    file = Path("..", "..", "data", "ONERA_RAW_ref_profile", "DataGIRAFERefProf2019.txt")
    girafe, names = Load_data(file)

    Convert_units_2_mGal()

    plot_T()
    plot_time_stamps()

    plot_az()

    girafe, n_epoch = Calc_Pm_and_C(girafe)

    plot_last_estimate_Pm_C()

    correct_bias_cl()
    
    hybridization()

    correct_bias_at()

    plot_s_n()

    plot_compare_result()
    
    plt.figure(figsize=(10,6))
    plt.plot(girafe.time, girafe.alpha, label="alpha Bidel")
    plt.plot(girafe.time, girafe.alpha2, label="alpha csol")
    plt.legend()

    
    plt.show()


    # pkl_file_path = '../Results/AT_FB_hybrid.pkl'
    # with open(pkl_file_path, 'wb') as pkl_file:
    #     pickle.dump(girafe, pkl_file)
    # print(f"Data saved to {pkl_file_path}")
    