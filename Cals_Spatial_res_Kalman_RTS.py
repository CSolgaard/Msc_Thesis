## Calculate Spatial FWHM and Spatial Resolution of Kalman Filters using the time domain method from Tims article. 
import numpy as np
import copy
import Direct_Strapdown as ds
import RTS as RTS
import Kalman_vol3_ubuntu as Kalman

import scienceplots
import matplotlib.pyplot as plt
plt.style.use(['science', 'grid'])
params = {'axes.labelsize': 'xx-large', 'axes.titlesize':'xx-large','xtick.labelsize':'xx-large', 
          'ytick.labelsize':'xx-large', 'legend.fontsize': 'xx-large','mathtext.fontset':'stix', 'font.family':'STIXGeneral'}
plt.rcParams.update(params)

import pandas as pd
import os
import warnings
from PIL import Image
from pathlib import Path
import pickle
# To ignore all warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import interp1d
from tqdm import tqdm


## ------------------- Lowpass Kalman Filter ---------------------
# Load IMAR DS Solution 

# file_path = "../Results/DS_300Hz_150s_3_stages.pkl"

# print(f"Loading DS results from: {file_path}")
# with open(file_path, 'rb') as file: 
#     DS = pickle.load(file)

# # Run normal Kalman Lowpass
# kf_orig, rts = Kalman.KalmanFilterLowpass(DS, 2)
# rts_orig = RTS.RTS_smoother(rts)

# # Plot Ref line to check result: 
# # Kalman.plot_ref_line(rts_orig, DS)

# # Ad a dirac delta function to raw signal: 
# deltafunc = np.zeros(len(DS["time"]))
# dirac = 10800
# deltafunc[dirac] = 1

# DS["dg"][:,2] = DS["dg"][:,2] + deltafunc

# # Run second iteration Kalman + RTS with Delta function implemented. 
# kf_new, rts = Kalman.KalmanFilterLowpass(DS, 2)
# rts_new = RTS.RTS_smoother(rts)

# diff = rts_new["profile"][:,0] - rts_orig["profile"][:,0]

# diff = kf_new["profile"][:,0] - kf_orig["profile"][:,0]
# delta_time = ds.SOW2HOD(400)

# # Calculate FWHM for left subplot
# norm_diff = diff/max(diff)
# half_max = 0.5 * np.max(norm_diff)
# half_max_indices = np.where(norm_diff >= half_max)[0]
# fwhm = DS["time"][np.max(half_max_indices)] - DS["time"][np.min(half_max_indices)]

# hwhm = DS["time"][dirac] - DS["time"][np.min(half_max_indices)]
# fwhm = hwhm * 2

# print(f"FWHM = {fwhm} -> Time Domain")

# plt.figure(figsize=(12,6))
# plt.subplot(2,1,1)
# plt.plot(rts_orig["time"], rts_orig["profile"][:,0], lw=2, label="Without Delta Function")
# plt.plot(rts_orig["time"], rts_new["profile"][:,0], lw=2, label=f"With Delta Function @ Time = {np.round(DS['time'][dirac])}s")
# plt.legend()
# plt.axvline(DS["time"][dirac], ls="--", lw=2)
# plt.ylabel(r" $\delta g$ [mGal]")
# plt.xlim(DS["time"][dirac-400], DS["time"][dirac+400])

# plt.subplot(2,1,2)
# plt.plot(rts_new["time"], norm_diff, lw=2)
# plt.plot(rts_new["time"][half_max_indices], norm_diff[half_max_indices], lw=2)
# plt.xlabel("Time Hour of Day [HOD]")
# plt.ylabel("Normalized Difference [mGal]")
# plt.axvline(DS["time"][dirac], ls="--", lw=2)
# plt.xlim(DS["time"][dirac-400], DS["time"][dirac+400])
# plt.tight_layout()

# plt.show()



## --------------------- Test using the spectral domane ---------------------
# diff = yo The output function 
# delta = yi The input function 
# H = fft(yo)/fft(yi)
def Calc_Transfer(method):

    if method == 1: 
        file_path = "../Results/DS_300Hz_150s_3_stages.pkl"
        with open(file_path, 'rb') as file: 
            DS = pickle.load(file)

        # Run normal Kalman Lowpass
        kf_orig, rts, _ = Kalman.KalmanFilterLowpass(DS, 2)
        rts_orig = RTS.RTS_smoother(rts)

        # Plot Ref line to check result: 
        # Kalman.plot_ref_line(rts_orig, DS)

        # Ad a dirac delta function to raw signal: 
        deltafunc = np.zeros(len(DS["time"]))
        dirac = 10800
        deltafunc[dirac] = 1
        DS["dg"][:,2] = DS["dg"][:,2] + deltafunc

        # Run second iteration Kalman + RTS with Delta function implemented. 
        kf_new, rts, _ = Kalman.KalmanFilterLowpass(DS, 2)
        rts_new = RTS.RTS_smoother(rts)

        diff = rts_new["profile"][:,0] - rts_orig["profile"][:,0]

    elif method == 2: 
        file_path = "../Results/DS_300Hz_150s_3_stages.pkl"
        with open(file_path, 'rb') as file: 
            DS = pickle.load(file)
        
        DS["vel_scalar"] = Kalman.vel_scalar(DS["vn"], DS["ve"], DS["vd"])
        
        BF = Kalman.Createdict()
        dg = ds.but2_v2(DS["dg"][450:-450, 2], BF["stages"], BF["ftc"], BF["srate"])
        Gravtie = Kalman.getGravTIE(DS["time"][450:-450], DS["vel_scalar"][450:-450], dg)
        kf, rts = Kalman.KalmanFilterTIE(DS, Gravtie, 2)
        
        rts_orig = RTS.RTS_smoother(rts)
        with open(file_path, 'rb') as file: 
            DS = pickle.load(file)
        
        DS["vel_scalar"] = Kalman.vel_scalar(DS["vn"], DS["ve"], DS["vd"])
        
        deltafunc = np.zeros(len(DS["time"]))
        
        dirac = 10800
        deltafunc[dirac] = 1
        DS["dg"][:,2] = DS["dg"][:,2] + deltafunc
        Gravtie = Kalman.getGravTIE(DS["time"][450:-450], DS["vel_scalar"][450:-450], dg)
        kf, rts = Kalman.KalmanFilterTIE(DS, Gravtie, 2)

        rts_new = RTS.RTS_smoother(rts)

        diff = rts_new["profile"][:,0] - rts_orig["profile"][:,0]

    # Cut out AOI 
    delta_cut = 1000
    deltafunc = deltafunc[10800-delta_cut:10800+delta_cut]
    diff = diff[10800-delta_cut:10800+delta_cut]

    # Compute the FFT of the input signal (Dirac delta function)
    fft_input = np.fft.fft(deltafunc)

    # Compute the FFT of the output signal (filtered signal)
    fft_output = np.fft.fft(diff)

    # Compute the frequency response
    H_fft = fft_output / fft_input

    # Extract the magnitude and phase of the frequency response
    magnitude = np.abs(H_fft)
    phase = np.angle(H_fft)

    # Generate frequency vector
    sampling_freq = 1  # Assuming unit sampling frequency
    freq_vector = np.fft.fftfreq(len(DS["time"][10800-delta_cut:10800+delta_cut]), d=1/sampling_freq)

    # Interpolate absolute values of H_fft
    interp_func = interp1d(freq_vector, magnitude, kind='linear')

    # Find frequency corresponding to half maximum
    half_max = max(magnitude) / 2
    # Increase the number of points for evaluation
    freq_vector_interp = np.linspace(freq_vector.min(), freq_vector.max(), num=len(freq_vector)*100)

    # Find frequency corresponding to half maximum using the interpolated function with more points
    half_max_frequency = abs(freq_vector_interp[np.argmin(np.abs(interp_func(freq_vector_interp) - half_max))])


    return freq_vector, interp_func, half_max_frequency, deltafunc, diff

freq_vector, interp_func, half_max_frequency, deltafunc, diff = Calc_Transfer(1)

from matplotlib.ticker import ScalarFormatter, NullLocator

# Plot the magnitude and phase responses
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.set_xscale('log')
# Plot the interpolated magnitude
ax1.plot(freq_vector[0:-1000], interp_func(freq_vector)[0:-1000], label=f'Kalman Lowpass', lw=2, color="blue")

# ax1.plot(freq_vector, interp_func(freq_vector), color='blue', label='Interpolated Magnitude', lw=2)
ax1.plot([half_max_frequency, half_max_frequency], [0, 1], lw=1, ls="--", color="black")
ax1.plot([0, half_max_frequency], [.5, .5], lw=1, ls="--", color="black")
ax1.set_ylim(0,1)
ax1.set_xlim(1e-4, 1e-1)

ax1.set_ylabel("Frequency Responce [.]")
ax1.set_xlabel("Frequency [Hz]")

ax1.xaxis.set_minor_locator(NullLocator())

# Manually set major ticks at desired locations
major_ticks = [0.0001, 0.001, 0.01, 0.1]
ax1.set_xticks(major_ticks)

# Set a custom formatter for major ticks
def custom_formatter(x, pos):
    if x == 0.0001:
        return '0.0001 [Hz]'
    elif x == 0.001:
        return '0.001 [Hz]'
    elif x == 0.01:
        return '0.01 [Hz]' 
    elif x == 0.1:
        return '0.1  [Hz]'
    else:
        return ''

ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.xaxis.set_major_formatter(custom_formatter)
ax1.text(0.11, 1.01, '10 [s]', verticalalignment='bottom', horizontalalignment='right', fontsize=16)
ax1.text(0.011, 1.01, '100 [s]', verticalalignment='bottom', horizontalalignment='right', fontsize=16)
ax1.text(0.0011, 1.01, '1000 [s]', verticalalignment='bottom', horizontalalignment='right', fontsize=16)
ax1.text(0.00011, 1.01, '10000 [s]', verticalalignment='bottom', horizontalalignment='right', fontsize=16)
ax1.text(0.00075, .51, 'Half Transmission', verticalalignment='bottom', horizontalalignment='right', fontsize=16)
ax1.text(half_max_frequency, .05, r'$f_{1/2} =$'+ f'{np.round(half_max_frequency, 5)} [Hz]', verticalalignment='bottom', horizontalalignment='right', fontsize=12, rotation=90)
ax1.text(half_max_frequency, 0.95, r'$2\cdot$FWHM ='+ f'{2*np.round(1/(2*half_max_frequency), 1)} [s]', verticalalignment='top', horizontalalignment='left', fontsize=12, rotation=270)

freq_vector, interp_func, half_max_frequency, deltafunc, diff = Calc_Transfer(2)
ax1.plot(freq_vector[0:-1000], interp_func(freq_vector)[0:-1000], label=f'Kalman Tie/Bias', lw=2, color="red", ls = "--")
plt.legend(loc="lower left")
plt.show()
print(f"FWHM = {1/(2*half_max_frequency)}")



# # half_max_frequency = freq_vector[np.argmin(np.abs(interp_func(freq_vector) - half_max))]

# print("Frequency corresponding to half of the maximum magnitude:", half_max_frequency)
# print(f"FWHM = {1/(2*half_max_frequency)} -> Spectral Domain")
# # Plot the magnitude and phase responses
# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax1.set_xscale('log')

# # Plot the interpolated magnitude
# ax1.plot(freq_vector[0:-1000], interp_func(freq_vector)[0:-1000], color='blue', label='Interpolated Magnitude')

# # Plot the original magnitude as a scatter plot
# ax1.scatter(freq_vector, magnitude, marker='o', color='red', label='Original Magnitude')
# ax1.axvline(half_max_frequency)
# ax1.axhline(.5)
# plt.legend()
# plt.show()



## --------------------- Calculate Spatial Resolution with Multiple Dirac --------------------
file_path = "../Results/DS_300Hz_150s_3_stages.pkl"

print(f"Loading DS results from: {file_path}")
with open(file_path, 'rb') as file: 
    DS = pickle.load(file)
DS["vel_scalar"] = ds.vel_scalar(DS["vn"], DS["ve"], DS["vd"])
min = 9700
max = 20200

Ref = {}                                        # Determined using visual inspection
Ref["all_lines"] = np.arange(min,max+1)
Ref["Line_1"] = np.arange(min+350,min+1950+1)
Ref["Line_2"] = np.arange(min+2250,min+4200+1)
Ref["Line_3"] = np.arange(min+4490,min+6050+1)
Ref["Line_4"] = np.arange(min+6400,min+8300+1)
Ref["Line_5"] = np.arange(min+8650,max-250+1)

Line = ["Line_1", "Line_2", "Line_3", "Line_4", "Line_5"]

# Ref["all"] = np.concatenate(np.array([Ref["Line_1"], Ref["Line_2"], Ref["Line_3"], Ref["Line_4"], Ref["Line_5"]]))
Ref["all"] = np.hstack([Ref[line] for line in Line])

delta_id = Ref["all"][::120]        # Take out every 120'th id for delta implementation

plt.figure(figsize=(12,6))
plt.plot(ds.SOW2HOD(DS["time"]), DS["fdg"][:,2], lw=2, label="Full Survey")
# for i in range(len(Line)): 
#     plt.plot(DS["time"][Ref[Line[i]]], DS["fdg"][Ref[Line[i]],2], lw=2, label=f"Profile {i+1}")
plt.scatter(ds.SOW2HOD(DS["time"])[Ref["all"]], DS["fdg"][Ref["all"],2], s=3, color="black", zorder=10, label="Refernce Profiles")
plt.scatter(ds.SOW2HOD(DS["time"])[delta_id], DS["fdg"][delta_id,2], s=50, color="red", zorder=10, marker='^', label="Dirac Delta Positions")
plt.xlabel("Time Hour of Day [h]")
plt.ylabel("Gravity Disturbance [mGal]")
plt.legend()
# plt.show()

vel = ds.but2_v2(DS["vel_scalar"], 3, 150, 1)
print(f"Mean Scalar Velocity: {np.mean(vel[delta_id])} m/s")

# Sample data (replace this with your actual data)
time_vector = DS["time"][delta_id]  # Example time vector
data = vel[delta_id]  # Example data corresponding to time vector

# Calculate differences in time vector to define subset boundaries
subset_indices = np.where(np.diff(time_vector) > 150)[0] + 1
subset_indices = np.insert(subset_indices, 0, 0)
subset_indices = np.append(subset_indices, len(time_vector))

for i in range(len(subset_indices) - 1):
    subset_data = data[subset_indices[i] : subset_indices[i+1]]
    subset_time = time_vector[subset_indices[i] : subset_indices[i+1]]
    print(f"Mean = {np.round(np.mean(subset_data), 2)}")
          
plt.figure(figsize=(12,6))
plt.plot(ds.SOW2HOD(DS["time"]), vel, lw=2, label="Full Flight scaalar Velocity (Lowpass)")
plt.scatter(ds.SOW2HOD(DS["time"])[Ref["all"]], vel[Ref["all"]], s=2, label="Reference Profiles", color="black", zorder=10)
plt.scatter(ds.SOW2HOD(DS["time"])[delta_id], vel[delta_id], s=50, color="red", zorder=10, marker='^', label="Dirac Delta Positions")
plt.xlabel("Time Hour of Day [h]")
plt.ylabel("Scalar Velocity [m/s]")
plt.legend()
plt.show()

def Calc_Spatial_res(delta_id, vel):
    file_path = "../Results/DS_300Hz_150s_3_stages.pkl"
    Spatial_res = np.array([])
    for dirac in tqdm(delta_id, colour="green"):
        with open(file_path, 'rb') as file: 
            DS = pickle.load(file)
        # Run normal Kalman Lowpass
        kf_orig, rts, _ = Kalman.KalmanFilterLowpass(DS, 2)
        rts_orig = RTS.RTS_smoother(rts)

        # Plot Ref line to check result: 
        # Kalman.plot_ref_line(rts_orig, DS)

        # Ad a dirac delta function to raw signal: 
        deltafunc = np.zeros(len(DS["time"]))
        deltafunc[dirac] = 1
        DS["dg"][:,2] = DS["dg"][:,2] + deltafunc

        # Run second iteration Kalman + RTS with Delta function implemented. 
        kf_new, rts, _ = Kalman.KalmanFilterLowpass(DS, 2)
        rts_new = RTS.RTS_smoother(rts)

        diff = rts_new["profile"][:,0] - rts_orig["profile"][:,0]

        # Cut out AOI 
        delta_cut = 1000
        deltafunc = deltafunc[dirac-delta_cut:dirac+delta_cut]
        diff = diff[dirac-delta_cut:dirac+delta_cut]

        # Compute the FFT of the input signal (Dirac delta function)
        fft_input = np.fft.fft(deltafunc)

        # Compute the FFT of the output signal (filtered signal)
        fft_output = np.fft.fft(diff)

        # Compute the frequency response
        H_fft = fft_output / fft_input

        # Extract the magnitude and phase of the frequency response
        magnitude = np.abs(H_fft)
        phase = np.angle(H_fft)

        # Generate frequency vector
        sampling_freq = 1  # Assuming unit sampling frequency
        freq_vector = np.fft.fftfreq(len(DS["time"][dirac-delta_cut:dirac+delta_cut]), d=1/sampling_freq)

        # Interpolate absolute values of H_fft
        interp_func = interp1d(freq_vector, magnitude, kind='linear')

        # Find frequency corresponding to half maximum
        half_max = np.max(magnitude) / 2
        # Increase the number of points for evaluation
        freq_vector_interp = np.linspace(freq_vector.min(), freq_vector.max(), num=len(freq_vector)*100)

        # Find frequency corresponding to half maximum using the interpolated function with more points
        half_max_frequency = abs(freq_vector_interp[np.argmin(np.abs(interp_func(freq_vector_interp) - half_max))])

        # Calculate FWHM 
        fwhm = 1/(2*half_max_frequency)

        # Calculate FWHM Spatial Resolution in [km]
        Spatial_res_ = (vel[dirac] * fwhm)/1000
        # Spatial_res_ = (100 * fwhm)/1000

        Spatial_res = np.append(Spatial_res, Spatial_res_)
    return Spatial_res
Spatial_res = Calc_Spatial_res(delta_id, vel)


# Sample data (replace this with your actual data)
time_vector = DS["time"][delta_id]  # Example time vector
data = Spatial_res  # Example data corresponding to time vector

# Calculate differences in time vector to define subset boundaries
subset_indices = np.where(np.diff(time_vector) > 150)[0] + 1
subset_indices = np.insert(subset_indices, 0, 0)
subset_indices = np.append(subset_indices, len(time_vector))
plt.figure(figsize=(10,5))
# Plot each subset separately
for i in range(len(subset_indices) - 1):
    subset_data = data[subset_indices[i] : subset_indices[i+1]]
    subset_time = time_vector[subset_indices[i] : subset_indices[i+1]]
    plt.scatter(ds.SOW2HOD(subset_time), subset_data, label=f"Profile {i+1}")
    plt.plot(ds.SOW2HOD(subset_time), subset_data, lw=1)
    plt.plot([ds.SOW2HOD(subset_time)[0], ds.SOW2HOD(subset_time)[-1]], 
             [np.mean(subset_data), np.mean(subset_data)], ls="--", lw=1, color="black")
    if np.mean(subset_data) > np.mean(Spatial_res):
        plt.text(ds.SOW2HOD(subset_time)[0], np.mean(subset_data)-.3, 
                    f"Mean = {np.round(np.mean(subset_data), 2)} [km]", fontsize=12, horizontalalignment='left')
    elif np.mean(subset_data) < np.mean(Spatial_res):
        plt.text(ds.SOW2HOD(subset_time)[0], np.mean(subset_data)+.4, 
            f"Mean = {np.round(np.mean(subset_data), 2)} [km]", fontsize=12, horizontalalignment='left')
# Add labels and legend
plt.xlabel("Time Hour of Day [h]")
plt.ylabel('Full-Wavelength Resolution [km]')
plt.legend(loc="lower right", fontsize=12)
plt.axhline(np.mean(Spatial_res), ls="--", lw=1, color="black")
plt.text(ds.SOW2HOD(DS["time"])[delta_id][-1], np.mean(Spatial_res), f"Mean = {np.round(np.mean(Spatial_res), 2)} [km]", fontsize=12, horizontalalignment='right', verticalalignment="bottom")

# Show the plot
plt.show()