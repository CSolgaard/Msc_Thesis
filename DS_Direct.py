# -----------------------Direct Strapdown - Direct method--------------------------
#  
# Script Library: Direct Strapdown - Direct Method
# Description: This script contains the methodology for computing gravity disturbance 
#              using specific force measurements from an Inertial Measurement Unit (IMU) 
#              and position data from Global Navigation Satellite System (GNSS).
# 
# ---------------------------------------------------------------------------------
# Author: Christian Solgaard (DTU) 02/01-2024 
# ---------------------------------------------------------------------------------


import numpy as np
from src import Direct_Strapdown as ds
from src import IMU_load as IMU_load
from src import Att_load as Att_load
from Kalman import Kalman as K
from tqdm import tqdm
import copy
from numba import jit
import scienceplots
import pickle
from scipy.interpolate import CubicSpline
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, TransformedBbox, BboxPatch, BboxConnector)
from matplotlib.transforms import Bbox

import matplotlib.pyplot as plt
# plt.style.use(['science','no-latex', 'grid'])
plt.style.use(['science','grid'])
from dataclasses import dataclass, asdict 

params = {'axes.labelsize': 'x-large', 'axes.titlesize':'xx-large','xtick.labelsize':'large', 
          'ytick.labelsize':'large', 'legend.fontsize': 'x-large','mathtext.fontset':'stix', 'font.family':'STIXGeneral'}
plt.rcParams.update(params)
from pathlib import Path
import pandas as pd
import os
from scipy.interpolate import splev, splrep
import warnings

# To ignore all warnings
warnings.filterwarnings("ignore")

# ---------------------- Preset varibles -------------------------
# Set GNSS antenna til -> IMU lever arm
lever_arm = np.array([ 5.250, -0.850, -1.960]) # (x-right,, y-fwd, z-down)

def Calc_ftc(FWHM): 
    return int((FWHM - 1)/.74)

# Set filter time coefficient (Butterworth)
prefilter = {}
prefilter["ftc"] = 2
prefilter["stage"] = 2
butterfilter = {}
butterfilter["ftc"] = 150 #Calc_ftc(40) #150
butterfilter["stage"] = 3


# ------------------------ Data Load --------------------------------
file1 = Path("..", "data", "airgravi2019_etalon", "imar", "inat_115.dat")
file2 = Path("..", "data", "airgravi2019_etalon", "gnss", "115_air1_ppp_1Hz.txt")
file3 = Path("..", "data", "airgravi2019_etalon", "115_inexp_nav_300Hz.dat")

imu = IMU_load.readIMAR(file1, "echo", "on")            # Load 300Hz IMU specific force
gnss = IMU_load.load_gnss(file2)                        # Load 1Hz GNSS PPP solution

nav, header = Att_load.readATT(file3, "echo", "on")     # Load 300Hz attitude solution
nav.rename(columns={"TINTRPL": "time"}, inplace=True)

coast = Path("..", "data", "France_coastline.txt")
names = ["X", "Y"]
coast = pd.read_csv(coast, header=0, delimiter=",", names=names)

# ------- New RC-Filter, also implemented in Direct_strapdown.py ---------
@jit
def RC_filter(data, stage, ftc, dt):
    """
    RC-filter from Dag Solheim

    Parameters:
    - data: Vector of data series
    - stage: Number of iterations, 1 iteration = forward + backward run
    - ftc: Filter time constant [s]
    - dt: Sample interval [s]

    Returns:
    - fdata: Vector of filtered data series
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

# --------------------- Data Preprocessing -------------------------------
# IMU Sampling rate
imu_tuple = {}
imu_tuple["dt"] = np.diff(imu.time)
imu_tuple["sfreq"] = np.round(1 / np.nanmean(imu_tuple["dt"]))
imu_tuple["srate"] = 1 / imu_tuple["sfreq"]

gnss_tuple = {}
gnss_tuple["dt"] = np.diff(gnss.time)
gnss_tuple["sfreq"] = np.round(1 / np.nanmean(gnss_tuple["dt"]))
gnss_tuple["srate"] = 1 / gnss_tuple["sfreq"]

# --------------------- Initialise Solution ----------------------------
solution = {}

# extract time stamps
solution["time"] = gnss.time

# trim time stamps to common time interval
idx = np.where((solution["time"] > imu.time.values[0]) & (solution["time"] < imu.time.values[-1]))
solution["time"] = solution["time"].values[idx]

# Sampling rate 
solution["dt"] = np.diff(solution["time"])
solution["sfreq"] = np.round(1 / np.nanmean(solution["dt"]))
solution["srate"] = 1 / solution["sfreq"]

# Extract Navigation Solution
nav_key = ["roll", "pitch", "yaw"]
for i in range(0, len(nav_key)): 
    solution[nav_key[i]] = ds.interpolate_DS(nav.time, nav[nav_key[i]], solution["time"], "linear", "extrapolate")



# ------------ Translate GNSS position too IMU Location -----------------
print("Translate GNSS Position to IMU Location")
nav_key = ["roll", "pitch", "yaw"]
for i in range(0, len(nav_key)): 
    gnss[nav_key[i]] = ds.interpolate_DS(nav.time, nav[nav_key[i]], gnss.time, "linear", "extrapolate")
    # gnss[nav_key[i]] = ds.RC_filter(gnss[nav_key[i]].values, prefilter["stage"], prefilter["ftc"], gnss_tuple["srate"])

# gnss.lat = ds.RC_filter(gnss.lat.values, prefilter["stage"], prefilter["ftc"], gnss_tuple["srate"])
# gnss.lon = ds.RC_filter(gnss.lon.values, prefilter["stage"], prefilter["ftc"], gnss_tuple["srate"])
# gnss.h = ds.RC_filter(gnss.h.values, prefilter["stage"], prefilter["ftc"], gnss_tuple["srate"])

gnss["olat"], gnss["olon"], gnss["oh"] = ds.pos_translate_v1(gnss.lat, gnss.lon, gnss.h, gnss.roll, gnss.pitch, gnss.yaw, lever_arm)

solution["lat"] = ds.interpolate_DS(gnss.time, gnss["olat"], solution["time"], "linear", "extrapolate")
solution["lon"] = ds.interpolate_DS(gnss.time, gnss["olon"], solution["time"], "linear", "extrapolate")
solution["h"] = ds.interpolate_DS(gnss.time, gnss["oh"], solution["time"], "linear", "extrapolate")

print("> Done")


# -------------- Derive Acceleration from GNSS ----------------------
gnss["imu_lat"] = gnss.olat
gnss["imu_lon"] = gnss.olon
gnss["imu_h"] = gnss.oh

# Echo
print('Deriving accelerations from GNSS')

# Set some variables
a = 6378137.0
ecc2 = 6.69437999014E-3
deg2rad = np.pi / 180

# Compute useful variables
cos_lat = np.cos(np.radians(gnss['imu_lat']))
sin_lat = np.sin(np.radians(gnss['imu_lat']))
term = 1.0 - ecc2 * sin_lat * sin_lat
R_E = a / np.sqrt(term)
R_N = R_E * (1.0 - ecc2) / term

# Fit cubic spline piecewise polynomial
k = 3  # cubic spline
pp_lat = CubicSpline(gnss['time'], gnss['imu_lat'] * deg2rad, bc_type='clamped')
pp_lon = CubicSpline(gnss['time'], gnss['imu_lon'] * deg2rad, bc_type='clamped')
pp_h = CubicSpline(gnss['time'], gnss['imu_h'], bc_type='clamped')

# Compute first order derivative
pp1_lat = pp_lat.derivative(nu=1)
pp1_lon = pp_lon.derivative(nu=1)
pp1_h = pp_h.derivative(nu=1)

# Evaluate velocity
gnss['vn'] = pp1_lat(gnss['time']) * R_N
gnss['ve'] = pp1_lon(gnss['time']) * R_E * cos_lat
gnss['vd'] = -pp1_h(gnss['time'])

# Interpolate velocity
solution['vel'] = np.column_stack((
    np.interp(solution['time'], gnss['time'], gnss['vn']),
    np.interp(solution['time'], gnss['time'], gnss['ve']),
    np.interp(solution['time'], gnss['time'], gnss['vd'])
))

# Compute second order derivative
pp2_lat = pp_lat.derivative(nu=2)
pp2_lon = pp_lon.derivative(nu=2)
pp2_h = pp_h.derivative(nu=2)

# Evaluate acceleration
gnss['accn'] = pp2_lat(gnss['time']) * R_N
gnss['acce'] = pp2_lon(gnss['time']) * R_E * cos_lat
gnss['accd'] = -pp2_h(gnss['time'])

solution["accn"] = RC_filter(gnss["accn"].values, prefilter["stage"], prefilter["ftc"], gnss_tuple["srate"])
solution["acce"] = RC_filter(gnss["acce"].values, prefilter["stage"], prefilter["ftc"], gnss_tuple["srate"])
solution["accd"] = RC_filter(gnss["accd"].values, prefilter["stage"], prefilter["ftc"], gnss_tuple["srate"])

temp_time = np.array(gnss["time"])[1:]

solution["gps_acc"] = np.zeros((len(solution["time"]), 3))
solution["gps_acc"][:,0] = ds.interpolate_DS(temp_time, solution["accn"].reshape(-1), solution["time"], "linear", "extrapolate")
solution["gps_acc"][:,1] = ds.interpolate_DS(temp_time, solution["acce"].reshape(-1), solution["time"], "linear", "extrapolate")
solution["gps_acc"][:,2] = ds.interpolate_DS(temp_time, solution["accd"].reshape(-1), solution["time"], "linear", "extrapolate")



# --------------- Transform IMU accelerations into NED-frame ----------------

print("Interpolating IMU Accelerations")

for key, value in nav.items(): 
    imu_tuple[key] = ds.interpolate_DS(nav.time, value, 
                                       imu.time, "linear", False, "extrapolate")

print("Rotating solution Accelerations")
imu_att = np.vstack([imu_tuple["roll"], imu_tuple["pitch"], imu_tuple["yaw"]]).T
solution_bacc = np.vstack([imu.bacc1, imu.bacc2, imu.bacc3]).T

# Transform Accelerations 
imu_nacc = ds.b2n_v1(imu.time, solution_bacc, imu_att)
imu_tuple["nacc_"] = imu_nacc.T  

imu_plot = {}
imu_plot["time"] = imu.time
imu_plot["dacc"] = imu_tuple["nacc_"][:,2] 

# Prefilter accelerations
print("Pre-filter IMU readings and interpolate to Solution epoch")

solution["imu_acc"] = np.zeros([len(solution["time"]), 3])
imu_tuple["nacc"] = np.zeros([len(imu.time)-1, 3])
for i in range(0,3): 
    nacc_temp = ds.RC_filter(imu_tuple["nacc_"][:,i], prefilter["stage"], prefilter["ftc"], imu_tuple["srate"])
    imu_tuple["nacc"][:,i] = nacc_temp.reshape(-1)
    solution["imu_acc"][:,i] = ds.interpolate_DS(imu.time[1:], imu_tuple["nacc"][:,i], solution["time"], "linear", "extrapolate")

print("> Done")


# ------------------------- Compute Transport Rate ----------------------
print("Computing Transport-Rate (Eotvos and Coriolis) Effect")

solution["vn"] = solution["vel"][:,0]
solution["ve"] = solution["vel"][:,1]
solution["vd"] = solution["vel"][:,2]

vel = np.vstack([solution["vn"], solution["ve"], solution["vd"]]).T
pos = np.vstack([solution["lat"], solution["lon"], solution["h"]]).T

# Compute Transport Rate 
solution["tacc"] = ds.transport_rate_v2(solution["time"], vel, pos)

solution["acc_corr"] = solution["imu_acc"] - solution["tacc"]
print("> Done")


# ----------------------------- Form Gravity -----------------------------------
print("Derive Gravity Disturbance")
solution["g"] = solution["gps_acc"] - solution["acc_corr"]

# Compute normal gravity 
gamma, _, _ = ds.normal_gravity_precise_v1(solution["lat"].reshape(-1,1), 
                                           solution["lon"].reshape(-1,1), 
                                           solution["h"].reshape(-1,1), 3)
solution["gamma_down"] = gamma.down
solution["gamma_north"] = gamma.north
solution["gamma_east"] = gamma.east



# Derive gravity disturbance
# solution["dg"] = (solution["g"][:,2] - solution["gamma_down"].reshape(-1))*10**5
north_dg = (solution["g"][:,0] - solution["gamma_north"].reshape(-1))*10**5
east_dg = (solution["g"][:,1] - solution["gamma_east"].reshape(-1))*10**5
down_dg = (solution["g"][:,2] - solution["gamma_down"].reshape(-1))*10**5

north_dg = np.nan_to_num(north_dg, nan=0) 
east_dg = np.nan_to_num(east_dg, nan=0) 
down_dg = np.nan_to_num(down_dg, nan=0) 

# solution["dg"] = np.nan_to_num(solution["dg"], nan=0)
solution["dg"] = np.array([north_dg.T, east_dg.T, down_dg.T]).T
print("> Done")

# ----------------------- Butterworth Lowpass filter -----------------------

north_fdg = ds.but2_v2(solution["dg"][:,0], butterfilter["stage"], butterfilter["ftc"], solution["srate"])
east_fdg = ds.but2_v2(solution["dg"][:,1], butterfilter["stage"], butterfilter["ftc"], solution["srate"])
down_fdg = ds.but2_v2(solution["dg"][:,2], butterfilter["stage"], butterfilter["ftc"], solution["srate"])

solution["fh"] = ds.but2_v2(solution["h"], butterfilter["stage"], butterfilter["ftc"], solution["srate"])


# ---------------------- Test 4'th Order Bessel ----------------------------
# from scipy import signal

# tau = 10.0

# # Calculate the corresponding cutoff frequency
# cutoff_frequency = 1 / (2 * np.pi * tau)

# order = 4
# b, a = signal.bessel(order, cutoff_frequency, analog=False, btype='low', output='ba')
# north_fdg = signal.filtfilt(b, a, solution["dg"][:,0])
# east_fdg = signal.filtfilt(b, a, solution["dg"][:,1])
# down_fdg = signal.filtfilt(b, a, solution["dg"][:,2])

solution["fdg"] = np.array([north_fdg.T, east_fdg.T, down_fdg.T]).T
# ------------------------- Print Results ----------------------------------
def plot_ref_line():
    import pyproj
    from pyproj import Transformer
    survey = {}
    survey["time"] = solution["time"]
    survey["dg"] = solution["fdg"][:,2]

    pipeline = "+ellps=GRS80 +proj=pipeline +step +proj=utm +zone=30"
    transform_object = Transformer.from_pipeline(pipeline)
    geodetic_corr = [solution["lon"], solution["lat"], solution["h"]]
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

    @dataclass
    class Verification_line: 
        name: str
        index: np.array
        dist: np.array
        dg: np.array
        time: np.array
        h: np.array
        roll: np.array
        pitch: np.array
        yaw: np.array
        h_reduced: np.array
        dg_bias: np.array
    ref_line_dist = np.arange(10,140,.5)
    Line = ["Line_1", "Line_2", "Line_3", "Line_4", "Line_5"]
    RMS_line = {}
    RMS_line["dist"] = ref_line_dist

    H_ref = solution["fh"][10050]
    solution["h_reduced"] = solution["fh"] - H_ref
    solution["dg_bias"] = solution["h_reduced"] * (-.3086)

    def flip_line(line, i): 
        if i in [1, 3]: 
            return np.flip(line)
        else:
            return line
        
    # Test Haversine Distance instead of direct line. 
    def haversine_distance(lat1, lon1, lat_array, lon_array):
        """
        Calculate the great circle distance between a reference point 
        and an array of other points on the earth (specified in decimal degrees)
        """
        # Convert decimal degrees to radians
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat_array_rad, lon_array_rad = np.radians(lat_array), np.radians(lon_array)
        
        # Haversine formula
        dlat = lat_array_rad - lat1_rad
        dlon = lon_array_rad - lon1_rad
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat_array_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Radius of earth in kilometers is 6371
        km = 6371 * c
        return km*1000

    ref_lon = solution["lon"][min+300]
    ref_lat = solution["lat"][min+300]

    for i in range(len(Line)): 
        dist_line = dist2ref(UTM_corr[0][Ref[Line[i]]], 
                            UTM_corr[1][Ref[Line[i]]], ref_x, ref_y)
        # dist_line = haversine_distance(ref_lat, ref_lon, solution["lat"][Ref[Line[i]]], solution["lon"][Ref[Line[i]]])


        RMS_line[Line[i]] = ds.interpolate_DS(dist_line*1e-3, solution["fdg"][Ref[Line[i]], 2], ref_line_dist, "linear", "extrapolate")

        # Line[i] = Verification_line(Line[i], Ref[Line[i]], 
        #                             dist_line, solution["fdg"][Ref[Line[i]], 2], 
        #                             solution["time"][Ref[Line[i]]], 
        #                             solution["fh"][Ref[Line[i]]], 
        #                             solution["roll"][Ref[Line[i]]], 
        #                             solution["pitch"][Ref[Line[i]]],
        #                             solution["yaw"][Ref[Line[i]]],
        #                             solution["h_reduced"][Ref[Line[i]]],
        #                             solution["dg_bias"][Ref[Line[i]]])
        h = flip_line(solution["fh"][Ref[Line[i]]], i)
        h_reduced = flip_line(solution["h_reduced"][Ref[Line[i]]], i)
        dg_bias = flip_line(solution["dg_bias"][Ref[Line[i]]], i)
        Line[i] = Verification_line(Line[i], Ref[Line[i]], 
                                    dist_line, solution["fdg"][Ref[Line[i]], 2], 
                                    solution["time"][Ref[Line[i]]], 
                                    h, 
                                    solution["roll"][Ref[Line[i]]], 
                                    solution["pitch"][Ref[Line[i]]],
                                    solution["yaw"][Ref[Line[i]]],
                                    h_reduced,
                                    dg_bias)
        
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
                print("Excluded Line:   Mean:   Min:    Max:    STD:    RMS:    RMSE")
            print(f"Profile {i}          {np.round(mean,2)}    {np.round(min,2)}   {np.round(max,2)}    {np.round(STD,2)}    {np.round(rms_,2)}    {np.round(rmse,2)}")
        return
    LOO_summary_stat(RMS_line)

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig, ax = plt.subplots(figsize=(10,5))
    for i in range(5): 
        ax.plot(Line[i].dist*1e-3, Line[i].dg, label =  f"Profile {i+1}", linewidth=3)

    ax.text(0.75, 0.08, f'RMSE: {rms_:.2f} [mGal]', transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # plt.grid()
    ax.plot(RMS_line["dist"], RMS_line["mean_line"], 'b--', linewidth=1, label="Mean reference")
    ax.legend(loc="lower right", fontsize="12")
    ax.set_xlabel("Distance from reference Point [km]")
    ax.set_ylabel("Gravity Disturbance [mGal]")
    # plt.title(f"Flight 115, Biscay Bay Reference Line: Attitude @ 300Hz\nButterworth, stages = 3, FWHM = {np.round(butterfilter['ftc']*0.74)} [s]")
    ax.set_xlim(5, 140)
    # save_path = "Results/Figures/Biscay_Reference_lines_@"+str(butterfilter["ftc"])+'s_'+str(butterfilter["stage"])+"_stages.eps"
    # plt.savefig(save_path)
    # plt.show()

    extent = [-10, 5, 42, 50]
    axin2 = fig.add_axes([0.17, 0.5, 0.35, 0.35], projection=ccrs.PlateCarree(), zorder=2)
    axin2.set_extent(extent)
    axin2.add_feature(cfeature.COASTLINE)
    axin2.add_feature(cfeature.BORDERS, linestyle=':')
    axin2.stock_img()
    axin2.plot(solution["lon"], solution["lat"], color="blue", lw=2)
    axin2.plot(solution["lon"][min+350:max-250+1], solution["lat"][min+350:max-250+1], color="red", lw=2)
    gl = axin2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', linewidth=0.5, color='gray')
    gl.xlabels_top = False
    gl.ylabels_right = False

    # for i in range(5): 
    #     plt.plot(Line[i].dist*1e-3, Line[i].dg-50, label =  f"Line$_{i+1}$", linewidth=3)

    # plt.text(0.75, 0.08, f'RMSE: {rms_:.2f} [mGal]', transform=plt.gca().transAxes,
    #         verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # # plt.grid()
    # plt.plot(RMS_line["dist"], RMS_line["mean_line"], 'b--', linewidth=1, label="Mean reference")
    # plt.legend(loc="lower right", fontsize="12")
    # plt.xlabel("Distance from reference Point [km]")
    # plt.ylabel("Gravity Disturbance [mGal]")
    # # plt.title(f"Flight 115, Biscay Bay Reference Line: Attitude @ 300Hz\nButterworth, stages = 3, FWHM = {np.round(butterfilter['ftc']*0.74)} [s]")
    # plt.xlim(5, 140)
    # save_path = "Results/Figures/Biscay_Reference_lines_@"+str(butterfilter["ftc"])+'s_'+str(butterfilter["stage"])+"_stages.eps"
    # plt.savefig(save_path)
    # plt.show()

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
    axins.plot(RMS_line["dist"], RMS_line["mean_line"], 'b--', linewidth=1)
    axins.set(xlim=(24,36), ylim=(-27.5+50,-12.5+50))
    

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
    save_path = "Results/Figures/Biscay_Reference_lines_@"+str(butterfilter["ftc"])+'s_'+str(butterfilter["stage"])+"_stages.eps"
    plt.savefig(save_path)


    # TEST HEIGHT OF REFERENCE

    fig, ax=plt.subplots(figsize=(10,5))
    for i in range(5): 
        ax.plot(Line[i].dist*1e-3, Line[i].h, label =  f"Profile {i+1}", linewidth=3, zorder=0)

    # ax.text(0.75, 0.08, f'RMSE: {rms_:.2f} [mGal]', transform=plt.gca().transAxes,
    #         verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # ax.plot(RMS_line["dist"], RMS_line["mean_line"], 'b--', linewidth=1, label="Mean reference", zorder=0)
    ax.legend(loc="lower right", fontsize="12")
    ax.set_xlabel("Distance From Reference Point [km]")
    ax.set_ylabel("Ellipsoidal Flight Height [m]")
    ax.set_ylim(475,500)
    ax.set_xlim(5, 140)


    fig, ax=plt.subplots(figsize=(10,5))
    for i in range(5): 
        ax.plot(Line[i].dist*1e-3, Line[i].h_reduced, label =  f"Profile {i+1}", linewidth=3, zorder=0)

    ax.legend(loc="upper left", fontsize="12")
    ax.set_xlabel("Distance From Reference Point [km]")
    ax.set_ylabel("Reduced Ellipsoidal Flight Height [m]")
    ax.set_ylim(-5,25)
    ax.set_xlim(5, 140)

    ax2 = ax.twinx()
    for i in range(5): 
        ax2.plot(Line[i].dist*1e-3, Line[i].h_reduced * .3086, linewidth=3, ls="--")
    ax2.set_ylabel("Gravity Gradient Bias [mGal]")
    ax2.set_ylim(0,10)



    fig, ax=plt.subplots(figsize=(10,5))
    for i in range(5): 
        ax.plot(Line[i].dist*1e-3, Line[i].dg + Line[i].dg_bias, label =  f"Profile {i+1}", linewidth=3, zorder=0)

    # ax.text(0.75, 0.08, f'RMSE: {rms_:.2f} [mGal]', transform=plt.gca().transAxes,
    #         verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # ax.plot(RMS_line["dist"], RMS_line["mean_line"], 'b--', linewidth=1, label="Mean reference", zorder=0)
    ax.legend(loc="lower right", fontsize="12")
    ax.set_xlabel("Distance From Reference Point [km]")
    ax.set_ylabel("Gravity Disturbance [mGal]")
    ax.set_xlim(5, 140)
    ax.set_ylim(-60,60)
    plt.title("Leveled to same Height.")

    # axins = inset_axes(ax, "100%", "100%", bbox_to_anchor=[.2, .55, .4, .4], bbox_transform=ax.transAxes, borderpad=0)
    insetPosition = Bbox.from_bounds(0.3, 0.55, 0.3, 0.3)  # Define position of inset axes
    axins = fig.add_axes(insetPosition)

    for i in range(5): 
        axins.plot(Line[i].dist*1e-3, Line[i].dg + Line[i].dg_bias, linewidth=2)
    # axins.plot(RMS_line["dist"], RMS_line["mean_line"], 'b--', linewidth=1)
    axins.set(xlim=(24,36), ylim=(-27.5+50,-12.5+50))
    
    my_mark_inset(ax, axins, loc1a=2, loc1b=2, loc2a=4, loc2b=4, fc="none", ec="0.5")

 
def plot_bathy():

    import netCDF4 as nc
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    filenc = Path("..", "data", "Bathymetry", "GEBCO_15_Jan_2024_1b6d6c7c6bdc", "gebco_2023_n50.0_s42.0_w-10.0_e5.0.nc")
    from matplotlib import gridspec
    dataset = nc.Dataset(filenc, "r")
    lat_bathy = dataset.variables['lat'][:]
    lon_bathy = dataset.variables['lon'][:]
    elevation_bathy = dataset.variables['elevation'][:]

    dataset.close()

    # Plot the elevation grid with a specific colormap
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 10))
    pc = ax.pcolormesh(lon_bathy, lat_bathy, elevation_bathy, shading='auto', cmap='viridis', transform=ccrs.PlateCarree())
    # plt.colorbar(pc, label='Elevation (m)')

    cbar = plt.colorbar(pc, label='Elevation (m)', shrink=0.4)  # Adjust the shrink parameter as needed
    pc.set_clim(-1000, 1000)

    # Add Cartopy features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_extent([min(lon_bathy), max(lon_bathy), min(lat_bathy), max(lat_bathy)])
    ax.stock_img()

    ax.plot(gnss.lon, gnss.lat, color="red", linewidth=2)

    # Set plot labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    # ax.set_title('Flight Path')
    plt.tight_layout()

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', linewidth=0.5, color='gray')
    gl.xlabels_top = False
    gl.ylabels_right = False

    save_path = "Results/Figures/Biscay_Survey_path_bathy.eps"
    plt.savefig(save_path)          
    plt.show()

def plot_ref_line_with_bathy():
    import netCDF4 as nc
    filenc = Path("..", "data", "Bathymetry", "GEBCO_15_Jan_2024_1b6d6c7c6bdc", "gebco_2023_n50.0_s42.0_w-10.0_e5.0.nc")
    dataset = nc.Dataset(filenc, "r")
    lat_bathy = dataset.variables['lat'][:]
    lon_bathy = dataset.variables['lon'][:]
    elevation_bathy = dataset.variables['elevation'][:]

    dataset.close()

    from scipy.interpolate import griddata
    def degrees_to_arc_minutes(degrees):
        return np.round(degrees * 60, 2)

    # Downsample bathymetry grid (adjust factor as needed)
    downsample_factor = 10
    lat_bathy_downsampled = lat_bathy[::downsample_factor]
    lon_bathy_downsampled = lon_bathy[::downsample_factor]
    elevation_bathy_downsampled = elevation_bathy[::downsample_factor, ::downsample_factor]


    lon_2d, lat_2d = np.meshgrid(lon_bathy_downsampled, lat_bathy_downsampled)

    # Flatten the 2D arrays for griddata
    lon_flat = lon_2d.flatten()
    lat_flat = lat_2d.flatten()
    elevation_flat = elevation_bathy_downsampled.flatten()


    # Interpolate elevation for GNSS coordinates
    elevation_interpolated = griddata((lon_flat, lat_flat),
                                    elevation_flat,
                                    (solution['lon'], solution['lat']),
                                    method='linear')

    # Add the interpolated elevation to the GNSS DataFrame
    solution['elevation_bathy'] = elevation_interpolated


    import pyproj
    from pyproj import Transformer
    survey = {}
    survey["time"] = solution["time"]
    survey["dg"] = solution["fdg"][:,2]

    pipeline = "+ellps=GRS80 +proj=pipeline +step +proj=utm +zone=30"
    transform_object = Transformer.from_pipeline(pipeline)
    geodetic_corr = [solution["lon"], solution["lat"], solution["h"]]
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

    @dataclass
    class Verification_line: 
        name: str
        index: np.array
        dist: np.array
        dg: np.array
        time: np.array
        h: np.array
        roll: np.array
        pitch: np.array
        yaw: np.array
        elevation: np.array
    ref_line_dist = np.arange(10,140,.5)
    Line = ["Line_1", "Line_2", "Line_3", "Line_4", "Line_5"]
    RMS_line = {}
    RMS_line["dist"] = ref_line_dist
    for i in range(len(Line)): 
        dist_line = dist2ref(UTM_corr[0][Ref[Line[i]]], 
                            UTM_corr[1][Ref[Line[i]]], ref_x, ref_y)
        RMS_line[Line[i]] = ds.interpolate_DS(dist_line*1e-3, solution["fdg"][Ref[Line[i]], 2], ref_line_dist, "linear", "extrapolate")

        Line[i] = Verification_line(Line[i], Ref[Line[i]], 
                                    dist_line, solution["fdg"][Ref[Line[i]], 2], 
                                    solution["time"][Ref[Line[i]]], 
                                    UTM_corr[2][Ref[Line[i]]], 
                                    solution["roll"][Ref[Line[i]]], 
                                    solution["pitch"][Ref[Line[i]]],
                                    solution["yaw"][Ref[Line[i]]], 
                                    solution['elevation_bathy'][Ref[Line[i]]])
        
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


    fig, axs = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]})

    # fig = plt.figure(figsize=(10,5))
    for i in range(5): 
        axs[0].plot(Line[i].dist*1e-3, Line[i].dg, label =  f"Profile {i+1}", linewidth=3)
        axs[1].plot(Line[i].dist*1e-3, Line[i].elevation, lw=3)

    axs[0].text(0.75, 1.5, f'RMSE: {rms_:.2f} [mGal]', transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
    # plt.grid()
    axs[0].plot(RMS_line["dist"], RMS_line["mean_line"], 'b--', linewidth=1, label="Mean reference")
    axs[0].legend(loc="lower right", fontsize="12")
    axs[1].set_xlabel("Distance from reference Point [km]")
    axs[0].set_ylabel("Gravity Disturbance [mGal]")
    # axs[0].set_title(f"Flight 115, Biscay Bay Reference Line: Attitude @ 300Hz\nButterworth, stages = 3, FWHM = {np.round(butterfilter['ftc']*0.74)} [s]")
    axs[0].set_xlim(5, 140)
    # save_path = "Results/Figures/Biscay_Reference_lines_@"+str(butterfilter["ftc"])+'s_'+str(butterfilter["stage"])+"_stages.eps"
    # plt.savefig(save_path)
    # 
    # Print or visualize the result
    
    axs[1].set_ylabel("Bathymetry [m]")
    # axs[1].legend()
    axs[1].set_xlim(5, 140)
    axs[1].text(.98, 0.195, f"Bathymetry @ {degrees_to_arc_minutes(np.diff(lat_bathy_downsampled)[0])}'", transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
    save_path = "Results/Figures/Biscay_Reference_lines_@"+str(butterfilter["ftc"])+'s_'+str(butterfilter["stage"])+"_stages_VS_Bathy.eps"
    plt.savefig(save_path)

    plt.show()


# imu_plot = {}
# imu_plot["time"] = imu.time
# imu_plot["dacc"] = imu_tuple["nacc_"][:,2] 


def plot_IMU_vs_dg(solution, dg):

    fig, ax1 = plt.subplots(figsize=(10,5))
    color = 'tab:orange'
    ax1.set_ylabel(r'Specific Force $f^{n}_{\mathrm{down}} - \gamma^n_{\mathrm{down}}$ [mGal]', color=color, weight='bold')
    line1, = ax1.plot(ds.SOW2HOD(dg["time"]), (dg["dacc"]+9.8)*1e5, label=r"IMU $f^{n}_{\mathrm{down}} - \gamma^n_{\mathrm{down}}$ 300 [Hz]", color=color, lw=1)
    ax1.tick_params(axis='y', labelcolor=color)

    ax1.set_xlabel('Time Hour Of Day [h]')
    ax1.set_xlim(min(ds.SOW2HOD(solution["time"])), max(ds.SOW2HOD(solution["time"])))
    ax1.set_ylim(min((dg["dacc"]+9.8)*1e5), max((dg["dacc"]+9.8)*1e5))
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Gravity Disturbance [mGal]', color=color, weight='bold')
    line2, = ax2.plot(ds.SOW2HOD(solution["time"]), solution["fdg"][:,2], color=color, lw=2, label=r"$\delta g^n_{\mathrm{down}}$ - Lowpassed")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(min((dg["dacc"]+9.8)*1e5), max((dg["dacc"]+9.8)*1e5))

    # Add legend
    plt.legend([line1, line2], [line1.get_label(), line2.get_label()], loc='upper left')

    insetPosition = Bbox.from_bounds(0.35, 0.15, 0.3, 0.3)  # Define position of inset axes
    axins = fig.add_axes(insetPosition)

    color = 'tab:orange'
    axins.set_facecolor(color)
    # axins.plot(ds.SOW2HOD(dg["time"]), (dg["dacc"]+9.8)*1e5, color=color, lw=1)
    axins.tick_params(axis='y', labelcolor=color)
    axins.set(xlim=(11,12), ylim=(-30,120))


    axins2 = axins.twinx()
    color = 'tab:blue'
    axins2.plot(ds.SOW2HOD(solution["time"]), solution["fdg"][:,2], color=color, lw=2)
    axins2.tick_params(axis='y', labelcolor=color)
    axins2.set(xlim=(11,12), ylim=(-30,120))


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
    
    my_mark_inset(ax2, axins, loc1a=2, loc1b=2, loc2a=1, loc2b=1, fc="none", ec="0.5")

    outname = "Results/Figures/Problem_description_figure.eps"
    plt.savefig(outname)



ans = input("Do you want to print the results? (y/n) ")
if ans.lower() == "y": 



    # solution["fdg"] = ds.but2_v2(solution["dg"][:,2], butterfilter["stage"], butterfilter["ftc"], solution["srate"])
    plt.figure(figsize=(12,4))
    plt.plot(ds.SOW2HOD(solution["time"][2000:-2000]), solution["fdg"][2000:-2000, 2], label=f"$\delta$g$_d$ - Butterworth @  FWHM = {np.round(butterfilter['ftc']*0.74)} [s]", linewidth=2)
    plt.legend(loc="lower left")
    plt.xlabel("Time Hour of Day [h]")
    plt.ylabel("Gravity Disturbance [mGal]")
    # plt.title(f"Direct method result: Down Component\nButterworth, stages = {butterfilter['stage']}, FWHM = {np.round(butterfilter['ftc']*0.74)} [s]")
    # plt.grid()
    output_file = 'Results/Figures/DS_300Hz_'+str(butterfilter["ftc"])+'s_'+str(butterfilter["stage"])+'_stages_Down_component.eps'
    plt.savefig(output_file)
    # plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(ds.SOW2HOD(solution["time"][2000:-2000]), solution["fdg"][2000:-2000, 0], label=f"$\delta$g$_n$ - Butterworth @ FWHM = {np.round(butterfilter['ftc']*0.74)} [s]", linewidth=2)
    plt.legend(loc="lower left")
    plt.xlabel("Time Hour of Day [h]")
    plt.ylabel("Gravity Disturbance [mGal]")
    # plt.title(f"Direct method result: North Component\nButterworth, stages = {butterfilter['stage']}, FWHM = {np.round(butterfilter['ftc']*0.74)} [s]")
    # plt.grid()
    output_file = 'Results/Figures/DS_300Hz_'+str(butterfilter["ftc"])+'s_'+str(butterfilter["stage"])+'_stages_North_component.eps'
    plt.savefig(output_file)
    # plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(ds.SOW2HOD(solution["time"][2000:-2000]), solution["fdg"][2000:-2000, 1], label=f"$\delta$g$_e$ - Butterworth @ FWHM = {np.round(butterfilter['ftc']*0.74)} [s]", linewidth=2)
    plt.legend(loc="lower left")
    plt.xlabel("Time Hour of Day [h]")
    plt.ylabel("Gravity Disturbance [mGal]")
    # plt.title(f"Direct method result: East Component\nButterworth, stages = {butterfilter['stage']}, FWHM = {np.round(butterfilter['ftc']*0.74)} [s]")
    # plt.grid()
    output_file = 'Results/Figures/DS_300Hz_'+str(butterfilter["ftc"])+'s_'+str(butterfilter["stage"])+'_stages_East_component.eps'
    plt.savefig(output_file)
    # plt.show()

    H_ref = solution["fh"][10050]
    solution["h_reduced"] = solution["fh"] - H_ref
    plt.figure(figsize=(12,6))
    # plt.plot(ds.SOW2HOD(solution["time"][10050:19950]), solution["h"][10050:19950], lw=2, label="Ellipsoidal Flight Height")
    # plt.plot(ds.SOW2HOD(solution["time"][10050:19950]), solution["fh"][10050:19950], lw=2, label="Lowpass Ellipsoidal Flight Height")
    plt.plot(ds.SOW2HOD(solution["time"][10050:19950]), solution["h_reduced"][10050:19950], lw=2, label="Reduced Lowpass Ellipsoidal Flight Height")
    plt.xlabel("Time Hour of Day [h]")
    plt.ylabel("Reduced Ellipsoidal Flight Height [m]")
    plt.legend()

    # Plot reference lines:
    plot_ref_line()

    # Plot Surevy Path and Bathymetry
    # plot_bathy()      # Runs really Slow

    # Plot Reference lines vs bathymetry
    plot_IMU_vs_dg(solution, imu_plot)
    plot_ref_line_with_bathy()

    # plot_IMU_vs_dg(solution, imu_plot)

elif ans.lower() == "n": 
    print("> Continued to save results as .pkl file")


# ------------------------ Save Solution -----------------------------

pkl_file_path = 'Results/DS_300Hz_'+str(butterfilter["ftc"])+'s_'+str(butterfilter["stage"])+'_stages.pkl'

# Save the solution dictionary to a pickle file
with open(pkl_file_path, 'wb') as pkl_file:
    pickle.dump(solution, pkl_file)

print(f'Data saved to {pkl_file_path}')



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
plt.figure(figsize=(10,5))

for i in range(len(Line)): 
    plt.plot(ds.SOW2HOD(solution["time"][Ref[Line[i]]]), solution["fdg"][Ref[Line[i]], 2], label=f"Profile {i + 1}", linewidth=2, zorder=10)
plt.plot(ds.SOW2HOD(solution["time"][2000:-2000]), solution["fdg"][2000:-2000, 2], label=f"$\delta$g$_d^n$ @ FWHM = {np.round(butterfilter['ftc']*0.74)} [s]", linewidth=2, zorder=0)
plt.legend(loc="lower left", ncol=2)
plt.ylim(-70+40, 70+40)
plt.xlabel("Time Hour of Day [h]")
plt.ylabel("Gravity Disturbance [mGal]")
output_file = 'Results/Figures/DS_300Hz_'+str(butterfilter["ftc"])+'s_'+str(butterfilter["stage"])+'_stages_Down_component_with_profiles.eps'
plt.savefig(output_file)
plt.show()

# # Specify the filename
# filename = "Results/Flight_position_ICGEM_input_1.txt"
# min = 0
# max = 9001
# np.savetxt(filename, np.column_stack((solution["lat"][min:max], solution["lon"][min:max], solution["h"][min:max])), fmt='%f', delimiter='\t')

# filename = "Results/Flight_position_ICGEM_input_2.txt"
# min = 9001
# max = 18001
# np.savetxt(filename, np.column_stack((solution["lat"][min:max], solution["lon"][min:max], solution["h"][min:max])), fmt='%f', delimiter='\t')

# filename = "Results/Flight_position_ICGEM_input_3.txt"
# min = 18001
# np.savetxt(filename, np.column_stack((solution["lat"][min:], solution["lon"][min:], solution["h"][min:])), fmt='%f', delimiter='\t')

# print("> Done")