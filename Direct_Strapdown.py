import numpy as np
import math
from tqdm import tqdm
import scipy as sc
from scipy import interpolate
from scipy.interpolate import splev, splrep
from dataclasses import dataclass
from numba import njit, jit
import pandas as pd
import os


def RMS(res):
    rms = math.sqrt((res.T@res)/len(res))
    return rms 

def pos_translate_v1(*args):
    """
      Procedure to translate position according to lever arm
      This is a python implementation of the Matlab code written by Tim Jensen 
      (DTU) 24/11/20
     
      -------------------------------------------------------------------------
     
      Input:
        lat         Nx1 array of latitude (GNSS position) [deg]
        lon         Nx1 array of longitude (GNSS position) [deg]
        h           Nx1 array of ellipsoidal height (GNSS position) [m]
        roll        Nx1 array of bank angle [deg]
        pitch       Nx1 array of elevation angle [deg]
        yaw         Nx1 array of heading angle [deg]
        lever_arm   3x1 array of lever arm along body-axes (IMU->GNSS) [m]
     
      Output:
        olat        Nx1 array of corrected latitude (IMU position) [deg]
        olon        Nx1 array of corrected longitude (IMU position) [deg]
        oh          Nx1 array of corrected height (IMU position) [deg]
     
      -------------------------------------------------------------------------
      Author: Christian Solgaard (DTU - Master student) 30/01/2023
    """
    
    
    # Check inputs - Tjek
    if len(args) != 7:
        raise ValueError("Expected 7 arguments, but got {}".format(len(args)))
    lat, lon, h, roll, pitch, yaw, lever_arm = args
    
    
    msg = "Error: Dimension mismatch!"
    no_epochs = len(lat)
    if len(lon) != no_epochs: 
        return print(msg)
    elif len(h) != no_epochs: 
        return print(msg)
    elif len(roll) != no_epochs: 
        return print(msg)
    elif len(pitch) != no_epochs: 
        return print(msg)
    elif len(yaw) != no_epochs: 
        return print(msg)
    
    # Set Parameters - Tjek
    class Ellipse:
        def __init__(self):
            self.a = 6378137.0 # [m]
            self.e2 = 6.69437999014*10**(-3) # e^2 [.]
            self.omega = 7292115.0*10**(-11) # [rad/s]

    rad2deg = 180/math.pi
    deg2rad = math.pi/180
    ellipse = Ellipse()
    
    ## Translate Position
    
    #Allocate space - Tjek 
    olat = np.array(no_epochs * [1*[np. nan]])
    olon = np.array(no_epochs * [1*[np. nan]])
    oh = np.array(no_epochs * [1*[np. nan]])
    
    #Derive some geometric parameters
    sin_lat = np.sin(lat*deg2rad)
    cos_lat = np.cos(lat*deg2rad)
    R_E = ellipse.a / np.sqrt(1 - ellipse.e2*sin_lat*sin_lat)
    R_N = R_E * ((1-ellipse.e2)/(1-ellipse.e2*sin_lat*sin_lat))
    
    #Form cosine and sine 
    cos_roll = np.cos(roll*deg2rad)
    sin_roll = np.sin(roll*deg2rad)
    cos_pitch = np.cos(pitch*deg2rad)
    sin_pitch = np.sin(pitch*deg2rad)
    cos_yaw = np.cos(yaw*deg2rad) 
    sin_yaw = np.sin(yaw*deg2rad)
    
    # Initialise transformation matrix 
    # Allocate memory
    C_b_n = np.zeros((3,3,no_epochs))
    
    C_b_n[0,0,:] = cos_pitch*cos_yaw
    C_b_n[0,1,:] = sin_roll*sin_pitch*cos_yaw - cos_roll*sin_yaw
    C_b_n[0,2,:] = cos_roll*sin_pitch*cos_yaw + sin_roll*sin_yaw
    C_b_n[1,0,:] = cos_pitch*sin_yaw
    C_b_n[1,1,:] = sin_roll*sin_pitch*sin_yaw + cos_roll*cos_yaw
    C_b_n[1,2,:] = cos_roll*sin_pitch*sin_yaw - sin_roll*cos_yaw
    C_b_n[2,0,:] = -sin_pitch
    C_b_n[2,1,:] = sin_roll*cos_pitch
    C_b_n[2,2,:] = cos_roll*cos_pitch


    # Move GNSS Solution to Reference Point 
    for n in tqdm(range(no_epochs), colour="green"): # np.arange(0, no_epochs):
        
        # Form Cartesian-to-Curvilinear transformation matrix
        T_r_p = np.array([[1/(R_N[n] + h[n]), 0, 0], [0, 1/(R_E[n] + h[n])/cos_lat[n], 0], [0, 0, -1]])

        # Move position
        l_n = np.dot(np.dot(T_r_p, C_b_n[:,:,n]), lever_arm)
        # l_n = np.dot(T_r_p, lever_arm)
        olat[n] = lat[n] - l_n[0]*rad2deg
        olon[n] = lon[n] - l_n[1]*rad2deg
        oh[n] = h[n] - l_n[2]
        
        # l_n = np.dot(np.dot(T_r_p, C_b_n[:,:,n]), lever_arm)
        # olat[n] = lat[n] - l_n[1]*rad2deg
        # olon[n] = lon[n] - l_n[0]*rad2deg
        # oh[n] = h[n] - (-l_n[2])



    return olat, olon, oh


def gnss_accelerations_v1(*args): 
    """
     Function that derives vertical accelerations from a time series of
     heights. This function is based upon the Matlab implementation written by 
     Tim E. Jensen 27/09-2019
    
     -------------------------------------------------------------------------
    
     Input:
       time    Nx1 array with time stamps [s]
       h       Nx1 array with heights [m]
       method  String can be either:
                 splines
                 difference
    
     Optional input:
       tout    Mx1 vector denoting time stamps with evaluated accelerations
               [s]
    
     Output:
       tout    Mx1 array with time stamps [s]
       acc     Mx1 array with accelerations [m/s^2]
    
     -------------------------------------------------------------------------
     Author: Christian Solgaard (DTU - Master student) 01/02-2023
    """

    # Check input parameters
    if len(args) < 3:
        raise ValueError("Expected at least 3 arguments, but got {}".format(len(args)))
        
    elif len(args) == 3: 
        time, h, method = args
        tout = time
    
    elif len(args) == 4: 
        time, h, method, tout = args
    
    # Derive Sampling rate 
    dt = np.diff(time)
    sfreq = round(1/np.nanmedian(dt))
    sfreq = sfreq * sfreq
    
    # ------------------------------------------------------------------------
    ## Derive sampling rate 
    if method.lower() == 'splines':
        # Fit cubic spline piecewise polynomial
        pp = sc.interpolate.CubicSpline(time, h)

        # Determine spline order
        k = 7
        pp = sc.interpolate.BSpline(time, h, k=k,extrapolate=None)

        # Compute second order derivative
        pp2 = pp.derivative(2)

        # Evaluate GNSS accelerations
        acc = -pp2(tout)

        # Remove NaN values
        idnan = np.isnan(acc)
        tout = np.delete(tout, np.where(idnan))
        acc = np.delete(acc, np.where(idnan))

    elif method.lower() == "difference":
            
        # Interpolate heights onto time stamps
        tint = np.arange(np.ceil(time[0]*sfreq), np.floor(time[-1]*sfreq)+1) / sfreq

        interp_spline = interpolate.UnivariateSpline(time, h, s=0)
        hint = interp_spline(tint)
        
        # Derive accelerations 
        #acc = - (hint[2:] + hint[:-3] - 2*hint[1:-2]) * sfreq2
        acc = -(hint[2:] + hint[:-2] - 2 * hint[1:-1]) * sfreq#(sfreq**2)
        
        # Adjust time stamps 
        tint = tint[1:-1]
        
        # Remove NaN values 
        tint = tint[np.logical_not(np.isnan(tint))]
        acc = acc[np.logical_not(np.isnan(acc))]
        
        # Interpolate onto desired time stamps 
        if len(args) < 4: 
            tout = tint 
        else :
            interp_spline = interpolate.interp1d(tint, acc, kind='linear', extrapolate=None)
            hint = interp_spline(tout)
            
    else :
        return print("Method not reckognized!")
    
    return acc, tout


def b2n_v1(*args): 
    """
      Procedure that rotates a vector from body-frame to navigation-frame
      This is a python implementation of the Matlab code written by Tim Jensen 
      (DTU) 24/11/20
      -------------------------------------------------------------------------
     
      Input:
        time    Nx1 array of time stamps
        bacc    Nx3 array of observations in body-frame
        att     Nx3 array of attitude (bank,elevation,heading) [deg]
     
      Output:
        nacc    Nx3 array of observations in navigation-frame
     
      -------------------------------------------------------------------------
      Author: Christian Solgaard (DTU - Master student) 30/01/2023
    """
    
    # Check input parameters
    if len(args) != 3:
        raise ValueError("Expected 3 arguments, but got {}".format(len(args)))
    time, bacc, att = args
    
    
    # Derive epochs
    no_epochs = len(time)
    
    # Check dimensions of accelerations
    ydim, xdim = bacc.shape
    if ydim != no_epochs and xdim != 3: 
        return print("Dimensions of bacc not consistent!")
    
    # Check dimensions of navigation estimates
    ydim, xdim = att.shape
    if ydim != no_epochs and xdim != 3: 
        return print("Dimensions of att not consistent!")
    
    
    ## Transform Accelerations into NED-Frame
    # Allocate space 
    nacc = np.array(no_epochs * [3*[np. nan]])
    
    # Compute some trigonometric funtions
    deg2rad = math.pi/180
    sin_roll = np.sin(att[:,0]*deg2rad)
    cos_roll = np.cos(att[:,0]*deg2rad)
    sin_pitch = np.sin(att[:,1]*deg2rad)
    cos_pitch = np.cos(att[:,1]*deg2rad)
    sin_yaw = np.sin(att[:,2]*deg2rad)
    cos_yaw = np.cos(att[:,2]*deg2rad)
    
    C_b_n = np.zeros((3,3,no_epochs))

    C_b_n[0,0,:] = cos_pitch*cos_yaw
    C_b_n[0,1,:] = sin_roll*sin_pitch*cos_yaw - cos_roll*sin_yaw
    C_b_n[0,2,:] = cos_roll*sin_pitch*cos_yaw + sin_roll*sin_yaw
    C_b_n[1,0,:] = cos_pitch*sin_yaw
    C_b_n[1,1,:] = sin_roll*sin_pitch*sin_yaw + cos_roll*cos_yaw
    C_b_n[1,2,:] = cos_roll*sin_pitch*sin_yaw - sin_roll*cos_yaw
    C_b_n[2,0,:] = -sin_pitch
    C_b_n[2,1,:] = sin_roll*cos_pitch
    C_b_n[2,2,:] = cos_roll*cos_pitch


    nacc = C_b_n[:,:,np.newaxis,:] * bacc[:,np.newaxis,:].T
    nacc = np.sum(nacc, axis=(1,2))
    nacc[np.isnan(nacc)] = 0

    # nacc_ = np.vstack([nacc[1,:], nacc[0,:], -nacc[2,:]])

    return nacc # nacc


def transport_rate_v2(*args):
    """
     Procedure to compute transport rate
     This is a python implementation of the Matlab code written by Tim Jensen 
     (DTU) 24/11/20
     -------------------------------------------------------------------------
    
     Input:
       time    Nx1 array of time stamps
       vel     Nx3 array of velocity (north,east,down) [m/s]
       pos     Nx3 array of position (lat,lon,h) [deg,deg,m]
    
     Output:
       tacc    Nx3 array of computed transport rate (north,east,down) [m/s^2]
    
     -------------------------------------------------------------------------
     Author: Christian  Solgaard (DTU - Master student) 31/01/2023
    """

    # Check input parameters
    if len(args) != 3:
        raise ValueError("Expected 3 arguments, but got {}".format(len(args)))
    time, vel, pos = args
    
    # Derive epochs 
    no_epochs = len(time)
    
    # Check dimensions of navigation estimates
    ydim, xdim = vel.shape
    if ydim != no_epochs and xdim != 3: 
        return print("Dimensions of vel not consistent!")
    
    ydim, xdim = pos.shape
    if ydim != no_epochs and xdim != 3: 
        return print("Dimensions of pos not consistent!")
    
    
    # Set Parameters - Tjek
    class Ellipse:
        def __init__(self):
            self.a = 6378137.0 # [m]
            self.e2 = 6.69437999014*10**(-3) # e^2 [.]
            self.omega = 7292115.0*10**(-11) # [rad/s]

    rad2deg = 180/math.pi
    deg2rad = math.pi/180
    ellipse = Ellipse()
    
    #--------------------------------------------------------------------------
    # Compute Transport Rate in NED-Frame 
    # Allocate space 
    tacc = np.array(no_epochs * [3*[np. nan]])
    
    # Compute some trigonometric functions
    sin_phi = np.sin(pos[:,0]*deg2rad)
    cos_phi = np.cos(pos[:,0]*deg2rad)
    
    # Compute radii of curvature
    term = 1 - ellipse.e2 * sin_phi * sin_phi
    R_E = ellipse.a / np.sqrt(term)
    R_N = R_E * ((1 - ellipse.e2)/term)
    
    # Compute Earth Rotation Rate, Form skew-symmetric matrix
    Omega_ie = np.zeros((3,3,no_epochs))
    Omega_ie[0,1,:] = -ellipse.omega * (-sin_phi)
    Omega_ie[1,0,:] = ellipse.omega * (-sin_phi)
    Omega_ie[1,2,:] =  - ellipse.omega * cos_phi
    Omega_ie[2,1,:] = ellipse.omega * cos_phi

    Omega_en = np.zeros((3,3,no_epochs))
    Omega_en[0,1,:] = -(-vel[:,1]*sin_phi/(R_E+pos[:,2])/cos_phi)
    Omega_en[0,2,:] = -vel[:,0]/(R_N+pos[:,2])
    Omega_en[1,0,:] = -vel[:,1]*sin_phi/(R_E+pos[:,2])/cos_phi
    Omega_en[1,2,:] = -(vel[:,1]/(R_E+pos[:,2]))
    Omega_en[2,0,:] = -(-vel[:,0]/(R_N+pos[:,2]))
    Omega_en[2,1,:] =  vel[:,1]/(R_E+pos[:,2])

    Omega = 2*Omega_ie + Omega_en
    tacc = Omega[:,:,np.newaxis,:] * vel[:,np.newaxis,:].T
    tacc = np.sum(tacc, axis=(1,2)).T

    tacc[np.isnan(tacc)] = 0

    return tacc


@njit
def but2_v2(*args):
    """
     2nd order multi-stage forward/backward butterworth filter.
     This is a python implementation of the Matlab code by Tim Jensen (DTU) 29/05/2020, 
     Which is further based on the code by Rene Forsberg, Nov. 1997
    
     -------------------------------------------------------------------------
    
     Input:
       data    Vector of data series
       stage   Number of iterations, 1 iteration = forwar+backward run
       ftc     Filter time constant [s]
       dt      Sample interval [s]
    
     Output:
       fdata   Vector of filtered data series
    
     -------------------------------------------------------------------------
     Author:    Christian Solgaard (DTU - Master student) 27/02-2023
    """
    
    # Check input parameters
    if len(args)!= 4: 
        raise ValueError("Expected 4 arguments, but got {}".format(len(args)))
    data, stage, ftc, dt = args

    edge = 200
    nmax = len(data)
    tc = ftc/dt
    mu = np.mean(data)
    
    n1 = edge
    n2 = nmax + edge + 1
    n3 = nmax + 2*edge
    
    p = np.empty(n3) # Zero padding og original signal 
    p[:n1] = 0.0
    p[n1:n2-1] = data-mu
    p[n2:] = 0.0
    
    nmax = n3
    kk = np.cos(np.pi/tc) / np.sin(np.pi/tc)
    k0 = kk*kk + np.sqrt(2.0)*kk + 1
    k1 = 2 * (1 - kk*kk)
    k2 = kk*kk - np.sqrt(2.0)*kk + 1
    
    for j in range(stage):
        x1 = p[0]
        x2 = p[0]
        y1 = p[0]
        y2 = p[0]
        
        for i in range(nmax): # Forward pass
            xi = p[i]
            p[i] = (x2 + 2*x1 + xi - k2*y2 - k1*y1) / k0
            x2 = x1
            x1 = xi
            y2 = y1
            y1 = p[i]
        
        x1 = p[nmax-1]
        x2 = p[nmax-1]
        y1 = p[nmax-1]
        y2 = p[nmax-1]
        
        for i in range(nmax-1, -1, -1): # Backward pass
            xi = p[i]
            p[i] = (x2 + 2*x1 + xi - k2*y2 - k1*y1) / k0
            x2 = x1
            x1 = xi
            y2 = y1
            y1 = p[i]
    
    fdata = p[n1:n2-1] + mu
    return fdata


def normal_gravity_precise_v1(*args):
    """
     normal_gravity_precise - routine that performs an exact computation of
     the normal gravity vector according to WGS84. Input are geodetic
     coordinates (WGS84) and output is along the north, east and down axes of
     a local coordinate frame (n-frame). The down axis is defined as being
     perpendicular to the reference ellipsoid.
    
     This implementation is based upon the Matlab code written by 
     Tim Jensen (DTU) 13/03/18
    
     -------------------------------------------------------------------------
    
     Input:
       lat     Geodetic latitude [deg]
       lon     Geodetic longitude [deg]
       h       Ellipsoidal height [m]
    
     Optional input:
       lf      Integer value specifying resolving axes frame:
                   0: Ellipsoidal-harmonic (u,beta,gamma)
                   1: Rectangular (x,y,z)
                   2: Spherical (r,lat,lon)
                   3: Geodetic (north,east,down) <- DEFAULT
    9
     Output:
       g       tuple structure with 3 grids corresponding to the three
               components of the specified reference frame  [m/s^2]
    
     -------------------------------------------------------------------------
     Author: Christian Solgaard (DTU - Master student) 02/02-2023
    """
    
    # Check input parameters
    if len(args) > 4:
        raise ValueError("Maximum four input arguments can be handled")
    elif len(args) < 4: 
        lf = 3
    elif len(args) < 3: 
        raise ValueError("At least three input arguments required!")
    lat, lon, h, lf = args
    
    
    # Get dimension of computation grid 
    nrow, ncol = lat.shape
    
    # Do some consistency checks 
    if lon.shape[0] != nrow: 
        return print("Latitude and longitude grids not consistent!")
    elif lon.shape[1] != ncol:
        return print("Latitude and longitude grids not consistent!")
    elif h.shape[0] != nrow: 
        return print("Latitude and height grids not consistent!")
    elif h.shape[1] != ncol: 
        return print("Latitude and height grids not consistent!")
    
    # Set Parameters 
    class GRS80:
        def __init__(self):
            self.a = 6378137.0 # Semi-major axis [m]
            self.GM = 3986005*10**(8) # Standard gravitational parameter (gravitational constant times mass) [m^3/s^2]
            self.omega = 7292115.0*10**(-11) # Earth rotation rate (rad/s)
            self.b = 6356752.3141 # Semi-minor axis [m]
            self.ecc2 = 0.00669438002290 # First eccentricity squared [.]
            self.E = 521854.0097 # Linear eccentricity [.]

    # Derfine conversion factors
    rad2deg = 180/math.pi
    deg2rad = math.pi/180
    GRS80 = GRS80()
    
    # ------------------------------------------------------------------------

    # Compute normal gravity in ellipsoidal reference frame (u, beta, lambda)
    a2 = GRS80.a**2
    b2 = GRS80.b**2
    omega2 = GRS80.omega**2
    
    # Derive sine and cosine of latitude 
    lat = lat*deg2rad
    lon = lon*deg2rad
    sinlat = np.sin(lat)
    coslat = np.cos(lat) 
    sinlon = np.sin(lon) 
    coslon = np.cos(lon)
    
    # Compute radius of curvature of prime vertical
    R_E = GRS80.a / np.sqrt(1 - GRS80.ecc2 * sinlat * sinlat)
    
    # Compute rectangular coordinates 
    x = (R_E + h) * coslat * coslon
    y = (R_E + h) * coslat * sinlon
    z = (R_E*(1-GRS80.ecc2) + h) * sinlat
    
    # Pre-compute some variables 
    x2 = x**2
    y2 = y**2
    z2 = z**2
    E2 = GRS80.E**2
    
    # Compute ellipsoidal coordinates 
    u_ = np.sqrt( 0.5 * ( x2 + y2 + z2 - E2 ) * ( 1 + np.sqrt( 1 + ( 4*E2*z2 / ( x2 + y2 + z2 - E2 )**2 ) ) ) )
    u2 = u_**2
    v2 = u2 + E2
    v = np.sqrt(v2)
    
    # Compute reduced latitude
    beta_ = np.arctan(z * v / u_ / np.sqrt(x2 + y2))
    sinbeta = np.sin(beta_) 
    cosbeta = np.cos(beta_) 
    
    # Compute other variables
    w = np.sqrt(u2 + E2*(sinbeta**2))/v
    q = 0.5 * ( ( 1 + ( 3*u2 / E2 ) ) * np.arctan( GRS80.E / u_ ) - ( 3*u_ / GRS80.E ) )
    q0 = 0.5 * ( ( 1 + ( 3*b2 / E2 ) ) * np.arctan( GRS80.E / GRS80.b ) - ( 3*GRS80.b / GRS80.E ) ) # scalar - depends only on ellipse
    q1 = 3 * ( 1 + ( u2 / E2 ) ) * ( 1 - (u_/GRS80.E)*np.arctan( GRS80.E / u_ ) ) - 1
    
    # Compute u and beta components of normal gravity
    # Initialize g as a data-class 
    @dataclass
    class g:
        u: np.array
        beta: np.array
        lambda_: np.array
        x: np.array
        y: np.array
        z: np.array
        r: np.array
        lat: np.array
        north: np.array
        east: np.array
        down: np.array
        
    def create_g_instance(kwargs):
        g_fields = {
            "u": np.array([]),
            "beta": np.array([]),
            "lambda_": np.array([]),
            "x": np.array([]),
            "y": np.array([]),
            "z": np.array([]),
            "r": np.array([]),
            "lat": np.array([]),
            "north": np.array([]),
            "east":np.array([]),
            "down": np.array([])
        }
        g_fields.update(kwargs)
        return g(**g_fields)

    # Compute u and beta components of normal gravity
    u = ( (-GRS80.GM/v2) - (omega2*a2*GRS80.E/v2)*(q1/q0)*(0.5*(sinbeta**2)-(1/6)) + omega2*u_*(cosbeta**2) ) / w
    beta = ( (omega2*a2/v)*(q/q0)*sinbeta*cosbeta - omega2*v*sinbeta*cosbeta ) / w
    
    # Check whether or not to continue
    if lf == 0: 
        lambda_ = np.zeros(u.shape)
        return create_g_instance({"u":u, "beta":beta, "lambda_":lambda_}), u_, beta_ 
    # create_g_instance({"x": x, "y": y})
    g1 = create_g_instance({"u":u, "beta":beta})
    # Transform components into a rectangular system (x,y,z)
    x = ( u_/w/v ) * cosbeta * coslon * g1.u + (-1/w) * sinbeta * coslon * g1.beta
    y = ( u_/w/v ) * cosbeta * sinlon * g1.u + (-1/w) * sinbeta * sinlon * g1.beta
    z = (1/w) * sinbeta * g1.u + ( u/w/v ) * cosbeta * g1.beta
    
    # Check whether or not to continue
    if lf == 1: 
        return create_g_instance({"x": x, "y" : y, "z" : z}), u_, beta_ 
    
    # Transform components into a spherical coordinate system (r,phi_E,lambda)
    g2 = create_g_instance({"x" : x, "y" : y, "z": z})
    
    # Compute geocentric latitude phi_E from Torge (4.11a)
    phi_E = np.arctan( (1 - GRS80.ecc2) * np.tan(lat) )
    sinphi_E = np.sin(phi_E)
    cosphi_E = np.cos(phi_E)
    
    # Radial component
    r = cosphi_E * coslon * g2.x + cosphi_E * sinlon * g2.y + sinphi_E * g2.z
    
    # North component
    lat_ = -sinphi_E * coslon * g2.x + (-sinphi_E) * sinlon * g2.y + cosphi_E * g2.z
    
    # Check whether or not to continue
    if lf == 2: 
        lon_ = np.zeros(r.shape)
        return create_g_instance({"r" : r, "lat" : lat_, "lon" : lon_}), u_, beta_ 
    
    g3 = create_g_instance({"r" : r, "lat" : lat_})
    # Project components onto geodetic normal line
    # Compute angular difference
    alpha = lat - phi_E
    
    # Compute components in local frame
    north = - g3.r * np.sin(alpha) + g3.lat * np.cos(alpha)
    east = np.zeros(north.shape)
    down = - g3.r * np.cos(alpha) - g3.lat * np.sin(alpha)
    
    return create_g_instance({"north" : north, "east" : east, "down" : down}), u_, beta_
    

def interpolate_DS(*args):
     """
     Interpolation function, based on the scipy package 
     Inputs: 
          var1:               ... (np.array)
          var2:               ... (np.array)
          var3:               ... (np.array)
          kind:               ... (str)
          bounds_error:       ... (str)
          fill_value:         ... (str)
     
    -------------------------------------------------------------------------
     Author: Christian Solgaard (DTU - Master student) 02/02-2023
     """

    # paramns based on varible inputs. 
     if len(args) == 6:
          var1, var2, var3, kind, bounds_error, fill_value = args
          interp_spline = interpolate.interp1d(var1, var2, kind=kind, 
                                              bounds_error=bounds_error, fill_value=fill_value)
          interp = interp_spline(var3)
          return interp
     else: 
          var1, var2, var3, kind, fill_value = args
          interp_spline = interpolate.interp1d(var1, var2, kind=kind, fill_value=fill_value)
          interp = interp_spline(var3)
          return interp


def movmean(array, window_size):
    """
    A simple implementation of a lowpass filter. 
    The lowpass filter is based on a moving mean method, computed using 
    a kernel window and convolution. 
    ------------------------------------------------------------------------

    Input: 
        array:              data [N x 1], (np.array)
        window_size:        size of moving kernal, (int)

    Output: 
        moving_averages:    data [N-window_size x 1], (np.array) 
    
    -------------------------------------------------------------------------
     Author: Christian Solgaard (DTU - Master student) 14/03-2023
    """

    # Convert the input array to a numpy array
    arr = np.array(array)
    
    # Use numpy's convolve function to calculate the moving average
    weights = np.ones(window_size) / window_size
    moving_averages = np.convolve(arr, weights, mode='valid')
    
    return moving_averages


def cutoff_bound(*args): 
    """ 
    Procedure to clean and remove divergence effects from numerical 
    differentation, originating from acceleration calculation from 
    GNSS: 

    --------------------------------------------------------------
    Input: 
        arr:        np.array(), [1, N]

    Output: 
        arr:        np.array(), [1, N-x]
            where x, is length of the removed values. 
        x:          int, number of elements removed from array.

    ---------------------------------------------------------------
    Author: Christian Solgaard (DTU - Master student) 03/03-2023
    """

    # Check input parameters
    if len(args) != 1:
        raise ValueError("Expected 1 arguments, but got {}".format(len(args)))
    arr = args[0]

    # Numerical diff the array 
    darr = np.diff(arr).reshape(-1)

    threshold_value  = 0.001
    idx = len(darr)
    while True: 
        if idx >= 0 and np.abs(darr[idx-1]) <= threshold_value: 
            if np.abs(np.mean(darr[idx-100:idx])) <= threshold_value:
                break
            else:  
                idx -= 1
        idx -= 1

    if idx >= 0: 
        arr_out = arr[:idx+1]
        index = idx+1
        print(f"The threshold value {threshold_value} was found at index {idx+1}.")
    else: 
        print(f"The threshold value {threshold_value} was not found in the array. no cutoff was applied.")
        
    return arr_out, index   


def bias_drift_corr(*args): 
    """
    Procedure to calculate the bias and drift correction of 
    gravity disturbance, derived using the Direct Strapdown method 

    ---------------------------------------------------------------

    Input: 
        dg:         np.array(), Gravity disurbance array
        time:       np.array(), Time vector [SOW], (gnss_time)
        vel:        np.array(), Velocity (scalar) , [N,1], gnss_time

    Output:    
        dg_corr:    np.array(), Corrected gravity disturbance.  
    
    
    ---------------------------------------------------------------
    Author: Christian Solgaard (DTU - Master student) 03/03-2023
    """

    # Check input parameters
    if len(args) != 3:
        raise ValueError("Expected 3 arguments, but got {}".format(len(args)))
    dg, time, vel = args

    # Check input dimensions 
    no_epochs = len(vel)

    if no_epochs != dg.shape[0]: 
        raise ValueError("Dimensions of dg and vel, not consistent")
    elif dg.shape[0] != time.shape[0]: 
        raise ValueError("Dimensions of dg and time, not consistent")
 
    # Determine Stationary periode 
    idx = vel == 0 # Value should be 0, as stationary measurments in needed
    idx_start = np.array([])
    a = False
    b = False
    for i in range(len(idx)): 
        if idx[i] == True: 
            idx_start = np.append(idx_start, i)
        else: 
            if len(idx_start) >= 100:
                break
            else: 
                a = True
    idx_end = np.array([])
    for i in range(len(idx) - 1, -1,-1): 
        if idx[i] == True: 
            idx_end = np.append(idx_end, i)
        else: 
            if len(idx_end) >= 100: 
                break 
            else: 
                b = True
    if a: print("Stationary time not suffiscient, additional start points included")
    if b: print("Stationary time not suffiscient, additional end points included")

    idx_end = np.int64(idx_end); idx_start = np.int64(idx_start)
    idx = np.append(idx_start, np.flip(idx_end))
    time_ref = time[idx]
    dg_ref = dg[idx]

    dg_ref_1 = dg[idx_start]
    dg_ref_2 = dg[np.flip(idx_end)]

    # Calculate reference dg 
    dg_ref = np.mean(np.append(dg_ref_1, dg_ref_2))

    # Set reference time: 
    t1 = time[0]
    t2 = time[-1]

    k1 = np.mean(dg_ref_1 - dg_ref)
    k2 = np.mean(dg_ref_2 - dg_ref)

    # Implement bias calculation for every epoch during non-stationary flight
    k = np.zeros(len(dg)) # Allocate Space
    k = k1 + ((time - t1)/(t2 - t1))*(k2 - k1)  # Bias determination jf. (Johann et al. 2019)

    dg_corr = dg.flatten() - k
    
    return dg_corr


def decyear(*args): 
    """ 
    Procedure to convert UTCDate and UTCTime to Decimal year. 
    --------------------------------------------------------------
    
    Input: 
        UTCDate:    array, stucture: %d/%m/%Y
        UTCTime:    array, structure: %H:%M:%S.%f

    Output: 
        Dec_year:   array. 

    ---------------------------------------------------------------
    Author: Christian Solgaard (DTU - Master student) 17/03-2023
    """

    # Check input parameters
    if len(args) != 2:
        raise ValueError("Expected 2 arguments, but got {}".format(len(args)))
    UTCDate, UTCTime = args

    timestamp_str = np.core.defchararray.add(UTCDate.values.astype(str), ' ')
    timestamp_str = np.core.defchararray.add(timestamp_str, UTCTime.values.astype(str))
    dec_year = pd.to_datetime(timestamp_str, format="%d/%m/%Y %H:%M:%S.%f")

    # # apply the decimal year conversion to the timestamp column
    year_start = pd.to_datetime(dec_year.year, format="%Y")
    year_end = pd.to_datetime(dec_year.year + 1, format="%Y")
    dec_year = dec_year.year + ((dec_year - year_start).total_seconds() / (year_end - year_start).total_seconds())

    return dec_year

def PPP_freq(ifile): 
    """ 
    Procedure to extract the PPP solution Frequency from filename  
    --------------------------------------------------------------
    
    Input: 
        ifile:    str. Path to file

    Output: 
        freq:   int. 

    ---------------------------------------------------------------
    Author: Christian Solgaard (DTU - Master student) 09/11-2023
    """
    input_filename = os.path.basename(ifile)
    freq = 1/int(input_filename.split('_')[-1][0])
    return freq

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



def gnss_accelerations_v2(time, lat, lon, h):

    result = {}

    # Set some variables
    ellipsoid_a = 6378137.0
    ellipsoid_ecc2 = 6.69437999014E-3
    deg2rad = np.pi / 180

    # Compute useful variables
    cos_lat = np.cos(np.deg2rad(lat))
    sin_lat = np.sin(np.deg2rad(lat))
    term = 1.0 - ellipsoid_ecc2 * sin_lat * sin_lat
    R_E = ellipsoid_a / np.sqrt(term)
    R_N = R_E * (1.0 - ellipsoid_ecc2) / term

    # Fit cubic spline piecewise polynomial
    k = 3  # spline order
    tck_lat = splrep(time, lat * deg2rad, k=k)
    tck_lon = splrep(time, lon * deg2rad, k=k)
    tck_h = splrep(time, h, k=k)

    # Compute first order derivative
    derivative_order = 1
    pp1_lat = splev(time, tck_lat, der=derivative_order)
    pp1_lon = splev(time, tck_lon, der=derivative_order)
    pp1_h = splev(time, tck_h, der=derivative_order)

    # Evaluate velocity
    result['vn'] = pp1_lat * R_N
    result['ve'] = pp1_lon * R_E * cos_lat
    result['vd'] = -pp1_h

    # Compute secound order derivative 
    derivative_order = 2
    pp2_lat = splev(time, tck_lat, der=derivative_order)
    pp2_lon = splev(time, tck_lon, der=derivative_order)
    pp2_h = splev(time, tck_h, der=derivative_order)

    # Evaluate acceleration 
    result["accn"] = pp2_lat * R_N
    result["acce"] = pp2_lon * R_E * cos_lat
    result["accd"] = -pp2_h

    return result


def RMSE_calc(res): 
    RMSE = np.sqrt(np.sum(res.T*res)/(len(res)))
    return RMSE



@njit
def approximate_expm(A, order):
    """
    Approximate the matrix exponential of A using a Taylor series expansion.
    
    Parameters:
    A (array-like): Input matrix.
    order (int): Order of the Taylor series expansion (higher order for more accuracy).
    
    Returns:
    expm_A (numpy.ndarray): Approximation of the matrix exponential.
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
    A = np.zeros((6,6))
    A[0:3, 0:3] = -F * tau
    A[0:3, 3:] = PSD * np.abs(tau)
    A[3:, 0:3] = np.zeros((nvar, nvar))
    A[3:, 3:] = F.T * tau

    # B = expm(A)
    B = approximate_expm(A, 20)    

    # The lower-right partition is the transpose of the transition matrix
    Phi = B[id1:id2, id1:id2].T

    # Compute Q using Brown and Hwang (3.9.25)
    Q = Phi * B[0:nvar, id1:id2]
    
    return Phi, Q


# def rewritePDF2Tuple(PDF, index_start, index_end): 
#     temp = {}
#     for key, value in PDF.items():
#         temp[key] = value[index_start:index_end]
#     return temp

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




def pos_translate_v2_no_rotation(*args):
    """
      Procedure to translate position according to lever arm 
      GNSS -> GIRAFE 
     
      -------------------------------------------------------------------------
     
      Input:
        lat         Nx1 array of latitude (GNSS position) [deg]
        lon         Nx1 array of longitude (GNSS position) [deg]
        h           Nx1 array of ellipsoidal height (GNSS position) [m]
        lever_arm   3x1 array of lever arm along body-axes (IMU->GNSS) [m]
     
      Output:
        olat        Nx1 array of corrected latitude (IMU position) [deg]
        olon        Nx1 array of corrected longitude (IMU position) [deg]
        oh          Nx1 array of corrected height (IMU position) [deg]
     
      -------------------------------------------------------------------------
      Author: Christian Solgaard (DTU - Master student) 16/02/2023
    """
    
    # Check inputs - Tjek
    if len(args) != 4:
        raise ValueError("Expected 4 arguments, but got {}".format(len(args)))
    lat, lon, h, lever_arm = args
    
    
    msg = "Error: Dimension mismatch!"
    no_epochs = len(lat)
    if len(lon) != no_epochs: 
        return print(msg)
    elif len(h) != no_epochs: 
        return print(msg)
    
    # Set Parameters - Tjek
    class Ellipse:
        def __init__(self):
            self.a = 6378137.0 # [m]
            self.e2 = 6.69437999014*10**(-3) # e^2 [.]
            self.omega = 7292115.0*10**(-11) # [rad/s]

    rad2deg = 180/math.pi
    deg2rad = math.pi/180
    ellipse = Ellipse()
    
    ## Translate Position
    
    #Allocate space - Tjek 
    olat = np.array(no_epochs * [1*[np. nan]])
    olon = np.array(no_epochs * [1*[np. nan]])
    oh = np.array(no_epochs * [1*[np. nan]])
    
    #Derive some geometric parameters
    sin_lat = np.sin(lat*deg2rad)
    cos_lat = np.cos(lat*deg2rad)
    R_E = ellipse.a / np.sqrt(1 - ellipse.e2*sin_lat*sin_lat)
    R_N = R_E * ((1-ellipse.e2)/(1-ellipse.e2*sin_lat*sin_lat))

    # Move GNSS Solution to Reference Point 
    for n in tqdm(range(no_epochs), colour="green"): 
        
        # Form Cartesian-to-Curvilinear transformation matrix
        T_r_p = np.array([[1/(R_N[n] + h[n]), 0, 0], [0, 1/(R_E[n] + h[n])/cos_lat[n], 0], [0, 0, -1]])

        # Move position
        l_n = np.dot(T_r_p, lever_arm)

        olat[n] = lat[n] - l_n[0]*rad2deg
        olon[n] = lon[n] - l_n[1]*rad2deg
        oh[n] = h[n] - l_n[2]
        
    return olat, olon, oh