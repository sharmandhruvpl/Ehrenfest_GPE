import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import argparse
import os
import time
import json
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import h5py

# Constants
hbar = 1.0  # Reduced Planck's constant (atomic units)
mass = 1.0  # Proton mass in atomic units

# Optimized Hamiltonian
def hamiltonian(x, y, parameters):
    """Compute the Hamiltonian with parameter dict - optimized but mathematically identical."""
    # Calculate z directly - avoid function call overhead
    if parameters['z_choice'] == 'constant':
        z = parameters['z_val']
    else:
        z = np.sqrt(x**2 + np.sqrt(1 - parameters['e']**2) * y**2)

    # Calculate off-diagonal term directly
    off_diag = x - 1j * np.sqrt(1 - parameters['e']**2) * y
    
    # Create and return Hamiltonian - no change in physics
    return 0.5*parameters['s'] * np.array([
        [parameters['a']*z, off_diag], 
        [np.conjugate(off_diag), -z]
    ], dtype=complex)

# z-function - simplified direct calculation
def z_func(x, y, parameters):
    """Compute z value with parameter dict."""
    if parameters['z_choice'] == 'constant':
        return parameters['z_val']
    else:
        # Direct calculation
        return np.sqrt(x**2 + np.sqrt(1 - parameters['e']**2) * y**2)

# Optimized eigenvectors calculation - NO CACHING
def eigenvectors(x, y, parameters):
    """Calculate eigenvectors of Hamiltonian - optimized but exact."""
    # Compute Hamiltonian
    H = hamiltonian(x, y, parameters)
    
    # Compute eigenvalues and eigenvectors directly
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    # Normalize the eigenvectors - critical for quantum accuracy
    eigenvectors[:, 0] /= np.linalg.norm(eigenvectors[:, 0])
    eigenvectors[:, 1] /= np.linalg.norm(eigenvectors[:, 1])
    
    # Return as separate arrays for lower and upper states
    psi_lower = eigenvectors[:, 0]
    psi_upper = eigenvectors[:, 1]
    
    return psi_lower, psi_upper

# Berry phase calculation - NO CACHING, direct calculation each time
def berry_phase(x, y, parameters, state='lower'):
    """
    Calculate the analytical Berry phase for elliptical intersection.
    Uses the closed-form expression with alpha = a + 1.

    Lower state: pi * (-alpha + sqrt(4 + alpha^2)) / sqrt(4 + alpha^2)
    Upper state: 2*pi - lower

    Parameters:
    - x, y: Position coordinates (not used, kept for interface compatibility)
    - parameters: Simulation parameters
    - state: Either 'lower' or 'upper' to specify which state

    Returns:
    - The exact Berry phase value
    """
    a = parameters.get('a', 1.0)
    alpha = a + 1

    # Lower state Berry phase: pi * (-alpha + sqrt(4 + alpha^2)) / sqrt(4 + alpha^2)
    sqrt_term = np.sqrt(4 + alpha**2)
    phase_lower = np.pi * (-alpha + sqrt_term) / sqrt_term

    if state == 'lower':
        return phase_lower
    else:  # upper
        return 2 * np.pi - phase_lower

def _berry_curvature_original(x, y, z, a, e, s):
    """Berry curvature for elliptical intersection - zero like conical intersection."""
    return 0.0

# Berry curvature calculation - zero for elliptical intersection (like conical)
def berry_curvature(x, y, parameters):
    """
    Berry curvature for elliptical intersection.
    Returns 0 (same as conical intersection case).

    Parameters:
    - x, y: Coordinates in space
    - parameters: Dictionary containing simulation parameters

    Returns:
    - 0.0 (Berry curvature is zero for this intersection type)
    """
    return 0.0

# Non-Adiabatic Coupling calculation - optimized but exact
def nonadiabatic_coupling(x, y, state_i, state_j, parameters, delta=1e-4):
    """
    Compute first-order non-adiabatic coupling d_ij^x and d_ij^y for states i and j.
    Optimized implementation with no caching.
    """
    # Get eigenvectors at current position 
    psi_lower, psi_upper = eigenvectors(x, y, parameters)
    
    # Select states
    if state_i == 'lower':
        psi_i = psi_lower
    else:  # state_i == 'upper'
        psi_i = psi_upper
    
    if state_j == 'lower':
        psi_j = psi_lower
    else:  # state_j == 'upper'
        psi_j = psi_upper
    
    # Get eigenvectors at shifted positions - calculate once per direction
    psi_lower_dx, psi_upper_dx = eigenvectors(x + delta, y, parameters)
    psi_lower_dy, psi_upper_dy = eigenvectors(x, y + delta, parameters)
    
    # Select j states at shifted positions
    if state_j == 'lower':
        psi_j_dx = psi_lower_dx
        psi_j_dy = psi_lower_dy
    else:  # state_j == 'upper'
        psi_j_dx = psi_upper_dx
        psi_j_dy = psi_upper_dy
    
    # Calculate NAC components with finite differences
    d_ij_x = np.real(np.vdot(psi_i, psi_j_dx) - np.vdot(psi_i, psi_j)) / delta
    d_ij_y = np.real(np.vdot(psi_i, psi_j_dy) - np.vdot(psi_i, psi_j)) / delta
    
    return d_ij_x, d_ij_y

def berry_connection(x, y, parameters, delta=1e-4):
    """Calculate Berry connection A = -i⟨ψ|∇ψ⟩."""
    try:
        psi_lower, _ = eigenvectors(x, y, parameters)
        
        # Calculate wavefunctions at neighboring points
        psi_dx, _ = eigenvectors(x + delta, y, parameters)
        psi_dy, _ = eigenvectors(x, y + delta, parameters)
        
        # Ensure phase consistency
        if np.real(np.vdot(psi_lower, psi_dx)) < 0:
            psi_dx = -psi_dx
        if np.real(np.vdot(psi_lower, psi_dy)) < 0:
            psi_dy = -psi_dy
        
        # Calculate derivatives
        dpsi_dx = (psi_dx - psi_lower) / delta
        dpsi_dy = (psi_dy - psi_lower) / delta
        
        # Berry connection components
        Ax = np.real(-1j * np.vdot(psi_lower, dpsi_dx))
        Ay = np.real(-1j * np.vdot(psi_lower, dpsi_dy))
        
        return Ax, Ay
    except Exception as e:
        print(f"Warning: Berry connection calculation failed: {e}")
        return 0.0, 0.0  # Return zeros as fallback

def analytical_berry_connection(x, y, z, parameters):
    """
    Calculate Berry connection analytically for both eigenstates.
    
    Parameters:
    -----------
    x, y, z : float
        Coordinates in parameter space
    parameters : dict
        Dictionary containing 'a', 'e', and 's' values
        
    Returns:
    --------
    tuple
        (Ax_lower, Ay_lower, Ax_upper, Ay_upper)
    """
    # Extract parameters
    a = parameters['a']
    e = parameters['e']
    s = parameters['s']
    
    # Common terms - precompute to avoid repetition
    x2 = x**2
    y2 = y**2
    xy_term = x2 + (1-e**2)*y2
    sqrt_term = np.sqrt(x2 + (1-e**2)*y2 + 0.25*(1+a)**2*z**2)
    
    # Use proper complex number notation (1j instead of (0.,1.))
    complex_y = x + 1j*np.sqrt(1-e**2)*y
    complex_y_conj = x - 1j*np.sqrt(1-e**2)*y
    
    # Berry connection components for LOWER eigenstate
    # The formula is kept exactly the same, just fixing the complex number notation
    Ax_lower = -1j*((-0.5*((-2.*x*(0.5*z + 0.5*a*z - 1.*sqrt_term))/((1.*xy_term)*sqrt_term) - (2.*x*(0.5*z + 0.5*a*z - 1.*sqrt_term)**2)/(1.*xy_term)**2))/(1. + (1.*(0.5*z + 0.5*a*z - 1.*sqrt_term)**2)/(1.*xy_term))**2 - (1.*(-0.5*z - 0.5*a*z + 1.*sqrt_term)*((0.5*(-0.5*z - 0.5*a*z + 1.*sqrt_term)*((-2.*x*(0.5*z + 0.5*a*z - 1.*sqrt_term))/((1.*xy_term)*sqrt_term) - (2.*x*(0.5*z + 0.5*a*z - 1.*sqrt_term)**2)/(1.*xy_term)**2))/((1.*x + 1j*np.sqrt(1. - 1.*e**2)*y)*(1. + (1.*(0.5*z + 0.5*a*z - 1.*sqrt_term)**2)/(1.*xy_term))**1.5) - (1.*x)/((1.*x + 1j*np.sqrt(1. - 1.*e**2)*y)*sqrt_term*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*sqrt_term)**2)/(1.*xy_term))) + (1.*(-0.5*z - 0.5*a*z + 1.*sqrt_term))/((1.*x + 1j*np.sqrt(1. - 1.*e**2)*y)**2*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*sqrt_term)**2)/(1.*xy_term)))))/((1.*x - 1j*np.sqrt(1. - 1.*e**2)*y)*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*sqrt_term)**2)/(1.*xy_term))))

    Ay_lower = -1j*((-0.5*((-2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*sqrt_term))/((1.*xy_term)*sqrt_term) - (2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*sqrt_term)**2)/(1.*xy_term)**2))/(1. + (1.*(0.5*z + 0.5*a*z - 1.*sqrt_term)**2)/(1.*xy_term))**2 - (1.*(-0.5*z - 0.5*a*z + 1.*sqrt_term)*((0.5*(-0.5*z - 0.5*a*z + 1.*sqrt_term)*((-2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*sqrt_term))/((1.*xy_term)*sqrt_term) - (2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*sqrt_term)**2)/(1.*xy_term)**2))/((1.*x + 1j*np.sqrt(1. - 1.*e**2)*y)*(1. + (1.*(0.5*z + 0.5*a*z - 1.*sqrt_term)**2)/(1.*xy_term))**1.5) - (1.*(1. - 1.*e**2)*y)/((1.*x + 1j*np.sqrt(1. - 1.*e**2)*y)*sqrt_term*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*sqrt_term)**2)/(1.*xy_term))) + (1j*np.sqrt(1. - 1.*e**2)*(-0.5*z - 0.5*a*z + 1.*sqrt_term))/((1.*x + 1j*np.sqrt(1. - 1.*e**2)*y)**2*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*sqrt_term)**2)/(1.*xy_term)))))/((1.*x - 1j*np.sqrt(1. - 1.*e**2)*y)*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*sqrt_term)**2)/(1.*xy_term))))

    # Berry connection components for UPPER eigenstate
    Ax_upper = -1j*((-0.5*((2.*x*(0.5*z + 0.5*a*z + 1.*sqrt_term))/((1.*xy_term)*sqrt_term) - (2.*x*(0.5*z + 0.5*a*z + 1.*sqrt_term)**2)/(1.*xy_term)**2))/(1. + (1.*(0.5*z + 0.5*a*z + 1.*sqrt_term)**2)/(1.*xy_term))**2 + (1.*(0.5*z + 0.5*a*z + 1.*sqrt_term)*((-0.5*(0.5*z + 0.5*a*z + 1.*sqrt_term)*((2.*x*(0.5*z + 0.5*a*z + 1.*sqrt_term))/((1.*xy_term)*sqrt_term) - (2.*x*(0.5*z + 0.5*a*z + 1.*sqrt_term)**2)/(1.*xy_term)**2))/((1.*x + 1j*np.sqrt(1. - 1.*e**2)*y)*(1. + (1.*(0.5*z + 0.5*a*z + 1.*sqrt_term)**2)/(1.*xy_term))**1.5) + (1.*x)/((1.*x + 1j*np.sqrt(1. - 1.*e**2)*y)*sqrt_term*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z + 1.*sqrt_term)**2)/(1.*xy_term))) - (1.*(0.5*z + 0.5*a*z + 1.*sqrt_term))/((1.*x + 1j*np.sqrt(1. - 1.*e**2)*y)**2*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z + 1.*sqrt_term)**2)/(1.*xy_term)))))/((1.*x - 1j*np.sqrt(1. - 1.*e**2)*y)*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z + 1.*sqrt_term)**2)/(1.*xy_term))))

    Ay_upper = -1j*((-0.5*((2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z + 1.*sqrt_term))/((1.*xy_term)*sqrt_term) - (2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z + 1.*sqrt_term)**2)/(1.*xy_term)**2))/(1. + (1.*(0.5*z + 0.5*a*z + 1.*sqrt_term)**2)/(1.*xy_term))**2 + (1.*(0.5*z + 0.5*a*z + 1.*sqrt_term)*((-0.5*(0.5*z + 0.5*a*z + 1.*sqrt_term)*((2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z + 1.*sqrt_term))/((1.*xy_term)*sqrt_term) - (2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z + 1.*sqrt_term)**2)/(1.*xy_term)**2))/((1.*x + 1j*np.sqrt(1. - 1.*e**2)*y)*(1. + (1.*(0.5*z + 0.5*a*z + 1.*sqrt_term)**2)/(1.*xy_term))**1.5) + (1.*(1. - 1.*e**2)*y)/((1.*x + 1j*np.sqrt(1. - 1.*e**2)*y)*sqrt_term*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z + 1.*sqrt_term)**2)/(1.*xy_term))) - (1j*np.sqrt(1. - 1.*e**2)*(0.5*z + 0.5*a*z + 1.*sqrt_term))/((1.*x + 1j*np.sqrt(1. - 1.*e**2)*y)**2*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z + 1.*sqrt_term)**2)/(1.*xy_term)))))/((1.*x - 1j*np.sqrt(1. - 1.*e**2)*y)*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z + 1.*sqrt_term)**2)/(1.*xy_term))))

    return np.real(Ax_lower), np.real(Ay_lower), np.real(Ax_upper), np.real(Ay_upper)

# Second-order NAC calculation - optimized but mathematically identical
def second_order_nac(x, y, state_i, state_j, parameters, delta=1e-4):
    """Compute second-order non-adiabatic couplings d_ij^xx, d_ij^xy, d_ij^yy."""
    # Get eigenvectors at current position
    psi_lower, psi_upper = eigenvectors(x, y, parameters)
    
    # Select states
    if state_i == 'lower':
        psi_i = psi_lower
    else:  # state_i == 'upper'
        psi_i = psi_upper
    
    if state_j == 'lower':
        psi_j = psi_lower
    else:  # state_j == 'upper'
        psi_j = psi_upper
    
    # Compute wavefunctions at shifted positions - compute once per position
    psi_lower_xp, psi_upper_xp = eigenvectors(x + delta, y, parameters)
    psi_lower_xm, psi_upper_xm = eigenvectors(x - delta, y, parameters)
    psi_lower_yp, psi_upper_yp = eigenvectors(x, y + delta, parameters)
    psi_lower_ym, psi_upper_ym = eigenvectors(x, y - delta, parameters)
    psi_lower_pp, psi_upper_pp = eigenvectors(x + delta, y + delta, parameters)
    psi_lower_pm, psi_upper_pm = eigenvectors(x + delta, y - delta, parameters)
    psi_lower_mp, psi_upper_mp = eigenvectors(x - delta, y + delta, parameters)
    psi_lower_mm, psi_upper_mm = eigenvectors(x - delta, y - delta, parameters)
    
    # Select j states at shifted positions
    if state_j == 'lower':
        psi_j_xp = psi_lower_xp
        psi_j_xm = psi_lower_xm
        psi_j_yp = psi_lower_yp
        psi_j_ym = psi_lower_ym
        psi_j_pp = psi_lower_pp
        psi_j_pm = psi_lower_pm
        psi_j_mp = psi_lower_mp
        psi_j_mm = psi_lower_mm
    else:  # state_j == 'upper'
        psi_j_xp = psi_upper_xp
        psi_j_xm = psi_upper_xm
        psi_j_yp = psi_upper_yp
        psi_j_ym = psi_upper_ym
        psi_j_pp = psi_upper_pp
        psi_j_pm = psi_upper_pm
        psi_j_mp = psi_upper_mp
        psi_j_mm = psi_upper_mm
    
    # Second derivatives using finite differences
    d2_psi_j_dx2 = (psi_j_xp - 2 * psi_j + psi_j_xm) / (delta**2)
    d2_psi_j_dy2 = (psi_j_yp - 2 * psi_j + psi_j_ym) / (delta**2)
    d2_psi_j_dxdy = (psi_j_pp - psi_j_pm - psi_j_mp + psi_j_mm) / (4 * delta**2)
    
    # Compute second-order couplings
    d_ij_xx = np.real(np.vdot(psi_i, d2_psi_j_dx2))
    d_ij_xy = np.real(np.vdot(psi_i, d2_psi_j_dxdy))
    d_ij_yy = np.real(np.vdot(psi_i, d2_psi_j_dy2))
    
    return d_ij_xx, d_ij_xy, d_ij_yy

# Optimized Hamiltonian derivatives
def dh_dx(x, y, parameters, delta=1e-6):
    """Calculate x-derivative of Hamiltonian with optimized finite difference."""
    # Calculate Hamiltonians once for each position
    H_plus = hamiltonian(x + delta, y, parameters)
    H_minus = hamiltonian(x - delta, y, parameters)
    
    # Use central difference for better accuracy
    return (H_plus - H_minus) / (2 * delta)

def dh_dy(x, y, parameters, delta=1e-6):
    """Calculate y-derivative of Hamiltonian with optimized finite difference."""
    # Calculate Hamiltonians once for each position
    H_plus = hamiltonian(x, y + delta, parameters)
    H_minus = hamiltonian(x, y - delta, parameters)
    
    # Use central difference for better accuracy
    return (H_plus - H_minus) / (2 * delta)


def evolve_psi(psi, H_old, H_new, dt, x_old, y_old, x_new, y_new, vx, vy, parameters):
    """
    Evolve quantum state using improved RK4 integrator.
    
    Parameters:
    - psi: Current quantum state
    - H_old, H_new: Hamiltonians at old and new positions
    - dt: Time step
    - x_old, y_old, x_new, y_new: Position coordinates
    - vx, vy: Velocity components
    - parameters: Simulation parameters
    
    Returns:
    - Updated quantum state
    """
    try:
        # Calculate midpoint position and Hamiltonian
        x_mid = 0.5 * (x_old + x_new)
        y_mid = 0.5 * (y_old + y_new)
        H_mid = 0.5 * (H_old + H_new)
        
        # Check for NaN values in Hamiltonians
        if np.any(np.isnan(H_old)) or np.any(np.isnan(H_new)) or np.any(np.isnan(H_mid)):
            raise ValueError("NaN detected in Hamiltonian")
        
        # Fourth-order Runge-Kutta for TDSE
        k1 = tdse(psi, H_old, x_old, y_old, vx, vy, parameters)
        k2 = tdse(psi + 0.5 * dt * k1, H_mid, x_mid, y_mid, vx, vy, parameters)
        k3 = tdse(psi + 0.5 * dt * k2, H_mid, x_mid, y_mid, vx, vy, parameters)
        k4 = tdse(psi + dt * k3, H_new, x_new, y_new, vx, vy, parameters)
        
        # Check for NaN values in k coefficients
        if (np.any(np.isnan(k1)) or np.any(np.isnan(k2)) or 
            np.any(np.isnan(k3)) or np.any(np.isnan(k4))):
            raise ValueError("NaN detected in RK4 coefficients")
        
        # RK4 update
        psi_new = psi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Verify state is valid
        if not np.all(np.isfinite(psi_new)):
            raise ValueError("Non-finite values in quantum state")
        
        # Calculate norm
        norm = np.linalg.norm(psi_new)
        
        # Check norm is reasonable
        if norm < 1e-10:
            raise ValueError("State norm too small")
        
        # Return normalized state
        return psi_new / norm
        
    except Exception as e:
        if parameters.get('debug', False):
            print(f"Warning: State evolution failed: {e}")
        
        # If evolution fails, return normalized original state
        return psi / np.linalg.norm(psi)


def tdse(psi, H, x, y, vx, vy, parameters):
    """
    Time-dependent Schrödinger equation with improved NAC handling.
    
    Parameters:
    - psi: Quantum state
    - H: Hamiltonian matrix
    - x, y: Position coordinates
    - vx, vy: Velocity components
    - parameters: Simulation parameters
    
    Returns:
    - Time derivative of quantum state
    """
    # Complex unit
    iota = 1j
    
    # Standard TDSE term: -i*H*psi
    result = -iota * np.dot(H, psi)
    
    # Add NAC terms if enabled
    if parameters.get('nac', False):
        # Get adiabatic states
        psi_lower, psi_upper = eigenvectors(x, y, parameters)
        
        # Project state onto adiabatic basis
        c_lower = np.vdot(psi_lower, psi)
        c_upper = np.vdot(psi_upper, psi)
        
        # Only include NAC if populations are significant
        if abs(c_lower) > 1e-5 and abs(c_upper) > 1e-5:
            # Calculate NAC vectors
            d_lu_x, d_lu_y = nonadiabatic_coupling(x, y, 'lower', 'upper', parameters)
            
            # Velocity coupling term (dot product of velocity and NAC)
            coupling_term = vx * d_lu_x + vy * d_lu_y
            
            # Avoid numerical issues with phase factor
            if abs(c_upper) < 1e-10:
                phase_factor = 1.0
            else:
                # Calculate phase difference between amplitudes
                phase_factor = np.exp(iota * np.angle(c_lower/c_upper))
            
            # NAC contribution to TDSE
            nac_term = iota * coupling_term * (
                c_upper * psi_lower * phase_factor - 
                c_lower * psi_upper * np.conjugate(phase_factor)
            )
            
            # Add NAC term with safety check
            if np.all(np.isfinite(nac_term)):
                result += nac_term
    
    return result


def initialize_trajectory(parameters, is_prelooping, theta=0.0):
    """
    Initialize trajectories with physically motivated initial conditions.
    For regular trajectories: random kinetic energy between 5-10 units.
    For pre-looping trajectories: keep standard initialization.
    """
    # Pre-looping trajectories - NO CHANGES to this part
    if is_prelooping:
        r_min = parameters.get('r_min', 0.5)
        r_max = parameters.get('r_max', 5.0)
        r = np.random.uniform(r_min, r_max)
        
        if theta == 0.0:
            theta = np.random.uniform(0, 2 * np.pi)
        
        e = parameters.get('e', 0.0)
        x0 = r * np.cos(theta)
        y0 = r * np.sin(theta)
        if abs(e) > 1e-6:
            y0 /= np.sqrt(1 - e**2 + 1e-10)
            
        # Standard velocity initialization for pre-looping trajectories
        H0 = hamiltonian(x0, y0, parameters)
        eigvals, _ = eigh(H0)
        energy_gap = abs(eigvals[1] - eigvals[0])
        v_scale = np.sqrt(abs(energy_gap) / mass + 1e-6)
        
        # Berry curvature-guided velocity for pre-looping
        B = berry_curvature(x0, y0, parameters)
        B_sign = np.sign(B) if abs(B) > 1e-10 else 1
        velocity_magnitude = v_scale * 0.5
        
        vx0 = -B_sign * velocity_magnitude * np.sin(theta)
        vy0 = B_sign * velocity_magnitude * np.cos(theta)
        
        if abs(e) > 1e-6:
            vy0 *= np.sqrt(1 - e**2)
    
    # Regular trajectories - MODIFIED for random kinetic energy between 5-10
    else:
        # Position initialization in rectangular region (unchanged)
        xmin = parameters.get('xmin', 4.0)
        xmax = parameters.get('xmax', 7.0)
        ymin = parameters.get('ymin', 4.0)
        ymax = parameters.get('ymax', 7.0)
        x0 = np.random.uniform(xmin, xmax)
        y0 = np.random.uniform(ymin, ymax)
        
        # Get kinetic energy range from parameters
        min_kinetic = parameters.get('min_initial_kinetic', 5.0)
        max_kinetic = parameters.get('max_initial_kinetic', 10.0)
        
        # Generate random kinetic energy in the specified range
        target_kinetic = np.random.uniform(min_kinetic, max_kinetic)
        
        # Calculate velocity required for this kinetic energy: KE = 0.5*m*v²
        velocity_magnitude = np.sqrt(2 * target_kinetic / mass)
        
        # Random direction with our calculated speed
        angle = np.random.uniform(0, 2 * np.pi)
        vx0 = velocity_magnitude * np.cos(angle)
        vy0 = velocity_magnitude * np.sin(angle)
    
    return float(x0), float(y0), float(vx0), float(vy0)














def init_state(x, y, parameters, is_prelooping=False):
    """
    Initialize quantum state based on selected basis type.
    
    For pre-looping trajectories:
    - Apply the analytical Berry phase at t=0 only
    - No artificial dynamics are used to accumulate phase
    
    For regular trajectories:
    - Initialize in selected state with optional random phase
    
    Parameters:
    - x, y: Position coordinates
    - parameters: Simulation parameters
    - is_prelooping: Whether to apply the Berry phase at initialization
    
    Returns:
    - Normalized initial quantum state
    """
    # MODIFIED: Choose state_type based on trajectory type
    traj_type = 'prelooping' if is_prelooping else 'regular'
    state_type = parameters['state_types'][traj_type]
    
    # Select whether to use upper state based on trajectory type
    use_upper_state = parameters['use_upper_state'][traj_type]
    
    if state_type == 'adiabatic':
        # Initialize in adiabatic (energy eigenstate) basis
        psi_lower, psi_upper = eigenvectors(x, y, parameters)
        
        # Select initial state based on parameters
        state = psi_upper.copy() if use_upper_state else psi_lower.copy()
        
        # For pre-looping: Apply the Berry phase at t=0
        if is_prelooping and parameters.get('geometric', True):
            which_state = 'upper' if use_upper_state else 'lower'
            phase_value = berry_phase(x, y, parameters, state=which_state)
            state *= np.exp(1j * phase_value)
        elif parameters.get('random_phase', True):
            # Optional random phase for regular trajectories
            state *= np.exp(1j * np.random.uniform(0, 2*np.pi))
            
    elif state_type == 'diabatic':
        # Initialize in diabatic (fixed) basis
        if use_upper_state:
            state = np.array([0, 1], dtype=complex)
        else:
            state = np.array([1, 0], dtype=complex)
            
        # For pre-looping: Still apply the Berry phase
        if is_prelooping and parameters.get('geometric', True):
            # Need adiabatic states to calculate proper Berry phase
            psi_lower, psi_upper = eigenvectors(x, y, parameters)
            which_state = 'upper' if use_upper_state else 'lower'
            phase_value = berry_phase(x, y, parameters, state=which_state)
            state *= np.exp(1j * phase_value)
        elif parameters.get('random_phase', True):
            state *= np.exp(1j * np.random.uniform(0, 2*np.pi))
            
    elif state_type == 'gaussian':
        # Initialize as a Gaussian wavepacket
        psi_lower, psi_upper = eigenvectors(x, y, parameters)
        
        # Width parameter from settings
        width = parameters.get('gaussian_width', 0.5)
        
        # Gaussian envelope
        r = np.sqrt(x**2 + y**2)
        envelope = np.exp(-r**2 / (2 * width**2))
        
        # Select state based on parameters
        if use_upper_state:
            state = envelope * psi_upper
        else:
            state = envelope * psi_lower
            
        # Apply phase as needed
        if is_prelooping and parameters.get('geometric', True):
            which_state = 'upper' if use_upper_state else 'lower'
            phase_value = berry_phase(x, y, parameters, state=which_state)
            state *= np.exp(1j * phase_value)
        elif parameters.get('random_phase', True):
            state *= np.exp(1j * np.random.uniform(0, 2*np.pi))
    
    # Ensure proper normalization
    return state / np.linalg.norm(state)








def force(psi, x, y, vx, vy, parameters, traj_type):
    """
    Compute total force based on quantum-classical dynamics, with selectable forces
    based on the trajectory type.
    
    Parameters:
    - psi: Current quantum state.
    - x, y: Position coordinates.
    - vx, vy: Velocity components.
    - parameters: Dictionary containing simulation parameters.
    - traj_type: 'regular' or 'prelooping'.
    
    Returns:
    - fx_total, fy_total: Total physical force components.
    - force_components: Dictionary containing individual force components.
    """
    # MODIFIED: Get the correct set of enabled forces for the current trajectory type
    enabled_forces = parameters['forces_enabled'][traj_type]

    force_components = {
        'ehrenfest': [0.0, 0.0], 'berry': [0.0, 0.0], 'nac': [0.0, 0.0],
        'nac2': [0.0, 0.0], 'geometric': [0.0, 0.0]
    }
    fx_total, fy_total = 0.0, 0.0

    # These are needed for almost all force calculations
    psi_lower, psi_upper = eigenvectors(x, y, parameters)
    c_lower = np.vdot(psi_lower, psi)
    c_upper = np.vdot(psi_upper, psi)
    pop_lower = np.abs(c_lower)**2
    pop_upper = np.abs(c_upper)**2

    # Ehrenfest force
    if enabled_forces.get('ehrenfest', False):
        dh_x = dh_dx(x, y, parameters)
        dh_y = dh_dy(x, y, parameters)
        fx_ehr = -np.real(np.vdot(psi, np.dot(dh_x, psi)))
        fy_ehr = -np.real(np.vdot(psi, np.dot(dh_y, psi)))
        force_components['ehrenfest'] = [fx_ehr, fy_ehr]
        fx_total += fx_ehr
        fy_total += fy_ehr
    
    # Berry curvature force
    if enabled_forces.get('berry', False):
        pop_diff = pop_lower - pop_upper
        B = berry_curvature(x, y, parameters)
        fx_berry = B * pop_diff * vy
        fy_berry = -B * pop_diff * vx
        force_components['berry'] = [fx_berry, fy_berry]
        fx_total += fx_berry
        fy_total += fy_berry
    
    # Non-adiabatic coupling forces
    if enabled_forces.get('nac', False) or enabled_forces.get('nac2', False):
        if pop_lower > 1e-10 and pop_upper > 1e-10:
            coherence = np.real(np.conj(c_lower) * c_upper)
            d_lu_x, d_lu_y = nonadiabatic_coupling(x, y, 'lower', 'upper', parameters)
            
            if enabled_forces.get('nac', False):
                fx_nac = -2 * coherence * d_lu_x
                fy_nac = -2 * coherence * d_lu_y
                force_components['nac'] = [fx_nac, fy_nac]
                fx_total += fx_nac
                fy_total += fy_nac
            
            if enabled_forces.get('nac2', False):
                d_lu_xx, d_lu_xy, d_lu_yy = second_order_nac(x, y, 'lower', 'upper', parameters)
                fx_nac2 = -coherence * d_lu_xx
                fy_nac2 = -coherence * d_lu_yy
                force_components['nac2'] = [fx_nac2, fy_nac2]
                fx_total += fx_nac2
                fy_total += fy_nac2
    
    # Geometric force from Berry connection
    if enabled_forces.get('geometric', False):
        Ax, Ay = berry_connection(x, y, parameters)
        delta = 1e-5
        Ax1, _ = berry_connection(x + delta, y, parameters)
        Ax2, _ = berry_connection(x - delta, y, parameters)
        _, Ay1 = berry_connection(x, y + delta, parameters)
        _, Ay2 = berry_connection(x, y - delta, parameters)
        dAx_dx = (Ax1 - Ax2) / (2 * delta)
        dAy_dy = (Ay1 - Ay2) / (2 * delta)
        fx_geometric = -(pop_lower * dAx_dx + pop_upper * dAx_dx)
        fy_geometric = -(pop_lower * dAy_dy + pop_upper * dAy_dy)
        force_components['geometric'] = [fx_geometric, fy_geometric]
        fx_total += fx_geometric
        fy_total += fy_geometric
    
    return fx_total, fy_total, force_components


def evolve_classical(x, y, vx, vy, psi, dt, H, parameters, traj_type):
    """
    Evolve classical variables with Velocity Verlet, using forces for the correct traj_type.
    """
    energy_old = np.real(np.vdot(psi, np.dot(H, psi))) + 0.5 * mass * (vx**2 + vy**2)
    min_dt = dt * 1e-3
    current_dt = dt
    max_retries = 5
    
    for attempt in range(max_retries):
        # MODIFIED: Pass traj_type to the force function
        fx, fy, _ = force(psi, x, y, vx, vy, parameters, traj_type)
        
        vx_half = vx + 0.5 * fx / mass * current_dt
        vy_half = vy + 0.5 * fy / mass * current_dt
        x_new = x + vx_half * current_dt
        y_new = y + vy_half * current_dt
        
        H_new = hamiltonian(x_new, y_new, parameters)
        psi_new = evolve_psi(psi, H, H_new, current_dt, x, y, x_new, y_new, vx_half, vy_half, parameters)
        
        # MODIFIED: Pass traj_type to the force function again for the new position
        fx_new, fy_new, _ = force(psi_new, x_new, y_new, vx_half, vy_half, parameters, traj_type)
        
        vx_new = vx_half + 0.5 * fx_new / mass * current_dt
        vy_new = vy_half + 0.5 * fy_new / mass * current_dt
        
        energy_new = np.real(np.vdot(psi_new, np.dot(H_new, psi_new))) + 0.5 * mass * (vx_new**2 + vy_new**2)
        energy_scale = max(abs(energy_old), abs(energy_new), 1e-6)
        rel_energy_error = abs(energy_new - energy_old) / energy_scale
        threshold = parameters.get('energy_threshold', 1e-4)
        
        if rel_energy_error < threshold or current_dt <= min_dt:
            return x_new, y_new, vx_new, vy_new, H_new, psi_new
        
        current_dt *= 0.5
        if current_dt < min_dt:
            current_dt = min_dt

    # Fallback if all retries fail
    fx, fy, _ = force(psi, x, y, vx, vy, parameters, traj_type)
    vx_half = vx + 0.5 * fx / mass * current_dt
    vy_half = vy + 0.5 * fy / mass * current_dt
    x_new = x + vx_half * current_dt
    y_new = y + vy_half * current_dt
    H_new = hamiltonian(x_new, y_new, parameters)
    psi_new = evolve_psi(psi, H, H_new, current_dt, x, y, x_new, y_new, vx_half, vy_half, parameters)
    fx_new, fy_new, _ = force(psi_new, x_new, y_new, vx_half, vy_half, parameters, traj_type)
    vx_new = vx_half + 0.5 * fx_new / mass * current_dt
    vy_new = vy_half + 0.5 * fy_new / mass * current_dt
    
    if parameters.get('debug', False):
        print(f"WARNING: Energy conservation failed. Using minimum dt={current_dt:.2e} after all retries.")
    
    return x_new, y_new, vx_new, vy_new, H_new, psi_new


def simulate_trajectory(traj_index, parameters):
    """
    Simulate a single quantum-classical trajectory.
    """
    num_pre = parameters.get('num_prelooping', 0)
    is_prelooping = (traj_index < num_pre)
    traj_type = 'prelooping' if is_prelooping else 'regular'
    
    ns = parameters['ns']
    dt = parameters['dt']
    
    # Initialize data storage dictionary for the trajectory
    traj = {
        'type': traj_type,
        'positions': np.zeros((ns, 2)), 'momenta': np.zeros((ns, 2)),
        'velocities': np.zeros((ns, 2)), 'psi_t': np.zeros((ns, 2), dtype=complex),
        'energies': np.zeros(ns), 'populations': np.zeros((ns, 2)),
        'adiabatic_pops': np.zeros((ns, 2)), 'berry_curvatures': np.zeros(ns),
        'forces': np.zeros((ns, 2)), 'forces_ehrenfest': np.zeros((ns, 2)),
        'forces_berry': np.zeros((ns, 2)), 'forces_nac': np.zeros((ns, 2)),
        'forces_nac2': np.zeros((ns, 2)), 'forces_geometric': np.zeros((ns, 2)),
        'warnings': []
    }
    
    # Initialize classical and quantum variables
    x0, y0, vx0, vy0 = initialize_trajectory(parameters, is_prelooping)
    x, y, vx, vy = x0, y0, vx0, vy0
    traj['initial_position'] = [x0, y0]
    traj['initial_velocity'] = [vx0, vy0]
    traj['is_prelooping'] = is_prelooping
    
    H = hamiltonian(x, y, parameters)
    psi = init_state(x0, y0, parameters, is_prelooping)
    
    if is_prelooping:
        traj['lower_cone_berry_phase'] = berry_phase(x0, y0, parameters, state='lower')
        traj['upper_cone_berry_phase'] = berry_phase(x0, y0, parameters, state='upper')
    
    # Main simulation loop
    for step in range(ns):
        # Record data for the current step
        traj['positions'][step] = [x, y]
        traj['velocities'][step] = [vx, vy]
        traj['momenta'][step] = [mass * vx, mass * vy]
        traj['psi_t'][step] = psi
        traj['energies'][step] = np.real(np.vdot(psi, np.dot(H, psi)))
        
        psi_lower, psi_upper = eigenvectors(x, y, parameters)
        c_lower = np.vdot(psi_lower, psi)
        c_upper = np.vdot(psi_upper, psi)
        traj['populations'][step] = [np.abs(psi[0])**2, np.abs(psi[1])**2]
        traj['adiabatic_pops'][step] = [np.abs(c_lower)**2, np.abs(c_upper)**2]
        traj['berry_curvatures'][step] = berry_curvature(x, y, parameters)
        
        # MODIFIED: Pass traj_type to the force function
        fx, fy, force_components = force(psi, x, y, vx, vy, parameters, traj_type)
        traj['forces'][step] = [fx, fy]
        traj['forces_ehrenfest'][step] = force_components['ehrenfest']
        traj['forces_berry'][step] = force_components['berry']
        traj['forces_nac'][step] = force_components['nac']
        traj['forces_nac2'][step] = force_components['nac2']
        traj['forces_geometric'][step] = force_components['geometric']
        
        # Evolve to the next step
        if step < ns - 1:
            try:
                # MODIFIED: Pass traj_type to the evolution function
                x, y, vx, vy, H, psi = evolve_classical(x, y, vx, vy, psi, dt, H, parameters, traj_type)
            except Exception as e:
                traj['warnings'].append(f"Evolution error at step {step}: {e}")
                break
    
    return traj


def save_trajectories(results, parameters, output_dir, run_id):
    """
    Save all trajectory data to a single HDF5 file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(output_dir, f"quantum_dynamics_{run_id}_{timestamp}.h5")
    
    with h5py.File(filename, 'w') as f:
        param_group = f.create_group('parameters')
        for key, value in parameters.items():
            if isinstance(value, (list, dict, tuple)):
                param_group.attrs[key] = json.dumps(value)
            else:
                try:
                    param_group.attrs[key] = value
                except TypeError:
                    param_group.attrs[key] = str(value)
        
        for i, traj in enumerate(results):
            traj_group = f.create_group(f'trajectory_{i}')
            for key, value in traj.items():
                if key == 'psi_t':
                    traj_group.create_dataset('psi_t_real', data=np.real(value))
                    traj_group.create_dataset('psi_t_imag', data=np.imag(value))
                elif isinstance(value, (list, np.ndarray)):
                     if key == 'warnings':
                         if value:
                             traj_group.create_dataset(key, data=np.string_(value))
                     else:
                        traj_group.create_dataset(key, data=value)
                else:
                    traj_group.attrs[key] = value
    
    print(f"Trajectory data saved to {filename}")
    return filename


def load_trajectories(filename):
    """
    Load all trajectory data from an HDF5 file.
    """
    results = []
    parameters = {}
    
    with h5py.File(filename, 'r') as f:
        param_group = f['parameters']
        for key, value in param_group.attrs.items():
            try:
                if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                    parameters[key] = json.loads(value)
                else:
                    parameters[key] = value
            except (json.JSONDecodeError, TypeError):
                parameters[key] = value
        
        for name in f:
            if name.startswith('trajectory_'):
                traj_group = f[name]
                traj = {}
                # Load attributes
                for key, value in traj_group.attrs.items():
                    traj[key] = value
                # Load datasets
                for key, dset in traj_group.items():
                    if key == 'psi_t_real':
                        psi_real = dset[()]
                        psi_imag = traj_group['psi_t_imag'][()]
                        traj['psi_t'] = psi_real + 1j * psi_imag
                    elif key == 'psi_t_imag':
                        continue
                    elif key == 'warnings':
                        traj[key] = [w.decode('utf-8') for w in dset[()]]
                    else:
                        traj[key] = dset[()]
                results.append(traj)
    
    print(f"Loaded {len(results)} trajectories from {filename}")
    return results, parameters

def run_simulation_batch(parameters, output_dir='quantum_dynamics_data', run_id=None):
    """
    Run a batch of trajectories with given parameters and save results
    
    Parameters:
    - parameters: Dictionary of simulation parameters
    - output_dir: Directory to save output files
    - run_id: Unique identifier for this simulation run
    
    Returns:
    - filename: Path to saved data file
    """
    # Generate a unique run ID if not provided
    if run_id is None:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"Starting simulation batch {run_id} with {parameters['nt']} trajectories")
    print(f"Parameters: a={parameters['a']}, e={parameters['e']}, s={parameters['s']}")
    
    # Initialize results list
    results = []
    
    # Get number of available CPUs
    num_cpus = mp.cpu_count()
    use_cpus = min(num_cpus, 8)  # Limit to 8 CPUs maximum
    
    print(f"Using {use_cpus} CPUs out of {num_cpus} available")
    
    # Run simulations
    start_time = time.time()
    
    # Determine whether to use parallel processing based on trajectory count
    if parameters['nt'] >= 10 and use_cpus > 1:
        # Parallel processing for many trajectories
        with ProcessPoolExecutor(max_workers=use_cpus) as executor:
            traj_indices = list(range(parameters['nt']))
            traj_params = [parameters] * parameters['nt']
            
            results = list(executor.map(simulate_trajectory, traj_indices, traj_params))
    else:
        # Sequential processing for fewer trajectories
        for i in range(parameters['nt']):
            traj = simulate_trajectory(i, parameters)
            results.append(traj)
            if (i+1) % 10 == 0:
                print(f"Completed {i+1}/{parameters['nt']} trajectories")
    
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Save results to file
    filename = save_trajectories(results, parameters, output_dir, run_id)
    
    return filename

def generate_param_file(param_dict, output_dir='quantum_dynamics_data', filename=None):
    """
    Generate a parameter file that can be used by the batch runner
    
    Parameters:
    - param_dict: Dictionary containing simulation parameters
    - output_dir: Directory to save the parameter file
    - filename: Optional filename for the parameter file
    
    Returns:
    - param_file_path: Path to the generated parameter file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"params_{timestamp}.json"
    
    param_file_path = os.path.join(output_dir, filename)
    
    with open(param_file_path, 'w') as f:
        json.dump(param_dict, f, indent=2)
    
    print(f"Parameter file generated: {param_file_path}")
    return param_file_path


def parse_command_line():
    """Parse command line arguments for running in batch mode"""
    # Using ArgumentDefaultsHelpFormatter to show default values in help message
    parser = argparse.ArgumentParser(description='Quantum Dynamics Simulation for HPC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Main operation modes
    parser.add_argument('--run', action='store_true', help='Run simulation with specified parameters')
    parser.add_argument('--generate-params', action='store_true', help='Generate a parameter file and exit')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode to set parameters')
    
    # Parameter sources
    parser.add_argument('--param-file', type=str, help='Path to parameter JSON file to load')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='quantum_dynamics_data', help='Directory for output files')
    parser.add_argument('--run-id', type=str, help='Unique identifier for this run (optional)')
    
    # System parameters
    parser.add_argument('--a', type=float, default=1.0, help='Constant a for Hamiltonian')
    parser.add_argument('--e', type=float, default=0.8, help='Eccentricity parameter (0 <= e < 1)')
    parser.add_argument('--s', type=float, default=1.0, help='Scaling factor s')
    parser.add_argument('--z-choice', type=str, choices=['constant', 'function'], default='constant', help='Is z constant or a function of x,y')
    parser.add_argument('--z-val', type=float, default=2.0, help='Constant value for z (if z-choice is constant)')
    
    # Trajectory parameters
    parser.add_argument('--nt', type=int, default=100, help='Total number of trajectories')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step for integration')
    parser.add_argument('--Ttotal', type=float, default=100.0, help='Total simulation time')
    
    # Position initialization - FIXED DEFAULTS
    parser.add_argument('--xmin', type=float, default=4.0, help='Minimum x for regular trajectory initialization')
    parser.add_argument('--ymin', type=float, default=4.0, help='Minimum y for regular trajectory initialization')
    parser.add_argument('--xmax', type=float, default=7.0, help='Maximum x for regular trajectory initialization')
    parser.add_argument('--ymax', type=float, default=7.0, help='Maximum y for regular trajectory initialization')
    parser.add_argument('--r-min', type=float, default=0.5, help='Minimum radius for pre-looping trajectories')
    parser.add_argument('--r-max', type=float, default=5.0, help='Maximum radius for pre-looping trajectories')
    parser.add_argument('--num-prelooping', type=int, default=0, help='Number of pre-looping trajectories')
    
    # ADDED MISSING ENERGY PARAMETERS
    parser.add_argument('--min-initial-kinetic', type=float, default=5.0, help='Minimum initial kinetic energy for regular trajectories')
    parser.add_argument('--max-initial-kinetic', type=float, default=10.0, help='Maximum initial kinetic energy for regular trajectories')
    
    # ADDED MISSING GAUSSIAN PARAMETER
    parser.add_argument('--gaussian-width', type=float, default=0.5, help='Width parameter for Gaussian wavepacket initialization')
    
    # Separate state initialization parameters for regular and pre-looping trajectories
    state_types = ['adiabatic', 'diabatic', 'gaussian']
    parser.add_argument('--regular-state-type', type=str, choices=state_types, default='adiabatic', 
                        help='Basis for initial state of regular trajectories')
    parser.add_argument('--prelooping-state-type', type=str, choices=state_types, default='adiabatic', 
                        help='Basis for initial state of pre-looping trajectories')
    parser.add_argument('--regular-upper-state', action='store_true', 
                        help='Initialize regular trajectories in the upper state (default is lower)')
    parser.add_argument('--prelooping-upper-state', action='store_true', 
                        help='Initialize pre-looping trajectories in the upper state (default is lower)')

    # Force selection arguments
    all_forces = ['ehrenfest', 'berry', 'nac', 'nac2', 'geometric']
    parser.add_argument('--regular-forces', nargs='+', default=['ehrenfest'], choices=all_forces,
                        help='List of forces to enable for regular trajectories.')
    parser.add_argument('--prelooping-forces', nargs='+', default=['ehrenfest', 'berry', 'nac', 'geometric'], choices=all_forces,
                        help='List of forces to enable for pre-looping trajectories.')
    
    # ADDED MISSING CONFIGURATION PARAMETERS
    parser.add_argument('--energy-threshold', type=float, default=1e-4, help='Energy conservation threshold')
    parser.add_argument('--random-phase', action='store_true', default=True, help='Add random phase to initial states')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    return parser.parse_args()


def parameters_from_args(args):
    """Convert command line arguments to a parameters dictionary"""
    params = {
        'a': args.a,
        'e': args.e,
        's': args.s,
        'z_choice': args.z_choice,
        'z_val': args.z_val,
        'nt': args.nt,
        'dt': args.dt,
        'Ttotal': args.Ttotal,
        'ns': int(np.ceil(args.Ttotal / args.dt)),
        'r_min': args.r_min,
        'r_max': args.r_max,
        'xmin': args.xmin,
        'ymin': args.ymin,
        'xmax': args.xmax,
        'ymax': args.ymax,
        'num_prelooping': args.num_prelooping,
        
        # ADDED MISSING PARAMETERS
        'min_initial_kinetic': args.min_initial_kinetic,
        'max_initial_kinetic': args.max_initial_kinetic,
        'gaussian_width': args.gaussian_width,
        'energy_threshold': args.energy_threshold,
        'random_phase': args.random_phase,
        'debug': args.debug
    }

    # Create the nested dictionary for force settings
    all_forces = ['ehrenfest', 'berry', 'nac', 'nac2', 'geometric']
    params['forces_enabled'] = {
        'regular': {force: (force in args.regular_forces) for force in all_forces},
        'prelooping': {force: (force in args.prelooping_forces) for force in all_forces}
    }
    
    # Create dictionaries for state types and upper states
    params['state_types'] = {
        'regular': args.regular_state_type,
        'prelooping': args.prelooping_state_type
    }
    
    params['use_upper_state'] = {
        'regular': args.regular_upper_state,
        'prelooping': args.prelooping_upper_state
    }
    
    return params


def collect_interactive_parameters():
    """Collect all simulation parameters through interactive prompts"""
    print("\n--- Quantum Dynamics Simulation Interactive Setup ---")
    
    parameters = {}
    
    print("\n[System Parameters]")
    parameters['a'] = float(input("Enter constant 'a' for Hamiltonian (default: 1.0): ") or "1.0")
    parameters['e'] = float(input("Enter eccentricity 'e' (0 <= e < 1) (default: 0.8): ") or "0.8")
    parameters['s'] = float(input("Enter scaling factor 's' (default: 1.0): ") or "1.0")
    parameters['z_choice'] = input("Is z constant or a function? ('constant' or 'function', default: 'constant'): ").lower() or 'constant'
    if parameters['z_choice'] == 'constant':
        parameters['z_val'] = float(input("Enter constant value for z (default: 2.0): ") or "2.0")

    print("\n[Trajectory & Time Parameters]")
    parameters['nt'] = int(input("Enter number of trajectories (default: 100): ") or "100")
    parameters['dt'] = float(input("Enter time step (default: 0.01): ") or "0.01")
    parameters['Ttotal'] = float(input("Enter total time (default: 100.0): ") or "100.0")
    parameters['ns'] = int(np.ceil(parameters['Ttotal'] / parameters['dt']))

    print("\n[Regular Trajectory Initialization Region]")
    # FIXED DEFAULTS TO MATCH CODE EXPECTATIONS
    parameters['xmin'] = float(input("Enter x_min (default: 4.0): ") or "4.0")
    parameters['xmax'] = float(input("Enter x_max (default: 7.0): ") or "7.0")
    parameters['ymin'] = float(input("Enter y_min (default: 4.0): ") or "4.0")
    parameters['ymax'] = float(input("Enter y_max (default: 7.0): ") or "7.0")

    # ADDED MISSING ENERGY CONFIGURATION
    print("\n[Initial Energy Configuration]")
    parameters['min_initial_kinetic'] = float(input("Minimum initial kinetic energy for regular trajectories (default: 5.0): ") or "5.0")
    parameters['max_initial_kinetic'] = float(input("Maximum initial kinetic energy for regular trajectories (default: 10.0): ") or "10.0")

    print("\n[Pre-Looping Trajectory Parameters]")
    parameters['num_prelooping'] = int(input(f"Enter number of pre-looping trajectories (0 to {parameters['nt']}, default: 0): ") or "0")
    parameters['num_prelooping'] = min(max(parameters['num_prelooping'], 0), parameters['nt'])
    if parameters['num_prelooping'] > 0:
        parameters['r_min'] = float(input("Enter minimum radius (default: 0.5): ") or "0.5")
        parameters['r_max'] = float(input("Enter maximum radius (default: 5.0): ") or "5.0")

    # Initial state configuration for regular and pre-looping trajectories
    print("\n[Initial State Configuration]")
    state_types = ['adiabatic', 'diabatic', 'gaussian']
    
    print("Regular Trajectories:")
    reg_state_choice = input("  Choose initial state (1=lower, 2=upper, default: 1): ") or "1"
    reg_state_type_choice = input("  Select basis type (1=adiabatic, 2=diabatic, 3=gaussian, default: 1): ") or "1"
    
    if parameters['num_prelooping'] > 0:
        print("Pre-looping Trajectories:")
        pre_state_choice = input("  Choose initial state (1=lower, 2=upper, default: 1): ") or "1"
        pre_state_type_choice = input("  Select basis type (1=adiabatic, 2=diabatic, 3=gaussian, default: 1): ") or "1"
    else:
        pre_state_choice = "1"
        pre_state_type_choice = "1"
    
    # ADDED MISSING GAUSSIAN WIDTH PARAMETER
    parameters['gaussian_width'] = float(input("Gaussian wavepacket width (default: 0.5): ") or "0.5")
    
    # Create dictionaries for state types and upper states
    parameters['state_types'] = {
        'regular': state_types[int(reg_state_type_choice) - 1],
        'prelooping': state_types[int(pre_state_type_choice) - 1]
    }
    
    parameters['use_upper_state'] = {
        'regular': (reg_state_choice == "2"),
        'prelooping': (pre_state_choice == "2")
    }
    
    print(f"-> Regular trajectories: {parameters['state_types']['regular']} basis, {'upper' if parameters['use_upper_state']['regular'] else 'lower'} state")
    if parameters['num_prelooping'] > 0:
        print(f"-> Pre-looping trajectories: {parameters['state_types']['prelooping']} basis, {'upper' if parameters['use_upper_state']['prelooping'] else 'lower'} state")

    # Force selection
    print("\n[Force Selection]")
    all_forces = ['ehrenfest', 'berry', 'nac', 'nac2', 'geometric']
    print(f"Available forces: {', '.join(all_forces)}")
    
    reg_forces_input = input("Enter forces for REGULAR trajectories (space-separated, default: ehrenfest): ") or "ehrenfest"
    reg_forces_list = reg_forces_input.lower().split()
    
    pre_forces_input = input("Enter forces for PRE-LOOPING trajectories (space-separated, default: ehrenfest berry nac geometric): ") or "ehrenfest berry nac geometric"
    pre_forces_list = pre_forces_input.lower().split()
    
    parameters['forces_enabled'] = {
        'regular': {force: (force in reg_forces_list) for force in all_forces},
        'prelooping': {force: (force in pre_forces_list) for force in all_forces}
    }
    print(f"-> Regular forces set to: {parameters['forces_enabled']['regular']}")
    print(f"-> Pre-looping forces set to: {parameters['forces_enabled']['prelooping']}")

    print("\n[Advanced & Output Settings]")
    # MADE THESE CONFIGURABLE INSTEAD OF HARDCODED
    parameters['energy_threshold'] = float(input("Energy conservation threshold (default: 1e-4): ") or "1e-4")
    parameters['random_phase'] = input("Add random phase to initial states? (y/n, default: y): ").lower() != 'n'
    parameters['debug'] = input("Show detailed warnings? (y/n, default: n): ").lower() == 'y'
    
    output_dir = input("Output directory (default: quantum_dynamics_data): ") or "quantum_dynamics_data"
    run_id = input("Run ID (optional, default: auto-generated): ") or None
    
    if input("Save these parameters to a file? (y/n, default: y): ").lower() != 'n':
        generate_param_file(parameters, output_dir)
        
    return parameters, output_dir, run_id



def main():
    """Main entry point for the program"""
    args = parse_command_line()
    
    # Determine operation mode
    if args.interactive:
        # Interactive mode - collect parameters via prompts
        parameters, output_dir, run_id = collect_interactive_parameters()
        
    elif args.generate_params:
        # Generate parameter file only
        parameters = parameters_from_args(args)
        output_dir = args.output_dir
        generate_param_file(parameters, output_dir)
        print("Parameter file generated. Exiting.")
        return
        
    elif args.param_file:
        # Load parameters from file
        with open(args.param_file, 'r') as f:
            parameters = json.load(f)
        output_dir = args.output_dir
        run_id = args.run_id
        
    else:
        # Use command line arguments
        parameters = parameters_from_args(args)
        output_dir = args.output_dir
        run_id = args.run_id
    
    # Run the simulation
    if args.run or args.interactive or args.param_file:
        # Ensure ns is calculated
        if 'ns' not in parameters:
            parameters['ns'] = int(np.ceil(parameters['Ttotal'] / parameters['dt']))
        
        filename = run_simulation_batch(parameters, output_dir, run_id)
        print(f"Simulation complete. Results saved to: {filename}")
    else:
        print("No action specified. Use --run, --generate-params, or --interactive")
        print("For help, use --help")

if __name__ == "__main__":
    main()

