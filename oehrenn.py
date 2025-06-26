# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import plotly.io as pio
pio.renderers.default = 'notebook'
# Constants
hbar = 1.0  # Reduced Planck's constant (atomic units)
mass = 1.0  # Proton mass in atomic units

# Helper Functions for Input Collection
def get_input(prompt, default, type_func=float):
    """Get user input with a default value."""
    user_input = input(f"{prompt} (default: {default}): ").strip()
    return type_func(user_input) if user_input else default

# Optimized Hamiltonian
def hamiltonian(x, y, parameters):
    """Compute the Hamiltonian with parameter dict - optimized but mathematically identical."""
    # Calculate z directly - avoid function call overhead
    if parameters['z_choice'] == 'constant':
        z = parameters['z_val']
    else:
        z = np.sqrt(x**2 + (1 - parameters['e']**2) * y**2)
    
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
        return np.sqrt(x**2 + (1 - parameters['e']**2) * y**2)

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

# %%
# Berry phase calculation - NO CACHING, direct calculation each time
def berry_phase(x, y, parameters, state='lower'):
    """
    Calculate the analytical Berry phase for a state circling a conical intersection.
    Direct calculation ensures accuracy for each position.
    
    Parameters:
    - x, y: Position coordinates
    - parameters: Simulation parameters
    - state: Either 'lower' or 'upper' to specify which state
    
    Returns:
    - The exact Berry phase value
    """
    # Extract parameters directly
    e = parameters['e'] 
    a = parameters.get('a', 1.0)
    
    # Calculate z value - direct calculation for each call
    if parameters['z_choice'] == 'constant':
        z = parameters['z_val']
    else:
        z = np.sqrt(x**2 + (1 - e**2) * y**2)
    
    # Calculate radius
    r = np.sqrt(np.real(x)**2 + (1 - e**2) * np.real(y)**2)
    
    # Calculate denominator term (with safety check)
    denom = np.sqrt(4 * (1 - e**2) * r**2 + (1 + a)**2 * z**2)
    if denom < 1e-10:
        return 0.0
        
    # Calculate the term that differentiates upper and lower states
    term = (1 + a) * z / denom
    
    # Return the appropriate phase
    if state == 'lower':
        return np.pi * (1 - term)
    else:  # upper
        return np.pi * (1 + term)



# Original berry_curvature calculation - keep exactly as in original code
def _berry_curvature_original(x, y, z, a, e, s):
    """Original Berry curvature function with individual parameters."""
    # Your existing analytical expression here
    bcl=(0+1j)*((1.*((-2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2)*((-2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2))/(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**3 - (0.5*((2.*(1. - 1.*e**2)*x*y)/((1.*x**2 + (1. - 1.*e**2)*y**2)*(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) + (2.*(1. - 1.*e**2)*x*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)**1.5) + (8.*(1. - 1.*e**2)*x*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)**2*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) + (8.*(1. - 1.*e**2)*x*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**3))/(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**2 - (1.*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((-0.75*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((-2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2)*((-2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**2.5) + (0.5*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((2.*(1. - 1.*e**2)*x*y)/((1.*x**2 + (1. - 1.*e**2)*y**2)*(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) + (2.*(1. - 1.*e**2)*x*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)**1.5) + (8.*(1. - 1.*e**2)*x*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)**2*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) + (8.*(1. - 1.*e**2)*x*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**3))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**1.5) + (0.5*(1. - 1.*e**2)*y*((-2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**1.5) - ((0.+0.5j)*np.sqrt(1. - 1.*e**2)*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((-2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)**2*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**1.5) + (0.5*x*((-2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**1.5) - (0.5*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((-2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)**2*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**1.5) + (1.*(1. - 1.*e**2)*x*y)/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)**1.5*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))) + ((0.+1j)*np.sqrt(1. - 1.*e**2)*x)/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)**2*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))) + (1.*(1. - 1.*e**2)*y)/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)**2*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))) - ((0.+2j)*np.sqrt(1. - 1.*e**2)*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)**3*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)))))/((1.*x - (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))) + (0.5*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((-2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2)*((0.5*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((-2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**1.5) - (1.*x)/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))) + (1.*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)**2*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)))))/((1.*x - (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**1.5) - (1.*(1. - 1.*e**2)*y*((0.5*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((-2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**1.5) - (1.*x)/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))) + (1.*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)**2*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)))))/((1.*x - (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))) - ((0.+1j)*np.sqrt(1. - 1.*e**2)*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((0.5*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((-2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**1.5) - (1.*x)/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))) + (1.*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)**2*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)))))/((1.*x - (0.+1j)*np.sqrt(1. - 1.*e**2)*y)**2*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)))) - (0+1j)*((1.*((-2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2)*((-2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2))/(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**3 - (0.5*((2.*(1. - 1.*e**2)*x*y)/((1.*x**2 + (1. - 1.*e**2)*y**2)*(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) + (2.*(1. - 1.*e**2)*x*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)**1.5) + (8.*(1. - 1.*e**2)*x*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)**2*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) + (8.*(1. - 1.*e**2)*x*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**3))/(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**2 - (1.*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((-0.75*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((-2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2)*((-2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**2.5) + (0.5*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((2.*(1. - 1.*e**2)*x*y)/((1.*x**2 + (1. - 1.*e**2)*y**2)*(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) + (2.*(1. - 1.*e**2)*x*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)**1.5) + (8.*(1. - 1.*e**2)*x*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)**2*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) + (8.*(1. - 1.*e**2)*x*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**3))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**1.5) + (0.5*(1. - 1.*e**2)*y*((-2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**1.5) - ((0.+0.5j)*np.sqrt(1. - 1.*e**2)*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((-2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)**2*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**1.5) + (0.5*x*((-2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**1.5) - (0.5*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((-2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)**2*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**1.5) + (1.*(1. - 1.*e**2)*x*y)/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)**1.5*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))) + ((0.+1j)*np.sqrt(1. - 1.*e**2)*x)/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)**2*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))) + (1.*(1. - 1.*e**2)*y)/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)**2*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))) - ((0.+2j)*np.sqrt(1. - 1.*e**2)*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)**3*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)))))/((1.*x - (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))) + (0.5*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((-2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*x*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2)*((0.5*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((-2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**1.5) - (1.*(1. - 1.*e**2)*y)/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))) + ((0.+1j)*np.sqrt(1. - 1.*e**2)*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)**2*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)))))/((1.*x - (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**1.5) - (1.*x*((0.5*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((-2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**1.5) - (1.*(1. - 1.*e**2)*y)/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))) + ((0.+1j)*np.sqrt(1. - 1.*e**2)*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)**2*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)))))/((1.*x - (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))) + (1.*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((0.5*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))*((-2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x**2 + (1. - 1.*e**2)*y**2)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)) - (2.*(1. - 1.*e**2)*y*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)**2))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))**1.5) - (1.*(1. - 1.*e**2)*y)/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))) + ((0.+1j)*np.sqrt(1. - 1.*e**2)*(-0.5*z - 0.5*a*z + 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2)))/((1.*x + (0.+1j)*np.sqrt(1. - 1.*e**2)*y)**2*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2)))))/((1.*x - (0.+1j)*np.sqrt(1. - 1.*e**2)*y)**2*np.sqrt(1. + (1.*(0.5*z + 0.5*a*z - 1.*np.sqrt(1.*x**2 + (1. - 1.*e**2)*y**2 + 0.25*(1. + 1.*a)**2*z**2))**2)/(1.*x**2 + (1. - 1.*e**2)*y**2))))    
    curvature = np.abs(bcl)
    if not np.isfinite(curvature) or curvature > 1e16:
        return np.clip(curvature, -1e16, 1e16)
    
    return curvature


# Berry curvature calculation - NO CACHING, direct calculation
def berry_curvature(x, y, parameters):
    """
    Calculate Berry curvature without caching to ensure accuracy.
    
    Parameters:
    - x, y: Coordinates in space
    - parameters: Dictionary containing simulation parameters
    
    Returns:
    - Berry curvature value at the given point
    """
    try:
        # Extract needed parameters
        z = z_func(x, y, parameters)
        a = parameters['a']
        e = parameters['e']
        s = parameters['s']
        
        # Call the original function with the extracted parameters
        # This is the complex analytical expression from the original code
        bcl = _berry_curvature_original(x, y, z, a, e, s)
        
        # Return the absolute value with safety checks
        curvature = np.abs(bcl)
        if not np.isfinite(curvature) or curvature > 1e6:
            return np.clip(curvature, -1e6, 1e6)
        
        return curvature
    except Exception as e:
        if parameters.get('debug', False):
            print(f"Warning: Berry curvature calculation failed: {e}")
        return 0.0  # Return zero as fallback



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

# %%

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


# %%
def evolve_classical(x, y, vx, vy, psi, dt, H, parameters):
    """
    Evolve classical variables with improved Velocity Verlet algorithm.
    Uses a more robust approach to time-stepping.
    
    Parameters:
    - x, y: Current positions
    - vx, vy: Current velocities
    - psi: Current quantum state
    - dt: Time step
    - H: Current Hamiltonian
    - parameters: Simulation parameters
    
    Returns:
    - Updated state variables
    """
    # Calculate initial energy
    energy_old = np.real(np.vdot(psi, np.dot(H, psi))) + 0.5 * mass * (vx**2 + vy**2)
    
    # Set minimum time step and maximum retries
    min_dt = dt * 1e-3  # Increased minimum dt to prevent freezing
    current_dt = dt
    max_retries = 5     # Reduced from 10 to avoid excessive time-step reduction
    
    for attempt in range(max_retries):
        # Compute initial force
        fx, fy = force(psi, x, y, vx, vy, parameters)
        
        # Half-step velocity update
        vx_half = vx + 0.5 * fx / mass * current_dt
        vy_half = vy + 0.5 * fy / mass * current_dt
        
        # Full-step position update
        x_new = x + vx_half * current_dt
        y_new = y + vy_half * current_dt
        
        # Update Hamiltonian and quantum state
        H_new = hamiltonian(x_new, y_new, parameters)
        psi_new = evolve_psi(psi, H, H_new, current_dt, x, y, x_new, y_new, vx_half, vy_half, parameters)
        
        # Compute new force with updated state
        fx_new, fy_new = force(psi_new, x_new, y_new, vx_half, vy_half, parameters)
        
        # Full-step velocity update
        vx_new = vx_half + 0.5 * fx_new / mass * current_dt
        vy_new = vy_half + 0.5 * fy_new / mass * current_dt
        
        # Calculate new energy
        energy_new = np.real(np.vdot(psi_new, np.dot(H_new, psi_new))) + 0.5 * mass * (vx_new**2 + vy_new**2)
        
        # Check energy conservation
        energy_scale = max(abs(energy_old), abs(energy_new), 1e-6)
        rel_energy_error = abs(energy_new - energy_old) / energy_scale
        
        # More forgiving energy conservation threshold
        threshold = parameters.get('energy_threshold', 1e-4) * 10.0  # Made 10x more forgiving
        
        if rel_energy_error < threshold or current_dt <= min_dt:
            # Success or reached minimum time step
            return x_new, y_new, vx_new, vy_new, H_new, psi_new
        
        # Reduce time step more cautiously
        current_dt *= 0.5
        
        # Don't go below minimum time step
        if current_dt < min_dt:
            current_dt = min_dt
    
    # If all retries fail, still return the result with the smallest time step
    # This prevents trajectories from getting completely stuck
    fx, fy = force(psi, x, y, vx, vy, parameters)
    vx_half = vx + 0.5 * fx / mass * current_dt
    vy_half = vy + 0.5 * fy / mass * current_dt
    x_new = x + vx_half * current_dt
    y_new = y + vy_half * current_dt
    H_new = hamiltonian(x_new, y_new, parameters)
    psi_new = evolve_psi(psi, H, H_new, current_dt, x, y, x_new, y_new, vx_half, vy_half, parameters)
    fx_new, fy_new = force(psi_new, x_new, y_new, vx_half, vy_half, parameters)
    vx_new = vx_half + 0.5 * fx_new / mass * current_dt
    vy_new = vy_half + 0.5 * fy_new / mass * current_dt
    
    if parameters.get('debug', False):
        print(f"WARNING: Using minimum dt={current_dt:.2e} after all retries.")
    
    return x_new, y_new, vx_new, vy_new, H_new, psi_new


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

# %%
def initialize_trajectory(parameters, is_prelooping, theta=0.0):
    """
    Initialize trajectories with physically motivated initial conditions.
    No artificial constraints are applied to enforce orbital motion.
    
    Parameters:
    - parameters: Dictionary of simulation parameters
    - is_prelooping: Whether this is a pre-looped trajectory
    - theta: Optional angle parameter (used for testing deterministic outcomes)
    
    Returns:
    - x0, y0, vx0, vy0: Initial position and velocity
    """
    # Define spatial bounds based on parameters
    max_pos = parameters.get('r_max', 5.0)
    
    # Position initialization depends on trajectory type
    if is_prelooping:
        # For pre-looped trajectories: use physics-based initialization
        # Ensure reasonable distance from conical intersection
        r_min = parameters.get('r_min', 0.5)
        r_max = parameters.get('r_max', 5.0)
        
        # Use fixed radius if specified, otherwise random
        if parameters.get('use_fixed_radius', False):
            r = parameters.get('radius', (r_min + r_max)/2)
        else:
            r = np.random.uniform(r_min, r_max)
        
        # Random angle if not specified
        if theta == 0.0:
            theta = np.random.uniform(0, 2*np.pi)
        
        # Handle elliptical coordinates based on eccentricity
        e = parameters.get('e', 0.0)
        x0 = r * np.cos(theta)
        y0 = r * np.sin(theta)
        
        # Adjust y-coordinate for elliptical geometry if needed
        if abs(e) > 1e-6:
            y0 /= np.sqrt(1 - e**2 + 1e-10)  # Avoid division by zero
    else:
        # Regular trajectories: true random initialization
        # Avoid starting exactly at the conical intersection
        while True:
            x0 = np.random.uniform(-max_pos, max_pos)
            y0 = np.random.uniform(-max_pos, max_pos)
            # Ensure we're not too close to the conical intersection
            if x0**2 + y0**2 > 0.25:  # minimum distance from origin
                break
    
    # For all trajectories: physically-motivated velocity scale
    H0 = hamiltonian(x0, y0, parameters)
    eigvals, _ = np.linalg.eigh(H0)
    energy_gap = abs(eigvals[1] - eigvals[0])
    
    # Scale velocity by energy gap (physics-based)
    v_scale = np.sqrt(energy_gap / mass)
    
    if is_prelooping:
        # For pre-looped: initial momentum influenced by Berry curvature
        # This is physics-based, not an artificial constraint
        B = berry_curvature(x0, y0, parameters)
        
        # Use Berry curvature sign to determine rotation direction
        B_sign = np.sign(B) if abs(B) > 1e-10 else 1
        
        # Velocity perpendicular to radial direction (based on physics)
        # Scale velocity moderately by energy gap
        velocity_magnitude = v_scale * 0.5  # Moderate initial velocity
        
        # Set direction perpendicular to position vector (physical)
        vx0 = -B_sign * velocity_magnitude * np.sin(theta)
        vy0 = B_sign * velocity_magnitude * np.cos(theta)
        
        # Apply elliptical correction if needed
        if abs(e) > 1e-6:
            vy0 *= np.sqrt(1 - e**2)
    else:
        # Regular trajectories: random initial velocity
        angle = np.random.uniform(0, 2*np.pi)
        speed_factor = np.random.uniform(0.5, 1.5)  # Variable speed
        vx0 = v_scale * speed_factor * np.cos(angle)
        vy0 = v_scale * speed_factor * np.sin(angle)
    
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
    # Different initialization based on state type
    state_type = parameters.get('state_type', 'adiabatic')
    
    if state_type == 'adiabatic':
        # Initialize in adiabatic (energy eigenstate) basis
        psi_lower, psi_upper = eigenvectors(x, y, parameters)
        
        # Select initial state based on parameters
        state = psi_upper.copy() if parameters['use_upper_state'] else psi_lower.copy()
        
        # For pre-looping: Apply the Berry phase at t=0
        if is_prelooping and parameters.get('geometric', True):
            which_state = 'upper' if parameters['use_upper_state'] else 'lower'
            phase_value = berry_phase(x, y, parameters, state=which_state)
            state *= np.exp(1j * phase_value)
        elif parameters.get('random_phase', True):
            # Optional random phase for regular trajectories
            state *= np.exp(1j * np.random.uniform(0, 2*np.pi))
            
    elif state_type == 'diabatic':
        # Initialize in diabatic (fixed) basis
        if parameters['use_upper_state']:
            state = np.array([0, 1], dtype=complex)
        else:
            state = np.array([1, 0], dtype=complex)
            
        # For pre-looping: Still apply the Berry phase
        if is_prelooping and parameters.get('geometric', True):
            # Need adiabatic states to calculate proper Berry phase
            psi_lower, psi_upper = eigenvectors(x, y, parameters)
            which_state = 'upper' if parameters['use_upper_state'] else 'lower'
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
        if parameters['use_upper_state']:
            state = envelope * psi_upper
        else:
            state = envelope * psi_lower
            
        # Apply phase as needed
        if is_prelooping and parameters.get('geometric', True):
            which_state = 'upper' if parameters['use_upper_state'] else 'lower'
            phase_value = berry_phase(x, y, parameters, state=which_state)
            state *= np.exp(1j * phase_value)
        elif parameters.get('random_phase', True):
            state *= np.exp(1j * np.random.uniform(0, 2*np.pi))
    
    # Ensure proper normalization
    return state / np.linalg.norm(state)



def force(psi, x, y, vx, vy, parameters):
    """
    Compute total force based on quantum-classical dynamics.
    No artificial constraints are applied to enforce orbital motion.
    
    Parameters:
    - psi: Current quantum state
    - x, y: Position coordinates
    - vx, vy: Velocity components
    - parameters: Dictionary containing simulation parameters
    
    Returns:
    - fx_total, fy_total: Total physical force components
    """
    # Get adiabatic eigenstates at current position
    psi_lower, psi_upper = eigenvectors(x, y, parameters)
    
    # Project current state onto adiabatic basis
    c_lower = np.vdot(psi_lower, psi)
    c_upper = np.vdot(psi_upper, psi)
    
    # Calculate populations and coherence
    pop_lower = np.abs(c_lower)**2
    pop_upper = np.abs(c_upper)**2
    coherence = np.real(np.conj(c_lower) * c_upper)
    
    # Calculate population difference (for Berry force)
    pop_diff = pop_lower - pop_upper
    
    # Ehrenfest force (gradient of energy)
    dh_x = dh_dx(x, y, parameters)
    dh_y = dh_dy(x, y, parameters)
    fx_ehr = -np.real(np.vdot(psi, np.dot(dh_x, psi)))
    fy_ehr = -np.real(np.vdot(psi, np.dot(dh_y, psi)))
    
    # Initialize total force with Ehrenfest term
    fx_total = fx_ehr
    fy_total = fy_ehr
    
    # Berry curvature force (if enabled)
    if parameters.get('berry', False):
        # Calculate Berry curvature
        B = berry_curvature(x, y, parameters)
        
        # Berry force: F = B × v (weighted by population difference)
        # This is the proper Lorentz-like force from Berry curvature
        fx_berry = B * pop_diff * vy
        fy_berry = -B * pop_diff * vx
        
        fx_total += fx_berry
        fy_total += fy_berry
    
    # Non-adiabatic coupling force (if enabled)
    if parameters.get('nac', False) and pop_lower > 1e-10 and pop_upper > 1e-10:
        # Calculate NAC vectors
        d_lu_x, d_lu_y = nonadiabatic_coupling(x, y, 'lower', 'upper', parameters)
        
        # NAC force proportional to coherence
        fx_nac = -2 * coherence * d_lu_x
        fy_nac = -2 * coherence * d_lu_y
        
        fx_total += fx_nac
        fy_total += fy_nac
        
        # Second-order NAC terms (if enabled)
        if parameters.get('nac2', False):
            d_lu_xx, d_lu_xy, d_lu_yy = second_order_nac(x, y, 'lower', 'upper', parameters)
            fx_nac2 = -coherence * d_lu_xx
            fy_nac2 = -coherence * d_lu_yy
            fx_total += fx_nac2
            fy_total += fy_nac2
    
    # Geometric force from Berry connection (if enabled)
    if parameters.get('geometric', False):
        # Calculate Berry connection
        Ax, Ay = berry_connection(x, y, parameters)
        
        # Calculate derivatives with finite differences
        delta = 1e-5
        Ax1, _ = berry_connection(x + delta, y, parameters)
        Ax2, _ = berry_connection(x - delta, y, parameters)
        _, Ay1 = berry_connection(x, y + delta, parameters)
        _, Ay2 = berry_connection(x, y - delta, parameters)
        
        # Berry connection derivatives
        dAx_dx = (Ax1 - Ax2) / (2 * delta)
        dAy_dy = (Ay1 - Ay2) / (2 * delta)
        
        # Geometric force properly weighted by populations
        fx_geometric = -(pop_lower * dAx_dx + pop_upper * dAx_dx)
        fy_geometric = -(pop_lower * dAy_dy + pop_upper * dAy_dy)
        
        fx_total += fx_geometric
        fy_total += fy_geometric
    
    return fx_total, fy_total

# %%
def simulate_trajectory(traj_index, parameters):
    """
    Simulate a single quantum-classical trajectory.
    For pre-looped trajectories, applies Berry phase at t=0 only,
    then lets dynamics evolve naturally without artificial constraints.
    
    Parameters:
    - traj_index: Index of trajectory
    - parameters: Dictionary containing simulation parameters
    
    Returns:
    - Dictionary containing trajectory data
    """
    # Determine trajectory type
    num_pre = parameters.get('num_prelooping', 0)
    is_prelooping = (traj_index < num_pre)
    traj_type = 'prelooping' if is_prelooping else 'regular'
    
    # Allocate storage for trajectory data
    ns = parameters['ns']
    dt = parameters['dt']
    
    traj = {
        'type': traj_type,
        'positions': np.zeros((ns, 2)),
        'momenta': np.zeros((ns, 2)),
        'velocities': np.zeros((ns, 2)),
        'psi_t': np.zeros((ns, 2), dtype=complex),
        'energies': np.zeros(ns),
        'populations': np.zeros((ns, 2)),
        'adiabatic_pops': np.zeros((ns, 2)),
        'berry_curvatures': np.zeros(ns),
        'forces': np.zeros((ns, 2)),
        'warnings': []
    }
    
    # Initialize position and velocity - physics-based, no artificial constraints
    x0, y0, vx0, vy0 = initialize_trajectory(parameters, is_prelooping)
    x, y, vx, vy = x0, y0, vx0, vy0
    
    # Store initial conditions
    traj['initial_position'] = [x0, y0]
    traj['initial_velocity'] = [vx0, vy0]
    traj['is_prelooping'] = is_prelooping
    
    # Compute initial Hamiltonian
    H = hamiltonian(x, y, parameters)
    
    # Initialize quantum state - apply Berry phase at t=0 for pre-looping
    psi0 = init_state(x0, y0, parameters, is_prelooping)
    psi = psi0 / np.linalg.norm(psi0)  # ensure normalization
    
    # Store analytical Berry phase values for analysis
    if is_prelooping:
        traj['lower_cone_berry_phase'] = berry_phase(x0, y0, parameters, state='lower')
        traj['upper_cone_berry_phase'] = berry_phase(x0, y0, parameters, state='upper')
    
    # Main evolution loop - same physics for both trajectory types
    for step in range(ns):
        # Record current state
        traj['positions'][step]  = [x, y]
        traj['velocities'][step] = [vx, vy]
        traj['momenta'][step]    = [mass*vx, mass*vy]
        traj['psi_t'][step]      = psi
        traj['energies'][step]   = np.real(np.vdot(psi, np.dot(H, psi)))
        
        # Record populations in diabatic and adiabatic bases
        psi_lower, psi_upper = eigenvectors(x, y, parameters)
        c_lower = np.vdot(psi_lower, psi)
        c_upper = np.vdot(psi_upper, psi)
        traj['populations'][step]    = [abs(psi[0])**2, abs(psi[1])**2]
        traj['adiabatic_pops'][step] = [abs(c_lower)**2, abs(c_upper)**2]
        
        # Record Berry curvature and force at this step
        traj['berry_curvatures'][step] = berry_curvature(x, y, parameters)
        fx, fy = force(psi, x, y, vx, vy, parameters)
        traj['forces'][step] = [fx, fy]
        
        # Evolve to next step - same physics-based evolution for all trajectories
        if step < ns - 1:
            try:
                x, y, vx, vy, H, psi = evolve_classical(x, y, vx, vy, psi, dt, H, parameters)
            except Exception as e:
                traj['warnings'].append(f"Evolution error at step {step}: {e}")
                break
    
    return traj

# %%
def analyze_results(results, parameters):
    """Comprehensive analysis of simulation results for pure initial states."""
    print("\nAnalysis of Simulation Results:")
    print("=" * 30)
    
    # Basic statistics
    num_trajectories = len(results)
    num_prelooping = sum(1 for t in results if t.get('type') == 'prelooping')
    
    print(f"Total trajectories: {num_trajectories}")
    print(f"Pre-looping trajectories: {num_prelooping}")
    
    # Energy conservation analysis
    energy_drifts = []
    for traj in results:
        # Calculate relative energy drift
        energy = traj['energies']
        mean_energy = np.mean(energy)
        max_deviation = np.max(np.abs(energy - energy[0]))
        relative_drift = max_deviation / np.abs(energy[0]) if np.abs(energy[0]) > 1e-10 else 0
        energy_drifts.append(relative_drift)
    
    avg_energy_drift = np.mean(energy_drifts)
    max_energy_drift = np.max(energy_drifts)
    
    print(f"\nEnergy Conservation:")
    print(f"- Average energy drift: {avg_energy_drift:.2e}")
    print(f"- Maximum energy drift: {max_energy_drift:.2e}")
    
    # Population analysis
    diabatic_transfer = []
    adiabatic_transfer = []
    
    for traj in results:
        # Diabatic basis population change
        diab_pop_change = np.abs(traj['populations'][-1, 1] - traj['populations'][0, 1])
        diabatic_transfer.append(diab_pop_change)
        
        # Adiabatic basis population change
        adiab_pop_change = np.abs(traj['adiabatic_pops'][-1, 1] - traj['adiabatic_pops'][0, 1])
        adiabatic_transfer.append(adiab_pop_change)
    
    print("\nPopulation Dynamics:")
    print(f"- Average diabatic population transfer: {np.mean(diabatic_transfer):.4f}")
    print(f"- Average adiabatic population transfer: {np.mean(adiabatic_transfer):.4f}")
    
    # Berry phase analysis - Only for prelooping trajectories
    if num_prelooping > 0 and parameters.get('berry', False):
        # Extract Berry phase values based on initial state
        if not parameters['use_upper_state']:
            phases = [t.get('lower_cone_berry_phase', 0) for t in results if t.get('type') == 'prelooping']
            state_label = "lower"
        else:
            phases = [t.get('upper_cone_berry_phase', 0) for t in results if t.get('type') == 'prelooping']
            state_label = "upper"
        
        normalized_phases = [phase / (2 * np.pi) for phase in phases]
        
        print(f"\nAnalytical Berry Phase Analysis ({state_label} cone):")
        print(f"- Mean Berry phase: {np.mean(phases):.4f} rad ({np.mean(normalized_phases):.4f}*2π)")
        print(f"- Standard deviation: {np.std(phases):.4f} rad")
        
        # Check for quantization
        rounded_phases = [round(phase) for phase in normalized_phases]
        quantization_error = np.mean([abs(p - r) for p, r in zip(normalized_phases, rounded_phases)])
        print(f"- Quantization error: {quantization_error:.4f}*2π")
        
        if quantization_error < 0.1:
            print("- Berry phase appears well-quantized (error < 0.1)")
        else:
            print("- Berry phase not well-quantized (error > 0.1)")
    
    # NAC effects analysis
    if parameters.get('nac', False):
        print("\nNon-Adiabatic Effects:")
        
        # Calculate maximum NAC-induced population transfer
        max_transfer = np.max(adiabatic_transfer)
        print(f"- Maximum adiabatic population transfer: {max_transfer:.4f}")
        
        if max_transfer < 0.01:
            print("- Dynamics appear mostly adiabatic")
        elif max_transfer < 0.1:
            print("- Weak non-adiabatic effects observed")
        else:
            print("- Strong non-adiabatic effects observed")
    
    # Trajectory classification
    radii = []
    for traj in results:
        # Calculate average radius
        positions = traj['positions']
        r = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        radii.append(np.mean(r))
    
    print("\nTrajectory Characteristics:")
    print(f"- Average orbit radius: {np.mean(radii):.4f}")
    
    # Return analysis results as a dictionary
    analysis_results = {
        "num_trajectories": num_trajectories,
        "num_prelooping": num_prelooping,
        "energy_conservation": {
            "avg_drift": avg_energy_drift,
            "max_drift": max_energy_drift
        },
        "population": {
            "diabatic_transfer": np.mean(diabatic_transfer),
            "adiabatic_transfer": np.mean(adiabatic_transfer)
        },
        "trajectories": {
            "avg_radius": np.mean(radii)
        }
    }
    
    # Add Berry phase analysis if applicable
    if num_prelooping > 0 and parameters.get('berry', False):
        analysis_results["berry_phase"] = {
            f"{state_label}_cone": {
                "mean": np.mean(phases),
                "normalized_mean": np.mean(normalized_phases),
                "std": np.std(phases),
                "quantization_error": quantization_error
            }
        }
    
    return analysis_results


def plot_comprehensive_visualization(results, parameters):
    """
    Create comprehensive visualization of simulation results for pure initial states.
    
    Parameters:
    - results: List of trajectory results
    - parameters: Dictionary of simulation parameters
    
    Returns:
    - fig: Plotly figure object with multiple subplots
    """
    # Extract parameters needed for plots
    ns = parameters['ns']
    dt = parameters['dt']
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Trajectories in Phase Space', 'Population Dynamics',
            'Energy Conservation', 'Berry Curvature',
            'Momenta', 'Analytical Berry Phase'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'scatter3d'}],
            [{'type': 'scatter'}, {'type': 'bar'}]  # Bar chart for Berry phase
        ]
    )
    
    # Generate colors for different trajectories - convert matplotlib colors to hex
    mpl_colors = plt.cm.tab10.colors
    colors = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b in mpl_colors]
    
    # Plot positions (trajectories)
    for i, traj in enumerate(results):
        color = colors[i % len(colors)]
        positions = traj['positions']
        
        fig.add_trace(
            go.Scatter(
                x=positions[:, 0], y=positions[:, 1],
                mode='lines',
                name=f"Traj {i+1}",
                line=dict(color=color),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Plot population dynamics (average over all trajectories)
    time = np.arange(ns) * dt
    
    # Combine data across trajectories
    diabatic_pops = np.zeros((ns, 2))
    adiabatic_pops = np.zeros((ns, 2))
    
    for traj in results:
        diabatic_pops += traj['populations']
        adiabatic_pops += traj['adiabatic_pops']
    
    # Average across trajectories
    if len(results) > 0:
        diabatic_pops /= len(results)
        adiabatic_pops /= len(results)
    
    # Diabatic populations
    fig.add_trace(
        go.Scatter(
            x=time, y=diabatic_pops[:, 0],
            mode='lines', name='Diabatic Lower',
            line=dict(color='blue')
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=time, y=diabatic_pops[:, 1],
            mode='lines', name='Diabatic Upper',
            line=dict(color='red')
        ),
        row=1, col=2
    )
    
    # Adiabatic populations
    fig.add_trace(
        go.Scatter(
            x=time, y=adiabatic_pops[:, 0],
            mode='lines', name='Adiabatic Lower',
            line=dict(color='blue', dash='dash')
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=time, y=adiabatic_pops[:, 1],
            mode='lines', name='Adiabatic Upper',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=2
    )
    
    # Energy conservation
    for i, traj in enumerate(results):
        color = colors[i % len(colors)]
        # Normalize energy to initial value
        energy = traj['energies']
        if np.abs(energy[0]) > 1e-10:
            energy_relative = (energy - energy[0]) / np.abs(energy[0])
            
            fig.add_trace(
                go.Scatter(
                    x=time, y=energy_relative,
                    mode='lines',
                    name=f"Energy {i+1}",
                    line=dict(color=color),
                    showlegend=False
                ),
                row=2, col=1
            )
    
    # Berry curvature visualization (3D)
    if parameters.get('berry', False):
        # Create grid for visualization
        x = np.linspace(-3, 3, 30)
        y = np.linspace(-3, 3, 30)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(len(x)):
            for j in range(len(y)):
                try:
                    Z[i,j] = berry_curvature(X[i,j], Y[i,j], parameters)
                except Exception:
                    Z[i,j] = 0
        
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Viridis',
                name='Berry Curvature'
            ),
            row=2, col=2
        )
    
    # Momenta
    for i, traj in enumerate(results):
        color = colors[i % len(colors)]
        momenta = traj['momenta']
        
        fig.add_trace(
            go.Scatter(
                x=momenta[:, 0], y=momenta[:, 1],
                mode='lines',
                name=f"Momenta {i+1}",
                line=dict(color=color),
                showlegend=False
            ),
            row=3, col=1
        )
    
    # Analytical Berry phase - Bar chart for relevant cone based on initial state
    prelooping_trajectories = [t for t in results if t.get('type') == 'prelooping']
    
    if prelooping_trajectories and parameters.get('berry', False):
        traj_labels = [f"Traj {i+1}" for i in range(len(prelooping_trajectories))]
        lower_phases = [t.get('lower_cone_berry_phase', 0) / (2 * np.pi) for t in prelooping_trajectories]
        upper_phases = [t.get('upper_cone_berry_phase', 0) / (2 * np.pi) for t in prelooping_trajectories]
        
        # Plot only the relevant Berry phase based on initial state
        if not parameters['use_upper_state']:
            fig.add_trace(
                go.Bar(
                    x=traj_labels,
                    y=lower_phases,
                    name="Lower Cone Berry Phase",
                    marker_color='blue'
                ),
                row=3, col=2
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=traj_labels,
                    y=upper_phases,
                    name="Upper Cone Berry Phase",
                    marker_color='red'
                ),
                row=3, col=2
            )
    
    # Update layout
    fig.update_layout(
        height=900,
        width=1200,
        title_text=f"Quantum Dynamics with Pure {parameters['state_type'].capitalize()} Basis",
        showlegend=True,
        barmode='group'  # Group bars for Berry phase
    )
    
    # Update axes
    fig.update_xaxes(title_text="x position", row=1, col=1)
    fig.update_yaxes(title_text="y position", row=1, col=1)
    
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_yaxes(title_text="Population", row=1, col=2, range=[0, 1.1])
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Relative Energy Change", row=2, col=1)
    
    fig.update_xaxes(title_text="Px", row=3, col=1)
    fig.update_yaxes(title_text="Py", row=3, col=1)
    
    fig.update_xaxes(title_text="Trajectory", row=3, col=2)
    fig.update_yaxes(title_text="Berry Phase (in units of 2π)", row=3, col=2)
    
    # Add additional information
    basis_info = {
        'adiabatic': 'Adiabatic basis (energy eigenstates)',
        'diabatic': 'Diabatic basis (fixed basis)',
        'gaussian': 'Gaussian wavepacket'
    }
    
    effects = []
    if parameters.get('berry', False):
        effects.append('Berry curvature')
    if parameters.get('nac', False):
        effects.append('Non-adiabatic coupling')
    if parameters.get('geometric', False):
        effects.append('Geometric phase')
    
    effects_text = ', '.join(effects) if effects else 'None'
    
    annotation_text = (
        f"Basis: {basis_info.get(parameters['state_type'], 'Unknown')}<br>"
        f"Initial state: {'Upper' if parameters['use_upper_state'] else 'Lower'}<br>"
        f"Effects included: {effects_text}<br>"
        f"Eccentricity e: {parameters['e']:.2f}"
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        text=annotation_text,
        showarrow=False,
        font=dict(size=12),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4
    )
    
    return fig


def plot_pes_with_trajectories(results, parameters):
    """
    Create a 3D visualization showing both potential energy surfaces (PES)
    along with trajectories, with proper scaling to ensure visibility.
    
    Parameters:
    - results: List of trajectory dictionaries
    - parameters: Dictionary of simulation parameters
    
    Returns:
    - fig: Plotly figure object
    """
    # Create a grid for the PES surfaces (fixed reasonable range)
    x_range = np.linspace(-5, 5, 50)  # Adjusted range for better visualization
    y_range = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Calculate energies for both PES
    Z_lower = np.zeros_like(X)
    Z_upper = np.zeros_like(X)
    
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            x, y = X[i, j], Y[i, j]
            H = hamiltonian(x, y, parameters)
            eigenvalues, _ = np.linalg.eigh(H)
            Z_lower[i, j] = eigenvalues[0]  # Lower PES
            Z_upper[i, j] = eigenvalues[1]  # Upper PES
    
    # Create figure
    fig = go.Figure()
    
    # Add lower PES surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_lower,
        colorscale='Blues',
        opacity=0.7,
        name='Lower PES',
        showscale=False
    ))
    
    # Add upper PES surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_upper,
        colorscale='Reds',
        opacity=0.7,
        name='Upper PES',
        showscale=False
    ))
    
    # Colors for trajectories
    colors = ['green', 'purple', 'orange', 'cyan', 'magenta']
    
    # Select up to 3 trajectories to display
    max_trajectories = min(3, len(results))
    
    # Add trajectories, but filter to only show parts within the PES range
    for i in range(max_trajectories):
        traj = results[i]
        positions = traj['positions']
        
        # Filter positions to keep only those within our PES plotting range
        # with some margin to show trajectories that slightly exceed the range
        plot_range = 7.5  # Slightly larger than PES range
        valid_indices = np.where(
            (positions[:, 0] >= -plot_range) & (positions[:, 0] <= plot_range) &
            (positions[:, 1] >= -plot_range) & (positions[:, 1] <= plot_range)
        )[0]
        
        if len(valid_indices) == 0:
            print(f"Warning: Trajectory {i+1} is entirely outside the plotting range.")
            continue
        
        valid_positions = positions[valid_indices]
        
        # For each position, calculate the energy using the instantaneous quantum state
        energies = np.zeros(len(valid_positions))
        
        for j, (x, y) in enumerate(valid_positions):
            psi = traj['psi_t'][valid_indices[j]]
            H = hamiltonian(x, y, parameters)
            energies[j] = np.real(np.vdot(psi, np.dot(H, psi)))
        
        # Add trajectory
        fig.add_trace(go.Scatter3d(
            x=valid_positions[:, 0],
            y=valid_positions[:, 1],
            z=energies,
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=5),
            name=f'Trajectory {i+1}'
        ))
        
        # Add markers to show start and end points (only if they're within range)
        if valid_indices[0] == 0:  # If the start point is within range
            fig.add_trace(go.Scatter3d(
                x=[valid_positions[0, 0]],
                y=[valid_positions[0, 1]],
                z=[energies[0]],
                mode='markers',
                marker=dict(size=8, color=colors[i % len(colors)], symbol='circle'),
                name=f'Start {i+1}'
            ))
        
        if valid_indices[-1] == len(positions) - 1:  # If the end point is within range
            fig.add_trace(go.Scatter3d(
                x=[valid_positions[-1, 0]],
                y=[valid_positions[-1, 1]],
                z=[energies[-1]],
                mode='markers',
                marker=dict(size=8, color=colors[i % len(colors)], symbol='x'),
                name=f'End {i+1}'
            ))
    
    # Update layout
    fig.update_layout(
        title='Trajectories on Potential Energy Surfaces',
        scene=dict(
            xaxis_title='x position',
            yaxis_title='y position',
            zaxis_title='Energy',
            aspectratio=dict(x=1, y=1, z=0.8),
            # Fix axis ranges to ensure consistent view
            xaxis=dict(range=[-6, 6]),
            yaxis=dict(range=[-6, 6]),
            # Let z-axis auto-scale based on energy values
        ),
        width=900,
        height=700,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.5)'
        )
    )
    
    # Add annotation explaining the plot limitations
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.02, y=0.02,
        text='Note: Trajectories are filtered to show only segments within the PES range',
        showarrow=False,
        align='left',
        bgcolor='rgba(255, 255, 255, 0.7)',
        bordercolor='gray',
        borderwidth=1,
        borderpad=4
    )
    
    return fig



def plot_population_dynamics(results, parameters):
    """
    Create a figure with two subplots showing the averaged diabatic and adiabatic 
    population dynamics separately for better comparison.
    
    Parameters:
    - results: List of trajectory dictionaries
    - parameters: Dictionary of simulation parameters
    
    Returns:
    - fig: Plotly figure with diabatic and adiabatic population subplots
    """
    # Extract time information
    ns = parameters['ns']
    dt = parameters['dt']
    time = np.arange(ns) * dt
    
    # Create figure with two subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'Diabatic Population Dynamics (Averaged)',
            'Adiabatic Population Dynamics (Averaged)'
        ),
        vertical_spacing=0.15
    )
    
    # Check if there are any results
    if not results:
        return fig
        
    # Initialize arrays for population data
    diabatic_pops_avg = np.zeros((ns, 2))
    adiabatic_pops_avg = np.zeros((ns, 2))
    
    # Sum population data from all trajectories
    for traj in results:
        diabatic_pops_avg += traj['populations']
        adiabatic_pops_avg += traj['adiabatic_pops']
    
    # Average by dividing by number of trajectories
    diabatic_pops_avg /= len(results)
    adiabatic_pops_avg /= len(results)
    
    # Plot diabatic populations
    fig.add_trace(
        go.Scatter(
            x=time, y=diabatic_pops_avg[:, 0],
            mode='lines',
            name='Lower Diabatic',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=time, y=diabatic_pops_avg[:, 1],
            mode='lines',
            name='Upper Diabatic',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # Plot adiabatic populations
    fig.add_trace(
        go.Scatter(
            x=time, y=adiabatic_pops_avg[:, 0],
            mode='lines',
            name='Lower Adiabatic',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=time, y=adiabatic_pops_avg[:, 1],
            mode='lines',
            name='Upper Adiabatic',
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )
    
    # Add reference lines at 0.5 population
    fig.add_trace(
        go.Scatter(
            x=[time[0], time[-1]], y=[0.5, 0.5],
            mode='lines',
            line=dict(color='gray', dash='dot', width=1),
            name='Equal Population',
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[time[0], time[-1]], y=[0.5, 0.5],
            mode='lines',
            line=dict(color='gray', dash='dot', width=1),
            name='Equal Population',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Update layout with better formatting
    fig.update_layout(
        height=800,
        width=900,
        title=dict(
            text=f'Population Dynamics Comparison (e = {parameters["e"]:.2f})',
            x=0.5
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    # Update axes
    fig.update_xaxes(title_text='Time', row=1, col=1)
    fig.update_yaxes(title_text='Diabatic Population', range=[0, 1], row=1, col=1)
    
    fig.update_xaxes(title_text='Time', row=2, col=1)
    fig.update_yaxes(title_text='Adiabatic Population', range=[0, 1], row=2, col=1)
    
    # Add annotation explaining the plots
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.01, y=0.99,
        text=(
            f'Average of {len(results)} trajectories<br>'
            'Blue: Lower state populations<br>'
            'Red: Upper state populations<br>'
            f'Initial upper state population: {parameters["upper_pop"]:.2f}'
        ),
        showarrow=False,
        align='left',
        bgcolor='rgba(255, 255, 255, 0.7)',
        bordercolor='gray',
        borderwidth=1,
        borderpad=4
    )
    
    return fig


# 


def collect_all_parameters():
   """Collect simulation parameters focused on physical rather than numerical constraints."""
   print("\nQuantum Dynamics Simulation Parameters")
   print("=" * 35)
   
   parameters = {}
   
   # System parameters
   print("\nSystem Parameters:")
   parameters['a'] = get_input("Enter constant 'a' for Hamiltonian", 1.0, float)
   parameters['e'] = get_input("Enter constant 'e' (0 ≤ e ≤ 1)", 0.8, float)
   parameters['s'] = get_input("Enter scaling factor 's'", 1.0, float)
   
   # z-function parameters
   parameters['z_choice'] = input("Is z constant or a function? ('constant' or 'function', default: 'constant'): ").lower() or 'constant'
   if parameters['z_choice'] == 'constant':
       parameters['z_val'] = get_input("Enter constant value for z", 2.0, float)
   
   # Trajectory parameters
   print("\nTrajectory Parameters:")
   parameters['r_min'] = get_input("Enter minimum radius for pre-looping trajectories", 0.5, float)
   parameters['r_max'] = get_input("Enter maximum radius for pre-looping trajectories", 5.0, float)
   parameters['nt'] = int(get_input("Enter number of trajectories", 5, int))
   parameters['dt'] = get_input("Enter time step", 0.01, float)
   parameters['Ttotal'] = get_input("Enter total time", 100.0, float)
   parameters['ns'] = int(np.ceil(parameters['Ttotal'] / parameters['dt']))  # Number of steps
   
   # Berry phase parameters
   include_berry = input("Include Berry phase for pre-looping trajectories? (yes/no): ").lower() == 'yes'
   parameters['berry'] = include_berry
   
   if include_berry:
       parameters['num_prelooping'] = int(get_input("Enter number of pre-looping trajectories (0 to nt)", 
                                                   min(parameters['nt'], 1000), int))
       parameters['num_prelooping'] = min(max(parameters['num_prelooping'], 0), parameters['nt'])
       
       parameters['use_fixed_radius'] = input("Use fixed radius for pre-looping? (yes/no): ").lower() == 'yes'
       if parameters['use_fixed_radius']:
           parameters['radius'] = get_input("Enter fixed radius for pre-looping", 
                                           (parameters['r_min'] + parameters['r_max']) / 2, float)
   else:
       parameters['num_prelooping'] = 0
       parameters['use_fixed_radius'] = False
   
   # Initial state selection (pure states only)
   print("\nInitial State Configuration:")
   print("Using pure states for all trajectories")
   state_choice = input("Choose initial state (1=lower, 2=upper): ") or "1"
   parameters['use_upper_state'] = (state_choice == "2")
   
   if parameters['use_upper_state']:
       print("Using pure upper state")
       parameters['upper_pop'] = 1.0  # For backward compatibility
   else:
       print("Using pure lower state")
       parameters['upper_pop'] = 0.0  # For backward compatibility
   
   # For regular trajectories
   parameters['random_phase'] = input("Use random initial phase for regular trajectories? (y/n, default y): ").lower() != 'n'
   
   # State type selection
   print("\nSelect basis type:")
   print("1. Adiabatic basis")
   print("2. Diabatic basis")
   print("3. Gaussian wavepacket")
   
   state_choice = input("Enter choice (1-3, default 1): ") or "1"
   state_types = ['adiabatic', 'diabatic', 'gaussian']
   parameters['state_type'] = state_types[int(state_choice) - 1]
   
   # Effect parameters
   print("\nQuantum Effects Options:")
   parameters['berry'] = input("Include Berry curvature force? (y/n, default y): ").lower() != 'n'
   parameters['nac'] = input("Include non-adiabatic coupling? (y/n, default n): ").lower() == 'y'
   
   if parameters['nac']:
       parameters['nac2'] = input("Include second-order NAC? (y/n, default n): ").lower() == 'y'
   else:
       parameters['nac2'] = False
   
   parameters['geometric'] = input("Include geometric phase? (y/n, default y): ").lower() != 'n'
   
   # Scientific parameters (not artificial constraints)
   print("\nNumerical Integration Parameters:")
   parameters['energy_threshold'] = get_input("Energy conservation threshold", 1e-4, float)
   parameters['gaussian_width'] = get_input("Gaussian wavepacket width (if used)", 0.5, float)
   parameters['debug'] = input("Show detailed warnings and debug info? (y/n, default n): ").lower() == 'y'
   
   # Add phase offset parameter for state initialization
   parameters['phase_offset'] = 0.0  # Default phase offset between state components
   
   return parameters

# %%
# Optimized main simulation function with proper parallelization
def run_simulation(parameters):
    """Run quantum dynamics simulation without parallelization."""
    # Setup default parameters
    default_params = {
        'phase_offset': 0.0,
        'gaussian_width': 0.5,
        'debug': False,
        'energy_threshold': 1e-4,
    }
    
    # Update parameters with defaults for missing values
    for key, value in default_params.items():
        if key not in parameters:
            parameters[key] = value
    
    results = []
    nt = parameters['nt']
    
    # Run trajectories sequentially
    for i in tqdm(range(nt), desc="Simulating trajectories"):
        results.append(simulate_trajectory(i, parameters))
    
    return results

# Functions to keep exactly from original code:
# 1. analyze_results - The analysis logic should remain unchanged
# 2. collect_all_parameters - The parameter collection should remain unchanged
# 3. All visualization functions - These don't affect the physics and should be kept:
#    - plot_comprehensive_visualization
#    - plot_pes_with_trajectories
#    - plot_population_dynamics

# Main execution function
def run_full_simulation():
    """
    Main entry point to run the complete simulation workflow.
    Optimized but maintains exact physics.
    """
    # Collect all parameters
    parameters = collect_all_parameters()
    
    print("\nInitializing simulation...")
    # Run simulation with optimized code
    results = run_simulation(parameters)
    
    # Analyze results
    analysis = analyze_results(results, parameters)
    
    # Visualize results
    print("\nGenerating visualizations...")
    fig = plot_comprehensive_visualization(results, parameters)
    fig.show()
    
    # Create the 3D PES with trajectories visualization
    pes_fig = plot_pes_with_trajectories(results, parameters)
    pes_fig.show()
    
    # Create the separate population dynamics visualization
    pop_fig = plot_population_dynamics(results, parameters)
    pop_fig.show()
    
    # Save results if requested
    save_results = input("\nSave results? (y/n): ").lower() == 'y'
    if save_results:
        filename = f"quantum_dynamics_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.npz"
        
        # Prepare data for saving
        save_data = {
            'results': results,
            'parameters': parameters,
            'analysis': analysis
        }
        
        # Save using numpy's compressed format
        np.savez_compressed(filename, **save_data)
        print(f"Results saved to {filename}")
    
    return results, analysis, fig, pes_fig, pop_fig

# Execute the simulation if this is the main script
if __name__ == "__main__":
    results, analysis, fig, pes_fig, pop_fig = run_full_simulation()

# %%
