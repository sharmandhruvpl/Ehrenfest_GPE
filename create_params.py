# create_params.py
import json
import os
import numpy as np
from itertools import product

def create_parameters():
    """
    Create base parameter dictionary for quantum dynamics simulation.
    """
    # Initialize empty parameters dictionary
    parameters = {}
    
    # System Parameters
    parameters['a'] = 1.0         # Constant 'a' for Hamiltonian
    parameters['s'] = 1.0         # Scaling factor
    
    # Trajectory Parameters
    parameters['r_min'] = 0.05    # Minimum radius for trajectories
    parameters['r_max'] = 0.5     # Maximum radius for trajectories
    parameters['nt'] = 100        # Number of trajectories - reduced to 100
    parameters['dt'] = 0.005      # Time step
    parameters['Ttotal'] = 100.0  # Total simulation time
    
    # Calculate number of steps
    parameters['ns'] = int(np.ceil(parameters['Ttotal'] / parameters['dt']))
    
    # Berry Phase Parameters
    parameters['num_prelooping'] = 50  # Number of pre-looping trajectories - 50
    parameters['use_fixed_radius'] = False  # Use random radius between r_min and r_max
    
    # Initial State Configuration
    parameters['use_upper_state'] = True   # Use upper state
    parameters['upper_pop'] = 1.0          # Population of upper state
    parameters['random_phase'] = True      # Use random initial phase
    parameters['state_type'] = 'adiabatic' # Using adiabatic state type
    
    # Numerical Integration Parameters
    parameters['energy_threshold'] = 1e-4  # Energy conservation threshold
    parameters['gaussian_width'] = 0.5     # Gaussian wavepacket width
    parameters['debug'] = False            # Show detailed warnings and debug info
    
    return parameters

def generate_parameter_files():
    """Generate parameter files for all combinations of e, z, and forces."""
    # Create output directory
    os.makedirs("param_files", exist_ok=True)
    
    # Get base parameters
    base_params = create_parameters()
    
    # Define all variations
    e_values = [0.0, 0.4, 0.8]            # Eccentricity values
    
    # z configurations: (z_choice, z_val, name_suffix)
    z_configs = [
        ('constant', 0.0, 'z0.0'),        # z = 0 case
        ('constant', 0.05, 'z0.05'),      # z = 0.05 case
        ('function', None, 'zfunc')       # z as function
    ]
    
    # Force combinations: Berry, NAC, Geometric
    # Format: (berry, nac, nac2, geometric, name_suffix)
    force_combinations = [
        (True, False, False, True, "berry_geom"),        # Berry + Geometric
        (True, False, False, False, "berry_only"),       # Berry only
        (False, True, False, True, "nac_geom"),          # NAC + Geometric
        (False, True, True, False, "nac2_only"),         # NAC + NAC2 only (changed)
        (True, True, False, True, "berry_nac_geom"),     # Berry + NAC + Geometric
        (True, True, True, True, "all_forces"),          # All forces
        (False, False, False, True, "geom_only")         # Geometric only
    ]
    
    # Generate all combinations
    parameter_files = []
    
    for e, (z_choice, z_val, z_suffix), (berry, nac, nac2, geometric, force_suffix) in product(
        e_values, z_configs, force_combinations
    ):
        # Create a copy of base parameters
        params = base_params.copy()
        
        # Set e value
        params['e'] = e
        
        # Set z parameters
        params['z_choice'] = z_choice
        if z_choice == 'constant':
            params['z_val'] = z_val
        
        # Set force parameters
        params['berry'] = berry
        params['nac'] = nac
        params['nac2'] = nac2
        params['geometric'] = geometric
        
        # Create descriptive filename
        e_str = f"e{e:.1f}"
        
        filename = f"param_files/{e_str}_{z_suffix}_{force_suffix}.json"
        
        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(params, f, indent=2)
        
        parameter_files.append(filename)
    
    return parameter_files

if __name__ == "__main__":
    # Generate all parameter files
    param_files = generate_parameter_files()
    
    # Print summary
    print(f"Generated {len(param_files)} parameter files:")
    for i, filename in enumerate(param_files):
        base_name = os.path.basename(filename)
        print(f"{i+1}. {base_name}")
    
    print("\nParameter combinations include:")
    print("- Eccentricity (e): 0.0, 0.4, 0.8")
    print("- Z parameter: constant (0.0), constant (0.05), or function")
    print("- Force combinations: various combinations of Berry, NAC, NAC2, and Geometric forces")
    print("\nAll files use:")
    print("- 100 trajectories (50 pre-looping + 50 regular)")
    print("- Random radius between 0.05 and 0.5")
    print("- dt = 0.005, Ttotal = 100")
    print("- a = 1.0")
    print("- Adiabatic state representation")
