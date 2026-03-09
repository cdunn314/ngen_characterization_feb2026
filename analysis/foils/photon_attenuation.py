"""
Photon Mass Attenuation Coefficient Tables

This module reads elemental attenuation coefficient data from NIST XCOM text files
and provides easy access by element symbol.

Usage:
    from photon_attenuation import get_attenuation_coeffs, ATTENUATION_DATA
    
    # Get data for iron
    energy, mu_rho = get_attenuation_coeffs('Fe')
    
    # Or access the dictionary directly
    energy = ATTENUATION_DATA['Fe']['energy']  # MeV
    mu_rho = ATTENUATION_DATA['Fe']['mu_rho']  # cm²/g
"""

import numpy as np
from pathlib import Path
import json

# Elements available
ELEMENTS = ['Al', 'Cu', 'Fe', 'In', 'Mo', 'Nb', "Ni", 'Ti', 'Zr']


def parse_attenuation_file(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a NIST XCOM attenuation coefficient text file.
    
    Args:
        filepath: Path to the .txt file
        
    Returns:
        Tuple of (energy_MeV, mu_rho_cm2_per_g)
    """
    energies = []
    mu_rho = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip header lines and empty lines
            if not line or line.startswith('Energy') or line.startswith('(MeV)') or line.startswith('_'):
                continue
            
            # Handle edge markers (K, L, M, etc.) at the start of line
            parts = line.split()
            if len(parts) >= 3:
                # Check if first part is an edge marker (single letter)
                if len(parts[0]) == 1 and parts[0].isalpha():
                    # Edge marker present, skip it
                    energy_str = parts[1]
                    mu_str = parts[2]
                else:
                    energy_str = parts[0]
                    mu_str = parts[1]
                
                try:
                    energies.append(float(energy_str))
                    mu_rho.append(float(mu_str))
                except ValueError:
                    continue
    
    return np.array(energies), np.array(mu_rho)


def load_all_attenuation_data(data_dir: str = None) -> dict:
    """
    Load all elemental attenuation coefficient data.
    
    Args:
        data_dir: Directory containing the .txt files. Defaults to same directory as this module.
        
    Returns:
        Dictionary with element symbols as keys, containing 'energy' and 'mu_rho' arrays
    """
    if data_dir is None:
        data_dir = Path(__file__).parent
    else:
        data_dir = Path(data_dir)
    
    data = {}
    for element in ELEMENTS:
        filepath = data_dir / f"{element}.txt"
        if filepath.exists():
            energy, mu_rho = parse_attenuation_file(filepath)
            data[element] = {
                'energy': energy,      # MeV
                'mu_rho': mu_rho,      # cm²/g
            }
        else:
            print(f"Warning: {filepath} not found")
    
    return data


def get_attenuation_coeffs(element: str, data_dir: str = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the photon mass attenuation coefficients for an element.
    
    Args:
        element: Element symbol (e.g., 'Fe', 'Al', 'Cu')
        data_dir: Directory containing the .txt files
        
    Returns:
        Tuple of (energy_MeV, mu_rho_cm2_per_g)
        
    Example:
        energy, mu_rho = get_attenuation_coeffs('Fe')
    """
    if element not in ELEMENTS:
        raise ValueError(f"Element '{element}' not available. Choose from: {ELEMENTS}")
    
    if data_dir is None:
        data_dir = Path(__file__).parent
    else:
        data_dir = Path(data_dir)
    
    filepath = data_dir / f"{element}.txt"
    return parse_attenuation_file(filepath)


def interpolate_mu_rho(element: str, energy_MeV: float | np.ndarray, 
                        data_dir: str = None) -> float | np.ndarray:
    """
    Interpolate the mass attenuation coefficient at a specific energy.
    Uses log-log interpolation for better accuracy.
    
    Args:
        element: Element symbol
        energy_MeV: Energy or array of energies in MeV
        data_dir: Directory containing the .txt files
        
    Returns:
        Interpolated μ/ρ value(s) in cm²/g
    """
    energy_data, mu_rho_data = get_attenuation_coeffs(element, data_dir)
    
    # Log-log interpolation
    log_energy_data = np.log(energy_data)
    log_mu_rho_data = np.log(mu_rho_data)
    
    log_energy_query = np.log(energy_MeV)
    log_mu_rho_interp = np.interp(log_energy_query, log_energy_data, log_mu_rho_data)
    
    return np.exp(log_mu_rho_interp)


def save_to_npz(output_path: str = None, data_dir: str = None):
    """
    Save all attenuation data to a single .npz file for fast loading.
    
    Args:
        output_path: Path for the .npz file
        data_dir: Directory containing the .txt files
    """
    if output_path is None:
        output_path = Path(__file__).parent / "photon_attenuation_data.npz"
    
    data = load_all_attenuation_data(data_dir)
    
    # Flatten the data for npz storage
    save_dict = {}
    for element in data:
        save_dict[f"{element}_energy"] = data[element]['energy']
        save_dict[f"{element}_mu_rho"] = data[element]['mu_rho']
    
    np.savez(output_path, elements=ELEMENTS, **save_dict)
    print(f"Saved attenuation data to {output_path}")


def save_to_json(output_path: str = None, data_dir: str = None):
    """
    Save all attenuation data to a human-readable JSON file.
    
    Args:
        output_path: Path for the .json file
        data_dir: Directory containing the .txt files
    """
    if output_path is None:
        output_path = Path(__file__).parent / "photon_attenuation_data.json"
    
    data = load_all_attenuation_data(data_dir)
    
    # Build JSON structure
    json_data = {
        "description": "Photon mass attenuation coefficients (μ/ρ) from NIST XCOM database",
        "units": {
            "energy": "MeV",
            "mu_rho": "cm²/g"
        },
        "elements": {}
    }
    
    for element in data:
        json_data["elements"][element] = {
            "energy": data[element]['energy'].tolist(),
            "mu_rho": data[element]['mu_rho'].tolist()
        }
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Saved attenuation data to {output_path}")


def load_from_json(json_path: str = None) -> dict:
    """
    Load attenuation data from JSON file.
    
    Args:
        json_path: Path to the .json file
        
    Returns:
        Dictionary with element symbols as keys
    """
    if json_path is None:
        json_path = Path(__file__).parent / "photon_attenuation_data.json"
    
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    data = {}
    for element, values in json_data["elements"].items():
        data[element] = {
            'energy': np.array(values['energy']),
            'mu_rho': np.array(values['mu_rho']),
        }
    
    return data


def load_from_npz(npz_path: str = None) -> dict:
    """
    Load attenuation data from .npz file.
    
    Args:
        npz_path: Path to the .npz file
        
    Returns:
        Dictionary with element symbols as keys
    """
    if npz_path is None:
        npz_path = Path(__file__).parent / "photon_attenuation_data.npz"
    
    npz_data = np.load(npz_path)
    elements = list(npz_data['elements'])
    
    data = {}
    for element in elements:
        data[element] = {
            'energy': npz_data[f"{element}_energy"],
            'mu_rho': npz_data[f"{element}_mu_rho"],
        }
    
    return data


# Pre-load data when module is imported
ATTENUATION_DATA = load_all_attenuation_data()


if __name__ == "__main__":
    # Create the .json file for human-readable storage
    save_to_json()
    
    # Also create .npz file for faster loading
    save_to_npz()
    
    # Test loading
    print("\nAvailable elements:", ELEMENTS)
    print("\nExample - Iron (Fe):")
    energy, mu_rho = get_attenuation_coeffs('Fe')
    print(f"  Energy range: {energy.min():.3e} - {energy.max():.3e} MeV")
    print(f"  μ/ρ range: {mu_rho.min():.3e} - {mu_rho.max():.3e} cm²/g")
    
    # Test interpolation
    test_energy = 0.662  # Cs-137 gamma energy in MeV
    mu_at_662keV = interpolate_mu_rho('Fe', test_energy)
    print(f"\n  μ/ρ at {test_energy*1000:.0f} keV: {mu_at_662keV:.4f} cm²/g")
