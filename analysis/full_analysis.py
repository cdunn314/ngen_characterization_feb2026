"""
Script to run full analysis in a loop.

This script:
1. Executes foil_analysis_NaI.ipynb notebook
2. Executes diamond_analysis.ipynb notebook
3. Performs the full analysis from full_analysis.ipynb
4. Repeats steps 1-3 in a loop
"""

import subprocess
import sys
from pathlib import Path
import json
import numpy as np
import openmc
import openmc.lib
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import curve_fit
import os
import time
import h5py
sys.path.append(str(Path(__file__).parent / '../analysis/foils'))
sys.path.append(str(Path(__file__).parent / '../analysis/diamond'))
sys.path.append(str(Path(__file__).parent / '../neutronics'))
from neutron_source import nGen_generator
from experiment_model import create_experiment_model, get_xs_from_tallies

def run_notebook(notebook_path: Path):
    """Run a Jupyter notebook using nbconvert."""
    print(f"\n{'='*80}")
    print(f"Running notebook: {notebook_path}")
    print(f"{'='*80}\n")

    
    try:
        result = subprocess.run(
            [
                "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                str(notebook_path)
            ],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ Successfully executed: {notebook_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error executing notebook: {notebook_path.name}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: jupyter nbconvert not found. Please install it with:")
        print("  pip install nbconvert jupyter")
        sys.exit(1)


def get_expected_DT_neutron_energies(angles, voltage=120, target_thickness='thick'):
    """Calculate expected DT neutron energies at given angles."""
    if target_thickness == 'thin':
        parameters = np.array([[50, 14.04814, 0.47679, 0.00834],
                                [100, 14.06732, 0.67488, 0.01719],
                                [200, 14.10711, 0.95596, 0.03320],
                                [300, 14.14704, 1.17282, 0.04923],
                                [400, 14.18670, 1.35640, 0.06527],
                                [500, 14.22569, 1.51899, 0.08249]])
    elif target_thickness == 'thick':
        parameters = np.array([[50, 14.06520, 0.42329, 0.00682],
                                [100, 14.07883, 0.57613, 0.01222],
                                [150, 14.08942, 0.66776, 0.01600],
                                [200, 14.09680, 0.72427, 0.01908],
                                [250, 14.10286, 0.76661, 0.02167],
                                [300, 14.10803, 0.80001, 0.02374],
                                [325, 14.10723, 0.79477, 0.02347]])
    angles = np.array(angles)

    E_0 = np.interp(voltage, parameters[:,0], parameters[:,1])
    E_1 = np.interp(voltage, parameters[:,0], parameters[:,2])
    E_2 = np.interp(voltage, parameters[:,0], parameters[:,3])

    energies = E_0 + E_1 * np.cos(np.radians(angles)) + E_2 * np.cos(np.radians(angles))**2
    return energies


def get_expected_DT_neutron_yields(angles, voltage=120, target_thickness='thick'):
    """Calculate expected DT neutron yields at given angles."""
    if target_thickness == 'thin':
        parameters = np.array([[20, 1.0, 0.0220, 0.00025],
                                [30, 1.0, 0.0227, -0.0093],
                                [40, 1.0, 0.0310, 0.0007],
                                [50, 1.0, 0.0344, 0.0010],
                                [60, 1.0, 0.0518, -0.0035],
                                [70, 1.0, 0.0407, 0.0011],
                                [100, 1.0, 0.0482, 0.0011],
                                [150, 1.0, 0.0599, 0.0009],
                                [200, 1.0, 0.0678, 0.0005],
                                [250, 1.0, 0.0685, -0.0104],
                                [300, 1.0, 0.0818, 0.0005],
                                [350, 1.0, 0.0904, 0.0028],
                                [400, 1.0, 0.1003, -0.0008],
                                [450, 1.0, 0.1140, -0.0101],
                                [500, 1.0, 0.1273, -0.0187]])
    elif target_thickness == 'thick':
        parameters = np.array([[50, 1, 0.03003, 0.00035],
                                [100, 1, 0.04087, 0.00062],
                                [150, 1, 0.04727, 0.00083],
                                [200, 1, 0.05124, 0.00096],
                                [250, 1, 0.05419, 0.00110],
                                [300, 1, 0.05651, 0.00119],
                                [325, 1, 0.05616, 0.00119]])
    angles = np.array(angles)
    Y_0 = np.interp(voltage, parameters[:,0], parameters[:,1])
    Y_1 = np.interp(voltage, parameters[:,0], parameters[:,2])
    Y_2 = np.interp(voltage, parameters[:,0], parameters[:,3])

    yields = Y_0 + Y_1 * np.cos(np.radians(angles)) + Y_2 * np.cos(np.radians(angles))**2
    return yields


def energy_model(theta, E_0, E_1, E_2):
    """Energy as a function of angle with cosine expansion."""
    return E_0 + E_1 * np.cos(np.radians(theta)) + E_2 * np.cos(np.radians(theta))**2


def calculate_uncertainty_band(theta, popt, pcov):
    """Calculate 1-sigma uncertainty band using linear error propagation."""
    cos_theta = np.cos(np.radians(theta))
    cos2_theta = cos_theta**2
    
    # J = [∂f/∂E_0, ∂f/∂E_1, ∂f/∂E_2]
    J = np.column_stack([np.ones_like(theta), cos_theta, cos2_theta])
    
    # Variance at each point: σ²(y) = J * Σ * J^T
    variance = np.sum(J @ pcov * J, axis=1)
    return np.sqrt(variance)


def add_foil_xs_to_processed_data(foil_xs_dict, json_path=None):
    """
    Add foil cross-section data to the processed_data.json file.
    
    Parameters
    ----------
    foil_xs_dict : dict
        Dictionary with foil names as keys and numpy arrays of cross-sections as values.
        Example: {'Aluminum-1_foil_-90deg_na': array([...]), ...}
    json_path : Path, optional
        Path to the processed_data.json file. Defaults to ../data/processed_data.json.
    """
    if json_path is None:
        json_path = Path(__file__).parent / '../data/processed_data.json'
    
    # Load existing data
    if json_path.exists():
        with open(json_path, 'r') as f:
            processed_data = json.load(f)
    else:
        processed_data = {}
    
    # Convert numpy arrays to lists for JSON serialization
    foil_xs_serializable = {}
    for key, value in foil_xs_dict.items():
        if isinstance(value, np.ndarray):
            foil_xs_serializable[key] = value.tolist()
        else:
            foil_xs_serializable[key] = value
    
    # Add or update the foil cross-sections
    processed_data['foil_cross_sections'] = foil_xs_serializable
    
    # Write back to file
    with open(json_path, 'w') as f:
        json.dump(processed_data, f, indent=4)
    
    print(f"Added {len(foil_xs_dict)} foil cross-sections to {json_path}")


def read_foil_xs_from_processed_data(json_path=None):
    """
    Read foil cross-section data from the processed_data.json file.
    
    Parameters
    ----------
    json_path : Path, optional
        Path to the processed_data.json file. Defaults to ../data/processed_data.json.
    
    Returns
    -------
    dict
        Dictionary with foil names as keys and numpy arrays of cross-sections as values.
        Example: {'Aluminum-1_foil_-90deg_na': array([...]), ...}
    """
    if json_path is None:
        json_path = Path(__file__).parent / '../data/processed_data.json'
    
    with open(json_path, 'r') as f:
        processed_data = json.load(f)
    
    if 'foil_cross_sections' not in processed_data:
        print(f"No foil cross-sections found in {json_path}")
        return {}
    
    # Convert lists back to numpy arrays
    foil_xs_dict = {}
    for key, value in processed_data['foil_cross_sections'].items():
        if isinstance(value, list):
            foil_xs_dict[key] = np.array(value)
        else:
            foil_xs_dict[key] = value
    
    print(f"Read {len(foil_xs_dict)} foil cross-sections from {json_path}")
    return foil_xs_dict


def plot_source_directions(particle_directions, energies, output_dir=Path('.'), iteration=0):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    # create colormap for energies
    norm = plt.Normalize(min(energies), max(energies))
    cmap = plt.get_cmap('viridis')

    for direction, energy in zip(particle_directions, energies):
        u, v, w = direction
        color = cmap(norm(energy))
        axes[0].plot([0,u], [0,v], '-', color=color, alpha=0.3)
        axes[1].plot([0,v], [0,w], '-', color=color, alpha=0.3)
        axes[2].plot([0,u], [0,w], '-', color=color, alpha=0.3)
        ax2.plot([0,u], [0,v], [0,w], '-', color=color, alpha=0.3)
    axes[0].set_title('U vs V')
    axes[0].set_xlabel('U')
    axes[0].set_ylabel('V')
    axes[1].set_title('V vs W')
    axes[1].set_xlabel('V')
    axes[1].set_ylabel('W')
    axes[2].set_title('U vs W')
    axes[2].set_xlabel('U')
    axes[2].set_ylabel('W')

    ax2.set_xlabel('U')
    ax2.set_ylabel('V')
    ax2.set_zlabel('W')

    # add colorbar outside of plots
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Make room for colorbar
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    # Add colorbar spanning full width
    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Neutron Energy (eV)')

    # add colorbar to 3D plot
    cbar2 = fig2.colorbar(sm, ax=ax2, orientation='vertical', fraction=0.02, pad=0.1)
    cbar2.set_label('Neutron Energy (eV)')

    fig2.tight_layout()

    fig.savefig(output_dir / f'source_directions_iteration_{iteration}.png')
    fig2.savefig(output_dir / f'source_directions_3D_iteration_{iteration}.png')
    return


def read_diamond_spectra_from_h5(h5_filename):

    with h5py.File(h5_filename, 'r') as f:
        print(f'Description: {f.attrs["description"]}')
        print(f'\nAvailable angles:')
        
        loaded_data = {}
        for name in f.keys():
            grp = f[name]
            loaded_data[name] = {
                'angle': grp.attrs['angle'],
                'energy_bins': grp['energy_bins'][:],
                'spectrum': grp['spectrum'][:]
            }
    return loaded_data



def perform_full_analysis(iteration=0, output_dir=None, 
                          openmc_model_path=Path(__file__).parent / '../neutronics/isotropic_source'):
    """
    Perform the full analysis workflow.
    
    Parameters:
    -----------
    iteration : int
        The iteration number for this analysis run
    output_dir : Path, optional
        Directory to save output plots. If None, plots are not saved.
    """
    print(f"\n{'#'*80}")
    print(f"# Starting Full Analysis - Iteration {iteration}")
    print(f"{'#'*80}\n")
    
    # run diamond analysis notebook
    diamond_notebook_path = Path(__file__).parent / "diamond" / "diamond_analysis.ipynb"
    # if not run_notebook(diamond_notebook_path):
    #     print("Error: Diamond analysis notebook failed. Skipping this iteration.")
    #     return

    # build neutron source from diamond spectra
    source_center = [0, 0, 0]
    diamond_spectra = read_diamond_spectra_from_h5('../data/diamond_processed_spectra.h5')
    neutron_sources = nGen_generator(diamond_spectra, center=source_center, reference_uvw=(-1, 0, 0))

    # build OpenMC model with this source and run it
    model, foil_cell_volumes = create_experiment_model(
        read_from_json=True,
        irdff_energy_groups=np.array([0, 2, 3, 6, 9, 12, 15]) * 1e6, # energy group boundaries in eV
        source=None,
        source_center=source_center,
        dd_dt_ratio=0.0,
        diamond_detector_distance=14.1 + 1.70, # 14.1 cm from source to detector face, about 1.70 cm from detector face to diamond face
        num_particles_per_batch=int(1e4)
    )
    model.export_to_model_xml()
    # if iteration==0:
    #     model.plot_geometry()
    model.run(threads=14)

    foil_xs_dict = get_xs_from_tallies(
        statepoint_path = Path("statepoint.100.h5"),
        foil_cell_volumes = foil_cell_volumes,
    )

    print(foil_xs_dict)

    add_foil_xs_to_processed_data(foil_xs_dict)




    # nai_notebook_path = Path(__file__) / "foils" / "foil_analysis"
    # try:
    #     run_notebook(nai_notebook_path)
    # except Exception as e:
    #     print(e)
    



    

    

    
    # # Load processed data
    # processed_data_file = Path(__file__).parent.parent / '../data' / 'processed_data.json'
    
    # foil_rates = {}
    # all_angles = []
    # all_rates = []
    # all_rate_errs = []
    
    # with open(processed_data_file, "r") as f:
    #     processed_data = json.load(f)
    #     diamond_angles = processed_data['diamond']['angles']
    #     neutron_energies = processed_data['diamond']['neutron_energies']['values']
    #     neutron_energy_errs = processed_data['diamond']['neutron_energies']['errors']
    #     n_alpha_rates = processed_data['diamond']['n_alpha_rates']['values']
    #     n_alpha_rate_errs = processed_data['diamond']['n_alpha_rates']['errors']
    #     sim_n_alpha_rates, sim_n_alpha_rate_errs = get_simulation_n_alpha_rate(diamond_angles, model_path=openmc_model_path)

    #     neutron_rates = np.array(n_alpha_rates) / np.array(sim_n_alpha_rates)
    #     neutron_rate_errs = neutron_rates * np.sqrt(
    #         (np.array(n_alpha_rate_errs) / np.array(n_alpha_rates))**2 +
    #         (np.array(sim_n_alpha_rate_errs) / np.array(sim_n_alpha_rates))**2
    #     )
    
    #     all_rates.append(neutron_rates)
    #     all_rate_errs.append(neutron_rate_errs)
    #     all_angles.append(diamond_angles)
    
    #     for detector in processed_data['foils']:
    #         foil_rates[detector] = {}
    #         for nuclide in processed_data['foils'][detector]:
    #             temp_angles = []
    #             temp_rates = []
    #             temp_rate_errs = []
    #             for angle in processed_data['foils'][detector][nuclide]:
    #                 temp_angles.append(float(angle))
    #                 temp_rates.append(processed_data['foils'][detector][nuclide][angle]['rate'])
    #                 temp_rate_errs.append(processed_data['foils'][detector][nuclide][angle]['rate_err'])
    #             foil_rates[detector][nuclide] = {
    #                 "angles": temp_angles,
    #                 "rates": temp_rates,
    #                 "rate_errs": temp_rate_errs
    #             }
    #             all_rates.append(temp_rates)
    #             all_rate_errs.append(temp_rate_errs)
    #             all_angles.append(temp_angles)
    
    # all_rates = np.concatenate(all_rates)
    # all_rate_errs = np.concatenate(all_rate_errs)
    # all_angles = np.abs(np.concatenate(all_angles))
    
    # sort_ind = np.argsort(all_angles)
    # all_angles = all_angles[sort_ind]
    # all_rates = all_rates[sort_ind]
    # all_rate_errs = all_rate_errs[sort_ind]
    
    # print("✓ Data loaded and processed")
    
    # # Gaussian Process for neutron rates
    # X = all_angles.reshape(-1, 1)
    # y = all_rates
    # y_err = all_rate_errs
    
    # kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=30.0, length_scale_bounds=(1, 100))
    # gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err**2, n_restarts_optimizer=10, normalize_y=True)
    # gp.fit(X, y)
    
    # X_pred = np.linspace(0, 180, 200).reshape(-1, 1)
    # y_pred, y_std = gp.predict(X_pred, return_std=True)
    
    # print(f"✓ Gaussian Process fitted")
    # print(f"  Optimized kernel: {gp.kernel_}")
    # print(f"  Log-marginal-likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.3f}")
    
    # # Plot neutron rates
    # if output_dir:
    #     fmts = {"diamond": "s", "Nb93": "^", "Zr90": "o"}
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     ax.errorbar(np.abs(diamond_angles), neutron_rates, yerr=neutron_rate_errs, 
    #                fmt=fmts["diamond"], label="Diamond Detector")
    #     for detector in foil_rates:
    #         for nuclide in foil_rates[detector]:
    #             angles = foil_rates[detector][nuclide]['angles']
    #             rates = foil_rates[detector][nuclide]['rates']
    #             rate_errs = foil_rates[detector][nuclide]['rate_errs']
    #             ax.errorbar(np.abs(angles), rates, yerr=rate_errs, fmt=fmts[nuclide],
    #                        label=f"{detector} - {nuclide}", capsize=4)
        
    #     ax.plot(X_pred, y_pred, 'b-', label='GP mean prediction', linewidth=2)
    #     ax.fill_between(X_pred.ravel(), y_pred - 1.96*y_std, y_pred + 1.96*y_std,
    #                     alpha=0.1, color='blue', label='95% confidence interval')
    #     ax.set_xlabel('Angle (degrees)')
    #     ax.set_ylabel('Neutron Rate (n/s)')
    #     ax.legend()
    #     ax.set_ylim(bottom=0)
    #     ax.grid(alpha=0.3)
    #     plt.tight_layout()
    #     plt.savefig(output_dir / f'neutron_rates_iter{iteration}.png', dpi=150)
    #     plt.close()
    #     print(f"✓ Saved neutron rates plot")
    
    # # Least squares fitting for neutron energy
    # angles = np.abs(np.array(diamond_angles))
    # y_energy = np.array(neutron_energies)
    # y_energy_err = np.array(neutron_energy_errs)
    
    # popt, pcov = curve_fit(energy_model, angles, y_energy, sigma=y_energy_err, absolute_sigma=True)
    # E_0_fit, E_1_fit, E_2_fit = popt
    # E_0_err, E_1_err, E_2_err = np.sqrt(np.diag(pcov))
    
    # print(f"\n✓ Fitted neutron energy parameters:")
    # print(f"  E_0 = {E_0_fit:.5f} ± {E_0_err:.5f} MeV")
    # print(f"  E_1 = {E_1_fit:.5f} ± {E_1_err:.5f} MeV")
    # print(f"  E_2 = {E_2_fit:.5f} ± {E_2_err:.5f} MeV")
    
    # angle_pred = np.linspace(angles.min(), angles.max(), 200)
    # energy_pred = energy_model(angle_pred, *popt)
    # energy_std = calculate_uncertainty_band(angle_pred, popt, pcov)
    
    # residuals = y_energy - energy_model(angles, *popt)
    # chi_square = np.sum((residuals / y_energy_err)**2)
    # dof = len(y_energy) - len(popt)
    # reduced_chi_square = chi_square / dof
    
    # print(f"\n✓ Goodness of fit:")
    # print(f"  χ² = {chi_square:.2f}")
    # print(f"  Reduced χ² = {reduced_chi_square:.2f}")
    # print(f"  Degrees of freedom = {dof}")
    
    # # Plot neutron energy
    # if output_dir:
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     ax.errorbar(angles, y_energy, yerr=y_energy_err, 
    #                fmt='ko', markersize=6, capsize=4, alpha=0.6, label='Measurements', zorder=3)
    #     ax.plot(angle_pred, energy_pred, 'r-', label='Least squares fit', linewidth=2, zorder=2)
    #     ax.fill_between(angle_pred, energy_pred - 1.96*energy_std, energy_pred + 1.96*energy_std,
    #                     alpha=0.2, color='red', label='95% confidence interval', zorder=1)
        
    #     angles_extrap = np.linspace(0, 180, 100)
    #     energy_pred_extrap = energy_model(angles_extrap, *popt)
    #     ax.plot(angles_extrap, energy_pred_extrap, 'r--', label='Extrapolated Fit', 
    #            linewidth=1.5, alpha=0.7)
        
    #     plot_angles = np.linspace(0, 180, 200)
    #     expected_energies_thick = get_expected_DT_neutron_energies(plot_angles, voltage=120, 
    #                                                                target_thickness='thick')
    #     expected_energies_thin = get_expected_DT_neutron_energies(plot_angles, voltage=120, 
    #                                                               target_thickness='thin')
    #     ax.plot(plot_angles, expected_energies_thin, 'g--', label='Thin Target Expected', 
    #            linewidth=2, alpha=0.7)
    #     ax.plot(plot_angles, expected_energies_thick, 'b--', label='Thick Target Expected', 
    #            linewidth=2, alpha=0.7)
        
    #     ax.set_xlabel('Angle (degrees)', fontsize=12)
    #     ax.set_ylabel('Neutron Energy (MeV)', fontsize=12)
    #     ax.set_title(f'Neutron Energy vs Angle - Iteration {iteration} (χ²/dof = {reduced_chi_square:.2f})', 
    #                 fontsize=14)
    #     ax.legend(fontsize=10, loc='best')
    #     ax.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.savefig(output_dir / f'neutron_energy_iter{iteration}.png', dpi=150)
    #     plt.close()
    #     print(f"✓ Saved neutron energy plot")
    
    # # Create source generator
    # sys.path.insert(0, str(Path(__file__).parent / '../neutronics'))
    # from neutron_source import nGen_generator
    
    
    
    # print(f"\n✓ Created source generators")
    
    # # Verify source generation
    # n_iterations_per_batch = 1000
    # batch_seed = 12345 + iteration  # Use different seed for each iteration
    
    # sphere = openmc.Sphere(r=100.0, boundary_type='vacuum')
    # cell = openmc.Cell(region=-sphere, fill=None)
    # geometry = openmc.Geometry(openmc.Universe(cells=[cell]))
    # materials = openmc.Materials()
    # settings = openmc.Settings()
    # settings.run_mode = 'fixed source'
    # settings.source = source_generator
    # settings.batches = 100
    # settings.inactive = 0
    # settings.particles = n_iterations_per_batch
    # model = openmc.Model(geometry=geometry, settings=settings, materials=materials)
    
    # # Export model to temporary directory
    # temp_model_dir = Path(__file__).parent / f'neutronics/temp_model_iter{iteration}'
    # temp_model_dir.mkdir(exist_ok=True)
    # original_dir = Path.cwd()
    # os.chdir(temp_model_dir)
    # model.export_to_model_xml()
    
    # particle_directions = []
    # energies = []
    
    # try:
    #     openmc.lib.init()
    #     particles = openmc.lib.sample_external_source(n_samples=n_iterations_per_batch, 
    #                                                   prn_seed=batch_seed)
        
    #     for particle in particles:
    #         u, v, w = particle.u
    #         particle_directions.append((u, v, w))
    #         energies.append(particle.E)
        
    #     print(f"✓ Sampled {len(particle_directions):,} particles from source")
        
    # except Exception as e:
    #     print(f"✗ Error sampling particles: {e}")
    # finally:
    #     try:
    #         openmc.lib.finalize()
    #     except:
    #         pass
    #     os.chdir(original_dir)


    # # Plot source directions
    # if output_dir:
    #     plot_source_directions(particle_directions, energies, output_dir=output_dir, iteration=iteration)
    #     print(f"✓ Saved source direction plots")

    

    
    # # Run isotropic source simulation
    # print(f"\n✓ Creating experiment model with isotropic source")
    # from experiment_model import create_experiment_model
    
    # model = create_experiment_model(source=isotropic_source)
    
    # directory = Path(__file__).parent.parent / 'neutronics' / f'isotropic_source_iter{iteration}'
    # directory.mkdir(parents=True, exist_ok=True)
    
    # os.chdir(directory)
    # model.export_to_model_xml()
    
    # print(f"✓ Running OpenMC simulation for iteration {iteration}")
    # model.run(threads=14)
    # print(f"✓ Simulation complete")
    
    # os.chdir(original_dir)
    
    # # Summary statistics
    # print(f"\n{'='*80}")
    # print(f"Iteration {iteration} Summary:")
    # print(f"{'='*80}")
    # print(f"Neutron rate GP log-likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.3f}")
    # print(f"Energy fit reduced χ²: {reduced_chi_square:.3f}")
    # print(f"Number of particles sampled: {len(particle_directions):,}")
    # print(f"{'='*80}\n")
    
    # return {
    #     'iteration': iteration,
    #     'gp_log_likelihood': gp.log_marginal_likelihood(gp.kernel_.theta),
    #     'reduced_chi_square': reduced_chi_square,
    #     'E_0': E_0_fit,
    #     'E_1': E_1_fit,
    #     'E_2': E_2_fit,
    #     'n_particles': len(particle_directions)
    # }


def main(n_iterations=2):
    """Main loop for running the full analysis."""
    # Setup paths
    script_dir = Path(__file__).parent
    experiment_model_path = script_dir / '../neutronics/experiment_model.py'
    foil_nb_path = script_dir / 'foils' / 'foil_analysis_NaI.ipynb'
    diamond_nb_path = script_dir / 'diamond' / 'diamond_analysis.ipynb'
    output_dir = script_dir / 'loop_output'
    isotropic_simulations_dir = script_dir / '../neutronics/isotropic_source'
    output_dir.mkdir(exist_ok=True)

    sys.path.append(str(experiment_model_path.parent))
    import experiment_model  # Ensure experiment_model is importable
    
    
    print(f"\n{'#'*80}")
    print(f"# Starting Analysis Loop - {n_iterations} iterations")
    print(f"{'#'*80}\n")
    
    results = []
    
    for i in range(n_iterations):
        iteration_start = time.time()
        
        print(f"\n{'*'*80}")
        print(f"* ITERATION {i+1}/{n_iterations}")
        print(f"{'*'*80}\n")

        home_dir = os.getcwd()
        print(f'Current Directory: {home_dir}')
        openmc_model = experiment_model.create_experiment_model(num_particles_per_batch=1e4)
        os.chdir(isotropic_simulations_dir)
        openmc_model.export_to_model_xml()

        print(f"✓ Running OpenMC simulation for iteration {i}")
        openmc_model.plot_geometry()
        openmc_model.run(threads=14)
        print(f"✓ Simulation complete")

        print(f'Changing back to script directory: {script_dir}')
        os.chdir(home_dir)
        print(f'Current Directory after simulation: {os.getcwd()}')

        # Step 1: Run foil analysis notebook
        if not run_notebook(foil_nb_path):
            print(f"Warning: Foil analysis notebook failed for iteration {i}")
        
        # Step 2: Run diamond analysis notebook
        if not run_notebook(diamond_nb_path):
            print(f"Warning: Diamond analysis notebook failed for iteration {i}")
        
        # Step 3: Perform full analysis
        try:
            result = perform_full_analysis(iteration=i, output_dir=output_dir)
            results.append(result)
        except Exception as e:
            print(f"✗ Error in full analysis for iteration {i}: {e}")
            import traceback
            traceback.print_exc()
        
        iteration_time = time.time() - iteration_start
        print(f"\n⏱  Iteration {i+1} completed in {iteration_time:.1f} seconds")
    
    # Save results summary
    results_file = output_dir / 'analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'#'*80}")
    print(f"# Analysis Loop Complete!")
    print(f"{'#'*80}")
    print(f"Results saved to: {results_file}")
    print(f"Plots saved to: {output_dir}")
    print(f"\nFinal Results Summary:")
    print(f"{'='*80}")
    for r in results:
        print(f"Iteration {r['iteration']}: χ²/dof = {r['reduced_chi_square']:.3f}, "
              f"E_0 = {r['E_0']:.5f} MeV")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # main()
    perform_full_analysis()
