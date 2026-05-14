import openmc
from experiment_model import create_experiment_model
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt

curr_dir = os.getcwd()
energies = np.arange(13.0, 15.6, 0.1) * 1e6 # energies from 13 MeV to 15 MeV in 0.1 MeV increments

energy_response_directory = Path("energy_response_diamond")
energy_response_directory.mkdir(exist_ok=True)

nrows = int(np.floor(np.sqrt(len(energies))))
ncols = int(np.ceil(len(energies) / nrows))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))
axes = axes.flatten()

fig2, ax2 = plt.subplots()
# create a colormap for the different energies
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(energies.min(), energies.max())

for i, energy in enumerate(energies):


    directory = energy_response_directory / f"energy_{energy/1e3:.0f}keV"
    directory.mkdir(exist_ok=True)

    source_center = [507.25, 217, 10.16 + 90.5 + (11/16*2.54) + 3 * 2.54 + 20]
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point(source_center)
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Discrete([energy], [1.0])
    source.particle = 'neutron'

    # build OpenMC model with this source and run it
    model, foil_cell_volumes = create_experiment_model(
        read_from_json=True,
        irdff_energy_groups=np.array([0, 16e6]), # single energy group for total cross-section
        source=source,
        source_center=source_center,
        dd_dt_ratio=0.0,
        diamond_detector_distance=14.1, # 14.1 cm from source to detector face, not to diamond face
        num_particles_per_batch=int(1e6)
    )

    os.chdir(directory)
    if not Path('model.xml').exists():
        model.export_to_model_xml()
    if not Path('statepoint.100.h5').exists():
        model.run(threads=14)

    # get diamond spectrum
    with openmc.StatePoint('statepoint.100.h5') as sp:
        spectrum_tally = sp.get_tally(name='spectrum tally')

    tally_n_alpha_rates = spectrum_tally.get_reshaped_data(value='mean').squeeze()[:,:,1]
    simulation_n_alpha_spectra = {}
    cells = model.geometry.get_all_material_cells()
    tally_cell_ids = spectrum_tally.find_filter(openmc.CellFilter).bins
    sim_energy_bins = spectrum_tally.find_filter(openmc.EnergyFilter).bins
    sim_energy_bins = np.append(sim_energy_bins[:,0], sim_energy_bins[-1,1])

    color = cmap(norm(energy))
    for j,tally_cell_id in enumerate(tally_cell_ids):
        cell = cells[tally_cell_id]
        cell_angle = int(cell.name[len("Diamond_detector_"):-len("deg")])
        if cell_angle == 90:
            print(f'Tally cell ID: {tally_cell_id}, Cell angle: {cell_angle}')
            simulation_n_alpha_spectra[cell_angle] = tally_n_alpha_rates[i, :]

            ax2.stairs(simulation_n_alpha_spectra[cell_angle], sim_energy_bins/1e6, color=color)
            ax2.set_xlabel('Energy (MeV)')
            ax2.set_ylabel('Counts')
            ax2.set_title(f'Angle: {cell_angle}')
            ax2.set_yscale('log')

            axes[i].stairs(simulation_n_alpha_spectra[cell_angle], sim_energy_bins/1e6)
            axes[i].set_xlabel('Energy (MeV)')
            axes[i].set_ylabel('Counts')
            axes[i].set_title(f'Angle: {cell_angle} deg, Energy: {energy/1e6:.1f} MeV')
            axes[i].set_xlim(energy*0.95/1e6, energy*1.05/1e6)
            axes[i].set_yscale('log')
    # plt.show()


    os.chdir(curr_dir)
plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Incident Neutron Energy (MeV)',
             ax=ax2)
fig.tight_layout()
fig2.tight_layout()
fig2.savefig("diamond_energy_response_spectra.png", dpi=300)

plt.show()

    

     


