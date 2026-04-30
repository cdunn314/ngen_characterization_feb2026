import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../neutronics')
from process_irdff import process_irdff

irdff_xs = process_irdff('Zr90')

fig, ax = plt.subplots()
ax.plot(irdff_xs[16].x, irdff_xs[16].y)
ax.set_xlabel('Energy (eV)')
ax.set_ylabel('Cross Section (barns)')
ax.set_xlim(1.3e7, 1.55e7)

import json
from scipy.interpolate import interp1d

angles = []
foil_xs = []
energies = []
with open('../data/processed_data.json', 'r') as f:
    processed_data = json.load(f)

angle_to_energy = interp1d(processed_data["diamond"]["angles"], processed_data["diamond"]["neutron_energies"]["values"])

for foil in processed_data["foil_cross_sections"]:
    foil_split = foil.split('_')
    if "Zirconium" in foil_split[0]:
        if "under" not in foil_split[-2]:
            angle = int(foil_split[-2][:-3])
            foil_xs.append(processed_data["foil_cross_sections"][foil])
            angles.append(angle)
            energies.append(angle_to_energy(angle))
            print(f"Angle: {angles[-1]}, Interpolated Energy: {energies[-1]}, Cross Section: {foil_xs[-1]}")


inds = np.argsort(angles)
angles = np.array(angles)[inds]
foil_xs = np.array(foil_xs)[inds]
energies = np.array(energies)[inds]

fig2, ax2 = plt.subplots()
ax2.plot(angles, foil_xs*1e24)
ax2.set_xlabel('Angle (deg)')
ax2.set_ylabel('Foil Cross Section (barn)')

fig3, ax3 = plt.subplots()
ax3.plot(irdff_xs[16].x/(1e6), irdff_xs[16].y, '-b', label='IRDFF Cross Section')
ax3.plot(energies, foil_xs*1e24, '.r', markersize=10, label='Foil Cross Section')
ax3.set_xlabel('Energy (MeV)')
ax3.set_ylabel('Cross Section (barns)')
ax3.set_xlim(12, 16)
ax3.legend()

import openmc
with openmc.StatePoint('statepoint.100.h5') as sp:
    foil_spectrum_tally = sp.get_tally(name="foil spectrum tally")
fluxes = foil_spectrum_tally.get_reshaped_data(value='mean').squeeze()
print('fluxes shape: ', fluxes.shape)

cell_ids = foil_spectrum_tally.filters[0].bins
energy_bins = foil_spectrum_tally.filters[1].bins
print(energy_bins)
plot_energy_bins = np.array(energy_bins)[:,0]
print(plot_energy_bins)
print(energy_bins[-1][-1])
plot_energy_bins = np.append(plot_energy_bins, energy_bins[-1][-1])
print(energy_bins.shape)

model = openmc.Model.from_model_xml('model.xml')
cells = model.geometry.get_all_material_cells()

nrows = int(np.floor(np.sqrt(len(cell_ids))))
ncols = int(np.ceil(np.sqrt(len(cell_ids))))
nrows=2
ncols=3
fig4, axes4 = plt.subplots(nrows=nrows, ncols=ncols, figsize=[20, 12])
axes4 = axes4.flatten()
counter = 0
for i, cell_id in enumerate(cell_ids):
    cell_name = cells[cell_id].name
    if "Zirconium" in cell_name and "under" not in cell_name:
        ax = axes4[counter]
        ax.stairs(fluxes[i,:], plot_energy_bins)
        ax.set_xlabel('Energy')
        ax.set_ylabel('Flux (n-cm/source)')
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_title(cell_name)
        ax.set_xlim(1.3e7, 1.55e7)
        counter += 1


fig5, ax5 = plt.subplots()
for i,cell_id in enumerate(cell_ids):
    cell_name = cells[cell_id].name
    if "Zirconium" in cell_name and "under" not in cell_name:
        ax5.stairs(fluxes[i,:], plot_energy_bins, label=cell_name)
ax5.set_xlabel('Energy')
ax5.set_ylabel('Flux (n-cm/source)')
# ax.set_xscale('log')
# ax.set_yscale('log')
ax5.set_xlim(1.3e7, 1.55e7)
ax5.legend()


plt.show()