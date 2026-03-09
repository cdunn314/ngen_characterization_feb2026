import numpy as np
import openmc
import openmc.lib
import sys
sys.path.append('../')
from neutron_source import nGen_generator
from matplotlib import pyplot as plt

batch_seed = 12344
n_iterations_per_batch = 1000

verbose = False

neutron_source = nGen_generator(
    energies = [13.0e6, 14.0e6, 15.0e6, 16.0e6],
    angles = [0, 45, 90, 180],
    spectra = [[1.0, 0.0, 0.0, 0.0],
               [0.0, 0.5, 0.0, 0.0],
               [0.0, 0.0, 0.1, 0.0],
               [0.0, 0.0, 0.0, 0.01]]
)

# create super basic model
sphere = openmc.Sphere(r=100.0, boundary_type='vacuum')
cell = openmc.Cell(region=-sphere, fill=None)
geometry = openmc.Geometry(openmc.Universe(cells=[cell]))
materials = openmc.Materials()
settings = openmc.Settings()
settings.run_mode = 'fixed source'
settings.source = neutron_source
settings.batches = 100
settings.inactive = 0
settings.particles = n_iterations_per_batch
model = openmc.Model(geometry=geometry, settings=settings, materials=materials)
model.export_to_model_xml()

batch_num = 0

particle_directions = []
energies = []

try:
    # Initialize OpenMC for this batch
    openmc.lib.init()
    particles = openmc.lib.sample_external_source(
        n_samples=n_iterations_per_batch, prn_seed=batch_seed)
    
    # Process particles from this batch
    batch_counts = 0
    for particle in particles:
        u,v,w = particle.u
        particle_directions.append((u,v,w))
        energies.append(particle.E)
        if verbose:
            print(f"Particle direction: u={u}, v={v}, w={w}, energy={particle.E}")
        batch_counts += 1

    if verbose:
        print(f"  Particles processed in batch: {batch_counts:,}")
    
except Exception as e:
    print(f"Error in batch {batch_num + 1}: {e}")
finally:
    # Clean up OpenMC for this batch
    try:
        openmc.lib.finalize()
    except:
        pass  # In case finalize fails

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# create colormap for energies
norm = plt.Normalize(min(energies), max(energies))
cmap = plt.get_cmap('viridis')

for direction, energy in zip(particle_directions, energies):
    u, v, w = direction
    color = cmap(norm(energy))
    axes[0].plot([0,u], [0,v], '-', color=color, alpha=0.3)
    axes[1].plot([0,v], [0,w], '-', color=color, alpha=0.3)
    axes[2].plot([0,u], [0,w], '-', color=color, alpha=0.3)
axes[0].set_title('U vs V')
axes[0].set_xlabel('U')
axes[0].set_ylabel('V')
axes[1].set_title('V vs W')
axes[1].set_xlabel('V')
axes[1].set_ylabel('W')
axes[2].set_title('U vs W')
axes[2].set_xlabel('U')
axes[2].set_ylabel('W')

# add colorbar outside of plots
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
# Make room for colorbar
plt.tight_layout()
fig.subplots_adjust(bottom=0.20)
# Add colorbar spanning full width
cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.03])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Neutron Energy (eV)')
plt.show()
        