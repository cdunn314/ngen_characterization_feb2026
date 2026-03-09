# building the MIT-VaultLab neutron generator
# angular and energy distribution

from pathlib import Path
from collections.abc import Iterable
import pandas as pd
import numpy as np

try:
    import h5py
    import openmc
    from openmc import IndependentSource
except ModuleNotFoundError:
    pass


def nGen_generator(
    neutron_spectra: dict,
    center=(0, 0, 0), reference_uvw=(0, 0, 1)
) -> "Iterable[IndependentSource]":
    """
    Builds the StarFire nGen neutron generator in OpenMC
    with data tabulated from John Ball and Shon Mackie characterization
    via diamond detectors

    Parameters
    ----------
    center : tuple, optional
        coordinate position of the source (it is a point source),
        by default (0, 0, 0)
    reference_uvw : tuple, optional
        direction for the polar angle (tuple or list of versors)
    it is the same for the openmc.PolarAzimuthal class
    more specifically, polar angle = 0 is the direction of the D accelerator
    towards the Zr-T target, by default (0, 0, 1)

    Returns
    -------
        list of openmc neutron sources with angular and energy distribution
        and total strength of 1
    """
    # polar bins
    angles = []
    yields = []
    spectra = []
    energies = []
    for name, data in neutron_spectra.items():
        angle = data['angle']
        angles.append(angle)
        yield_ = data['spectrum'].sum()
        yields.append(yield_)
        spectra.append(data['spectrum'])
        energies.append(data['energy_bins'])

    spectra = np.array(spectra)
    energies = np.array(energies)
    print("angles:", angles)

    # sort by angle
    sort_indices = np.argsort(angles)
    angles = np.array(angles)[sort_indices]
    yields = np.array(yields)[sort_indices]
    spectra = spectra[sort_indices]
    energies = energies[sort_indices]

    pbins = np.cos([np.deg2rad(float(a)) for a in angles] + [np.pi])

    print("pbins:", pbins)


    # yield values for strengths
    yields = np.array(yields) * np.diff(pbins)
    print("yields shape: ", yields.shape)
    yields /= np.sum(yields)
    print("yields:", yields)

    # azimuthal values
    phi = openmc.stats.Uniform(a=0, b=2 * np.pi)

    all_sources = []
    for i, angle in enumerate(pbins[:-1]):


        # cos(polar angle) distribution
        mu = openmc.stats.Uniform(a=pbins[i + 1], b=pbins[i])

        space = openmc.stats.Point(center)
        angle = openmc.stats.PolarAzimuthal(mu=mu, phi=phi, reference_uvw=reference_uvw)
        energy = openmc.stats.Tabular(
            energies[i], spectra[i], interpolation="linear-linear"
        )
        strength = yields[i]

        my_source = openmc.IndependentSource(
            space=space,
            angle=angle,
            energy=energy,
            strength=strength,
            particle="neutron",
        )

        all_sources.append(my_source)

    return all_sources