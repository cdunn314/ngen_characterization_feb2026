import openmc
from matplotlib import pyplot as plt
from pathlib import Path

file_path = Path(__file__).parent.resolve()
def process_irdff(nuclide: str):
    if nuclide=='Zr90':
        path = file_path / 'IRDFF-II/dos-irdff2-4025.acef'
    elif nuclide=='Nb93':
        path = file_path / 'IRDFF-II/dos-irdff2-4125.acef'
    elif nuclide=='In113':
        path = file_path / 'IRDFF-II/dos-irdff2-4930.acef'
    elif nuclide=='In115':
        path = file_path / 'IRDFF-II/dos-irdff2-4931.acef'
    elif nuclide=='Ni58':
        path = file_path / 'IRDFF-II/dos-irdff2-2825.acef'
    elif nuclide=='Fe56':
        path = file_path / 'IRDFF-II/dos-irdff2-2631.acef'
    elif nuclide=='Mo92':
        path = file_path / 'IRDFF-II/dos-irdff2-4225.acef'
    elif nuclide=='Ti46':
        path = file_path / 'IRDFF-II/dos-irdff2-2225.acef'
    elif nuclide=='Ti47':
        path = file_path / 'IRDFF-II/dos-irdff2-2228.acef'
    elif nuclide=='Ti48':
        path = file_path / 'IRDFF-II/dos-irdff2-2231.acef'
    elif nuclide=='Cu63':
        path = file_path / 'IRDFF-II/dos-irdff2-2925.acef'
    elif nuclide=='Cu65':
        path = file_path / 'IRDFF-II/dos-irdff2-2931.acef'
    elif nuclide=='Al27':
        path = file_path / 'IRDFF-II/dos-irdff2-1325.acef'
    else:
        raise ValueError(f"Nuclide {nuclide} not supported in IRDFF processing.")
    
    # From Paul Romano's jupyter notebook on processing IRDFF data
    # https://nbviewer.org/gist/paulromano/842b1df9eb2003747e2fd6d95514129a

    # Get ACE file and assign variables for NXS, JXS, and XSS arrays
    ace_table = openmc.data.ace.get_table(path)
    nxs = ace_table.nxs
    jxs = ace_table.jxs
    xss = ace_table.xss

    # Get MT values and locators for cross section blocks
    lmt = jxs[3]
    nmt = nxs[4]
    lxs = jxs[6]
    mts = xss[lmt : lmt+nmt].astype(int)
    locators = xss[lxs : lxs+nmt].astype(int)

    cross_sections = {}
    for mt, loca in zip(mts, locators):
        # Determine starting index on energy grid
        nr = int(xss[jxs[7] + loca - 1])
        if nr == 0:
            breakpoints = None
            interpolation = None
        else:
            breakpoints = xss[jxs[7] + loca : jxs[7] + loca + nr].astype(int)
            interpolation = xss[jxs[7] + loca + nr : jxs[7] + loca + 2*nr].astype(int)

        # Determine number of energies in reaction
        ne = int(xss[jxs[7] + loca + 2*nr])

        # Read reaction cross section
        start = jxs[7] + loca + 1 + 2*nr
        energy = xss[start : start + ne] * 1e6
        xs = xss[start + ne : start + 2*ne]

        cross_sections[mt] = openmc.data.Tabulated1D(energy, xs, breakpoints, interpolation)
    
    return cross_sections


if __name__ == "__main__":
    zrsig = process_irdff('Zr90')
    nbsig = process_irdff('Nb93')

    plt.figure()
    plt.plot(zrsig[16].x, zrsig[16].y, label='Zr-90 (n,2n) Zr-89m')
    plt.plot(nbsig[11016].x, nbsig[11016].y, label='Nb-93 (n,2n) Nb-92m')
    plt.xlim(6e6, 16e6)
    plt.xlabel('Neutron Energy (eV)')
    plt.ylabel('Cross Section (barns)')
    plt.title('IRDFF (n,2n) Cross Sections')
    plt.legend()
    plt.grid(True)
    plt.show()