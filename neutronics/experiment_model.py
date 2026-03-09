import openmc
from openmc.model import RectangularParallelepiped as RPP
import json
import numpy as np
from process_irdff import process_irdff
from pathlib import Path
import os

niobium = openmc.Material(name='Niobium')
niobium.add_element('Nb', 1.0)
niobium.set_density('g/cm3', 8.57)

zirconium = openmc.Material(name='Zirconium')
zirconium.add_element('Zr', 1.0)
zirconium.set_density('g/cm3', 6.52)

indium = openmc.Material(name='Indium')
indium.add_element('In', 1.0)
indium.set_density('g/cm3', 7.31)

nickel = openmc.Material(name='Nickel')
nickel.add_element('Ni', 1.0)
nickel.set_density('g/cm3', 8.907)

iron = openmc.Material(name='Iron')
iron.add_element('Fe', 1.0)
iron.set_density('g/cm3', 7.874)

molybdenum = openmc.Material(name='Molybdenum')
molybdenum.add_element('Mo', 1.0)
molybdenum.set_density('g/cm3', 10.22)

copper = openmc.Material(name='Copper')
copper.add_element('Cu', 1.0)
copper.set_density('g/cm3', 8.94)

titanium = openmc.Material(name='Titanium')
titanium.add_element('Ti', 1.0)
titanium.set_density('g/cm3', 4.502)

aluminum = openmc.Material(name='Aluminum')
aluminum.add_element('Al', 1.0)
aluminum.set_density('g/cm3', 2.70)

# density from https://ddk.com/cvd-diamond/general-properties-of-cvd-diamond-2/
diamond = openmc.Material(name='Diamond')
diamond.add_element('C', 1.0)
diamond.set_density('g/cm3', 3.52)

file_path = Path(__file__).parent.resolve()
print(file_path)

NUCLIDES = {niobium:'Nb93', zirconium:'Zr90', indium:'In115', nickel:'Ni58', 
                iron:'Fe56', copper:'Cu65', titanium:'Ti48', molybdenum:'Mo92',
                aluminum:'Al27'}

NUCLIDES_BY_NAME = {niobium.name:'Nb93', zirconium.name:'Zr90', indium.name:'In115', nickel.name:'Ni58', 
                iron.name:'Fe56', copper.name:'Cu65', titanium.name:'Ti48', molybdenum.name:'Mo92',
                aluminum.name:'Al27'}
MT_NUMBERS = {niobium:{'Nb93(n,2n)Nb92m':11016}, 
                zirconium:{'Zr90(n,2n)Zr89':16}, 
                indium:{"In115(n,n')In115m":11004,
                        'In115(n,2n)In114m':11016,
                        'In115(n,gamma)In116m':11102}, 
                nickel:{'Ni58(n,p)Co58':103},
                iron:{'Fe56(n,p)Mn56':103},
                copper:{'Cu65(n,2n)Cu64':16},
                titanium:{'Ti48(n,p)Sc48':103},
                molybdenum:{'Mo92(n,p)Nb92m':11103},
                aluminum:{'Al27(n,p)Mg27':103,
                        'Al27(n,alpha)Na24':107}
}

ELEMENT_NAMES = {
    'Nb': "Niobium",
    'Zr': "Zirconium",
    'In': "Indium",
    'Ni': "Nickel",
    'Fe': "Iron",
    'Cu': "Copper",
    'Ti': "Titanium",
    'Mo': "Molybdenum",
    'Al': "Aluminum"
}

ELEMENT_SYMBOLS = {
    'Niobium': "Nb",
    'Zirconium': "Zr",
    'Indium': "In",
    'Nickel': "Ni",
    'Iron': "Fe",
    'Copper': "Cu",
    'Titanium': "Ti",
    'Molybdenum': "Mo",
    'Aluminum': "Al"
}

SYMBOLS_TO_MATERIALS = {
    "Nb": niobium,
    "Zr": zirconium,
    "In": indium,
    "Ni": nickel,
    "Fe": iron,
    "Cu": copper,
    "Ti": titanium,
    "Mo": molybdenum,
    "Al": aluminum
}

def create_foils(foil_angles, foil_distance, source_center,
                 foil_dict_list):
    # Foil dictlist should be of the form:
    # [{'name': 'Nb', 'material': nb_mat, 'thickness': 0.1}, ...]
    # where the first entry is the foil closest to the ring (furthest from source)
    distance_from_ring = 0.1
    foil_regions = []
    foil_cells = []
    for foil_dict in foil_dict_list:
        foil_cylinder = openmc.XCylinder(r=0.25*2.54,
                                         y0=source_center[1],
                                         z0=source_center[2])
        foil_front_plane = openmc.XPlane(x0=source_center[0] - foil_distance + distance_from_ring)
        foil_back_plane = openmc.XPlane(
            x0=source_center[0] - foil_distance + foil_dict['thickness'] + distance_from_ring
        )
        foil_region = -foil_cylinder & +foil_front_plane & -foil_back_plane
        for angle in foil_angles:
            if angle==90 and foil_dict['name']=='Zirconium':
                print("Creating Zr foil at 90 degrees. Check that this is correct since it is very close to the source.")
            rotated_foil_region = foil_region.rotate((0,0,angle),
                                                     pivot=source_center,
                                                     inplace=False)
            foil_cell = openmc.Cell(region=rotated_foil_region,
                                    fill=foil_dict['material'],
                                    name=f"{foil_dict['name']}_foil_{angle}deg")
            foil_cells.append(foil_cell)
            foil_regions.append(rotated_foil_region)
        distance_from_ring += foil_dict['thickness']

    return foil_cells, foil_regions

def create_diamond(diamond_angles, diamond_distances, source_center):
    diamond_cells = []
    diamond_regions = []
    for diamond_angle, diamond_distance in zip(diamond_angles, diamond_distances):
        diamond_rpp = RPP(
            source_center[0] - diamond_distance - 0.05,
            source_center[0] - diamond_distance,
            source_center[1] - 0.2,
            source_center[1] + 0.2,
            source_center[2] - 0.4,
            source_center[2] + 0.4
        )

        diamond_region = -diamond_rpp
        rotated_diamond_region = diamond_region.rotate((0,0,diamond_angle),
                                                    pivot=source_center,
                                                    inplace=False)
        diamond_cell = openmc.Cell(region=rotated_diamond_region,
                                fill=diamond,
                                name=f"Diamond_detector_{diamond_angle:.0f}deg")
        diamond_cells.append(diamond_cell)
        diamond_regions.append(rotated_diamond_region)
    return diamond_cells, diamond_regions


def get_foil_angles_from_json(json_path=file_path / '../data/general.json'):
    with open(json_path, 'r') as f:
        general_data = json.load(f)
    foil_list = general_data["neutron_detection"]["foils"]
    foil_angles = []
    for detector_dict in foil_list:
        for foil_dict in detector_dict["materials"]:
            foil_angles.append(foil_dict["angle"])
    # remove repeats
    foil_angles = list(set(foil_angles))
    foil_angles.sort()
    return foil_angles


def get_foil_info_from_json(json_path=file_path / '../data/general.json'):
    with open(json_path, 'r') as f:
        general_data = json.load(f)
    big_foil_list = general_data["neutron_detection"]["foils"]
    all_foil_dict = {}
    foil_angles = []
    foil_distances = []
    packet_positions = {}

    for detector_dict in big_foil_list:
        print(f"Processing detector dict from {detector_dict['detector_type']} data directory: {detector_dict['data_directory']}")
        for foil_dict in detector_dict["materials"]:
            angle = foil_dict["angle"]
            if angle not in all_foil_dict:
                all_foil_dict[angle] = []
                packet_positions[angle] = []
            new_foil_dict = {
                "name": foil_dict["designator"],
                "material": foil_dict["material"],
                "packet_position": foil_dict["packet_position"]
            }

            distance_dict = foil_dict["distance_to_source"]
            if distance_dict["unit"] == "cm":
                new_foil_dict["distance_to_source"] = distance_dict["value"]
            else:
                raise ValueError(f"Distance unit {distance_dict['unit']} not supported.")
            
            mass_dict = foil_dict["mass"]
            if mass_dict['unit'] == 'g':
                new_foil_dict["mass"] = mass_dict["value"]
            else:
                raise ValueError(f"Mass unit {mass_dict['unit']} not supported.")
            
            thickness_dict = foil_dict["thickness"]
            if thickness_dict['unit'] == 'cm':
                new_foil_dict["thickness"] = thickness_dict["value"]
            elif thickness_dict['unit'] == 'in' or thickness_dict['unit'] == 'inch' or thickness_dict['unit'] == '"':
                new_foil_dict["thickness"] = thickness_dict["value"] * 2.54
            else:
                raise ValueError(f"Thickness unit {thickness_dict['unit']} not supported.")
            
            all_foil_dict[angle].append(new_foil_dict)
            if isinstance(new_foil_dict["packet_position"], int):
                packet_positions[angle].append(foil_dict["packet_position"])
            elif isinstance(new_foil_dict["packet_position"], list) and len(new_foil_dict["packet_position"]) == 2:
                if not isinstance(packet_positions[angle], dict):
                    packet_positions[angle] = {}
                vertical_position = new_foil_dict["packet_position"][1]
                if vertical_position not in packet_positions[angle]:
                    packet_positions[angle][vertical_position] = []
                packet_positions[angle][vertical_position].append(new_foil_dict["packet_position"][0])
            else:
                print("is packet position list?", isinstance(new_foil_dict["packet_position"], list))
                raise ValueError(f"Packet position {new_foil_dict['packet_position']} not supported.")
            print(f"Added foil {new_foil_dict['name']} at angle {angle} with packet position {new_foil_dict['packet_position']} to all_foil_dict.")
    
    # organize each angle's foils by packet position
    for angle in all_foil_dict:
        if isinstance(packet_positions[angle], list):
            inds = np.argsort(packet_positions[angle])
            all_foil_dict[angle] = [all_foil_dict[angle][i] for i in inds]
        elif isinstance(packet_positions[angle], dict):
            new_foil_list = []
            for vertical_position in sorted(packet_positions[angle].keys()):
                horizontal_positions = packet_positions[angle][vertical_position]
                inds = np.argsort(horizontal_positions)
                sorted_horizontal_positions = [horizontal_positions[i] for i in inds]
                for horizontal_position in sorted_horizontal_positions:
                    for foil_dict in all_foil_dict[angle]:
                        if foil_dict["packet_position"] == [horizontal_position, vertical_position]:
                            new_foil_list.append(foil_dict)
            all_foil_dict[angle] = new_foil_list
        else:
            raise ValueError(f"Packet positions for angle {angle} are not organized as list or dict.")
    
    # Remove duplicate foils (same name) within each angle, keeping the first occurrence
    for angle in all_foil_dict:
        seen_names = set()
        unique_foils = []
        for foil_dict in all_foil_dict[angle]:
            if foil_dict["name"] not in seen_names:
                seen_names.add(foil_dict["name"])
                unique_foils.append(foil_dict)
            else:
                print(f"Removing duplicate foil {foil_dict['name']} at angle {angle}.")
        all_foil_dict[angle] = unique_foils
    
    print(f"Organized foil information from {json_path} into dictionary with angles as keys and lists of foil dictionaries as values.")
    return all_foil_dict


def create_foils_from_json(source_center, foil_distance, json_path=file_path / '../data/general.json'):
    all_foil_dict = get_foil_info_from_json(json_path)
    print("all_foil_dict:", all_foil_dict)
    foil_regions = []
    foil_cells = []
    foil_cell_names = []
    foil_cell_volumes = {}
    for angle in all_foil_dict.keys():
        distance_from_ring = 0.1
        # make dictionary to keep track of distance from ring for each packet position, 
        # since some angles have multiple foils at different vertical packet positions 
        # and we want to make sure they don't overlap
        distance_from_ring_dict = {}
        for foil_dict in all_foil_dict[angle]:
            print(foil_dict)
            if angle=='under':
                under_foil_distance = foil_dict['distance_to_source']
                # create a foil under the source, centered at the source x and y coordinates and at the specified distance from the source in z
                foil_cylinder = openmc.ZCylinder(r=0.25*2.54,
                                                x0=source_center[0],
                                                y0=source_center[1])
                foil_top_plane = openmc.ZPlane(z0=source_center[2] - under_foil_distance + foil_dict['thickness'] + distance_from_ring)
                foil_bottom_plane = openmc.ZPlane(z0=source_center[2] - under_foil_distance + distance_from_ring)
                # not actually rotated since it is under the source, but use the same variable name for consistency
                rotated_foil_region = -foil_cylinder & -foil_top_plane & +foil_bottom_plane
            else:
                if isinstance(foil_dict['packet_position'], list):
                    print("Inside list packet position case for foil:", foil_dict['name'], "at angle", angle)
                    vertical_position = foil_dict['packet_position'][1]
                    # initialize distance from ring for this vertical position if it doesn't exist yet
                    if distance_from_ring_dict.get(vertical_position) is None:
                        distance_from_ring_dict[vertical_position] = 0.1
            
                    if vertical_position == 'top':
                        z0 = source_center[2] + 0.5*2.54 + 0.1
                    elif vertical_position == 'bottom':
                        z0 = source_center[2] - 0.5*2.54 - 0.1
                    foil_cylinder = openmc.XCylinder(r=0.25*2.54,
                                                    y0=source_center[1],
                                                    z0=z0)
                    print("foil_cylinder", foil_cylinder)
                    foil_front_plane = openmc.XPlane(x0=source_center[0] - foil_distance + distance_from_ring_dict[vertical_position])
                    foil_back_plane = openmc.XPlane(
                        x0=source_center[0] - foil_distance + foil_dict['thickness'] + distance_from_ring_dict[vertical_position]
                    )
                    distance_from_ring_dict[vertical_position] += foil_dict['thickness']
                else:
                    # if there is only one vertical position for this angle, 
                    # just use the distance from ring variable and increment it for each foil at this angle
                    foil_cylinder = openmc.XCylinder(r=0.25*2.54,
                                                    y0=source_center[1],
                                                    z0=source_center[2])
                    foil_front_plane = openmc.XPlane(x0=source_center[0] - foil_distance + distance_from_ring)
                    foil_back_plane = openmc.XPlane(
                        x0=source_center[0] - foil_distance + foil_dict['thickness'] + distance_from_ring
                    )
                foil_region = -foil_cylinder & +foil_front_plane & -foil_back_plane

                rotated_foil_region = foil_region.rotate((0,0,angle),
                                                        pivot=source_center,
                                                        inplace=False)
            foil_cell = openmc.Cell(region=rotated_foil_region,
                                    fill=SYMBOLS_TO_MATERIALS[foil_dict['material']],
                                    name=f"{foil_dict['name']}_foil_{angle}deg")
            foil_cell_volumes[foil_cell.name] = foil_dict["thickness"] * np.pi * (0.25*2.54)**2
            foil_cells.append(foil_cell)
            foil_regions.append(rotated_foil_region)
            foil_cell_names.append(f"{foil_dict['name']}_foil_{angle}deg")
            distance_from_ring += foil_dict['thickness']
    
    print(f"Created foils with names: ")
    for name in foil_cell_names:
        print(name)

    return foil_cells, foil_regions, foil_cell_volumes


def get_diamond_info_from_json(json_path=file_path / '../data/general.json'):
    with open(json_path, 'r') as f:
        general_data = json.load(f)
    measurement_list = general_data["neutron_detection"]["diamond"]["measurements"]
    diamond_angles = []
    diamond_detector_distances = []
    for measurement_dict in measurement_list:
        diamond_angles.append(measurement_dict["angle"])
        distance_dict = measurement_dict["source_to_detector_distance"]

        if distance_dict["unit"] == "cm":
            diamond_detector_distances.append(distance_dict["value"])
        else:
            raise ValueError(f"Distance unit {distance_dict['unit']} not supported.")
    if len(list(set(diamond_angles))) != len(diamond_angles):
        raise ValueError("Duplicate diamond angles found in general.json.")
    return diamond_angles, diamond_detector_distances


def make_irdff_tallies(foil_cells, energy_groups=None):
    materials = []
    for cell in foil_cells:
        if cell.fill:
            materials.append(cell.fill)
    materials = list(set(materials))
    tallies = []
    for material in materials:
        nuclide = NUCLIDES[material]
        mt_dict = MT_NUMBERS[material]
        material_foil_cells = []
        for cell in foil_cells:
            if material == cell.fill:
                material_foil_cells.append(cell)
        foil_cell_filter = openmc.CellFilter(material_foil_cells)

        # Get IRDFF-II cross sections and apply them to an EnergyFunctionFilter
        cross_sections = process_irdff(nuclide)

        for mt_name, mt in mt_dict.items():
            # check that the cross section energies are in order
            true_i = 0
            for i in range(len(cross_sections[mt].x)-1):
                # if energies are out of order, raise error
                if cross_sections[mt].x[true_i] > cross_sections[mt].x[true_i+1]:
                    print(f"Energy at index {i} is out of order: {cross_sections[mt].x[true_i]} > {cross_sections[mt].x[true_i+1]}")
                    print("Cross section values around this energy:")
                    for j in range(max(0, true_i-2), min(len(cross_sections[mt].x), true_i+3)):
                        print(f"  Index {j}: Energy = {cross_sections[mt].x[j]}, Cross Section = {cross_sections[mt].y[j]}")
                    raise ValueError(f"Cross section energies for {nuclide} MT {mt} are not in ascending order.")
                # if there are duplicate energies, use second energy value (which should be the same as the first) and skip to next energy
                elif cross_sections[mt].x[true_i] == cross_sections[mt].x[true_i+1]:
                    print(f"Duplicate energy found at index {true_i}: {cross_sections[mt].x[true_i]}")
                    cross_sections[mt].x = list(cross_sections[mt].x[:true_i+1]) + list(cross_sections[mt].x[true_i+2:])
                    cross_sections[mt].y = list(cross_sections[mt].y[:true_i+1]) + list(cross_sections[mt].y[true_i+2:])
                else:
                    true_i += 1
            multiplier_filter = openmc.EnergyFunctionFilter.from_tabulated1d(cross_sections[mt])
            # multiply the cross section by the material number density, so that the reaction rate is output
            multiplier_filter.y *= material.get_nuclide_atom_densities(nuclide)[nuclide]

            tally = openmc.Tally(name=f'{material.name} {mt_name} IRDFF-II tally')
            tally.filters = [foil_cell_filter, multiplier_filter]
            if energy_groups is not None:
                energy_filter = openmc.EnergyFilter(energy_groups)
                tally.filters.append(energy_filter)
            tally.scores = ['flux']
            tallies.append(tally)
    return tallies


def get_irdff_tally_names(json_path: Path = file_path / '../data/general.json'):
    # only works if using general.json for tally info
    all_foil_dict = get_foil_info_from_json(json_path)
    
    tally_names = set()
    for angle in all_foil_dict:
        foil_elements = []
        for foil_dict in all_foil_dict[angle]:
            element_symbol = foil_dict["material"]
            material = SYMBOLS_TO_MATERIALS[element_symbol]
            for mt_name in MT_NUMBERS[material].keys():
                tally_name = f"{material.name} {mt_name} IRDFF-II tally"
                tally_names.add(tally_name)
    tally_names = list(tally_names)
    return tally_names


def get_xs_from_tallies(statepoint_path: Path, foil_cell_volumes: dict, json_path: Path = file_path / '../data/general.json'):
    """ Only works if using general.json for tally info"""
    irdff_tallies = {}
    with openmc.StatePoint(statepoint_path) as sp:
        flux_tally = sp.get_tally(name='foil flux tally')
        flux_tally_cell_ids = flux_tally.find_filter(openmc.CellFilter).bins
        geometry = sp.summary.geometry
        irdff_tally_names = get_irdff_tally_names(json_path)
        for name in irdff_tally_names:
            print("Getting tally with name:", name)
            irdff_tallies[name] = sp.get_tally(name=name)

    all_cells = geometry.get_all_cells()
    foil_xs_dict = {}
    for name, tally in irdff_tallies.items():
        foil_cell_ids = tally.find_filter(openmc.CellFilter).bins
        # find the part of the irdff_tally name inside parentheses, which should be the reaction type
        mt_name = name.split(' ')[-3]
        for c,cell_id in enumerate(foil_cell_ids):
            cell_name = all_cells[cell_id].name
            xs_key = cell_name + "_" + mt_name
            reaction_rate = tally.get_reshaped_data(value="mean").squeeze()
            if len(reaction_rate.shape) == 2:
                reaction_rate = reaction_rate[c, :]


            # get flux for that cell
            flux_index = np.where(np.array(flux_tally_cell_ids) == cell_id)[0][0]
            flux = flux_tally.get_reshaped_data(value="mean").squeeze()[flux_index, :]
            # flux *= foil_cell_volumes[cell_name]

            # get nuclide fraction
            material = all_cells[cell_id].fill
            try:
                nuclide = NUCLIDES_BY_NAME[material.name]
            except KeyError:
                print(f"Material {material} not found in NUCLIDES dictionary. Skipping cell {cell_name}.")
                print(NUCLIDES.keys())
                raise KeyError(f"Material {material} not found in NUCLIDES dictionary.")
                
            atom_density = material.get_nuclide_atom_densities(nuclide)[nuclide] / 1e-24  # convert from atoms/barn-cm to atoms/cm3

            # calculate cross section (xs)
            zero_flux_indices = np.where(flux <= 0)[0]
            if len(zero_flux_indices) > 0:
                print(f"Warning: Zero flux found for cell {cell_name} in tally {name}. Setting cross section to zero for these energy groups.")
                flux[zero_flux_indices] = 1e-30  # set to a very small number to avoid division by zero
            xs = reaction_rate / flux / atom_density
            foil_xs_dict[xs_key] = xs

    return foil_xs_dict



def create_experiment_model(foil_angles=None,
                            diamond_angles=None,
                            irdff_energy_groups=None,
                            source_center=[1000, 500, 100],
                            read_from_json=False,
                            source=None,
                            dd_dt_ratio=0.0,
                            foil_ring_inner_radius=12.0,
                            diamond_detector_distance=12.6,
                            foil_dict_list=[{'name': 'Niobium', 'material': niobium, 'thickness': 0.02*2.54},
                                            {'name': 'Zirconium', 'material': zirconium, 'thickness': 0.01*2.54}],
                            num_particles_per_batch=1e6
                            ):
    
    if foil_angles is None:
        foil_angles = []
    if diamond_angles is None:
        diamond_angles, diamond_detector_distances = get_diamond_info_from_json()
    else:
        diamond_detector_distances = [diamond_detector_distance] * len(diamond_angles)

    ngen_cylinder = openmc.XCylinder(r=2*2.54, 
                                     y0=source_center[1],
                                     z0=source_center[2])
    tip_cylinder = openmc.XCylinder(r=0.5*2.54,
                                   y0=source_center[1],
                                   z0=source_center[2])
    
    tip_front_plane = openmc.XPlane(x0=source_center[0] - 1.0)
    tip_back_plane = openmc.XPlane(x0=source_center[0] + 14.0)
    ngen_back_plane = openmc.XPlane(x0=source_center[0] + 60.0)


    foil_ring_inner_cyl = openmc.ZCylinder(r=foil_ring_inner_radius,
                                           x0=source_center[0],
                                           y0=source_center[1])
    foil_ring_outer_cyl = openmc.ZCylinder(r=foil_ring_inner_radius + 0.6,
                                           x0=source_center[0],
                                           y0=source_center[1])
    foil_ring_bottom_plane = openmc.ZPlane(z0=source_center[2] - 0.9 - 2.0)
    foil_ring_top_plane = openmc.ZPlane(z0=source_center[2] + 0.9)

    foil_ring_front_plane = openmc.XPlane(x0=source_center[0] + 10.85)
    foil_ring_back_plane = openmc.XPlane(x0=source_center[0] + 10.85 + 0.6)
    foil_ring_right_plane_1 = openmc.YPlane(y0=source_center[1] - 3.54/2)
    foil_ring_left_plane_1 = openmc.YPlane(y0=source_center[1] + 3.54/2)
    foil_ring_right_plane_2 = openmc.YPlane(y0=source_center[1] - 3.54/2 - 3.36)
    foil_ring_left_plane_2 = openmc.YPlane(y0=source_center[1] + 3.54/2 + 3.36)
    x0_plane = openmc.XPlane(x0=source_center[0])

    table_top_plane = openmc.ZPlane(z0=source_center[2] - 15.0)
    table_bottom_plane = openmc.ZPlane(z0=source_center[2] - 20.0)
    table_left_plane = openmc.YPlane(y0=source_center[1] - 30.0)
    table_right_plane = openmc.YPlane(y0=source_center[1] + 30.0)
    table_front_plane = openmc.XPlane(x0=source_center[0] - 10.0)
    table_back_plane = openmc.XPlane(x0=source_center[0] + 50.0)

    bounding_rpp = openmc.model.RectangularParallelepiped(
        xmin=source_center[0] - 100.0, xmax=source_center[0] + 100.0,
        ymin=source_center[1] - 100.0, ymax=source_center[1] + 100.0,
        zmin=source_center[2] - 100.0, zmax=source_center[2] + 100.0,
        boundary_type='vacuum'
    )

    hole_cylinder = openmc.XCylinder(r=0.5, y0=source_center[1], z0=source_center[2])

    tip_region = -tip_cylinder & +tip_front_plane & -tip_back_plane
    ngen_body_region = -ngen_cylinder & -ngen_back_plane & +tip_back_plane
    ngen_region = tip_region | ngen_body_region

    foil_ring_region_1 = +foil_ring_inner_cyl & -foil_ring_outer_cyl & \
                            +foil_ring_bottom_plane & -foil_ring_top_plane & \
                            -foil_ring_right_plane_2
    foil_ring_region_2 = +foil_ring_inner_cyl & -foil_ring_outer_cyl & \
                            +foil_ring_bottom_plane & -foil_ring_top_plane & \
                            +foil_ring_left_plane_2
    foil_ring_region_3 = +foil_ring_inner_cyl & -foil_ring_outer_cyl & \
                            +foil_ring_bottom_plane & -foil_ring_top_plane & \
                            -x0_plane & -foil_ring_left_plane_2 & +foil_ring_right_plane_2
    foil_ring_region_4 = +foil_ring_bottom_plane & -foil_ring_top_plane & \
                            +foil_ring_front_plane & -foil_ring_back_plane & \
                            +foil_ring_left_plane_1 & -foil_ring_left_plane_2
    foil_ring_region_5 = +foil_ring_bottom_plane & -foil_ring_top_plane & \
                            +foil_ring_front_plane & -foil_ring_back_plane & \
                            +foil_ring_right_plane_2 & -foil_ring_right_plane_1
    
    ring_hole_angles = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]

    ring_hole_regions = []
    for angle in ring_hole_angles:
        rotated_cyl = hole_cylinder.rotate((0,0,angle),
                                           pivot=source_center,
                                           inplace=False)
        ring_hole_region = -rotated_cyl & +foil_ring_inner_cyl & -foil_ring_outer_cyl
        ring_hole_regions.append(ring_hole_region)
    ring_hole_region = ring_hole_regions[0]
    for hole in ring_hole_regions[1:]:
        ring_hole_region = ring_hole_region | hole

    
    foil_ring_region = (foil_ring_region_1 | foil_ring_region_2 | foil_ring_region_3 | \
                        foil_ring_region_4 | foil_ring_region_5) & ~ring_hole_region

    
    table_region = -table_top_plane & +table_bottom_plane & \
                    +table_left_plane & -table_right_plane & \
                    +table_front_plane & -table_back_plane

    if read_from_json:
        foil_cells, foil_regions, foil_cell_volumes = create_foils_from_json(source_center, 
                                                          foil_ring_inner_radius, 
                                                          file_path / '../data/general.json')
    else:
        foil_cells, foil_regions = create_foils(foil_angles, foil_ring_inner_radius, source_center, foil_dict_list)
        foil_cell_volumes = None



    diamond_distances = np.array(diamond_detector_distances) + 0.16 # distance from detector silicon surface to diamond surface

    diamond_cells, diamond_regions = create_diamond(diamond_angles, diamond_distances, source_center)

    bounding_region = -bounding_rpp \
                       & ~ngen_region \
                       & ~foil_ring_region \
                       & ~table_region
    for foil_region in foil_regions:
        bounding_region = bounding_region & ~foil_region
    for diamond_region in diamond_regions:
        bounding_region = bounding_region & ~diamond_region
    
    pla = openmc.Material(name='PLA')
    pla.add_element('C', 3/9)
    pla.add_element('H', 4/9)
    pla.add_element('O', 2/9)
    pla.set_density('g/cm3', 1.24)

    # Using Wood (Southern Pine) from 
    # PNNL Materials Compendium
    # https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-15870Rev2.pdf
    # BUT NEED TO CHECK TYPE OF WOOD
    # Various densities listed for different types of wood
    wood = openmc.Material(name='Wood')
    wood.add_element('H', 0.462413)
    wood.add_element('C', 0.323396)
    wood.add_element('N', 0.002773)
    wood.add_element('O', 0.208782)
    wood.add_element('Mg', 0.000639)
    wood.add_element('S', 0.001211)
    wood.add_element('K', 0.000397)
    wood.add_element('Ca', 0.000388)
    wood.set_density('g/cm3', 0.64)


    ngen_cell = openmc.Cell(region=ngen_region,
                           fill=None,
                           name='Neutron Generator Cell')
    foil_ring_cell = openmc.Cell(region=foil_ring_region,
                           fill=pla,
                           name='Foil Ring Cell')
    table_cell = openmc.Cell(region=table_region,
                           fill=wood,
                           name='Table Cell')
    
    air_cell = openmc.Cell(region=bounding_region,
                           fill=None,
                           name='Air Cell')
    cells = [ngen_cell, foil_ring_cell, table_cell] + diamond_cells + foil_cells + [air_cell]
    # cells = [ngen_cell, foil_ring_cell, table_cell, air_cell]
    universe = openmc.Universe(cells=cells, name='Experiment Universe')
    geometry = openmc.Geometry(universe)
    geometry.remove_redundant_surfaces()

    materials = openmc.Materials([niobium, zirconium, pla, wood, diamond,
                                  indium, nickel, iron,
                                  molybdenum, copper, titanium,
                                  aluminum])
    if source is None:
        source = openmc.IndependentSource()
        source.space = openmc.stats.Point(source_center)
        source.angle = openmc.stats.Isotropic()
        if dd_dt_ratio == 0.0:
            source.energy = openmc.stats.Discrete([14.08e6], [1.0])
        elif dd_dt_ratio == 1.0:
            source.energy = openmc.stats.Discrete([2.45e6], [1.0])
        else:
            source.energy = openmc.stats.Discrete([2.45e6, 14.08e6], 
                                                [dd_dt_ratio, 1-dd_dt_ratio])
        source.particle = 'neutron'

    settings = openmc.Settings()
    settings.batches = 100
    settings.inactive = 0
    settings.particles = int(num_particles_per_batch)
    settings.run_mode = 'fixed source'
    settings.source = source


    foil_cell_filter = openmc.CellFilter(foil_cells)
    flux_tally = openmc.Tally(name='foil flux tally')
    flux_tally.filters = [foil_cell_filter]
    if np.any(irdff_energy_groups):
        flux_tally.filters.append(openmc.EnergyFilter(irdff_energy_groups))
    flux_tally.scores = ['flux']

    n2n_tally = openmc.Tally(name='n2n tally')
    n2n_tally.filters = [foil_cell_filter]
    n2n_tally.scores = ['(n,2n)']

    diamond_tally = openmc.Tally(name='diamond tally')
    diamond_cells_filter = openmc.CellFilter(diamond_cells)
    diamond_tally.filters = [diamond_cells_filter]
    diamond_tally.scores = ['flux', '(n,a)']

    irdff_tallies = make_irdff_tallies(foil_cells, energy_groups=irdff_energy_groups)

    tallies = openmc.Tallies([flux_tally, n2n_tally, diamond_tally] + irdff_tallies)
    # tallies = openmc.Tallies()

    cell_plot_colors = {
        ngen_cell: 'black',
        foil_ring_cell: 'tan',
        table_cell: 'saddlebrown',
        air_cell: 'white',
    }
    for cell in diamond_cells:
        cell_plot_colors[cell] = 'gray'
    for cell in foil_cells:
        if cell.name=='Zirconium-2_foil_90deg':
            cell_plot_colors[cell] = 'lightgreen'
        elif cell.name=='Indium-3_foil_0deg':
            cell_plot_colors[cell] = 'orange'
        elif cell.name=="Zirconium-3_foil_-90deg":
            cell_plot_colors[cell] = 'green'
        else:
            cell_plot_colors[cell] = 'brown'

    plot_colors = {
        niobium: 'blue',
        zirconium: 'green',
        indium: 'orange',
        nickel: 'purple',
        diamond: 'gray',
        pla: 'tan',
        wood: 'saddlebrown',
        copper: 'red',
        titanium: 'cyan',
        molybdenum: 'magenta',
        aluminum: 'silver',
    }
    plot_xy = openmc.Plot()
    plot_xy.basis = 'xy'
    plot_xy.origin = source_center
    plot_xy.width = (40, 40)
    plot_xy.pixels = (3000, 3000)

    plot_xy2 = openmc.Plot()
    plot_xy2.basis = 'xy'
    plot_xy2.origin = np.array(source_center) + np.array([0,0,0.5*2.54+0.1]) # shift up to be centered on the foil ring
    plot_xy2.width = (40, 40)
    plot_xy2.pixels = (3000, 3000)

    plot_xy3 = openmc.Plot()
    plot_xy3.basis = 'xy'
    plot_xy3.origin = np.array(source_center) + np.array([0,0,0.-(0.5*2.54+0.1)]) # shift up to be centered on the foil ring
    plot_xy3.width = (40, 40)
    plot_xy3.pixels = (3000, 3000)

    plot_xz = openmc.Plot()
    plot_xz.basis = 'xz'
    plot_xz.origin = source_center
    plot_xz.width = (50, 50)
    plot_xz.pixels = (4000, 4000)


    for plot in [plot_xy, plot_xy2, plot_xy3, plot_xz]:
        # plot.color_by = 'cell'
        # plot.colors = cell_plot_colors
        plot.color_by = 'material'
        plot.colors = plot_colors

    plots = openmc.Plots([plot_xy, plot_xy2, plot_xy3, plot_xz])

    model = openmc.Model(
        geometry=geometry,
        materials=materials,
        settings=settings,
        tallies=tallies,
        plots=plots
    )

    return model, foil_cell_volumes

if __name__ == "__main__":

    directory = file_path / 'experiment_model_test'
    directory.mkdir(exist_ok=True)
    os.chdir(directory)
    model = create_experiment_model(
        foil_angles=[15, 45, 90, 135],
        diamond_angles=[15, 45, 90, 135],
        foil_dict_list=[
            {'name': 'Iron', 'material': iron, 'thickness': 0.005 * 2 * 2.54},
            {'name': 'Niobium', 'material': niobium, 'thickness': 0.02*2.54},
            {'name': 'Zirconium', 'material': zirconium, 'thickness': 0.02*2.54},
            {'name': 'Indium', 'material': indium, 'thickness': 0.02*2.54},
            {'name': 'Nickel', 'material': nickel, 'thickness': 0.01 * 2 * 2.54},
            {'name': 'Copper', 'material': copper, 'thickness': 0.001 * 8},
            {'name': 'Titanium', 'material': titanium, 'thickness': 0.0127 * 2},
            {'name': 'Molybdenum', 'material': molybdenum, 'thickness': 0.0025 * 8 * 4},
            {'name': 'Aluminum', 'material': aluminum, 'thickness': 0.2},
            ],
        num_particles_per_batch=1e5,
        dd_dt_ratio=0.1,
    )
    model.export_to_model_xml(directory)
    model.plot_geometry(directory)
    model.run(threads=14)
    os.chdir(file_path)



