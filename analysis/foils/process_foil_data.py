from pathlib import Path
from libra_toolbox.neutron_detection.activation_foils.calibration import (
    CheckSource,
    ba133,
    co60,
    cs137,
    mn54,
    na22,
    ActivationFoil,
    nb93_n2n,
    zr90_n2n,
    Reaction,
    Nuclide
)
import libra_toolbox.neutron_detection.activation_foils.compass as compass
from libra_toolbox.neutron_detection.activation_foils.compass import (
    Measurement,
    CheckSourceMeasurement,
    SampleMeasurement,
)
from libra_toolbox.tritium.model import ureg
from datetime import date, datetime
import json
from zoneinfo import ZoneInfo
import copy
import numpy as np
import sys
script_dir = Path(__file__).parent.resolve()
sys.path.append(str(script_dir.parent.parent / "neutronics"))
from process_irdff import process_irdff
from experiment_model import SYMBOLS_TO_MATERIALS, MT_NUMBERS





script_path = Path(__file__).parent.resolve()

# Path to save the extracted files
output_path = Path("../../data/")
activation_foil_path = output_path

def read_foil_xs_from_processed_data(json_path='../../data/processed_data.json'):
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

# multigroup cross section data
def get_multigroup_cross_section(nuclide, mt, flux=None, energy_groups=None):
    xs = process_irdff(nuclide)
    if mt not in xs.keys():
        raise ValueError(f"MT {mt} not found in cross section data for nuclide {nuclide}. Available MTs: {xs.keys()}")
    continuous_energies = xs[mt].x
    continuous_xs = xs[mt].y
    if energy_groups is None:
        energy_groups = np.arange(1, 16, 1.0)*1e6
    # adjust continous_energies to be 0 if first energy is greater than 0, and adjust continuous_xs accordingly
    if continuous_energies[0] > energy_groups[0]:
        continuous_energies = np.insert(continuous_energies, 0, 0)
        continuous_xs = np.insert(continuous_xs, 0, 0)
    if flux is None:
        # default to flat flux spectrum, but this should be provided with OpenMC simulation
        flux = np.ones(len(continuous_xs)-1)

    multi_group_xs = np.zeros(len(energy_groups)-1)
    for i in range(len(energy_groups)-1):
        # integrate the continuous cross section over the energy group
        min_energy = energy_groups[i]
        max_energy = energy_groups[i+1]
        # print("continuous_energies:", continuous_energies)
        energy_mask = (continuous_energies >= min_energy) & (continuous_energies < max_energy)
        # print("min_energy:", min_energy, "max_energy:", max_energy)
        # print("energy_mask:", energy_mask)
        relevant_flux = flux[energy_mask[:-1]]  # flux should be defined on the same energy grid as the cross section
        relevant_xs = continuous_xs[energy_mask] 
        if len(relevant_xs) > len(relevant_flux):
            relevant_xs = relevant_xs[:-1]
        # print('relevant_flux shape:', relevant_flux.shape, 'relevant_xs shape:', relevant_xs.shape, 'energy_mask shape:', energy_mask.shape)
        numerator = np.trapezoid(relevant_flux * relevant_xs, x=continuous_energies[energy_mask])
        denominator = np.trapezoid(relevant_flux, x=continuous_energies[energy_mask])

        if denominator == 0:
            multi_group_xs[i] = 0
        else:
            multi_group_xs[i] = numerator / denominator
    multi_group_xs = multi_group_xs * 1e-24  # convert from barns to cm^2
    return multi_group_xs


def get_interpolated_cross_section(nuclide, mt, energy):
    xs = process_irdff(nuclide)
    if mt not in xs.keys():
        raise ValueError(f"MT {mt} not found in cross section data for nuclide {nuclide}. Available MTs: {xs.keys()}")
    continuous_energies = xs[mt].x
    continuous_xs = xs[mt].y
    # Interpolate the cross section at the given energy
    interpolated_xs = np.interp(energy, continuous_energies, continuous_xs)
    interpolated_xs = interpolated_xs * 1e-24  # convert from barns to cm^2
    return interpolated_xs


cu65n2n = Reaction(reactant=Nuclide("Cu65", atomic_mass=64.9278, abundance=0.3085),
                   product=Nuclide("Cu64", atomic_mass=63.9298, 
                                   energy=[511, 1345.77], intensity=[0.35 * 0.6152, 0.472*0.6152],
                                   half_life=(12.7006 * ureg.hour).to(ureg.s).magnitude),
                #    cross_section=get_multigroup_cross_section("Cu65", mt=16, flux=None)
                   cross_section=get_interpolated_cross_section("Cu65", mt=16, energy=14.08e6),
                   type="(n,2n)"
                   )

al27na = Reaction(reactant=Nuclide("Al27", atomic_mass=26.9815, abundance=1.0),
                   product=Nuclide("Na24", atomic_mass=23.99096, 
                                   energy=[1368.63], intensity=[0.9994],
                                   half_life=(14.956 * ureg.hour).to(ureg.s).magnitude),
                #    cross_section=get_multigroup_cross_section("Al27", mt=107, flux=None)
                     cross_section=get_interpolated_cross_section("Al27", mt=107, energy=14.08e6),
                     type="(n,alpha)"
                   )

in115inelastic = Reaction(reactant=Nuclide("In115", atomic_mass=114.9039, abundance=0.9572),
                   product=Nuclide("In115m", atomic_mass=114.9039, 
                                   energy=[336.24], intensity=[0.950 * 0.459],
                                   half_life=(4.486 * ureg.hour).to(ureg.s).magnitude),
                #    cross_section=get_multigroup_cross_section("In115", mt=11004, flux=None)
                    cross_section=get_interpolated_cross_section("In115", mt=11004, energy=14.08e6),
                    type="(n,n')"
                   )

in115ngamma = Reaction(reactant=Nuclide("In115", atomic_mass=114.9039, abundance=0.9572),
                     product=Nuclide("In116m", atomic_mass=115.9053, 
                                      energy=[1097.28], intensity=[0.585],
                                      half_life=(54.29 * ureg.minute).to(ureg.s).magnitude),
                 #    cross_section=get_multigroup_cross_section("In115", mt=11102, flux=None)
                      cross_section=get_interpolated_cross_section("In115", mt=11102, energy=14.08e6),
                        type="(n,gamma)"
                     )

mo92np = Reaction(reactant=Nuclide("Mo92", atomic_mass=91.9068, abundance=0.1465),
                   product=Nuclide("Nb92m", atomic_mass=91.9072, 
                                   energy=[934.44], intensity=[0.9915],
                                   half_life=(10.15 * ureg.day).to(ureg.s).magnitude),
                #    cross_section=get_multigroup_cross_section("Mo92", mt=11103, flux=None)
                    cross_section=get_interpolated_cross_section("Mo92", mt=11103, energy=14.08e6),
                    type="(n,p)"
                   )
ni58np = Reaction(reactant=Nuclide("Ni58", atomic_mass=57.9353, abundance=0.6808),
                   product=Nuclide("Co58", atomic_mass=57.9358, 
                                   energy=[511, 810.76], intensity=[0.30, 0.9945],
                                   half_life=(70.86 * ureg.day).to(ureg.s).magnitude),
                #    cross_section=get_multigroup_cross_section("Ni58", mt=103, flux=None)
                    cross_section=get_interpolated_cross_section("Ni58", mt=103, energy=14.08e6),
                    type="(n,p)"
                   )
ti48np = Reaction(reactant=Nuclide("Ti48", atomic_mass=47.9479, abundance=0.7372),
                   product=Nuclide("Sc48", atomic_mass=47.9522, 
                                   energy=[983.5, 1037.5, 1312.1], intensity=[1.00, 0.975, 1.00],
                                   half_life=(43.71 * ureg.hour).to(ureg.s).magnitude),
                #    cross_section=get_multigroup_cross_section("Ti48", mt=103, flux=None)
                    cross_section=get_interpolated_cross_section("Ti48", mt=103, energy=14.08e6),
                    type="(n,p)"
                   )

fe56np = Reaction(reactant=Nuclide("Fe56", atomic_mass=55.9349, abundance=0.91754),
                   product=Nuclide("Mn56", atomic_mass=55.9389, 
                                   energy=[846.76], intensity=[0.9885],
                                   half_life=(2.5789 * ureg.hour).to(ureg.s).magnitude),
                #    cross_section=get_multigroup_cross_section("Fe56", mt=103, flux=None)
                    cross_section=get_interpolated_cross_section("Fe56", mt=103, energy=14.08e6),
                    type="(n,p)"
                   )


elemental_reactions_dict = {
    "Zr": [zr90_n2n],
    "Nb": [nb93_n2n],
    "Cu": [cu65n2n],
    "Al": [al27na],
    "In": [in115inelastic, in115ngamma],
    "Mo": [mo92np],
    "Ni": [ni58np],
    "Ti": [ti48np],
    "Fe": [fe56np],
}

elemental_density_dict = {
    "Zr": 6.52,  # g/cm^3
    "Nb": 8.57,  # g/cm^3
    "Cu": 8.94,  # g/cm^3
    "Al": 2.70,  # g/cm^3
    "In": 7.31,   # g/cm^3
    "Mo": 10.22,  # g/cm^3
    "Ni": 8.907,  # g/cm^3
    "Ti": 4.502,   # g/cm^3
    "Fe": 7.874   # g/cm^3
}

################ Check Source Calibration Information ###################


def build_check_source_from_dict(check_source_dict: dict):
    """Build a CheckSource object from a dictionary."""
    if (check_source_dict["energies"] is not None and
          check_source_dict["intensities"] is not None and
          check_source_dict["half_life"] is not None):
        nuclide = Nuclide(
            name=check_source_dict["nuclide"],
            energy=check_source_dict["energies"],
            intensity=check_source_dict["intensities"],
            half_life=(check_source_dict["half_life"]["value"] 
                       * ureg.parse_units(check_source_dict["half_life"]["unit"])
                       ).to(ureg.s).magnitude
        ) 
    elif check_source_dict["nuclide"].lower() == "co60":
        nuclide = co60
    elif check_source_dict["nuclide"].lower() == "cs137":
        nuclide = cs137
    elif check_source_dict["nuclide"].lower() == "mn54":
        nuclide = mn54
    elif check_source_dict["nuclide"].lower() == "na22":
        nuclide = na22
    elif check_source_dict["nuclide"].lower() == "ba133":
        nuclide = ba133
    else:
        raise ValueError(
            f"Unknown nuclide: {check_source_dict['nuclide']}. "
            "Please provide a valid nuclide or energies/intensities/half_life."
        )
    activity_date = datetime.strptime(
            check_source_dict["activity"]["date"], "%Y-%m-%d")
    # Set the timezone to America/New_York
    activity_date = activity_date.replace(tzinfo=ZoneInfo("America/New_York"))
    check_source = CheckSource(
        nuclide=nuclide,
        activity=(check_source_dict["activity"]["value"] 
                  * ureg.parse_units(check_source_dict["activity"]["unit"])
                  ).to(ureg.Bq).magnitude,
        activity_date=activity_date
    )
    return check_source


def read_check_source_data_from_json(json_data: dict, measurement_directory_path: Path, key=None):
    """Read check source data from the general.json file."""
    check_source_dict = {}
    if key is not None:
        source_json_data = json_data["check_sources"][key]
    else:
        source_json_data = json_data["check_sources"]
    for check_source_name in source_json_data:
        check_source_data = source_json_data[check_source_name]
        directory = measurement_directory_path / check_source_data["directory"]
        check_source = build_check_source_from_dict(check_source_data)
        check_source_dict[check_source_name] = {
            "directory": directory,
            "check_source": check_source,
        }
    return check_source_dict


################# Background Information ###################

def read_background_data_from_json(json_data: dict, measurement_directory_path: Path, key=None):
    """Read background data from the general.json file."""
    if key is None:
        background_dir = measurement_directory_path / json_data["background_directory"]
    else:
        background_dir = measurement_directory_path / json_data["background_directory"][key]
    return background_dir




################ Foil Information ###################

def get_distance_to_source_from_dict(foil_dict: dict):
    distance_to_source_dict = foil_dict["distance_to_source"]
    # unit from string with pint
    unit = ureg.parse_units(distance_to_source_dict["unit"])
    return (distance_to_source_dict["value"] * unit).to(ureg.cm).magnitude
    

def get_mass_from_dict(foil_dict: dict):
    foil_mass = foil_dict["mass"]["value"]
    # unit from string with pint
    unit = ureg.parse_units(foil_dict["mass"]["unit"])
    return (foil_mass * unit).to(ureg.g).magnitude
    

def get_thickness_from_dict(foil_dict: dict):
    foil_thickness = foil_dict["thickness"]["value"]
    # unit from string with pint
    unit = ureg.parse_units(foil_dict["thickness"]["unit"])
    return (foil_thickness * unit).to(ureg.cm).magnitude

def get_angle_from_json(foil_dict: dict):
    angle = foil_dict.get("angle", None)
    if angle=='under':
        angle = np.nan
    return angle


def interpolate_mass_attenuation_coefficient(foil_element_symbol, energy):
    """Interpolate the mass attenuation coefficient for 
    a given foil element symbol and energy (keV)."""

    # Data from NIST XCOM database
    with open(script_path / 'photon_attenuation_data.json', 'r') as f:
        data = json.load(f)

    energies = np.array(data['elements'][foil_element_symbol]['energy'])  # MeV
    mu_rho = np.array(data['elements'][foil_element_symbol]['mu_rho'])  # cm²/g
    energies *= 1e3  # convert to keV

    # Interpolate the mass attenuation coefficient using log-log
    log_energies = np.log(energies)
    log_mu_rho = np.log(mu_rho)
    log_mass_attenuation_coefficient = np.interp(
        np.log(energy), 
        log_energies,  # energy values converted to keV
        log_mu_rho   # mass attenuation coefficient values
    )
    
    return np.exp(log_mass_attenuation_coefficient)  # in cm^2/g

def get_foil(foil_dict: dict):
    """Get information about a specific foil from the general data file.
    Args:
        json_data (dict): The loaded JSON data from the general.json file.
    Returns:
        ActivationFoil: An ActivationFoil object containing the foil's properties.
        distance_to_source (float): The distance from the foil to the neutron source in cm.
    """
    foil_element_symbol = foil_dict["material"]
    foil_designator = foil_dict.get("designator", None)
    

    # Get distance to generator
    distance_to_source = get_distance_to_source_from_dict(foil_dict)

    # Get mass
    foil_mass = get_mass_from_dict(foil_dict)

    # get foil thickness
    foil_thickness = get_thickness_from_dict(foil_dict)

    # get angle if it exists
    angle = get_angle_from_json(foil_dict)

    foil_density = elemental_density_dict.get(foil_element_symbol, None)
    if foil_density is None:
        raise ValueError(f"No density found for foil element symbol: {foil_element_symbol}")

    # Get foil name
    foil_name = foil_dict["designator"]
    if foil_name is None:
        foil_name = foil_element_symbol

    
    
    reactions = elemental_reactions_dict.get(foil_element_symbol, None)
    if reactions is None:
        raise ValueError(f"No reactions found for foil element symbol: {foil_element_symbol}")

    foils = []

    for reaction in reactions:
        foil = ActivationFoil(
            reaction=reaction,
            mass=foil_mass,
            name=foil_name + f" {reaction.reactant.name}{reaction.type}{reaction.product.name}",
            density=foil_density,
            thickness=foil_thickness,  # in cm
        )
        foil.mass_attenuation_coefficient = interpolate_mass_attenuation_coefficient(
            foil_element_symbol, reaction.product.energy[0])  # use the first gamma energy for interpolation
        foil.angle = angle
        foils.append(foil)

        print(f"Read in properties of {foil.name} foil")

    return foils, distance_to_source


def get_foil_source_dict_from_json(json_data: dict, measurement_directory_path: Path, key=None):
    """Read foil source data from the general.json file."""
    foils = json_data["materials"]
    foil_source_dict = {}
    for foil_dict in foils:
        foils_list, distance_to_source = get_foil(foil_dict)
        measurement_paths = {}
        if key is not None:
            measurement_subdirectories = foil_dict["measurement_directory"][key]
        else:
            measurement_subdirectories = foil_dict["measurement_directory"]
        for count_num, measurement_subdirectory in enumerate(measurement_subdirectories, start=1):
            measurement_paths[count_num] = (
                measurement_directory_path / measurement_subdirectory
            )
        # foil.name should be the same as the designator if it exists.
        # Otherwise is set to the element symbol. 
        for foil in foils_list:
            foil_source_dict[foil.name] = {
                "measurement_paths": measurement_paths,
                "foil": foil,
                "distance_to_source": distance_to_source,
            }
    return foil_source_dict



def get_data(download_from_raw=False, 
             data_url=None,
             check_source_dict=None,
             background_dir=None,
             foil_source_dict=None,
             h5_filename="activation_data.h5",
             detector_type="NaI"):
    with open("../../data/general.json", "r") as f:
        general_data = json.load(f)
        json_data_list = general_data["neutron_detection"]["foils"]
    
    # json_data is a list of dictionaries with foil, background and check source measurements
    # need to loop through list to find the one with the correct detector type
    detector_types = []
    for data_dict in json_data_list:
        if "detector_type" in data_dict:
            detector_types.append(data_dict["detector_type"])
        else:
            detector_types.append("NaI")  # Default detector type if not specified
    if not isinstance(detector_types, list):
        detector_type = [detector_types]
    
    if detector_type not in detector_types:
        raise ValueError(f"Detector type {detector_type} not found in general.json file. Available types: {detector_types}")
    
    # find which dictionary has the correct detector type
    json_data = None
    for data_dict in json_data_list:
        if "detector_type" in data_dict:
            if data_dict["detector_type"] == detector_type:
                json_data = data_dict
                break
    
    
    # get measurement directory path
    if isinstance(json_data["data_directory"], dict):
        if detector_type not in json_data["data_directory"].keys():
            raise ValueError(f"Detector type {detector_type} not found in data_directory of general.json file. Available types: {json_data['data_directory'].keys()}")
        measurement_directory_path = activation_foil_path / json_data["data_directory"][detector_type]
    else:
        measurement_directory_path = activation_foil_path / json_data["data_directory"]

    # get data download url
    if isinstance(data_url, str):
        pass
    elif isinstance(json_data["data_url"], dict):
        if detector_type not in json_data["data_url"].keys():
            raise ValueError(f"Detector type {detector_type} not found in data_url of general.json file. Available types: {json_data['data_url'].keys()}")
        data_url = json_data["data_url"][detector_type]
    else:
        data_url = json_data["data_url"]


    # Get the dictionaries for check sources, background, and foils
    if check_source_dict is None:
        check_source_dict = read_check_source_data_from_json(json_data, measurement_directory_path, key=None)
    if background_dir is None:
        background_dir = read_background_data_from_json(json_data, measurement_directory_path, key=None)
    if foil_source_dict is None:
        foil_source_dict = get_foil_source_dict_from_json(json_data, measurement_directory_path, key=None)
    if download_from_raw:
        # Download and extract foil data if not already done
        print(f"Checking if measurement directory exists at {measurement_directory_path}...")
        if measurement_directory_path.exists():
            print(f"Measurement directory {measurement_directory_path} already exists. Skipping download and extraction.")
        else:
            from download_raw_foil_data import download_and_extract_foil_data
            download_and_extract_foil_data(data_url, activation_foil_path, measurement_directory_path)
        # Process data
        check_source_measurements, background_meas = read_checksources_from_directory(
                                        check_source_dict, 
                                        background_dir, 
                                        detector_type=detector_type
                                        )
        foil_measurements = read_foil_measurements_from_dir(foil_source_dict, 
                                                            detector_type=detector_type)

        for measurement in check_source_measurements.values():
            measurement.detector_type = detector_type
        background_meas.detector_type = detector_type
        for foil_name in foil_measurements.keys():
            for measurement in foil_measurements[foil_name]["measurements"].values():
                measurement.detector_type = detector_type

        # save spectra to h5 for future, faster use
        print("Saving processed measurements to h5 file for future use...\n", 
                activation_foil_path,
                detector_type + '_' + h5_filename)
        save_measurements(check_source_measurements,
                        background_meas,
                        foil_measurements,
                        filepath=activation_foil_path / (detector_type + '_' + h5_filename))
    else:
        # Read measurements from h5 file
        measurements = Measurement.from_h5(activation_foil_path / (detector_type + '_' + h5_filename))
        foil_measurements = copy.deepcopy(foil_source_dict)
        check_source_measurements = {}
        # Get list of foil measurement names
        foil_measurement_names = []
        for foil_name in foil_source_dict.keys():
            for count_num in foil_source_dict[foil_name]["measurement_paths"]:
                foil_measurement_names.append(f"{foil_name} Count {count_num}")

            # Add empty measurements dictionary to foil_source_dict copy
            foil_measurements[foil_name]["measurements"] = {}
            
        for measurement in measurements:
            print(f"Processing {measurement.name} from h5 file...")
            # check if measurement is a check source measurement
            if measurement.name in check_source_dict.keys():
                # May want to change CheckSourceMeasurement in libra-toolbox to make this more seemless
                check_source_meas = CheckSourceMeasurement(measurement.name)
                check_source_meas.__dict__.update(measurement.__dict__)
                check_source_meas.check_source = check_source_dict[measurement.name]["check_source"]
                check_source_meas.detector_type = detector_type
                check_source_measurements[measurement.name] = check_source_meas
            elif measurement.name == "Background":
                background_meas = measurement
                background_meas.detector_type = detector_type
            elif measurement.name in  foil_measurement_names:
                # Extract foil name and count number from measurement name
                split_name = measurement.name.split(' ')
                count_num = int(split_name[-1])
                foil_name = " ".join(split_name[:-2])

                foil_meas = SampleMeasurement(measurement)
                foil_meas.__dict__.update(measurement.__dict__)
                foil_meas.foil = foil_source_dict[foil_name]["foil"]
                foil_meas.detector_type = detector_type
                foil_measurements[foil_name]["measurements"][count_num] = foil_meas
            else:
                print(f"Extra measurement included in h5 file: {measurement.name}")
            measurement.detector_type = detector_type   
        
    return check_source_measurements, background_meas, foil_measurements


def save_measurements(check_source_measurements,
                      background_meas,
                      foil_measurements,
                      filepath=activation_foil_path / "activation_data.h5"):
    """Save measurements to an h5 file."""
    print(f"Saving measurements to {filepath}...")
    # Ensure the directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    measurements = list(check_source_measurements.values())
    # Add background measurement to the list
    measurements.append(background_meas)
    # Add foil measurements to the list
    for foil_name in foil_measurements.keys():
        for count_num in foil_measurements[foil_name]["measurements"].keys():
            measurements.append(foil_measurements[foil_name]["measurements"][count_num])
    
    for i,measurement in enumerate(measurements):
        if i==0:
            mode = 'w'
        else:
            mode = 'a'
        measurement.to_h5(
            filename= filepath,
            mode=mode,
            spectrum_only=True
        )


def read_checksources_from_directory(
    check_source_measurements: dict, 
    background_dir: Path,
    detector_type="NaI"
):

    measurements = {}
    for name, values in check_source_measurements.items():
        print(f"Processing {name}...")
        meas = CheckSourceMeasurement.from_directory(values["directory"], name=name)
        meas.check_source = values["check_source"]
        meas.detector_type = detector_type
        measurements[name] = meas

    print(f"Processing background...")
    background_meas = Measurement.from_directory(
        background_dir,
        name="Background",
        info_file_optional=True,
    )
    background_meas.detector_type = detector_type
    return measurements, background_meas


def read_foil_measurements_from_dir(
    foil_measurements: dict,
    detector_type="NaI"
):

    for foil_name in foil_measurements.keys():
        foil_measurements[foil_name]["measurements"] = {}
        foil = foil_measurements[foil_name]["foil"]
        for count_num, measurement_path in foil_measurements[foil_name]["measurement_paths"].items():
            measurement_name = f"{foil_name} Count {count_num}"
            print(f"Processing {measurement_name}...")
            measurement = SampleMeasurement.from_directory(
                source_dir=measurement_path,
                name=measurement_name
            )
            measurement.foil = foil
            measurement.detector_type = detector_type
            foil_measurements[foil_name]["measurements"][count_num] = measurement

    return foil_measurements


# Get the irradiation schedule

with open("../../data/general.json", "r") as f:
    general_data = json.load(f)
irradiations = []
for generator in general_data["generators"]:
    if generator["enabled"] is False:
        continue
    for i, irradiation_period in enumerate(generator["periods"]):
        if i == 0:
            overall_start_time = datetime.strptime(
                irradiation_period["start"], "%m/%d/%Y %H:%M"
            )
        start_time = datetime.strptime(irradiation_period["start"], "%m/%d/%Y %H:%M")
        end_time = datetime.strptime(irradiation_period["end"], "%m/%d/%Y %H:%M")
        irradiations.append(
            {
                "t_on": (start_time - overall_start_time).total_seconds(),
                "t_off": (end_time - overall_start_time).total_seconds(),
            }
        )
time_generator_off = end_time
time_generator_off = time_generator_off.replace(tzinfo=ZoneInfo("America/New_York"))


def get_xs_from_xs_dict(foil_xs_dict, foil_name, foil):
    foil_angle = foil.angle
    if foil_angle is None or np.isnan(foil_angle):
        foil_angle = 'under'
    # get the element symbol from the foil nuclide Ex: Al27 -> Al
    foil_element_symbol = ''.join(filter(str.isalpha, foil.reaction.reactant.name))
    openmc_material = SYMBOLS_TO_MATERIALS[foil_element_symbol]
    reactions = MT_NUMBERS[openmc_material]
    reaction_xs_dict = {}
    for reaction in reactions:
        foil_designator, reaction_type = foil_name.split(' ')
        if reaction_type == reaction:
            xs_key = f"{foil_designator}_foil_{foil_angle}deg_{reaction}"
            print(f"Looking for cross section with key: {xs_key}")
            print(f"Available keys in foil_xs_dict: {list(foil_xs_dict.keys())}")
            reaction_xs_dict[reaction] = foil_xs_dict.get(xs_key, None)

    return reaction_xs_dict

def calculate_neutron_rate_from_foil(foil_measurements, 
                                     foil_name,
                                     background_meas,
                                     calibration_coeffs,
                                     efficiency_coeffs,
                                     search_width=330,
                                     irradiations=irradiations,
                                     time_generator_off=time_generator_off):
    neutron_rates = {}
    neutron_rate_errs = {}

    for count_num, measurement in foil_measurements[foil_name]["measurements"].items():

        neutron_rates[f"Count {count_num}"] = {}
        neutron_rate_errs[f"Count {count_num}"] = {}

        for detector in measurement.detectors:
            ch = detector.channel_nb

            gamma_emitted, gamma_emitted_err = measurement.get_gamma_emitted(
                background_measurement=background_meas,
                calibration_coeffs=calibration_coeffs[ch],
                efficiency_coeffs=efficiency_coeffs[ch],
                channel_nb=ch,
                search_width=search_width)

            
            neutron_rate = measurement.get_neutron_rate(
                channel_nb=ch,
                photon_counts=gamma_emitted,
                irradiations=irradiations,
                distance=foil_measurements[foil_name]["distance_to_source"],
                time_generator_off=time_generator_off,
                branching_ratio=foil_measurements[foil_name]["foil"].reaction.product.intensity
            )

            neutron_rate_err = measurement.get_neutron_rate(
                channel_nb=ch,
                photon_counts=gamma_emitted_err,
                irradiations=irradiations,
                distance=foil_measurements[foil_name]["distance_to_source"],
                time_generator_off=time_generator_off,
                branching_ratio=foil_measurements[foil_name]["foil"].reaction.product.intensity
            )
            neutron_rates[f"Count {count_num}"][ch] = neutron_rate
            neutron_rate_errs[f"Count {count_num}"][ch] = neutron_rate_err

    return neutron_rates, neutron_rate_errs


def calculate_neutron_flux_from_foil(foil_measurements, 
                                     foil_name,
                                     background_meas,
                                     calibration_coeffs,
                                     efficiency_coeffs,
                                     search_width=330,
                                     irradiations=irradiations,
                                     time_generator_off=time_generator_off,
                                     detector_efficiency=None,
                                     detector_efficiency_err=0.0,
):
    neutron_fluxes = {}
    neutron_flux_errs = {}

    for count_num, measurement in foil_measurements[foil_name]["measurements"].items():

        neutron_fluxes[f"Count {count_num}"] = {}
        neutron_flux_errs[f"Count {count_num}"] = {}

        for detector in measurement.detectors:
            ch = detector.channel_nb

            if isinstance(detector_efficiency, dict):
                det_eff = detector_efficiency[ch]
            else:
                det_eff = detector_efficiency
            if isinstance(detector_efficiency_err, dict):
                det_eff_err = detector_efficiency_err[ch]
            else:
                det_eff_err = detector_efficiency_err

            gamma_emitted, gamma_emitted_err = measurement.get_gamma_emitted(
                background_measurement=background_meas,
                calibration_coeffs=calibration_coeffs[ch],
                efficiency_coeffs=efficiency_coeffs[ch],
                channel_nb=ch,
                search_width=search_width,
                detection_efficiency=det_eff,
                detection_efficiency_err=det_eff_err)

            print(f"Gamma emitted for {foil_name} Count {count_num} Channel {ch}: {gamma_emitted} +/- {gamma_emitted_err}")
            
            neutron_flux = measurement.get_neutron_flux(
                channel_nb=ch,
                photon_counts=gamma_emitted,
                irradiations=irradiations,
                time_generator_off=time_generator_off,
                branching_ratio=foil_measurements[foil_name]["foil"].reaction.product.intensity
            )

            neutron_flux_err = measurement.get_neutron_flux(
                channel_nb=ch,
                photon_counts=gamma_emitted_err,
                irradiations=irradiations,
                time_generator_off=time_generator_off,
                branching_ratio=foil_measurements[foil_name]["foil"].reaction.product.intensity
            )
            neutron_fluxes[f"Count {count_num}"][ch] = neutron_flux
            neutron_flux_errs[f"Count {count_num}"][ch] = neutron_flux_err

    return neutron_fluxes, neutron_flux_errs


def calculate_neutron_flux_from_foil_with_xs(foil_measurements, 
                                     foil_name,
                                     background_meas,
                                     calibration_coeffs,
                                     efficiency_coeffs,
                                     search_width=330,
                                     irradiations=irradiations,
                                     time_generator_off=time_generator_off,
                                     detector_efficiency=None,
                                     detector_efficiency_err=0.0):
    neutron_fluxes = {}
    neutron_flux_errs = {}

    for count_num, measurement in foil_measurements[foil_name]["measurements"].items():

        neutron_fluxes[f"Count {count_num}"] = {}
        neutron_flux_errs[f"Count {count_num}"] = {}

        # Get cross section for the foil reaction from the processed_data.json file
        foil_xs_dict = read_foil_xs_from_processed_data()
        print('foil_xs_dict:', foil_xs_dict)
        xs_dict = get_xs_from_xs_dict(foil_xs_dict, foil_name, foil_measurements[foil_name]["foil"])
        print('xs_dict:', xs_dict)

        for reaction, xs in xs_dict.items():
            # Set the cross section for the foil reaction in the ActivationFoil object
            foil_measurements[foil_name]["foil"].reaction.cross_section = xs
            neutron_fluxes[f"Count {count_num}"][reaction] = {}
            neutron_flux_errs[f"Count {count_num}"][reaction] = {}

            for detector in measurement.detectors:
                ch = detector.channel_nb

                if isinstance(detector_efficiency, dict):
                    det_eff = detector_efficiency[ch]
                else:
                    det_eff = detector_efficiency
                if isinstance(detector_efficiency_err, dict):
                    det_eff_err = detector_efficiency_err[ch]
                else:
                    det_eff_err = detector_efficiency_err

                gamma_emitted, gamma_emitted_err = measurement.get_gamma_emitted(
                    background_measurement=background_meas,
                    calibration_coeffs=calibration_coeffs[ch],
                    efficiency_coeffs=efficiency_coeffs[ch],
                    channel_nb=ch,
                    search_width=search_width,
                    detection_efficiency=det_eff,
                    detection_efficiency_err=det_eff_err)

                print(f"Gamma emitted for {foil_name} Count {count_num} Channel {ch}: {gamma_emitted} +/- {gamma_emitted_err}")
                print("cross section for foil reaction:", foil_measurements[foil_name]["foil"].reaction.cross_section, "cm^2")
                
                neutron_flux = measurement.get_neutron_flux(
                    channel_nb=ch,
                    photon_counts=gamma_emitted,
                    irradiations=irradiations,
                    time_generator_off=time_generator_off,
                    branching_ratio=foil_measurements[foil_name]["foil"].reaction.product.intensity
                )

                neutron_flux_err = measurement.get_neutron_flux(
                    channel_nb=ch,
                    photon_counts=gamma_emitted_err,
                    irradiations=irradiations,
                    time_generator_off=time_generator_off,
                    branching_ratio=foil_measurements[foil_name]["foil"].reaction.product.intensity
                )
                neutron_fluxes[f"Count {count_num}"][reaction][ch] = neutron_flux
                neutron_flux_errs[f"Count {count_num}"][reaction][ch] = neutron_flux_err

    return neutron_fluxes, neutron_flux_errs
