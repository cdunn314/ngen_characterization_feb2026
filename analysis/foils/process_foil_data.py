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
from libra_toolbox.neutron_detection.activation_foils.explicit import get_chain
from datetime import date, datetime
import json
from zoneinfo import ZoneInfo
import copy
import numpy as np
from scipy.optimize import nnls
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
                                #    energy=[511, 810.76], intensity=[0.30, 0.9945],
                                   energy=[810.76], intensity=[0.9945],
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

def get_xs_from_xs_dict_no_foil_name(foil_xs_dict, angle, reaction):
    for name in foil_xs_dict.keys():
        name_split = name.split('_')
        reaction_i = name_split[-1]
        if reaction_i == reaction and name_split[-2] == f"{angle}deg":
            return foil_xs_dict[name]
    print(f"No cross section found for angle {angle} and reaction {reaction} in foil_xs_dict with keys: {list(foil_xs_dict.keys())}")
    return None

def calculate_neutron_rate_from_foil(foil_measurements, 
                                     foil_name,
                                     background_meas,
                                     calibration_coeffs,
                                     efficiency_coeffs,
                                     search_width=330,
                                     irradiations=irradiations,
                                     time_generator_off=time_generator_off,
                                     plot_spectra=False):
    neutron_rates = {}
    neutron_rate_errs = {}

    for count_num, measurement in foil_measurements[foil_name]["measurements"].items():

        neutron_rates[f"Count {count_num}"] = {}
        neutron_rate_errs[f"Count {count_num}"] = {}

        for detector in measurement.detectors:
            ch = detector.channel_nb
            if plot_spectra:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                hist, bin_edges = detector.get_energy_hist_background_substract(
                    background_detector=background_meas.detectors[ch],
                    bins=None,
                    live_or_real="live"
                )
                ax.stairs(hist, bin_edges, label=f"{foil_name} Count {count_num} Channel {ch}")
            else:
                ax = None
            gamma_emitted, gamma_emitted_err = measurement.get_gamma_emitted(
                background_measurement=background_meas,
                calibration_coeffs=calibration_coeffs[ch],
                efficiency_coeffs=efficiency_coeffs[ch],
                channel_nb=ch,
                search_width=search_width,
                ax=ax)

            
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
                                    plot_spectra=False
):
    neutron_fluxes = {}
    neutron_flux_errs = {}

    for count_num, measurement in foil_measurements[foil_name]["measurements"].items():

        neutron_fluxes[f"Count {count_num}"] = {}
        neutron_flux_errs[f"Count {count_num}"] = {}

        for detector in measurement.detectors:
            ch = detector.channel_nb

            if plot_spectra:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                hist, bin_edges = detector.get_energy_hist_background_substract(
                    background_detector=background_meas.detectors[ch],
                    bins=None,
                    live_or_real="live"
                )
                calibrated_bin_edges = np.polyval(calibration_coeffs[ch], bin_edges)
                ax.set_title(f"{foil_name} Count {count_num} Channel {ch}")
                ax.stairs(hist, calibrated_bin_edges, label=f"Spectrum")
            else:
                ax = None

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
                detection_efficiency_err=det_eff_err,
                ax=ax)

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

def get_correction_factor(measurement, channel_nb, irradiations, 
                                   time_generator_off, branching_ratio,
                                   total_efficiency=1.0):
    # This calculates the C_i factor from Equation 1 of 
    # "Determination of the Deuterium-Tritium (D-T) Generator Neutron Flux using 
    # Multi-foil Neutron Activation Analysis Method"
    # by D. Lee, B. Bucher, K. Krebs, E. Seabury, J. Wharton, July 2019
    # https://www.osti.gov/servlets/purl/1524045
    time_between_generator_off_and_start_of_counting = (
            measurement.start_time - time_generator_off
        ).total_seconds()

    detector = measurement.get_detector(channel_nb)

    f_time = (
        get_chain(irradiations, measurement.foil.reaction.product.decay_constant)
        * np.exp(
            -measurement.foil.reaction.product.decay_constant
            * time_between_generator_off_and_start_of_counting
        )
        * (
            1
            - np.exp(
                -measurement.foil.reaction.product.decay_constant
                * detector.real_count_time
            )
        )
        * (detector.live_count_time / detector.real_count_time)
        / measurement.foil.reaction.product.decay_constant
    )

    # Correction factor of gamma-ray self-attenuation in the foil
    if measurement.foil.thickness is None:
        f_self = 1
    else:
        f_self = (
            1
            - np.exp(
                -measurement.foil.mass_attenuation_coefficient
                * measurement.foil.density
                * measurement.foil.thickness
            )
        ) / (
            measurement.foil.mass_attenuation_coefficient
            * measurement.foil.density
            * measurement.foil.thickness
        )

    # Spectroscopic Factor to account for the branching ratio and the
    # total detection efficiency
    # total efficiency is by default 1 because efficiency is already accounted for in the get_gamma_emitted method, 
    # but can be included here if there are additional efficiency factors to consider
    # print("total efficiency:", total_efficiency)
    # print("branching ratio:", branching_ratio)
    f_spec = total_efficiency * np.array(branching_ratio)


    C = measurement.foil.nb_atoms * f_time * f_self * f_spec

    return C


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
    gamma_emitted_dict = {}
    gamma_emitted_err_dict = {}
    factors_dict = {}
    reactions_dict = {}

    for count_num, measurement in foil_measurements[foil_name]["measurements"].items():

        neutron_fluxes[f"Count {count_num}"] = {}
        neutron_flux_errs[f"Count {count_num}"] = {}
        gamma_emitted_dict[f"Count {count_num}"] = {}
        gamma_emitted_err_dict[f"Count {count_num}"] = {}
        factors_dict[f"Count {count_num}"] = {}
        reactions_dict[f"Count {count_num}"] = {}
        # Get cross section for the foil reaction from the processed_data.json file
        foil_xs_dict = read_foil_xs_from_processed_data()
        # print('foil_xs_dict:', foil_xs_dict)
        xs_dict = get_xs_from_xs_dict(foil_xs_dict, foil_name, foil_measurements[foil_name]["foil"])
        # print('xs_dict:', xs_dict)

        for reaction, xs in xs_dict.items():
            # Set the cross section for the foil reaction in the ActivationFoil object
            foil_measurements[foil_name]["foil"].reaction.cross_section = xs
            neutron_fluxes[f"Count {count_num}"][reaction] = {}
            neutron_flux_errs[f"Count {count_num}"][reaction] = {}
            gamma_emitted_dict[f"Count {count_num}"][reaction] = {}
            gamma_emitted_err_dict[f"Count {count_num}"][reaction] = {}
            factors_dict[f"Count {count_num}"][reaction] = {}
            reactions_dict[f"Count {count_num}"][reaction] = reaction

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


                # print(f"Gamma emitted for {foil_name} Count {count_num} Channel {ch}: {gamma_emitted} +/- {gamma_emitted_err}")
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

                factor = get_correction_factor(measurement, ch, irradiations, time_generator_off,
                                   branching_ratio=foil_measurements[foil_name]["foil"].reaction.product.intensity)
                
                neutron_fluxes[f"Count {count_num}"][reaction][ch] = neutron_flux
                neutron_flux_errs[f"Count {count_num}"][reaction][ch] = neutron_flux_err
                gamma_emitted_dict[f"Count {count_num}"][reaction][ch] = gamma_emitted
                gamma_emitted_err_dict[f"Count {count_num}"][reaction][ch] = gamma_emitted_err
                factors_dict[f"Count {count_num}"][reaction][ch] = factor

    return neutron_fluxes, neutron_flux_errs, gamma_emitted_dict, gamma_emitted_err_dict, factors_dict


def calculate_and_save_gamma_emitted(foil_measurements, 
                                      foil_name,
                                      background_meas,
                                      calibration_coeffs,
                                      efficiency_coeffs,
                                      search_width=330,
                                      detector_efficiency=None,
                                      detector_efficiency_err=0.0,
                                      json_path=None):
    """
    Calculate gamma emitted from foil measurements and save to processed_data.json.
    
    Similar to calculate_neutron_flux_from_foil_with_xs but only calculates 
    and saves the gamma emitted values without computing neutron flux.
    
    Parameters
    ----------
    foil_measurements : dict
        Dictionary of foil measurements with structure:
        {foil_name: {"foil": ActivationFoil, "measurements": {count_num: SampleMeasurement}}}
    foil_name : str
        Name of the foil to process
    background_meas : Measurement
        Background measurement
    calibration_coeffs : dict
        Energy calibration coefficients per channel
    efficiency_coeffs : dict
        Efficiency coefficients per channel
    search_width : float, optional
        Peak search width in keV. Default is 330.
    detector_efficiency : float or dict, optional
        Detection efficiency. If dict, keyed by channel number.
    detector_efficiency_err : float or dict, optional
        Detection efficiency uncertainty. If dict, keyed by channel number.
    json_path : Path, optional
        Path to the processed_data.json file. Defaults to ../../data/processed_data.json.
    
    Returns
    -------
    gamma_emitted_dict : dict
        Dictionary with structure (averaged across counts): 
        {foil_name: {reaction: {channel: {gamma_energy_keV: {"value": val, "error": err}}}}}
    """
    # Temporary storage for collecting values across counts
    temp_storage = {}  # {(ch, energy_key): [(value, error), ...]}
    
    foil = foil_measurements[foil_name]["foil"]
    
    # Get the reaction type from the foil
    reaction_name = foil.reaction.type if hasattr(foil.reaction, 'type') else "unknown"
    
    # Get gamma energies for this reaction
    gamma_energies = np.atleast_1d(foil.reaction.product.energy)

    for count_num, measurement in foil_measurements[foil_name]["measurements"].items():
        for detector in measurement.detectors:
            ch = detector.channel_nb

            if isinstance(detector_efficiency, dict):
                det_eff = detector_efficiency.get(ch, None)
            else:
                det_eff = detector_efficiency
            if isinstance(detector_efficiency_err, dict):
                det_eff_err = detector_efficiency_err.get(ch, 0.0)
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

            gamma_emitted = np.atleast_1d(gamma_emitted)
            gamma_emitted_err = np.atleast_1d(gamma_emitted_err)
            
            for i, gamma_energy in enumerate(gamma_energies):
                energy_key = f"{gamma_energy:.2f}_keV"
                key = (ch, energy_key)
                
                if key not in temp_storage:
                    temp_storage[key] = []
                
                val = float(gamma_emitted[i]) if i < len(gamma_emitted) else 0.0
                err = float(gamma_emitted_err[i]) if i < len(gamma_emitted_err) else 0.0
                temp_storage[key].append((val, err))

                # print(f"Gamma emitted for {foil_name} ({reaction_name}) Count {count_num} Channel {ch} "
                    #   f"@ {gamma_energy:.2f} keV: {val:.2f} +/- {err:.2f}")

    # Average across counts using inverse variance weighting
    gamma_emitted_dict = {}
    gamma_emitted_dict[foil_name] = {}
    gamma_emitted_dict[foil_name][reaction_name] = {}
    
    for (ch, energy_key), values_list in temp_storage.items():
        if ch not in gamma_emitted_dict[foil_name][reaction_name]:
            gamma_emitted_dict[foil_name][reaction_name][ch] = {}
        
        values = np.array([v[0] for v in values_list])
        errors = np.array([v[1] for v in values_list])
        
        # Weighted average (inverse variance weighting)
        # If any error is zero, use simple average
        if np.any(errors <= 0):
            avg_value = np.mean(values)
            avg_error = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        else:
            weights = 1.0 / (errors ** 2)
            avg_value = np.sum(weights * values) / np.sum(weights)
            avg_error = 1.0 / np.sqrt(np.sum(weights))
        
        gamma_emitted_dict[foil_name][reaction_name][ch][energy_key] = {
            "value": float(avg_value),
            "error": float(avg_error),
            "n_counts": len(values_list)
        }
        
        # print(f"Averaged gamma emitted for {foil_name} ({reaction_name}) Channel {ch} "
            #   f"@ {energy_key}: {avg_value:.2f} +/- {avg_error:.2f} (n={len(values_list)})")

    # Save to processed_data.json
    save_gamma_emitted_to_processed_data(gamma_emitted_dict, json_path)
    
    return gamma_emitted_dict


def calculate_and_save_all_gamma_emitted(foil_measurements,
                                          background_meas,
                                          calibration_coeffs,
                                          efficiency_coeffs,
                                          search_width=330,
                                          detector_efficiency=None,
                                          detector_efficiency_err=0.0,
                                          json_path=None):
    """
    Calculate gamma emitted for all foils and save to processed_data.json.
    
    Parameters
    ----------
    foil_measurements : dict
        Dictionary of foil measurements with structure:
        {foil_name: {"foil": ActivationFoil, "measurements": {count_num: SampleMeasurement}}}
    background_meas : Measurement
        Background measurement
    calibration_coeffs : dict
        Energy calibration coefficients per channel
    efficiency_coeffs : dict
        Efficiency coefficients per channel
    search_width : float, optional
        Peak search width in keV. Default is 330.
    detector_efficiency : float or dict, optional
        Detection efficiency. If dict, keyed by channel number.
    detector_efficiency_err : float or dict, optional
        Detection efficiency uncertainty. If dict, keyed by channel number.
    json_path : Path, optional
        Path to the processed_data.json file. Defaults to ../../data/processed_data.json.
    
    Returns
    -------
    all_gamma_emitted : dict
        Combined dictionary with all foils' gamma emitted data (averaged across counts).
        Structure: {foil_name: {reaction: {channel: {gamma_energy_keV: {"value": val, "error": err}}}}}
    """
    all_gamma_emitted = {}
    
    for foil_name in foil_measurements.keys():
        foil = foil_measurements[foil_name]["foil"]
        reaction_name = foil.reaction.type if hasattr(foil.reaction, 'type') else "unknown"
        
        # Get gamma energies for this reaction
        gamma_energies = np.atleast_1d(foil.reaction.product.energy)
        
        # Temporary storage for collecting values across counts
        temp_storage = {}  # {(ch, energy_key): [(value, error), ...]}
        
        for count_num, measurement in foil_measurements[foil_name]["measurements"].items():
            for detector in measurement.detectors:
                ch = detector.channel_nb

                if isinstance(detector_efficiency, dict):
                    det_eff = detector_efficiency.get(ch, None)
                else:
                    det_eff = detector_efficiency
                if isinstance(detector_efficiency_err, dict):
                    det_eff_err = detector_efficiency_err.get(ch, 0.0)
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

                gamma_emitted = np.atleast_1d(gamma_emitted)
                gamma_emitted_err = np.atleast_1d(gamma_emitted_err)
                
                for i, gamma_energy in enumerate(gamma_energies):
                    energy_key = f"{gamma_energy:.2f}_keV"
                    key = (ch, energy_key)
                    
                    if key not in temp_storage:
                        temp_storage[key] = []
                    
                    val = float(gamma_emitted[i]) if i < len(gamma_emitted) else 0.0
                    err = float(gamma_emitted_err[i]) if i < len(gamma_emitted_err) else 0.0
                    temp_storage[key].append((val, err))

                    # print(f"Gamma emitted for {foil_name} ({reaction_name}) Count {count_num} Channel {ch} "
                        #   f"@ {gamma_energy:.2f} keV: {val:.2f} +/- {err:.2f}")
        
        # Average across counts for this foil
        if foil_name not in all_gamma_emitted:
            all_gamma_emitted[foil_name] = {}
        all_gamma_emitted[foil_name][reaction_name] = {}
        
        for (ch, energy_key), values_list in temp_storage.items():
            if ch not in all_gamma_emitted[foil_name][reaction_name]:
                all_gamma_emitted[foil_name][reaction_name][ch] = {}
            
            values = np.array([v[0] for v in values_list])
            errors = np.array([v[1] for v in values_list])
            
            # Weighted average (inverse variance weighting)
            if np.any(errors <= 0):
                avg_value = np.mean(values)
                avg_error = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            else:
                weights = 1.0 / (errors ** 2)
                avg_value = np.sum(weights * values) / np.sum(weights)
                avg_error = 1.0 / np.sqrt(np.sum(weights))
            
            all_gamma_emitted[foil_name][reaction_name][ch][energy_key] = {
                "value": float(avg_value),
                "error": float(avg_error),
                "n_counts": len(values_list)
            }
            
            # print(f"Averaged gamma emitted for {foil_name} ({reaction_name}) Channel {ch} "
                #   f"@ {energy_key}: {avg_value:.2f} +/- {avg_error:.2f} (n={len(values_list)})")
    
    # Save all to processed_data.json
    save_gamma_emitted_to_processed_data(all_gamma_emitted, json_path)
    
    return all_gamma_emitted


def save_gamma_emitted_to_processed_data(gamma_emitted_dict, json_path=None):
    """
    Save gamma emitted data to the processed_data.json file.
    
    Parameters
    ----------
    gamma_emitted_dict : dict
        Dictionary with foil/reaction identifiers as keys and gamma emitted values as values.
        Structure: {foil_name: {reaction: {count_num: {channel: {gamma_energy_keV: {"value": val, "error": err}}}}}}
    json_path : Path, optional
        Path to the processed_data.json file. Defaults to ../../data/processed_data.json.
    """
    if json_path is None:
        json_path = Path(__file__).parent / '../../data/processed_data.json'
    
    # Load existing data
    if json_path.exists():
        with open(json_path, 'r') as f:
            processed_data = json.load(f)
    else:
        processed_data = {}
    
    # Convert numpy arrays to lists for JSON serialization
    gamma_emitted_serializable = {}
    for foil_name, reactions in gamma_emitted_dict.items():
        gamma_emitted_serializable[foil_name] = {}
        for reaction, counts in reactions.items():
            gamma_emitted_serializable[foil_name][reaction] = {}
            for count_num, channels in counts.items():
                gamma_emitted_serializable[foil_name][reaction][count_num] = {}
                for ch, energies in channels.items():
                    gamma_emitted_serializable[foil_name][reaction][count_num][str(ch)] = {}
                    for energy_key, data in energies.items():
                        gamma_emitted_serializable[foil_name][reaction][count_num][str(ch)][energy_key] = {
                            "value": float(data["value"]) if hasattr(data["value"], 'item') else data["value"],
                            "error": float(data["error"]) if hasattr(data["error"], 'item') else data["error"]
                        }
    
    # Add or update the gamma emitted data
    processed_data['gamma_emitted'] = gamma_emitted_serializable
    
    # Write back to file
    with open(json_path, 'w') as f:
        json.dump(processed_data, f, indent=4)
    
    # print(f"Saved gamma emitted data for {len(gamma_emitted_dict)} foils to {json_path}")


def read_gamma_emitted_from_processed_data(json_path=None):
    """
    Read gamma emitted data from the processed_data.json file.
    
    Parameters
    ----------
    json_path : Path, optional
        Path to the processed_data.json file. Defaults to ../../data/processed_data.json.
    
    Returns
    -------
    dict
        Dictionary with foil/reaction identifiers as keys and gamma emitted values.
    """
    if json_path is None:
        json_path = Path(__file__).parent / '../../data/processed_data.json'
    
    with open(json_path, 'r') as f:
        processed_data = json.load(f)
    
    if 'gamma_emitted' not in processed_data:
        print(f"No gamma emitted data found in {json_path}")
        return {}
    
    return processed_data['gamma_emitted']


def compute_timing_factor(foil, irradiations, time_generator_off, measurement):
    """
    Compute the timing factor (f_time) for a foil measurement.
    
    This factor accounts for:
    - Buildup during irradiation periods (via get_chain)
    - Decay between generator off and start of counting
    - Decay during counting
    - Dead time correction (live_time / real_time)
    
    Based on Equation 1 from Lee et al. DOI: 10.2172/1524045
    
    Parameters
    ----------
    foil : ActivationFoil
        The activation foil object with reaction/product info
    irradiations : list
        List of dictionaries with 't_on' and 't_off' keys for irradiation periods
    time_generator_off : datetime
        Time when the generator was turned off
    measurement : SampleMeasurement
        The measurement object containing timing info
    
    Returns
    -------
    float
        The timing factor f_time
    """
    decay_constant = foil.reaction.product.decay_constant
    
    # Time between generator off and start of counting
    t_delay = (measurement.start_time - time_generator_off).total_seconds()
    
    # Get detector timing info (use first detector)
    detector = measurement.detectors[0]
    live_time = detector.live_count_time
    real_time = detector.real_count_time
    
    # Compute f_time as in compass.py get_neutron_flux()
    f_time = (
        get_chain(irradiations, decay_constant)
        * np.exp(-decay_constant * t_delay)
        * (1 - np.exp(-decay_constant * real_time))
        * (live_time / real_time)
        / decay_constant
    )
    
    return f_time


def compute_self_attenuation_factor(foil):
    """
    Compute the gamma-ray self-attenuation correction factor for a foil.
    
    Parameters
    ----------
    foil : ActivationFoil
        The activation foil object
    
    Returns
    -------
    float
        The self-attenuation factor f_self
    """
    if foil.thickness is None:
        return 1.0
    
    mu_rho = foil.mass_attenuation_coefficient  # cm^2/g
    rho = foil.density  # g/cm^3
    t = foil.thickness  # cm
    
    exponent = mu_rho * rho * t
    if exponent < 1e-6:
        return 1.0  # Avoid numerical issues for very thin foils
    
    f_self = (1 - np.exp(-exponent)) / exponent
    return f_self


def build_response_matrix(foil_measurements_list, 
                          cross_section_dict,
                          irradiations,
                          time_generator_off,
                          energy_groups):
    """
    Build the response matrix A for the flux unfolding problem R = A * φ.
    
    Each row corresponds to a reaction measurement.
    Each column corresponds to an energy group.
    
    A_ij = N_i * σ_ij * f_time_i * f_self_i
    
    Where:
    - N_i is the number of target atoms for foil i
    - σ_ij is the cross section for reaction i in energy group j
    - f_time_i is the timing factor for measurement i
    - f_self_i is the self-attenuation factor for foil i
    
    Parameters
    ----------
    foil_measurements_list : list
        List of tuples: (foil_name, reaction_name, foil_obj, measurement_obj)
    cross_section_dict : dict
        Dictionary mapping (foil_name, reaction_name) to energy-group cross sections (array)
    irradiations : list
        List of irradiation periods
    time_generator_off : datetime
        Time when generator was turned off
    energy_groups : array
        Energy group boundaries (length n_groups + 1)
    
    Returns
    -------
    A : ndarray
        Response matrix of shape (n_reactions, n_energy_groups)
    reaction_labels : list
        Labels for each row of the matrix
    """
    n_reactions = len(foil_measurements_list)
    n_energy_groups = len(energy_groups) - 1
    
    A = np.zeros((n_reactions, n_energy_groups))
    reaction_labels = []
    
    for i, (foil_name, reaction_name, foil, measurement) in enumerate(foil_measurements_list):
        # Get cross sections for this reaction
        key = f"{foil_name}_{reaction_name}"
        if key in cross_section_dict:
            xs = np.array(cross_section_dict[key])
        else:
            print(f"Warning: Cross section not found for {key}, using zeros")
            xs = np.zeros(n_energy_groups)
        
        # Ensure xs has the right length
        if len(xs) != n_energy_groups:
            print(f"Warning: Cross section length mismatch for {key}: {len(xs)} vs {n_energy_groups}")
            if len(xs) == 1:
                # Single group - broadcast to all groups (simple approach)
                # In practice, you'd want energy-dependent cross sections
                xs = np.full(n_energy_groups, xs[0])
            else:
                xs = np.resize(xs, n_energy_groups)
        
        # Number of target atoms
        N = foil.nb_atoms
        
        # Timing factor
        f_time = compute_timing_factor(foil, irradiations, time_generator_off, measurement)
        
        # Self-attenuation factor
        f_self = compute_self_attenuation_factor(foil)
        
        # Build row of response matrix
        A[i, :] = N * xs * f_time * f_self
        
        reaction_labels.append(f"{foil_name}_{reaction_name}")
    
    return A, reaction_labels



def build_response_matrix_from_processed_data(angle, processed_data_json_filepath='../../data/processed_data.json'):

    with open(processed_data_json_filepath, 'r') as f:
        processed_data = json.load(f)

    foil_xs_dict = read_foil_xs_from_processed_data()
    responses = []
    responses_err = []
    response_matrix = []

    for detector_type in processed_data['foils']:
        for nuclide in processed_data['foils'][detector_type]:
            # print("processed_data['foils'][detector_type][nuclide].keys():", list(processed_data['foils'][detector_type][nuclide].keys()))
            if str(angle) in processed_data['foils'][detector_type][nuclide].keys():
                # if angle is present as a key without any sign change, use that
                angle_key = str(angle)
                xs_angle = angle
            elif str(-angle) in processed_data['foils'][detector_type][nuclide].keys():
                # if angle is only present as a key with a sign change, use that (e.g. -90 degrees is present but not 90 degrees for angle=90)
                angle_key = str(-angle)
                xs_angle = -angle
            else:
                continue  # skip if neither angle nor -angle is present as a key
            for reaction in processed_data['foils'][detector_type][nuclide][angle_key]['reactions']:
                # print(processed_data['foils'][detector_type][nuclide][angle_key]['reactions'][reaction])
                gamma_emitted = processed_data['foils'][detector_type][nuclide][angle_key]['reactions'][reaction]['gamma_emitted']
                gamma_emitted_err = processed_data['foils'][detector_type][nuclide][angle_key]['reactions'][reaction]['gamma_emitted_err']
                factors = processed_data['foils'][detector_type][nuclide][angle_key]['reactions'][reaction]['factors']
                number_of_measurements = processed_data['foils'][detector_type][nuclide][angle_key]['number_of_measurements']
                xs = get_xs_from_xs_dict_no_foil_name(foil_xs_dict, xs_angle, reaction)
                print("xs for reaction", reaction, "at angle", angle, "degrees:", xs, "cm^2")

                if len(gamma_emitted) == number_of_measurements:
                    # response is the number of gammas emitted divided by the correction factors,
                    # which should equal the flux times the cross section summed over all energy groups (i.e. the reaction rate)
                    response = np.array(gamma_emitted) / np.array(factors)
                    response_err = np.array(gamma_emitted_err) / np.array(factors)  # Propagate error through division assuming no error in factors for simplicity
                else:
                    pass
                    # raise ValueError(f"Length of gamma_emitted ({len(gamma_emitted)}) does not match number_of_measurements ({number_of_measurements}) for {nuclide} at angle {angle} degrees")
                
                responses.append(np.mean(response).squeeze())  # Use mean response across measurements for this reaction
                responses_err.append(np.sqrt(np.sum(response_err**2)/len(response_err)).squeeze())  # Use mean error across measurements for this reaction
                # append xs as new row in response matrix for this reaction
                response_matrix.append(xs)
    
    responses = np.array(responses)
    responses_err = np.array(responses_err)
    response_matrix = np.array(response_matrix)
    return response_matrix, responses, responses_err



def solve_flux_spectrum(R, R_err, A, energy_groups, regularization=0.0):
    """
    Solve the flux unfolding problem R = A * φ using non-negative least squares.
    
    Based on Equation 2 from INL/EXT-19-54045 (Sort_16030.pdf).
    
    Parameters
    ----------
    R : ndarray
        Measured reaction rates (number of decays measured), shape (n_reactions,)
    R_err : ndarray
        Uncertainties in R, shape (n_reactions,)
    A : ndarray
        Response matrix, shape (n_reactions, n_energy_groups)
    energy_groups : ndarray
        Energy group boundaries, length (n_energy_groups + 1)
    regularization : float, optional
        Regularization parameter (Tikhonov). Default is 0.0 (no regularization).
    
    Returns
    -------
    phi : ndarray
        Solved neutron flux spectrum, shape (n_energy_groups,)
    phi_err : ndarray
        Estimated uncertainties in phi (from residuals), shape (n_energy_groups,)
    residual : float
        Residual norm of the solution
    """
    n_reactions, n_groups = A.shape
    
    # Weight the system by measurement uncertainties
    # If R_err has zeros or very small values, use a minimum uncertainty
    weights = np.where(R_err > 0, 1.0 / R_err, 1.0)
    
    # Weighted least squares: minimize ||W(Aφ - R)||^2
    A_weighted = A * weights[:, np.newaxis]
    R_weighted = R * weights
    
    if regularization > 0:
        # Tikhonov regularization: add regularization term
        # Minimize ||W(Aφ - R)||^2 + λ||φ||^2
        # Equivalent to augmented system: [WA; sqrt(λ)I] φ = [WR; 0]
        sqrt_lambda = np.sqrt(regularization)
        A_aug = np.vstack([A_weighted, sqrt_lambda * np.eye(n_groups)])
        R_aug = np.concatenate([R_weighted, np.zeros(n_groups)])
        phi, residual = nnls(A_aug, R_aug)
    else:
        phi, residual = nnls(A_weighted, R_weighted)
    
    # Estimate uncertainties using covariance propagation
    # For overdetermined systems: Cov(φ) ≈ (A^T W^2 A)^{-1} * σ^2
    # where σ^2 is estimated from residuals
    try:
        W2 = np.diag(weights**2)
        ATA = A.T @ W2 @ A
        if regularization > 0:
            ATA += regularization * np.eye(n_groups)
        ATA_inv = np.linalg.inv(ATA)
        
        # Estimate variance from residuals
        residuals = R - A @ phi
        sigma2 = np.sum((residuals * weights)**2) / max(1, n_reactions - n_groups)
        
        phi_var = np.diag(ATA_inv) * sigma2
        phi_err = np.sqrt(np.maximum(phi_var, 0))
    except np.linalg.LinAlgError:
        print("Warning: Could not compute uncertainty estimates")
        phi_err = np.zeros(n_groups)
    
    return phi, phi_err, residual


def unfold_flux_from_measurements(foil_measurements,
                                  background_meas,
                                  calibration_coeffs,
                                  efficiency_coeffs,
                                  irradiations,
                                  time_generator_off,
                                  energy_groups,
                                  detector_efficiency=None,
                                  detector_efficiency_err=0.0,
                                  search_width=330,
                                  regularization=0.0,
                                  json_path=None):
    """
    Perform full flux unfolding from foil measurements.
    
    This function:
    1. Extracts gamma emitted from each foil measurement
    2. Saves gamma emitted data to processed_data.json
    3. Reads cross sections from processed_data.json
    4. Builds the response matrix
    5. Solves for the flux spectrum using NNLS
    
    Parameters
    ----------
    foil_measurements : dict
        Dictionary of foil measurements with structure:
        {foil_name: {"foil": ActivationFoil, "measurements": {count_num: SampleMeasurement}}}
    background_meas : Measurement
        Background measurement
    calibration_coeffs : dict
        Energy calibration coefficients per channel
    efficiency_coeffs : dict
        Efficiency coefficients per channel
    irradiations : list
        List of irradiation periods
    time_generator_off : datetime
        Time when generator was turned off
    energy_groups : ndarray
        Energy group boundaries
    detector_efficiency : float or dict, optional
        Detection efficiency
    detector_efficiency_err : float or dict, optional
        Detection efficiency uncertainty
    search_width : float, optional
        Peak search width in keV
    regularization : float, optional
        Regularization parameter for NNLS
    json_path : Path, optional
        Path to processed_data.json
    
    Returns
    -------
    phi : ndarray
        Unfolded flux spectrum
    phi_err : ndarray
        Flux uncertainties
    results_dict : dict
        Dictionary containing intermediate results
    """
    if json_path is None:
        json_path = Path(__file__).parent / '../../data/processed_data.json'
    
    # Step 1: Extract gamma emitted from each measurement
    gamma_emitted_dict = {}
    foil_measurements_list = []  # (foil_name, reaction_name, foil, measurement)
    R_list = []  # Reaction rates (gamma emitted)
    R_err_list = []  # Uncertainties
    
    for foil_name, foil_data in foil_measurements.items():
        foil = foil_data["foil"]
        gamma_emitted_dict[foil_name] = {}
        
        # Get the reaction type from the foil
        reaction_name = foil.reaction.type if hasattr(foil.reaction, 'type') else "unknown"
        gamma_emitted_dict[foil_name][reaction_name] = {}
        
        # Get gamma energies for this reaction
        gamma_energies = np.atleast_1d(foil.reaction.product.energy)
        
        for count_num, measurement in foil_data["measurements"].items():
            gamma_emitted_dict[foil_name][reaction_name][f"Count {count_num}"] = {}
            
            # Use first detector for simplicity (could average over detectors)
            for detector in measurement.detectors:
                ch = detector.channel_nb
                gamma_emitted_dict[foil_name][reaction_name][f"Count {count_num}"][ch] = {}
                
                if isinstance(detector_efficiency, dict):
                    det_eff = detector_efficiency.get(ch, None)
                else:
                    det_eff = detector_efficiency
                if isinstance(detector_efficiency_err, dict):
                    det_eff_err = detector_efficiency_err.get(ch, 0.0)
                else:
                    det_eff_err = detector_efficiency_err
                
                gamma_emitted, gamma_emitted_err = measurement.get_gamma_emitted(
                    background_measurement=background_meas,
                    calibration_coeffs=calibration_coeffs[ch],
                    efficiency_coeffs=efficiency_coeffs[ch],
                    channel_nb=ch,
                    search_width=search_width,
                    detection_efficiency=det_eff,
                    detection_efficiency_err=det_eff_err
                )
                
                # Store each gamma line separately using energy as key
                gamma_emitted = np.atleast_1d(gamma_emitted)
                gamma_emitted_err = np.atleast_1d(gamma_emitted_err)
                
                for i, gamma_energy in enumerate(gamma_energies):
                    energy_key = f"{gamma_energy:.2f}_keV"
                    gamma_emitted_dict[foil_name][reaction_name][f"Count {count_num}"][ch][energy_key] = {
                        "value": float(gamma_emitted[i]) if i < len(gamma_emitted) else 0.0,
                        "error": float(gamma_emitted_err[i]) if i < len(gamma_emitted_err) else 0.0
                    }
                
                # For matrix building, sum over gamma lines (use total for reaction rate)
                total_gamma = np.sum(gamma_emitted)
                total_gamma_err = np.sqrt(np.sum(gamma_emitted_err**2))
                
                # Add to lists for matrix building
                foil_measurements_list.append((foil_name, reaction_name, foil, measurement))
                R_list.append(total_gamma)
                R_err_list.append(total_gamma_err)
                
                # Only use first detector
                break
            # Only use first count
            break
    
    # Step 2: Save gamma emitted to processed_data.json
    save_gamma_emitted_to_processed_data(gamma_emitted_dict, json_path)
    
    # Step 3: Read cross sections from processed_data.json
    cross_section_dict = read_foil_xs_from_processed_data(str(json_path))
    
    # Step 4: Build response matrix
    A, reaction_labels = build_response_matrix(
        foil_measurements_list,
        cross_section_dict,
        irradiations,
        time_generator_off,
        energy_groups
    )
    
    R = np.array(R_list)
    R_err = np.array(R_err_list)
    
    print(f"Built response matrix A with shape: {A.shape}")
    print(f"Reaction rates R: {R}")
    print(f"Reaction labels: {reaction_labels}")
    
    # Step 5: Solve for flux spectrum
    phi, phi_err, residual = solve_flux_spectrum(
        R, R_err, A, energy_groups, regularization=regularization
    )
    
    print(f"Solved flux spectrum: {phi}")
    print(f"Residual norm: {residual}")
    
    results_dict = {
        "gamma_emitted": gamma_emitted_dict,
        "response_matrix": A,
        "reaction_labels": reaction_labels,
        "reaction_rates": R,
        "reaction_rate_errors": R_err,
        "residual": residual,
        "energy_groups": energy_groups
    }
    
    return phi, phi_err, results_dict
