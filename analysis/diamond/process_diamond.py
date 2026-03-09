import os

from libra_toolbox.neutron_detection.diamond.process_data import DataProcessor
from libra_toolbox.neutron_detection.activation_foils.compass import (sort_compass_files, 
                                                                      get_start_stop_time,
                                                                      get_spectrum_nbins) 
from pathlib import Path
import openmc
import json

file_path = Path(__file__).parent.resolve()

"""
CoMPASS Binary File Reader
Converted from C++ BinReader by Ryan Tang
https://codeberg.org/rtang/CoMPASS_BinReader
"""

import struct
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
from scipy.signal import find_peaks


@dataclass
class Data:
    """Data structure for a single CoMPASS event"""
    header: int = 0xCAE0
    board_id: int = 0
    channel: int = 0
    timestamp: int = 0  # in ps
    energy: int = 0
    energy_short: int = 0
    flags: int = 0
    waveform_code: int = 0
    n_sample: int = 0
    trace: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint16))
    
    def clear(self):
        self.header = 0xCAE0
        self.board_id = 0
        self.channel = 0
        self.timestamp = 0
        self.energy = 0
        self.energy_short = 0
        self.flags = 0
        self.waveform_code = 0
        self.n_sample = 0
        self.trace = np.array([], dtype=np.uint16)
    
    def __repr__(self):
        s = f"Header: 0x{self.header:X}\n"
        s += f"Board: {self.board_id}, Channel: {self.channel}\n"
        s += f"Energy: {self.energy}, TimeStamp: {self.timestamp} ps\n"
        s += f"Flags: 0x{self.flags:X}\n"
        if (self.header & 0x8) >= 1:
            s += f"Waveform code: {self.waveform_code}, nSample: {self.n_sample}\n"
        return s


class BinReader:
    """
    Reader for CoMPASS binary (.BIN) files.
    
    Usage:
        reader = BinReader('path/to/file.BIN')
        for event in reader:
            print(event.energy, event.timestamp)
        reader.close()
        
    Or with context manager:
        with BinReader('path/to/file.BIN') as reader:
            for event in reader:
                print(event.energy, event.timestamp)
    """
    
    def __init__(self, filename: Optional[str] = None):
        self.data = Data()
        self._file = None
        self._file_size = 0
        self._file_pos = 0
        self._end_of_file = False
        self._is_opened = False
        self._block_id = -1
        self._n_block = 0
        self._header_ok = False
        
        if filename is not None:
            self.open_file(filename)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Data:
        result = self.read_block()
        if result < 0:
            raise StopIteration
        return self.data
    
    def open_file(self, filename: str):
        """Open a CoMPASS binary file"""
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"Cannot read file: {filename}")
        
        self._file = open(filepath, 'rb')
        self._file.seek(0, 2)  # Seek to end
        self._file_size = self._file.tell()
        self._file.seek(0)  # Back to beginning
        
        self.data.clear()
        self._is_opened = True
        
        # Read the header (only at the beginning of file)
        header_bytes = self._file.read(2)
        if len(header_bytes) < 2:
            print("Header cannot read.")
            self._header_ok = False
            return
        
        self.data.header = struct.unpack('<H', header_bytes)[0]
        
        if (self.data.header >> 4) != 0xCAE:
            print(f"Header format not right: 0x{self.data.header:X}")
            self._header_ok = False
            return
        
        self._header_ok = True
        self._file_pos = self._file.tell()
    
    def close(self):
        """Close the file"""
        if self._file is not None:
            self._file.close()
        self._is_opened = False
        self.data.clear()
        self._file_size = 0
        self._file_pos = 0
        self._n_block = 0
        self._block_id = -1
        self._end_of_file = False
    
    def is_open(self) -> bool:
        return self._is_opened
    
    def is_end_of_file(self) -> bool:
        return self._end_of_file
    
    @property
    def file_pos(self) -> int:
        return self._file_pos
    
    @property
    def file_size(self) -> int:
        return self._file_size
    
    @property
    def block_id(self) -> int:
        return self._block_id
    
    @property
    def n_blocks(self) -> int:
        return self._n_block
    
    def read_block(self, skip_trace: bool = False) -> int:
        """
        Read a single event block from the file.
        
        Args:
            skip_trace: If True, skip waveform data for faster reading
            
        Returns:
            1 on success, -1 on end of file, -2 on header error
        """
        if self._file is None:
            return -1
        if self._end_of_file:
            return -1
        if not self._header_ok:
            return -2
        
        # See CoMPASS manual v19, P.67
        try:
            # Board ID (2 bytes)
            data = self._file.read(2)
            if len(data) < 2:
                self._end_of_file = True
                return -1
            self.data.board_id = struct.unpack('<H', data)[0]
            
            # Channel (2 bytes)
            data = self._file.read(2)
            self.data.channel = struct.unpack('<H', data)[0]
            
            # Timestamp (8 bytes)
            data = self._file.read(8)
            self.data.timestamp = struct.unpack('<Q', data)[0]
            
            # Energy (2 bytes) - if header bits 0 or 1 are set
            if (self.data.header & 0x3) != 0:
                data = self._file.read(2)
                self.data.energy = struct.unpack('<H', data)[0]
            
            # Skip 8 bytes if header bit 1 is set (some dummy data)
            if (self.data.header & 0x2) == 1:
                self._file.read(8)
            
            # Energy short (2 bytes) - if header bit 2 is set
            if (self.data.header & 0x4) != 0:
                data = self._file.read(2)
                self.data.energy_short = struct.unpack('<H', data)[0]
            
            # Flags (4 bytes)
            data = self._file.read(4)
            self.data.flags = struct.unpack('<I', data)[0]
            
            # Waveform data - if header bit 3 is set
            if (self.data.header & 0x8) == 1:
                # Waveform code (1 byte)
                data = self._file.read(1)
                self.data.waveform_code = struct.unpack('<B', data)[0]
                
                # Number of samples (4 bytes)
                data = self._file.read(4)
                self.data.n_sample = struct.unpack('<I', data)[0]
                
                if not skip_trace:
                    # Read trace data
                    trace_bytes = self._file.read(self.data.n_sample * 2)
                    self.data.trace = np.frombuffer(trace_bytes, dtype=np.uint16)
                else:
                    # Skip trace data
                    self._file.seek(self.data.n_sample * 2, 1)
                    self.data.trace = np.array([], dtype=np.uint16)
            
            self._block_id += 1
            self._file_pos = self._file.tell()
            
            if self._file_pos >= self._file_size:
                self._end_of_file = True
            
            return 1
            
        except struct.error:
            self._end_of_file = True
            return -1
    
    def scan_number_of_blocks(self) -> int:
        """
        Scan the entire file to count number of event blocks.
        Rewinds to beginning after scanning.
        """
        self._n_block = 0
        while self.read_block(skip_trace=True) != -1:
            self._n_block += 1
        
        print(f"Scan complete: number of data blocks: {self._n_block}")
        
        # Rewind to beginning
        self._file.seek(0)
        self._file.read(2)  # Skip header
        self._file_pos = self._file.tell()
        self._block_id = -1
        self._end_of_file = False
        
        return self._n_block
    
    def read_all_events(self, skip_trace: bool = True) -> dict:
        """
        Read all events and return as dictionary of arrays.
        
        Args:
            skip_trace: If True, skip waveform data for faster reading
            
        Returns:
            Dictionary with 'timestamps', 'energies', 'channels', 'board_ids', 'flags'
        """
        timestamps = []
        energies = []
        energies_short = []
        channels = []
        board_ids = []
        flags = []
        
        while self.read_block(skip_trace=skip_trace) != -1:
            timestamps.append(self.data.timestamp)
            energies.append(self.data.energy)
            energies_short.append(self.data.energy_short)
            channels.append(self.data.channel)
            board_ids.append(self.data.board_id)
            flags.append(self.data.flags)
        
        return {
            'timestamps': np.array(timestamps, dtype=np.uint64),
            'energies': np.array(energies, dtype=np.uint16),
            'energies_short': np.array(energies_short, dtype=np.uint16),
            'channels': np.array(channels, dtype=np.uint16),
            'board_ids': np.array(board_ids, dtype=np.uint16),
            'flags': np.array(flags, dtype=np.uint32),
        }


def read_compass_bin(filename: str, skip_trace: bool = True) -> dict:
    """
    Convenience function to read a CoMPASS binary file.
    
    Args:
        filename: Path to the .BIN file
        skip_trace: If True, skip waveform data
        
    Returns:
        Dictionary with event data arrays
    """
    with BinReader(filename) as reader:
        return reader.read_all_events(skip_trace=skip_trace)


if __name__ == "__main__":
    
    filename = '../../data/diamond/260225_Starfire_nGen_characterization/DAQ/-9deg/RAW/DataR_CH0@DT5730SB_14779_-9deg.BIN'
    print(f"Reading: {filename}")
    
    data = read_compass_bin(filename)
    print(f"Read {len(data['energies'])} events")
    print(f"Energy range: {data['energies'].min()} - {data['energies'].max()}")
    print(f"Timestamp range: {data['timestamps'].min()} - {data['timestamps'].max()} ps")


def get_events(measurement_path: Path, channel: int) -> dict:
    """
    Given a path to a measurement, read the CoMPASS binary file and return the event data as a dictionary of arrays.
    """
    print(os.listdir(measurement_path))
    data_filenames = sort_compass_files(measurement_path, filetype='.bin')
    print(f"Found data files: ", data_filenames)
    
    times = []
    energies = []
    if channel not in data_filenames:
        raise ValueError(f"Channel {channel} not found in measurement path: {measurement_path}")
    for filename in data_filenames[channel]:
        data = read_compass_bin(Path(measurement_path) / filename)
        times.append(data['timestamps'])
        energies.append(data['energies'])
    times = np.array(times).flatten()
    energies = np.array(energies).flatten()
    return times, energies

def get_diamond_measurement(measurement_path: Path, channel: int) -> DataProcessor:
    """
    Given a path to a measurement, process the data and return a DataProcessor object.
    """
    print(f"Processing measurement at path: {measurement_path}")
    
    time_values, energy_values = get_events(measurement_path, channel)
    measurement = DataProcessor()
    measurement.time_values = time_values
    measurement.energy_values = energy_values
    start_time, stop_time = get_start_stop_time(measurement_path)
    measurement.start_time = start_time
    measurement.stop_time = stop_time
    n_bins = get_spectrum_nbins(measurement_path.parent / 'settings.xml')
    measurement.channel_bins = np.arange(0, n_bins)
    
    return measurement

def get_measurements_from_json(channel: int):
    """
    Get the measurements from the general.json file and return a dictionary of DataProcessor objects.
    """

    with open(file_path / '../../data/general.json', 'r') as f:
        data = json.load(f)
    diamond_dict = data["neutron_detection"]["diamond"]

    measurements = {}
    for measurement_dict in diamond_dict['measurements']:
        name = f"{measurement_dict['angle']:.0f}deg"
        data_directory = measurement_dict['data_directory']
        measurement_path = Path(f'../../data/diamond/{data_directory}')
        measurements[name] = get_diamond_measurement(measurement_path, channel)
        measurements[name].angle = measurement_dict['angle']

    return measurements


def get_calibration_measurements_from_json(channel: int):
    """
    Get the calibration measurements from the general.json file and return a dictionary of DataProcessor objects.
    """

    with open(file_path / '../../data/general.json', 'r') as f:
        data = json.load(f)
    diamond_dict = data["neutron_detection"]["diamond"]

    calibration_measurements = {}
    for measurement_dict in diamond_dict['calibration_measurements']:
        name = measurement_dict['source']
        data_directory = measurement_dict['data_directory']
        measurement_path = Path(f'../../data/diamond/{data_directory}')
        calibration_measurements[name] = get_diamond_measurement(measurement_path, channel)
        calibration_measurements[name].energies = measurement_dict['energies']

    return calibration_measurements

def get_peaks(hist, name, expected_peaks, relative_prominence=None, relative_height=None, start_index=None):
    """
    Given a calibration measurement, get the peaks in the energy spectrum and return a list of peak energies.
    """

    if 'Ra226' in name:
        if relative_prominence is None:
            relative_prominence = 0.1
        if relative_height is None:
            relative_height = 0.1
        if start_index is None:
            start_index = 200
    hist_max = np.max(hist[start_index:])
    peak_indices, _ = find_peaks(hist[start_index:], 
                          prominence=relative_prominence * hist_max, 
                          height=relative_height * hist_max)  # Adjust height threshold as needed
    if len(peak_indices) != len(expected_peaks):
        print("Peak indices found: ", peak_indices)
        raise ValueError(f"Number of peaks found ({len(peak_indices)}) does not match number of expected energies ({len(expected_peaks)}) for radionuclide {name}.")
    peak_indices += start_index  # Adjust peak indices to account for the start index
    
    return peak_indices


def calibrate_diamond(calibration_measurements: dict, peak_kwargs: dict={}) -> tuple[list[float], list[float]]:
    """
    Given a dictionary of calibration measurements, return a list of calibration channels and a list of corresponding calibration energies. 
    """
    calibration_channels = []
    calibration_energies = []
    for name, measurement in calibration_measurements.items():
        hist, bins = np.histogram(measurement.energy_values, bins=measurement.channel_bins)
        energies = measurement.energies
        peak_indices = get_peaks(hist, name, energies, **peak_kwargs)
        calibration_channels.extend(measurement.channel_bins[peak_indices])
        calibration_energies.extend(energies)
    return calibration_channels, calibration_energies

    

def get_simulation_n_alpha_rate(angles: list[float], 
                                model_path: Path = file_path / '../../neutronics/isotropic_source') -> tuple[list[float], list[float]]:
    n_alpha_rates = []
    n_alpha_rate_errors = []
    model = openmc.model.Model.from_model_xml(model_path / 'model.xml')

    with openmc.StatePoint(model_path / 'statepoint.100.h5') as sp:
        tally = sp.get_tally(name='diamond tally')
        score_index = tally.scores.index('(n,a)')
        cell_filter = tally.find_filter(openmc.CellFilter)
        values = tally.get_reshaped_data(value='mean').squeeze()
        errors = tally.get_reshaped_data(value='std_dev').squeeze()
        for angle in angles:
            cell = model.geometry.get_cells_by_name(f'Diamond_detector_{angle:.0f}deg')[0]
            cell_index = list(cell_filter.bins).index(cell.id)
            n_alpha_rate = values[cell_index, score_index]
            n_alpha_rate_error = errors[cell_index, score_index]
            n_alpha_rates.append(n_alpha_rate)
            n_alpha_rate_errors.append(n_alpha_rate_error)


    return n_alpha_rates, n_alpha_rate_errors
    