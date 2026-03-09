import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from zoneinfo import ZoneInfo
from datetime import datetime

# read in csv file with measurement info
nGen_data = pd.read_csv('/home/cdunn314/libra/ngen_characterization/data/nGen-400 2025-09-29 14.38.42.csv')

# get relevant columns
elapsed_times = nGen_data['Elapsed (s)']
date_times = nGen_data['Time']
date_times = pd.to_datetime(date_times, format='%m/%d/%Y  %H:%M', errors='coerce')
# add timezone info
date_times = date_times.dt.tz_localize(ZoneInfo('America/New_York'), ambiguous='infer', nonexistent='shift_forward')

voltages = nGen_data['Anode Voltage']
currents = nGen_data['Anode Current']
power = voltages * currents / 1000  # in kW


def get_time_slice(start_time: datetime, stop_time: datetime):
    """
    Given start and stop times as datetime objects, return a slice of the data between those times.
    """
    mask = (date_times >= start_time) & (date_times <= stop_time)
    elapsed_times_slice = elapsed_times[mask] - elapsed_times[mask].iloc[0]  # reset to start at 0
    return elapsed_times_slice, voltages[mask], currents[mask], power[mask]


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(elapsed_times, voltages, label='Anode Voltage (V)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Voltage (kV)', color='tab:blue')
    ax2 = ax.twinx()
    ax2.plot(elapsed_times, currents, color='tab:orange', label='Anode Current (mA)')
    ax2.set_ylabel('Current (mA)', color='tab:orange')
    plt.title('nGen-400 Anode Voltage and Current Over Time')

    fig2, ax2 = plt.subplots(figsize=(10,6))
    ax2.plot(elapsed_times, power)  # power in kW
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Power (kW)')
    plt.title('nGen-400 Anode Power Over Time')
    plt.show()