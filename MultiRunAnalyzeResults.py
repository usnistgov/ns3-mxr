import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm
import re
from collections import defaultdict
import seaborn as sns
# Flags to control figure generation
ENABLE_THROUGHPUT_HEATMAP = True
ENABLE_THROUGHPUT_VS_TIME = False
ENABLE_ERROR_VS_TIME = False
ENABLE_CDF_SINR = False
ENABLE_DELAY_VS_TIME = False

ENABLE_MCS_VS_TIME = True
ENABLE_CUMULATIVE_ERRORS_VS_TIME = False

# Initialize the scenario_effect_throughput dictionary with nested structure
def nested_dict():
    return defaultdict(nested_dict)

base_folder = "ResultsGood"



# Normalize throughput for each app_id
def normalize_throughput(matrix, app_ids, txPower):
    normalized_matrix = matrix.copy()
    for i, app_id in enumerate(app_ids):
        # Extract values for this app_id across all scenarios and txPowers
        app_values = matrix[:, i*len(txPower):(i+1)*len(txPower)]
        # print(app_values)
        # Get the minimum and maximum throughput for this app_id
        min_throughput = np.min(app_values)
        max_throughput = np.max(app_values)
        # print("App:",app_id, " i:",i, " Min:", min_throughput, " Max:",max_throughput)
        
        # Take into account if all values are the same
        if max_throughput - min_throughput == 0: 
            normalized_app_values = 1
        else:
            # Normalize the values to range [0, 1]
            normalized_app_values = (app_values - min_throughput) / (max_throughput - min_throughput)
        
        # Update the matrix with normalized values
        normalized_matrix[:, i*len(txPower):(i+1)*len(txPower)] = normalized_app_values
    
    return normalized_matrix



def get_zero_throughput_periods(throughput_data, threshold=20):
    zero_periods = []
    start_time = None

    for i, row in throughput_data.iterrows():
        time, throughput = row[0], row[1]
        if throughput == 0:
            if start_time is None:
                start_time = time
        else:
            if start_time is not None and (time - start_time) >= threshold:
                zero_periods.append(start_time)
            start_time = None  # reset the start time

    # Check for the last chunk in case the file ends with a zero throughput period
    if start_time is not None and (throughput_data.iloc[-1, 0] - start_time) >= threshold:
        zero_periods.append(start_time)
        
    return zero_periods


def plot_cdf(data, label):
    sorted_data = np.sort(data)
    prob = np.arange(len(sorted_data)) / float(len(sorted_data))
    plt.plot(sorted_data, prob, label=label)

def plot_cdf_colored(data, label, color=None):  # Adding color parameter with a default value of None
    sorted_data = np.sort(data)
    prob = np.arange(len(sorted_data)) / float(len(sorted_data))
    plt.plot(sorted_data, prob, label=label, color=color)  # Using the color parameter here

# app_ids = [1, 2, 3, 4, 5, 6, 7, 610,650,6100,6500]
app_ids = [1]
# app_ids =[650,6100,6500]
# app_ids =[1, 2]
# app_ids =[1, 2]
# txPower = [0.0,10.0,20.0]
txPower = [0.0]
# scenarioIDs = list(range(1, 11))
scenarioIDs = [13]
rlc_mode = 'rlcAmEnabled'
runNumbers = list(range(1, 2))
# runNumbers = list(range(1, 11))


# Dictionaries to store the effect of scenarioID on throughput and PDR for each txPower for each app_id
scenario_effect_throughput = {app: {power: {} for power in txPower} for app in app_ids}
scenario_effect_PDR = {app: {power: {} for power in txPower} for app in app_ids}
throughput_values_for_cdf = {app: {power: [] for power in txPower} for app in app_ids}
# Dictionary to store the throughput value at each BS position
heatmap_data = {app: {power: {} for power in txPower} for app in app_ids}
decoding_errors_data = {}
# This dictionary will store the effect of scenario on total decoding errors for each app id and txPower
# scenario_effect_errors = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

scenario_effect_throughput = defaultdict(nested_dict)
scenario_effect_PDR = defaultdict(nested_dict)
scenario_effect_errors = defaultdict(nested_dict)
scenario_effect_delay = defaultdict(nested_dict)



import pandas as pd
from matplotlib.ticker import MultipleLocator
# Creating a list to store data
data = []



if ENABLE_CDF_SINR:
    for power in txPower:
        plt.figure()
        
        # Generate viridis colors
        colors = cm.viridis(np.linspace(0, 1, len(scenarioIDs)))
        
        for idx, scenarioID in enumerate(scenarioIDs):
            snr_file = f'{base_folder}/SurgeryFeedbackPos{scenarioID}_snr_{rlc_mode}rfh-app-1txPower{power}runNumber1.txt'
            snr_data = pd.read_csv(snr_file, sep=',', header=None)
            snr_data.replace(-np.inf, np.nan, inplace=True)
            # plot_cdf_colored(snr_data[1].dropna(), label=f'ScenarioID={scenarioID}', color=colors[idx])  # using the viridis colors
            plot_cdf(snr_data[1].dropna(), label=f'ScenarioID={scenarioID}')  # using the viridis colors
        
        plt.xlabel('SINR (dB)')
        plt.ylabel('Cumulative Probability')
        plt.title(f'CDF of SINR for txPower={power}')
        plt.legend()
        
        # Set major and minor ticks
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        
        plt.grid(True)
        plt.savefig(f'{base_folder}/Figures/cdf_snr_txPower{power}_Run1.png')
        plt.show()
        
if ENABLE_CDF_SINR:
    for power in txPower:
        plt.figure()
        
        # Generate viridis colors
        colors = cm.viridis(np.linspace(0, 1, len(scenarioIDs)))
        
        for idx, scenarioID in enumerate(scenarioIDs):
            snr_file = f'{base_folder}/SurgeryFeedbackPos{scenarioID}_snr_{rlc_mode}rfh-app-1txPower{power}runNumber1.txt'
            snr_data = pd.read_csv(snr_file, sep=',', header=None)
            snr_data.replace(-np.inf, np.nan, inplace=True)
            # plot_cdf_colored(snr_data[1].dropna(), label=f'ScenarioID={scenarioID}', color=colors[idx])  # using the viridis colors
            plot_cdf(snr_data[1].dropna(), label=f'ScenarioID={scenarioID}')  # using the viridis colors
        
        plt.xlabel('SINR (dB)')
        plt.ylabel('Cumulative Probability')
        plt.title(f'CDF of SINR for txPower={power}')
        plt.legend()
        plt.grid(True)

        # Set X-axis limit
        plt.xlim(-20, 4)  # Adjust the range as needed
        plt.ylim(0, 0.1)  # Adjust the range as needed

        # Add vertical dashed line at SINR = 0.59 dB
        plt.axvline(x=  1.03, color='r', linestyle='--')

        plt.savefig(f'{base_folder}/Figures/ZOOMEDcdf_snr_txPower{power}_Run1.png')
        plt.show()

def generate_cdf_plots(mcs_values_dict, base_folder):
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*']
    linestyles = ['-', '--', '-.', ':']
    colors = plt.cm.viridis(np.linspace(0, 1, len(mcs_values_dict)))

    for app_idx, (app_id, power_dict) in enumerate(mcs_values_dict.items()):
        for scenarioID, power_scenario_dict in power_dict.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            for power_idx, (power, scenario_mcs_values) in enumerate(power_scenario_dict.items()):
                sorted_mcs_values = np.sort(scenario_mcs_values)
                cdf = np.arange(len(sorted_mcs_values)) / float(len(sorted_mcs_values) - 1)
                ax.plot(sorted_mcs_values, cdf, label=f'txPower={power}', color=colors[app_idx], marker=markers[power_idx], markevery=0.1)

            ax.set_xlabel('MCS')
            ax.set_ylabel('CDF')
            ax.set_title(f'MCS CDF for rfh-app-{app_id}, Scenario {scenarioID}')
            ax.grid(True)
            ax.legend(loc='best')
            plt.tight_layout()
            plt.savefig(f'{base_folder}/Figures/MCS_CDF_rfh-app-{app_id}_Scenario{scenarioID}.png')
            plt.show()

# Example usage:



if ENABLE_THROUGHPUT_HEATMAP:
    # Matrix to store the average throughput for all scenarios, app_ids, and txPowers
    throughput_matrix = np.zeros((len(scenarioIDs), len(app_ids) * len(txPower)))
    
    for s, scenarioID in enumerate(scenarioIDs):
        for i, app_id in enumerate(app_ids):
            for j, power in enumerate(txPower):
                total_throughput = 0
                valid_runs = 0
                
                for runNumber in runNumbers:
                    # Construct the file name for the current run
                    throughput_file = f'{base_folder}/SurgeryFeedbackPos{scenarioID}_throughput_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt'
                    
                    # Check if the file exists
                    if os.path.exists(throughput_file):
                        throughput_data = pd.read_csv(throughput_file, sep=',', header=None)
                        
                        # Check if the data has enough rows
                        if len(throughput_data[2]) > 285:
                            total_throughput += throughput_data[2].iloc[285]
                            valid_runs += 1
                        else:
                            print(f"Throughput data not long enough in file: {throughput_file}")
                    
                # Calculate the average throughput if there are valid runs
                if valid_runs > 0:
                    average_throughput = total_throughput / valid_runs
                    throughput_matrix[s, i * len(txPower) + j] = average_throughput
                else:
                    print(f"No valid throughput data for Scenario {scenarioID}, App {app_id}, TxPower {power}, RunNumber {runNumber}")
                    
 # Generate combined labels for columns
    col_labels = [f"App{app_id}-Tx{power}" for app_id in app_ids for power in txPower]

    # Normalize the throughput_matrix for each app_id
    normalized_throughput_matrix = normalize_throughput(throughput_matrix, app_ids, txPower)

    # Convert the matrix to DataFrame for heatmap visualization
    df = pd.DataFrame(normalized_throughput_matrix, index=scenarioIDs, columns=col_labels)

    # print(df.isna().any().any())  # This will print True if there are any NaN values in the DataFrame.
        
    # Plotting the heatmap
    # Adjusting the figure size
    # Plotting the heatmap with separation for each app_id
    # Plotting the heatmap with separation for each app_id
    from matplotlib.colors import LinearSegmentedColormap

    # Create a custom colormap that transitions from red to green
    colors = [(1, 0, 0), (0, 1, 0)]  # Red to Green
    n_bins = [3]  # Discretizes the interpolation into bins
    cmap_name = 'custom_diverging'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

    # Plotting the heatmap with separation for each app_id and the custom colormap
    # Plotting the heatmap with separation for each app_id and the RdYlGn colormap
    # Plotting the heatmap with separation for each app_id and the RdYlGn colormap
    # Plotting the heatmap with separation for each app_id and the RdYlGn colormap, but without annotations
    # Plotting the heatmap with hierarchical x-axis
    # Plotting the heatmap
    # Adjusting the figure size
    fig, ax = plt.subplots(figsize=(20, 12))

    # Adjust the color map to be centered around 0.5
    cmap = plt.get_cmap('RdYlGn')

    sns.heatmap(df, cmap=cmap, linewidths=0, ax=ax, cbar_kws={'label': 'Normalized Througput per App Value'})

    # Setting titles and labels
    plt.title("Normalized throughput per app across Scenarios and Transmission Power")
    plt.xlabel("Transmission Power", fontsize=14)  # Adjust the fontsize parameter as desired
    plt.ylabel("Scenario ID", fontsize=14)  # Adjust the fontsize parameter as desired
    # Drawing vertical thick black lines to separate different app_ids
    for index in range(len(txPower), len(app_ids) * len(txPower), len(txPower)):
        plt.axvline(x=index, color='black', linewidth=2)

    # Drawing horizontal black lines to separate different scenarios
    for index in range(1, len(scenarioIDs)):
        plt.axhline(y=index, color='black', linewidth=2)

    # Drawing dashed vertical lines between txPowers for each app_id
    for i in range(len(app_ids)):
        for j in range(len(txPower)-1):
            ax.axvline(x=i*len(txPower) + j + 1, color='black', linewidth=1, linestyle='--')

    # Calculating tick positions for app_ids and txPower values
    app_id_ticks = [(i + 0.5) * len(txPower) - 0.5 for i in range(len(app_ids))]
    txPower_ticks = [i + 0.5 for i in range(len(txPower) * len(app_ids))]

    # Setting the secondary x-axis ticks and labels (for app_ids)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(app_id_ticks)
    ax2.set_xticklabels(app_ids)
    ax2.set_xlabel("App ID", fontsize=14)

    # Setting the primary x-axis ticks and labels (for txPower)
    ax.set_xticks(txPower_ticks)
    ax.set_xticklabels(txPower * len(app_ids), rotation=45)  # Repeat txPower values for each app_id with rotation

    # Saving the figure
    plt.tight_layout()
    plt.savefig(f'{base_folder}/Figures/Heatmap_combined_throughput.png')
    plt.show()


mcs_values_dict = {}
for app_id in tqdm(app_ids, desc="Processing app ids", ncols=100):
    for scenarioID in scenarioIDs:
        print("Scenario ID:",scenarioID)
        for runNumber in runNumbers:
            for power in txPower:
                # Load the throughput file
                throughput_file = f'{base_folder}/SurgeryFeedbackPos{scenarioID}_throughput_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt'
                throughput_data = pd.read_csv(throughput_file, sep=',', header=None)
                
                # Extract the throughput for timestep 585 from the third column
                try:
                    throughput_at_585 = throughput_data[2]
                    throughput_at_585 = throughput_at_585.iloc[285]
                except IndexError:
                    print("The file has fewer than 585 rows!")
                    print("throughputFile:", throughput_file)
                    print(throughput_data)
                    throughput_at_585 = None
                
                scenario_effect_throughput[app_id][power][scenarioID][runNumber] = throughput_at_585

                # Extract PDR from the last row
                pdr_value = throughput_data.iloc[-1, 3]
                scenario_effect_PDR[app_id][power][scenarioID][runNumber] = pdr_value

                #################### ALTERNATE CODE
                
                # Load the UE statistics file
                UEStatsFile = f'{base_folder}/SurgeryFeedbackPos{scenarioID}_UE_rlcAmEnabledrfh-app-{app_id}txPower{power}runNumber{runNumber}.txt'
                try:
                    UEStatsData = pd.read_csv(UEStatsFile, sep=',', header=0)
                except FileNotFoundError:
                    print(f"UE statistics file not found: {UEStatsFile}")
                    continue
                
                # Remove rows where MCS is 0
                UEStatsData = UEStatsData[UEStatsData['MCS'] != 0]          
                if ENABLE_MCS_VS_TIME:
                    # Convert SINR from linear to dB
                    UEStatsData['SINR'] = 10 * np.log10(UEStatsData['SINR'])

                    fig, ax1 = plt.subplots()
                    ax1.plot(UEStatsData['TIME'], UEStatsData['MCS'], label=f'MCS, txPower={power}')
                    ax1.set_xlabel('Time (s)')
                    ax1.set_ylabel('MCS', color='b')
                    ax1.tick_params(axis='y', labelcolor='b')

                    ax2 = ax1.twinx()
                    ax2.plot(UEStatsData['TIME'], UEStatsData['SINR'], 'r--', label=f'SINR')
                    ax2.set_ylabel('SINR (dB)', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')

                    fig.tight_layout()
                    plt.title(f'MCS and SINR vs Time for rfh-app-{app_id}, txPower={power}')
                    lines, labels = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax2.legend(lines + lines2, labels + labels2, loc=0)

                    plt.savefig(f'{base_folder}/Figures/Detailed/mcs_snr_vs_time_rfh-app-{app_id}_ScenarioID{scenarioID}_txPower{power}_Run{runNumber}.png')
                     # Initialize the data structure if needed
                    if app_id not in mcs_values_dict:
                        mcs_values_dict[app_id] = {}
                    if power not in mcs_values_dict[app_id]:
                        mcs_values_dict[app_id][power] = {}
                    if scenarioID not in mcs_values_dict[app_id][power]:
                        mcs_values_dict[app_id][power][scenarioID] = []

                    # Store MCS values for CDF plot
                    mcs_values_dict[app_id][power][scenarioID].extend(UEStatsData['MCS'].tolist())
                    
                    
                if ENABLE_CUMULATIVE_ERRORS_VS_TIME:
                    # Convert SINR from linear to dB
                    UEStatsData['SINR'] = 10 * np.log10(UEStatsData['SINR'])

                    # Calculate the cumulative sum of errors
                    UEStatsData['CumulativeErrors'] = UEStatsData['Corrupt'].cumsum()
                    # Calculate the cumulative sum of errors
                    total_error = UEStatsData['Corrupt'].cumsum().iloc[-1]  # Get the last value of the cumulative sum

                    # Store the total cumulative error in the data structure
                    scenario_effect_errors[app_id][power][scenarioID][runNumber] = total_error

                    fig, ax1 = plt.subplots()
                    ax1.plot(UEStatsData['TIME'], UEStatsData['CumulativeErrors'], label=f'Cumulative Errors, txPower={power}')
                    ax1.set_xlabel('Time (s)')
                    ax1.set_ylabel('Cumulative Errors', color='b')
                    ax1.tick_params(axis='y', labelcolor='b')

                    ax2 = ax1.twinx()
                    ax2.plot(UEStatsData['TIME'], UEStatsData['SINR'], 'r--', label=f'SINR')
                    ax2.set_ylabel('SINR (dB)', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')

                    fig.tight_layout()
                    plt.title(f'Cumulative Errors and SINR vs Time for rfh-app-{app_id}, txPower={power}')
                    lines, labels = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax2.legend(lines + lines2, labels + labels2, loc=0)

                    plt.savefig(f'{base_folder}/Figures/Detailed/cumulative_errors_snr_vs_time_rfh-app-{app_id}_ScenarioID{scenarioID}_txPower{power}_Run{runNumber}.png')
                    plt.show()
                ################### END ALTERNATE CODE
               

                # SNR file
                snr_file = f'{base_folder}/SurgeryFeedbackPos{scenarioID}_snr_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt'
                snr_data = pd.read_csv(snr_file, sep=',', header=None)
                snr_data.replace(-np.inf, np.nan, inplace=True)  # Replace -inf with NaN

                # Delay file
                # Load the delay file
                if ENABLE_DELAY_VS_TIME:
                    delay_file = f'{base_folder}/SurgeryFeedbackPos{scenarioID}_delay_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt'
                    try:
                        delay_data = pd.read_csv(delay_file, sep='\s+', header=0)
                    except FileNotFoundError:
                        print(f"Delay file not found: {delay_file}")
                        continue
                    # print(delay_data['delay(ms)'])
                    delay_data['delay(ms)'] = delay_data['delay(ms)'] / 1000000
                    
                    average_delay = delay_data['delay(ms)'].mean()
                    scenario_effect_delay[app_id][power][scenarioID][runNumber] = average_delay
                
                    fig, ax1 = plt.subplots()
                    ax1.plot(delay_data['time(s)'], delay_data['delay(ms)'], label=f'Delay, txPower={power}')
                    ax1.set_xlabel('Time (ms)')
                    ax1.set_ylabel('Delay (ms)', color='b')
                    ax1.tick_params(axis='y', labelcolor='b')
                    ax1.legend(loc="upper left")

                    ax2 = ax1.twinx()
                    ax2.plot(snr_data[0], snr_data[1], 'r--', label=f'SINR')
                    ax2.set_ylabel('SINR (dB)', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')
                    ax2.set_ylim(-100, snr_data[1].max() + 5)  # Set the y-axis limits


                    plt.title(f'Delay and SINR vs Time for rfh-app-{app_id}, txPower={power}, ScenarioID{scenarioID}, Run{runNumber}')
                    plt.grid(True)
                    plt.tight_layout()

                    plt.savefig(f'{base_folder}/Figures/Detailed/delay_snr_vs_time_rfh-app-{app_id}_ScenarioID{scenarioID}_txPower{power}_Run{runNumber}.png')
                    plt.show()

                # Number of decoding error
                if ENABLE_ERROR_VS_TIME:
                    filename = f"{base_folder}/log_true_true_{app_id}_{int(power)}_SurgeryFeedbackPos{scenarioID}_{runNumber}.txt"
            
                    # # Check if the file exists
                    if os.path.exists(filename):
                        with open(filename, 'r') as f:
                            content = f.readlines()
                            
                        total_error = 0
                        for line in content:
                            match = re.search(r'\+(\d+(\.\d+)?e\+\d+)ns Cannot receive because corrupted:(\d+(\.\d+)?)', line)
                            if match:
                                error = float(match.group(3))
                                total_error += error
                        
                        scenario_effect_errors[app_id][power][scenarioID][runNumber] = total_error

            
                    

                # THROUGHPUT VS TIME combined with SINR VS TIME
                if ENABLE_THROUGHPUT_VS_TIME:
                    fig, ax1 = plt.subplots()
                    ax1.plot(throughput_data[0], throughput_data[1], label=f'Throughput, txPower={power}')
                    ax1.set_xlabel('Time (s)')
                    ax1.set_ylabel('Throughput (Mbps)', color='b')
                    ax1.tick_params(axis='y', labelcolor='b')

                    ax2 = ax1.twinx()
                    ax2.plot(snr_data[0], snr_data[1], 'r--', label=f'SINR')
                    ax2.set_ylabel('SINR (dB)', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')
                    ax2.set_ylim(-100, snr_data[1].max() + 5)  # Set the y-axis limits

                    fig.tight_layout()
                    plt.title(f'Combined Throughput and SINR vs Time for rfh-app-{app_id}, txPower={power}')
                    lines, labels = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax2.legend(lines + lines2, labels + labels2, loc=0)

                    plt.savefig(f'{base_folder}/Figures/Detailed/combined_throughput_snr_rfh-app-{app_id}_ScenarioID{scenarioID}_txPower{power}_Run{runNumber}.png')
                    plt.show()
                    
                    # Check for zero throughput periods
                    zero_periods = get_zero_throughput_periods(throughput_data)
                    if zero_periods:
                        print(f"File: {throughput_file}")
                        print("Period(s) where throughput is equal to 0 for 20s or more starts at:")
                        for period_start in zero_periods:
                            print(f"  - {period_start} seconds")



# generate_cdf_plots(mcs_values_dict, base_folder)

from matplotlib.colors import to_hex, LinearSegmentedColormap
from rich.console import Console
from rich.table import Table
import numpy as np
from tqdm import tqdm
from matplotlib.colors import to_hex, LinearSegmentedColormap
from collections import defaultdict

import pandas as pd
from tabulate import tabulate
import numpy as np
from matplotlib.colors import to_hex, LinearSegmentedColormap
from tqdm import tqdm

import pandas as pd
from tabulate import tabulate
import numpy as np
from matplotlib.colors import to_hex, LinearSegmentedColormap
from tqdm import tqdm
from termcolor import colored

import pandas as pd
from tabulate import tabulate
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from termcolor import colored

import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from prettytable import PrettyTable
from termcolor import colored

import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from prettytable import PrettyTable
from termcolor import colored

import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import xlsxwriter

# Define the custom colormap
colors = ["#8b0000", "#ff0000", "#ff4500", "#ff6347", "#ffd700", "#adff2f", "#32cd32", "#008000", "#006400", "#2e8b57"]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=10)

def get_color(value):
    rgba = cmap(value)
    return rgba

# Create a Pandas Excel writer using XlsxWriter as the engine.
with pd.ExcelWriter("output.xlsx", engine='xlsxwriter') as writer:
    for app_id in app_ids:
        for power in txPower:
            pdr_values = defaultdict(list)
            for scenario in scenarioIDs:
                pdrs_for_runNumbers = list(scenario_effect_PDR[app_id][power][scenario].values())
                mean_pdr = np.mean(pdrs_for_runNumbers)
                std_pdr = np.std(pdrs_for_runNumbers)
                pdr_values[mean_pdr].append((scenario, mean_pdr, std_pdr))

            # Sort groups based on PDR values
            sorted_groups = sorted(pdr_values.items(), key=lambda x: x[0])
            
            # Assign colors to each group
            num_groups = len(sorted_groups)
            color_indices = np.linspace(0, 1, num_groups)
            color_mapping = {}
            for group, color_index in zip(sorted_groups, color_indices):
                pdr_value, scenarios = group
                color = get_color(color_index)
                for scenario in scenarios:
                    color_mapping[scenario[0]] = (color, scenario[1], scenario[2])
            
            # Create DataFrame
            df = pd.DataFrame(columns=["ScenarioID", "TX Power", "Average PDR", "Standard Deviation"])
            for scenario in scenarioIDs:
                color, mean_pdr, std_pdr = color_mapping[scenario]
                df = df.append({"ScenarioID": scenario, "TX Power": power, "Average PDR": mean_pdr, 
                                "Standard Deviation": std_pdr}, ignore_index=True)
            
            # Convert the dataframe to an XlsxWriter Excel object.
            sheet_name = f'App{app_id}_Power{power}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Get the xlsxwriter workbook and worksheet objects.
            workbook  = writer.book
            worksheet = writer.sheets[sheet_name]

            # Get the dimensions of the dataframe.
            num_rows, num_cols = df.shape

            # Create a format. Light red fill with dark red text.
            format1 = workbook.add_format({'bg_color':   '#FFC7CE',
                                           'font_color': '#9C0006'})

            # Apply a conditional format to the cell range.
            worksheet.conditional_format(1, 2, num_rows, 2, {'type':     '3_color_scale',
                                                 'min_color': "#F8696B",
                                                 'mid_color': "#FFEB84",
                                                 'max_color': "#63BE7B"})







from prettytable import PrettyTable
import numpy as np
from tqdm import tqdm

# Assuming scenario_effect_PDR, app_ids, txPower, and scenarioIDs are defined

for app_id in tqdm(app_ids, desc="Printing PDR values", ncols=100):
    table = PrettyTable()
    table.field_names = ["ScenarioID", "TX Power", "Average PDR", "Standard Deviation"]

    for power in txPower:
        for scenario in scenarioIDs:
            pdrs_for_runNumbers = list(scenario_effect_PDR[app_id][power][scenario].values())
            mean_pdr = np.mean(pdrs_for_runNumbers)
            std_pdr = np.std(pdrs_for_runNumbers)
            table.add_row([scenario, power, f"{mean_pdr:.4f}", f"{std_pdr:.4f}"])

    print(f"\n\nPDR for rfh-app-{app_id}")
    print(table)

# Error bar
# Grouped bar chart for effect of scenarioID on total decoding errors for each app id
# width = 0.2  # the width of the bars
# space = 0.2  # space between the groups of bars
# colors = plt.cm.viridis(np.linspace(0, 1, len(txPower)))

# for app_id in tqdm(app_ids, desc="Generating grouped bar charts", ncols=100):
#     labels = scenario_effect_errors[app_id][txPower[0]].keys()
#     x = np.arange(len(labels)) * (len(txPower) * width + space)  # adjusted for spacing between groups
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     for i, power in enumerate(txPower):
#         # Error values for the specific app_id and txPower
#         values = [scenario_effect_errors[app_id][power][scenario] for scenario in scenarioIDs]
#         ax.bar(x + i * width, values, width=width, label=f'txPower={power}', color=colors[i], edgecolor='black', alpha=0.85)
    
#     ax.set_xlabel('ScenarioID', fontsize=14)
#     ax.set_ylabel('Total Decoding Errors', fontsize=14)
#     ax.set_title(f'Effect of scenarioID on Decoding Errors for rfh-app-{app_id}', fontsize=16)
#     ax.set_xticks(x + (len(txPower) - 1) * width / 2)
#     ax.set_xticklabels(labels, fontsize=12)
#     ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
#     ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    
#     plt.tight_layout()
#     plt.savefig(f'{base_folder}/Figures/scenario_effect_on_decoding_errors_rfh-app-{app_id}_Run{runNumber}.png', bbox_inches='tight')
#     plt.show()




import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


############################################################
#        DELAY                                        #
############################################################
# width = 0.2  # the width of the bars
# space = 0.2  # space between the groups of bars
# colors = plt.cm.viridis(np.linspace(0, 1, len(txPower)))

# for app_id in tqdm(app_ids, desc="Generating grouped bar charts", ncols=100):
#     labels = scenario_effect_delay[app_id][txPower[0]].keys()
#     x = np.arange(len(labels)) * (len(txPower) * width + space)  # adjusted for spacing between groups
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     for i, power in enumerate(txPower):
#         # Compute average and std deviation of delay values across all runNumbers
#         mean_values = []
#         std_values = []
#         for scenario in scenarioIDs:
#             delays_for_runNumbers = list(scenario_effect_delay[app_id][power][scenario].values())
#             mean_delay = np.mean(delays_for_runNumbers)
#             std_delay = np.std(delays_for_runNumbers)
            
#             mean_values.append(mean_delay)
#             std_values.append(std_delay)
        
#         ax.bar(x + i * width, mean_values, yerr=std_values, width=width, label=f'txPower={power}', color=colors[i], edgecolor='black', alpha=0.85, capsize=5)
    
#     ax.set_xlabel('ScenarioID', fontsize=14)
#     ax.set_ylabel('Average Delay (ms)', fontsize=14)
#     ax.set_title(f'Effect of scenarioID on Delay for rfh-app-{app_id}', fontsize=16)
#     ax.set_xticks(x + (len(txPower) - 1) * width / 2)
#     ax.set_xticklabels(labels, fontsize=12)
    
#     # Place legend outside of the plot
#     ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
#     # Display a light grid
#     ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    
#     plt.tight_layout()
#     plt.savefig(f'{base_folder}/Figures/scenario_effect_on_delay_rfh-app-{app_id}.png', bbox_inches='tight')
#     plt.show()
############################################################
#        EDND DELAYU                                        #
############################################################


############################################################
#        THROUGHPUT                                        #
############################################################
width = 0.2  # the width of the bars
space = 0.2  # space between the groups of bars
colors = plt.cm.viridis(np.linspace(0, 1, len(txPower)))

for app_id in tqdm(app_ids, desc="Generating grouped bar charts", ncols=100):
    labels = scenario_effect_throughput[app_id][txPower[0]].keys()
    x = np.arange(len(labels)) * (len(txPower) * width + space)  # adjusted for spacing between groups
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, power in enumerate(txPower):
        # Compute average and std deviation of throughput values across all runNumbers
        mean_values = []
        std_values = []
        for scenario in scenarioIDs:
            throughputs_for_runNumbers = list(scenario_effect_throughput[app_id][power][scenario].values())
            mean_throughput = np.mean(throughputs_for_runNumbers)
            std_throughput = np.std(throughputs_for_runNumbers)
            
            mean_values.append(mean_throughput)
            std_values.append(std_throughput)
        
        ax.bar(x + i * width, mean_values, yerr=std_values, width=width, label=f'txPower={power}', color=colors[i], edgecolor='black', alpha=0.85, capsize=5)
    
    ax.set_xlabel('ScenarioID', fontsize=14)
    ax.set_ylabel('Average Throughput (Mbps)', fontsize=14)
    ax.set_title(f'Effect of scenarioID on Throughput for rfh-app-{app_id}', fontsize=16)
    ax.set_xticks(x + (len(txPower) - 1) * width / 2)
    ax.set_xticklabels(labels, fontsize=12)
    
    # Place legend outside of the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Display a light grid
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f'{base_folder}/Figures/scenario_effect_on_throughput_rfh-app-{app_id}.png', bbox_inches='tight')
    plt.show()



import matplotlib.pyplot as plt
import numpy as np

width = 0.2  # the width of the boxes
space = 0.5  # space between the groups of boxes

for app_id in tqdm(app_ids, desc="Generating grouped box plots", ncols=100):
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define the position of boxplots
    scenario_positions = np.arange(len(scenarioIDs)) * (len(txPower) * width + space)

    # Style
    colors = plt.cm.viridis(np.linspace(0, 1, len(txPower)))

    box_data = []
    box_labels = []
    box_colors = []
    
    for i, scenario in enumerate(scenarioIDs):
        for j, power in enumerate(txPower):
            try:
                throughputs = [scenario_effect_throughput[app_id][power][scenario][run] for run in runNumbers]
                box_data.append(throughputs)
                box_labels.append(scenario_positions[i] + j * width)
                box_colors.append(colors[j])
            except KeyError:
                continue

    bp = ax.boxplot(box_data, positions=box_labels, widths=width, patch_artist=True)

    # Styling the box plots
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')

    for whisker in bp['whiskers']:
        whisker.set(color='black', linestyle='-')

    for cap in bp['caps']:
        cap.set(color='black', linestyle='-')

    for median in bp['medians']:
        median.set(color='red', linewidth=1.5)

    for flier in bp['fliers']:
        flier.set(marker='o', color='black', alpha=0.5)

    # Setting labels
    ax.set_xticks(scenario_positions + (len(txPower) - 1) * width / 2)
    ax.set_xticklabels(scenarioIDs, rotation=0, fontsize=12)
    ax.set_title(f'Effect of scenarioID and txPower on Throughput for rfh-app-{app_id}', fontsize=16)
    ax.set_ylabel('Throughput (Mbps)', fontsize=14)

    # Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=color, lw=4) for color in colors]
    ax.legend(custom_lines, [f'txPower={power}' for power in txPower], loc='upper left', bbox_to_anchor=(1, 1))

    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{base_folder}/Figures/scenario_effect_on_throughput_grouped_boxplot_rfh-app-{app_id}.png', bbox_inches='tight')
    plt.show()

############################################################
#       END  THROUGHPUT                                        #
############################################################

############################################################
#        PDR                                                #
############################################################
width = 0.2  # the width of the bars
space = 0.2  # space between the groups of bars
colors = plt.cm.viridis(np.linspace(0, 1, len(txPower)))

for app_id in tqdm(app_ids, desc="Generating grouped bar charts for PDR", ncols=100):
    labels = scenario_effect_PDR[app_id][txPower[0]].keys()
    x = np.arange(len(labels)) * (len(txPower) * width + space)  # adjusted for spacing between groups
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, power in enumerate(txPower):
        # Compute average and std deviation of PDR values across all runNumbers
        mean_values = []
        std_values = []
        for scenario in scenarioIDs:
            pdrs_for_runNumbers = list(scenario_effect_PDR[app_id][power][scenario].values())
            mean_pdr = np.mean(pdrs_for_runNumbers)
            std_pdr = np.std(pdrs_for_runNumbers)
            
            mean_values.append(mean_pdr)
            std_values.append(std_pdr)
        
        ax.bar(x + i * width, mean_values, yerr=std_values, width=width, label=f'txPower={power}', color=colors[i], edgecolor='black', alpha=0.85, capsize=5)
    
    ax.set_xlabel('ScenarioID', fontsize=14)
    ax.set_ylabel('Average PDR', fontsize=14)
    ax.set_title(f'Effect of scenarioID on PDR for rfh-app-{app_id}', fontsize=16)
    ax.set_xticks(x + (len(txPower) - 1) * width / 2)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{base_folder}/Figures/scenario_effect_on_PDR_rfh-app-{app_id}.png', bbox_inches='tight')
    plt.show()

# Boxplot
import matplotlib.pyplot as plt
import numpy as np

width = 0.2  # the width of the boxes
space = 0.5  # space between the groups of boxes

for app_id in tqdm(app_ids, desc="Generating grouped box plots for PDR", ncols=100):
    fig, ax = plt.subplots(figsize=(14, 8))
    scenario_positions = np.arange(len(scenarioIDs)) * (len(txPower) * width + space)
    colors = plt.cm.viridis(np.linspace(0, 1, len(txPower)))
    box_data = []
    box_labels = []
    box_colors = []
    
    for i, scenario in enumerate(scenarioIDs):
        for j, power in enumerate(txPower):
            try:
                pdrs = [scenario_effect_PDR[app_id][power][scenario][run] for run in runNumbers]
                box_data.append(pdrs)
                box_labels.append(scenario_positions[i] + j * width)
                box_colors.append(colors[j])
            except KeyError:
                continue

    bp = ax.boxplot(box_data, positions=box_labels, widths=width, patch_artist=True)

    # Styling the box plots
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    for whisker in bp['whiskers']:
        whisker.set(color='black', linestyle='-')
    for cap in bp['caps']:
        cap.set(color='black', linestyle='-')
    for median in bp['medians']:
        median.set(color='red', linewidth=1.5)
    for flier in bp['fliers']:
        flier.set(marker='o', color='black', alpha=0.5)

    ax.set_xticks(scenario_positions + (len(txPower) - 1) * width / 2)
    ax.set_xticklabels(scenarioIDs, rotation=0, fontsize=12)
    ax.set_title(f'Effect of scenarioID and txPower on PDR for rfh-app-{app_id}', fontsize=16)
    ax.set_ylabel('PDR', fontsize=14)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=color, lw=4) for color in colors]
    ax.legend(custom_lines, [f'txPower={power}' for power in txPower], loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{base_folder}/Figures/scenario_effect_on_PDR_grouped_boxplot_rfh-app-{app_id}.png', bbox_inches='tight')
    plt.show()

############################################################
#       END  PDR                                            #
############################################################


###########################################################
#    Decoding Error                                    #
###########################################################
width = 0.2  # the width of the bars
space = 0.2  # space between the groups of bars
colors = plt.cm.viridis(np.linspace(0, 1, len(txPower)))

for app_id in tqdm(app_ids, desc="Generating grouped bar charts for Decoding Error", ncols=100):
    labels = scenario_effect_errors[app_id][txPower[0]].keys()
    x = np.arange(len(labels)) * (len(txPower) * width + space)  # adjusted for spacing between groups
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, power in enumerate(txPower):
        # Compute average and std deviation of Decoding Error values across all runNumbers
        mean_values = []
        std_values = []
        for scenario in scenarioIDs:
            errors_for_runNumbers = list(scenario_effect_errors[app_id][power][scenario].values())
            mean_error = np.mean(errors_for_runNumbers)
            std_error = np.std(errors_for_runNumbers)
            
            mean_values.append(mean_error)
            std_values.append(std_error)
        
        ax.bar(x + i * width, mean_values, yerr=std_values, width=width, label=f'txPower={power}', color=colors[i], edgecolor='black', alpha=0.85, capsize=5)
    
    ax.set_xlabel('ScenarioID', fontsize=14)
    ax.set_ylabel('Average Decoding Error', fontsize=14)
    ax.set_title(f'Effect of scenarioID on Decoding Error for rfh-app-{app_id}', fontsize=16)
    ax.set_xticks(x + (len(txPower) - 1) * width / 2)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{base_folder}/Figures/scenario_effect_on_decoding_error_rfh-app-{app_id}.png', bbox_inches='tight')
    plt.show()

# Boxplot
import matplotlib.pyplot as plt
import numpy as np

width = 0.2  # the width of the boxes
space = 0.5  # space between the groups of boxes

for app_id in tqdm(app_ids, desc="Generating grouped box plots for Decoding Error", ncols=100):
    fig, ax = plt.subplots(figsize=(14, 8))
    scenario_positions = np.arange(len(scenarioIDs)) * (len(txPower) * width + space)
    colors = plt.cm.viridis(np.linspace(0, 1, len(txPower)))
    box_data = []
    box_labels = []
    box_colors = []
    
    for i, scenario in enumerate(scenarioIDs):
        for j, power in enumerate(txPower):
            try:
                errors = [scenario_effect_errors[app_id][power][scenario][run] for run in runNumbers]
                box_data.append(errors)
                box_labels.append(scenario_positions[i] + j * width)
                box_colors.append(colors[j])
            except KeyError:
                continue

    bp = ax.boxplot(box_data, positions=box_labels, widths=width, patch_artist=True)

    # Styling the box plots
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    for whisker in bp['whiskers']:
        whisker.set(color='black', linestyle='-')
    for cap in bp['caps']:
        cap.set(color='black', linestyle='-')
    for median in bp['medians']:
        median.set(color='red', linewidth=1.5)
    for flier in bp['fliers']:
        flier.set(marker='o', color='black', alpha=0.5)

    ax.set_xticks(scenario_positions + (len(txPower) - 1) * width / 2)
    ax.set_xticklabels(scenarioIDs, rotation=0, fontsize=12)
    ax.set_title(f'Effect of scenarioID and txPower on Decoding Error for rfh-app-{app_id}', fontsize=16)
    ax.set_ylabel('Decoding Error', fontsize=14)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=color, lw=4) for color in colors]
    ax.legend(custom_lines, [f'txPower={power}' for power in txPower], loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{base_folder}/Figures/scenario_effect_on_decoding_error_grouped_boxplot_rfh-app-{app_id}.png', bbox_inches='tight')
    plt.show()

# ############################################################
# #       END  Decoding Error                                #
# ############################################################



# CDF plots for throughput per app for each tx power
colors = plt.cm.viridis(np.linspace(0, 1, len(txPower)))

for app_id in tqdm(app_ids, desc="Generating CDF plots", ncols=100):
    plt.figure()
    for power in txPower:
        data = throughput_values_for_cdf[app_id][power]
        sorted_data = np.sort(data)
        prob = np.arange(len(sorted_data)) / float(len(sorted_data))
        plt.plot(sorted_data, prob, label=f'txPower={power}')
    
    plt.xlabel('Throughput (Mbps)')
    plt.ylabel('Cumulative Probability')
    plt.title(f'CDF of Throughput for rfh-app-{app_id}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{base_folder}/Figures/cdf_throughput_rfh-app-{app_id}_Run{runNumber}.png')
    plt.show()
    
    
    


