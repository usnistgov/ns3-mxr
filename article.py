import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from scipy import stats
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines
import itertools
import gc
import matplotlib.ticker as ticker

ENABLE_CDF_SINR = True
ENABLE_MCS_GENERAL = False
ENABLE_DELAY_PER_SCENARIO = False
ENABLE_DELAY_GENERAL = True
ENABLE_THROUGHPUT_GENERAL = False
ENABLE_THROUGHPUT_PER_SCENARIO = False
# NO DELAY DONE
ENABLE_THROUGHPUT_INDIVIDUAL = False
ENABLE_DELAY_INDIVIDUAL = False
ENABLE_MCS_SINR_INDIVIDUAL = False
ENABLE_SINR_BEAMINTERVAL_INDIVIDUAL = False # NOT DONE


throughputCollected = False

# Assuming these are defined somewhere in your script:
base_folder = "ResultsWithTraces"
app_ids = [6250]
# app_ids = [7,6,610]
# app_ids = [6]
scenarioIDs = [1,2,3,4,5,6,7,8,9,10]
# scenarioIDs = [5]
runNumbers = [1,2,3,4,5]
# runNumbers = [1]
# runNumbers = [3,4]
txPower = [20.0]
beamformingIntervals = [1]
# beamformingIntervals = [1,2]
codebookFiles = ["1x16.txt"]
rlc_mode = 'rlcAmEnabled'  # Example RLC mode, replace with actual variable if needed

# Initialize your data structures
scenario_effect_throughput = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
scenario_effect_PDR = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
throughput_values_per_combination = defaultdict(list)

# Define the base folders
figures_folder = os.path.join(base_folder, 'Figures')


app_throughput_map = {
    1: '1.27',  
    2: '0.84', 
    3: '0.96', 
    4: '1.20', 
    5: '1.28', 
    6: '2.18', 
    7: '0.46', 
    610: '22', 
    650: '110', 
    6100: '220', 
    6250: '545', 
    6500: '990', 
    # ... and so on for each app_id
}

# codebook_text_map = {
#     "1x16.txt": 'ULA - 1x16',  
#     "2x16.txt": 'URA - 2x16', 
# }

codebook_text_map = {
    "1x16.txt": 'ULA - 1x16',  
    
}


power_labels_gen = [f'{power}' for power in txPower]  # Legend labels as just the Tx Power values with units
codebook_labels_gen = list(codebook_text_map.values())
##############################################################
################### BOXPLOT STYLE ############################
##############################################################
meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor='firebrick')
displayMean = True
medianColor = dict(color='red')




########### To make a function


def collect_mcs_data_indiv(txPower, app_ids, codebookFiles, scenarioIDs, beamformingIntervals, runNumbers, base_folder):
    mcs_values_dict = {}
    total_iterations = len(app_ids) * len(codebookFiles) * len(scenarioIDs) * len(beamformingIntervals) * len(runNumbers) * len(txPower)
    progress_bar = tqdm(total=total_iterations, desc="Overall Progress", ncols=100)
    
    for app_id in app_ids:
        for codebookFile in codebookFiles:
            for scenarioID in scenarioIDs:
                for beamformingInterval in beamformingIntervals:
                    for runNumber in runNumbers:
                        for power in txPower:
                            progress_bar.set_description(f"App: {app_id}, Codebook: {codebookFile}, Scenario: {scenarioID}, BI: {beamformingInterval}, Run: {runNumber}, Power: {power}")
                            UEStatsFile = f'{base_folder}/{codebookFile}/BeamFInterval{beamformingInterval}/SurgeryFeedbackPos{scenarioID}_UE_rlcAmEnabledrfh-app-{app_id}txPower{power}runNumber{runNumber}.txt'
                            
                            try:
                                # Include 'TIME' in the columns to be read
                                UEStatsData = pd.read_csv(UEStatsFile, sep=',', usecols=['TIME', 'MCS'])
                                # UEStatsData = UEStatsData[UEStatsData['MCS'].notna() & (UEStatsData['MCS'] != 0)]
                                UEStatsData = UEStatsData[UEStatsData['MCS'].notna()]
                                mcs_values_dict[(app_id, codebookFile, scenarioID, beamformingInterval, power, runNumber)] = UEStatsData
                            except FileNotFoundError:
                                print(f"UE statistics file not found: {UEStatsFile}")
                            finally:
                                progress_bar.update(1)
    
    progress_bar.close()
    return mcs_values_dict





########################################################################################
########################################################################################
#                                                                                      #
#                                                                                      #
#                                 SINR                                                  #
#                                                                                      #
# ######################################################################################
########################################################################################
sinr_base_folder = os.path.join(figures_folder, 'SINR')
os.makedirs(sinr_base_folder, exist_ok=True)
detailed_sinr_folder = os.path.join(sinr_base_folder, 'Detailed')
os.makedirs(detailed_sinr_folder, exist_ok=True)
def plot_cdf(data, label, linestyle, color, marker, markevery=0.1):
    print(marker)
    
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
    plt.plot(sorted_data, yvals, label=label, linestyle=linestyle, color=color, marker=marker, markevery=markevery)






if ENABLE_CDF_SINR:
    

    # Create the SINR folder and subfolders
    sinr_base_folder = os.path.join(figures_folder, 'SINR')
    os.makedirs(sinr_base_folder, exist_ok=True)
    subfolders = ['codebookEffect', 'beamIntervalEffect', 'TxPowerEffect']
    for subfolder in subfolders:
        os.makedirs(os.path.join(sinr_base_folder, subfolder), exist_ok=True)

    # Define line styles for different parameter values
    linestyles = ['-', '--', '-.', ':']


    min_snr = float('inf')
    max_snr = -float('inf')
    for power in txPower:
        for beamformingInterval in beamformingIntervals:  # Ensure this is defined in the correct scope
            for i, codebookFile in enumerate(codebookFiles):
                for scenarioID in scenarioIDs:
                    snr_file = os.path.join(base_folder, codebookFile, f'BeamFInterval{beamformingInterval}',
                                            f'SurgeryFeedbackPos{scenarioID}_snr_{rlc_mode}rfh-app-1txPower{power}runNumber1.txt')
                    if os.path.exists(snr_file):
                        snr_data = pd.read_csv(snr_file, sep=',', header=None)
                        snr_data.replace(-np.inf, np.nan, inplace=True)
                        snr_values = snr_data[1].dropna()

                        # Update min and max SINR for setting axis limits
                        min_snr = min(min_snr, snr_values.min())
                        max_snr = max(max_snr, snr_values.max())



    ######################################################################################
    #                                                                                    #
    #                            CDF TX POWER EFFECT                                     #
    #                                                                                    #
    ######################################################################################
     # Define colors for scenarios if you have a limited set of scenarios
    scenario_colors = [plt.cm.tab10(i) for i in range(len(scenarioIDs))]
    from itertools import cycle
    # Define different markers for each scenario    
    markers = ['o', 's', 'v', '^', '<', '>', 'p', 'P', '*', 'X']  # Add more if needed
    marker_cycle = cycle(markers)

    scenario_legend_handles = []
    for j, scenarioID in enumerate(scenarioIDs):
        marker = next(marker_cycle)
        scenario_legend_handles.append(mlines.Line2D([], [], color=scenario_colors[j], marker=marker, label=f'{scenarioID}'))


    tx_power_linestyles = ['-', '--', '-.', ':']

    # Plot CDF for the effect of TxPower
    for codebookFile, beamformingInterval in tqdm(
        itertools.product(codebookFiles, beamformingIntervals),
        total=len(codebookFiles)*len(beamformingIntervals),
        desc="Generate SINR CDF figures for the effect of Tx Power",
        ncols=100):
            plt.figure(figsize=(10, 5))

            # This will hold the custom legend entries for TxPower
            tx_power_legend_handles = []

            for i, power in enumerate(txPower):
                linestyle = tx_power_linestyles[i % len(tx_power_linestyles)]
                # Add a legend handle for this TxPower with a black line
                tx_power_legend_handles.append(mlines.Line2D([], [], color='black', linestyle=linestyle, label=f'{power}'))

                for j,scenarioID in enumerate(scenarioIDs):
                   
                    marker = markers[j % len(markers)]
                    color = scenario_colors[j]
                    # print("For Scenario ID:",j, " Color is set to:", scenarioID % 10)
                    
                    snr_file = os.path.join(base_folder, codebookFile, f'BeamFInterval{beamformingInterval}',
                                            f'SurgeryFeedbackPos{scenarioID}_snr_{rlc_mode}rfh-app-1txPower{power}runNumber1.txt')
                  
                    if os.path.exists(snr_file):
                        snr_data = pd.read_csv(snr_file, sep=',', header=None)
                        snr_data.replace(-np.inf, np.nan, inplace=True)
                        snr_values = snr_data[1].dropna()

                        # Plot the CDF without a label to avoid duplicate legend entries
                        print(marker)
                        plot_cdf(snr_values, None, linestyle, color, marker=marker)

            plt.xlabel('SINR (dB)',fontsize=20)
            plt.ylabel('Cumulative Probability', fontsize=20)
            codebook_value = codebook_text_map.get(codebookFile, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                
            # plt.title(f'CDF of SINR for Tx Power = 20 dBm')
            plt.xlim(20, 60)
            ax = plt.gca()
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.grid(which='both', linestyle='--', linewidth=0.5)
            ax.tick_params(axis='x', labelsize=18)  # Set x-axis tick label font size
            ax.tick_params(axis='y', labelsize=18)  # Set y-axis tick label font size

            # Add legends to the plot
            # Legend for beamforming intervals
            # plt.legend(handles=beam_interval_legend_handles, loc='upper left', bbox_to_anchor=(1, 0.5), title='Beamforming Intervals', borderaxespad=0.)
            # Add legends to the plot
            # first_legend = plt.legend(handles=scenario_legend_handles, loc='upper left', bbox_to_anchor=(1, 1.05), title='Scenarios')
            # ax = plt.gca().add_artist(first_legend)
            legend = plt.legend(handles=scenario_legend_handles, loc='upper left',title='BS position',fontsize=11,ncol=2)
            legend.get_title().set_fontsize('16')  # Set font size for the legend title

            
            # plt.legend(handles=tx_power_legend_handles, )  # Adjust fontsize as needed

            # plt.legend(handles=tx_power_legend_handles, loc='upper left', bbox_to_anchor=(1, 0.5), title='Tx Power (dBm)')

            # Plot formatting
            # plt.xlabel('SINR (dB)')
            # plt.ylabel('Cumulative Probability')
            # plt.title(f'CDF of SINR - Codebook {codebookFile}, BI {beamformingInterval}')
            # plt.xlim(min_snr, max_snr)
            # ax = plt.gca()
            # ax.xaxis.set_major_locator(MultipleLocator(10))
            # ax.xaxis.set_minor_locator(MultipleLocator(5))
            # ax.grid(which='both', linestyle='--', linewidth=0.5)

            # # Add legends to the plot
            # # Legend for TxPower
            # first_legend = plt.legend(handles=scenario_legend_handles, loc='upper left', bbox_to_anchor=(1, 1.05), title='Scenarios')
            # ax = plt.gca().add_artist(first_legend)
            # plt.legend(handles=tx_power_legend_handles, loc='upper left', bbox_to_anchor=(1, 0.5), title='BI', borderaxespad=0.)

            # # Optional: If you want to add scenario colors to the legend, uncomment this block
            # scenario_legend_handles = [mlines.Line2D([], [], color=plt.cm.tab10(scenarioID % 10), label=f'Scenario {scenarioID}') for scenarioID in scenarioIDs]
            # ax.add_artist(plt.legend(handles=scenario_legend_handles, loc='upper left', bbox_to_anchor=(1, 1.05), title='Scenarios'))

            # Save the figure in the appropriate subfolder
            output_folder = os.path.join(sinr_base_folder, 'TxPowerEffect')
            os.makedirs(output_folder, exist_ok=True)
            plt.savefig(os.path.join(output_folder, f'cdf_snr_codebook{codebookFile}_BI{beamformingInterval}.pdf'), bbox_inches='tight')
            plt.close()

    
    ######################################################################################
    #                                                                                    #
    #                            CDF BEAMFORMING INTERVAL EFFECT                         #
    #                                                                                    #
    ######################################################################################
   
        
    # Plot CDF for the effect of beamforming intervals
    # for power in txPower:
    #     for codebookFile in codebookFiles:
    for power, codebookFile in tqdm(
        itertools.product(txPower, codebookFiles),
        total=len(txPower)*len(codebookFiles),
        desc="Generate SINR CDF figures for the effect of Beamforming Intervals",
        ncols=100):
            plt.figure(figsize=(10, 6))

            # This will hold the custom legend entries for beamforming intervals
            beam_interval_legend_handles = []
            # This will hold the custom legend entries
            

            for i, beamformingInterval in enumerate(beamformingIntervals):
                linestyle = linestyles[i % len(linestyles)]
                # Add a legend handle for this beamforming interval with a black line
                beam_interval_legend_handles.append(mlines.Line2D([], [], color='black', linestyle=linestyle, label=f'{beamformingInterval}s'))

                for j,scenarioID in enumerate(scenarioIDs):
                    color = scenario_colors[j]
                    # Add a legend handle for this scenario
                    
                    snr_file = os.path.join(base_folder, codebookFile, f'BeamFInterval{beamformingInterval}',
                                            f'SurgeryFeedbackPos{scenarioID}_snr_{rlc_mode}rfh-app-1txPower{power}runNumber1.txt')
                    if os.path.exists(snr_file):
                        snr_data = pd.read_csv(snr_file, sep=',', header=None)
                        snr_data.replace(-np.inf, np.nan, inplace=True)
                        snr_values = snr_data[1].dropna()

                        # Plot the CDF without a label to avoid duplicate legend entries
                        # plot_cdf(snr_values, None, linestyle, color)

            # Plot formatting
            plt.xlabel('SINR (dB)')
            plt.ylabel('Cumulative Probability')
            codebook_value = codebook_text_map.get(codebookFile, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
            
            plt.title(f'CDF of SINR - Codebook: {codebook_value}, Tx Power: {power} dBm')
            plt.xlim(min_snr, max_snr)
            ax = plt.gca()
            ax.xaxis.set_major_locator(MultipleLocator(10))
            ax.xaxis.set_minor_locator(MultipleLocator(5))
            ax.grid(which='both', linestyle='--', linewidth=0.5)

            # Add legends to the plot
            # Legend for beamforming intervals
            # plt.legend(handles=beam_interval_legend_handles, loc='upper left', bbox_to_anchor=(1, 0.5), title='Beamforming Intervals', borderaxespad=0.)
            # Add legends to the plot
            first_legend = plt.legend(handles=scenario_legend_handles, loc='upper left', bbox_to_anchor=(1, 1.05), title='Scenarios')
            ax = plt.gca().add_artist(first_legend)
            plt.legend(handles=beam_interval_legend_handles, loc='upper left', bbox_to_anchor=(1, 0.5), title='Beam Itvl')

            # Optional: If you want to add scenario colors to the legend, uncomment this block
            # scenario_legend_handles = [mlines.Line2D([], [], color=scenario_colors[i], label=f'Scenario {scenarioID}') for i, scenarioID in enumerate(scenarioIDs)]
            # ax.add_artist(plt.legend(handles=scenario_legend_handles, loc='upper left', bbox_to_anchor=(1, 1.05), title='Scenarios'))

            # Save the figure in the appropriate subfolder
            output_folder = os.path.join(sinr_base_folder, 'beamIntervalEffect')
            os.makedirs(output_folder, exist_ok=True)
            plt.savefig(os.path.join(output_folder, f'cdf_snr_codebook{codebookFile}_txPower{power}.png'), bbox_inches='tight')
            plt.close()



    ######################################################################################
    #                                                                                    #
    #                            CDF CODEBOOK EFFECT                                     #
    #                                                                                    #
    ######################################################################################
    # Plot CDF for the effect of codebooks
    # for power in txPower:
    #     for beamformingInterval in beamformingIntervals:
    
    for power, beamformingInterval in tqdm(
        itertools.product(txPower, beamformingIntervals),
        total=len(txPower)*len(beamformingIntervals),
        desc="Generate SINR CDF figures for the effect of Codebooks",
        ncols=100):

            plt.figure(figsize=(10, 6))
            
            # This will hold the custom legend entries
            codebook_legend_handles = []

            # for scenarioID in scenarioIDs:
            #     # Use a consistent color for each scenario
            #     color = plt.cm.tab10(scenarioID % 10)
            #     # Add a legend handle for this scenario
            #     scenario_legend_handles.append(mlines.Line2D([], [], color=color, label=f'Scenario {scenarioID}'))
        
            for i, codebookFile in enumerate(codebookFiles):
                codebook_value = codebook_text_map.get(codebookFile, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
            
                linestyle = linestyles[i % len(linestyles)]
                # Add a legend handle for this codebook with a black line
                codebook_legend_handles.append(mlines.Line2D([], [], color='black', linestyle=linestyle, label=f'{codebook_value}'))
                
                for j,scenarioID in enumerate(scenarioIDs):
                    color = scenario_colors[j]
                    snr_file = os.path.join(base_folder, codebookFile, f'BeamFInterval{beamformingInterval}',
                                            f'SurgeryFeedbackPos{scenarioID}_snr_{rlc_mode}rfh-app-1txPower{power}runNumber1.txt')
                    if os.path.exists(snr_file):
                        snr_data = pd.read_csv(snr_file, sep=',', header=None)
                        snr_data.replace(-np.inf, np.nan, inplace=True)
                        snr_values = snr_data[1].dropna()

                        # Plot the CDF
                        # plot_cdf(snr_values, None, linestyle, color)  # Do not assign label here

            # Plot formatting
            plt.xlabel('SINR (dB)')
            plt.ylabel('Cumulative Probability')
            plt.title(f'CDF of SINR for Tx Power: {power} dBm, Beam Itvl: {beamformingInterval}d')
            plt.xlim(min_snr, max_snr)
            ax = plt.gca()
            ax.xaxis.set_major_locator(MultipleLocator(10))
            ax.xaxis.set_minor_locator(MultipleLocator(5))
            ax.grid(which='both', linestyle='--', linewidth=0.5)
            
            # Add legends to the plot
            first_legend = plt.legend(handles=scenario_legend_handles, loc='upper left', bbox_to_anchor=(1, 1.05), title='Scenarios')
            ax = plt.gca().add_artist(first_legend)
            plt.legend(handles=codebook_legend_handles, loc='upper left', bbox_to_anchor=(1, 0.5), title='Codebooks')

            # Save the figure in the appropriate subfolder
            output_folder = os.path.join(sinr_base_folder, 'codebookEffect')
            os.makedirs(output_folder, exist_ok=True)
            plt.savefig(os.path.join(output_folder, f'cdf_snr_txPower{power}_beamf{beamformingInterval}.png'), bbox_inches='tight')
            plt.close()

########################################################################################
########################################################################################
#                                                                                      #
#                                                                                      #
#                                 MCS                                                  #
#                                                                                      #
# ######################################################################################
########################################################################################
def collect_mcs_data(txPower, app_ids, codebookFiles, scenarioIDs, beamformingIntervals, runNumbers, base_folder):
    mcs_values_dict = {}
    total_iterations = len(app_ids) * len(codebookFiles) * len(scenarioIDs) * len(beamformingIntervals) * len(runNumbers) * len(txPower)
    progress_bar = tqdm(total=total_iterations, desc="Overall Progress", ncols=100)
    for app_id in app_ids:
        for codebookFile in codebookFiles:
            for scenarioID in scenarioIDs:
                for beamformingInterval in beamformingIntervals:
                    for runNumber in runNumbers:
                        for power in txPower:
                            progress_bar.set_description(f"App: {app_id}, Codebook: {codebookFile}, Scenario: {scenarioID}, BI: {beamformingInterval}, Run: {runNumber}, Power: {power}")
                            UEStatsFile = f'{base_folder}/{codebookFile}/BeamFInterval{beamformingInterval}/SurgeryFeedbackPos{scenarioID}_UE_rlcAmEnabledrfh-app-{app_id}txPower{power}runNumber{runNumber}.txt'
                            try:
                                UEStatsData = pd.read_csv(UEStatsFile, sep=',', usecols=['MCS'])
                                # UEStatsData = UEStatsData[UEStatsData['MCS'].notna() & (UEStatsData['MCS'] != 0)]
                                UEStatsData = UEStatsData[UEStatsData['MCS'].notna() ]
                                mcs_values_dict[(app_id, codebookFile, scenarioID, beamformingInterval, power, runNumber)] = UEStatsData
                            except FileNotFoundError:
                                print(f"UE statistics file not found: {UEStatsFile}")
                            finally:
                                progress_bar.update(1)
    progress_bar.close()
    return mcs_values_dict

# def collect_mcs_data(txPower, app_ids, codebookFiles, scenarioIDs, beamformingIntervals, runNumbers, base_folder):
#     """
#     Collect MCS data from files for different app IDs, scenario IDs, run numbers, transmission power levels,
#     beamforming intervals, and codebook files.

#     Parameters:
#     txPower (list): List of transmission power levels.
#     app_ids (list): List of application IDs.
#     codebookFiles (list): List of codebook file names.
#     scenarioIDs (list): List of scenario IDs.
#     beamformingIntervals (list): List of beamforming intervals.
#     runNumbers (list): List of run numbers.
#     base_folder (str): Base directory where the files are stored.

#     Returns:
#     dict: Dictionary with keys as tuples of (app_id, codebookFile, scenarioID, beamformingInterval, txPower, runNumber) and
#           values as Series of the MCS data.
#     """
#     mcs_values_dict = {}
#     for app_id in tqdm(app_ids, desc="Processing app ids", ncols=100):
#         for codebookFile in codebookFiles:
#             for scenarioID in scenarioIDs:
#                 for beamformingInterval in beamformingIntervals:
#                     for runNumber in runNumbers:
#                         for power in txPower:
#                             # Construct the file path
#                             UEStatsFile = f'{base_folder}/{codebookFile}/BeamFInterval{beamformingInterval}/SurgeryFeedbackPos{scenarioID}_UE_rlcAmEnabledrfh-app-{app_id}txPower{power}runNumber{runNumber}.txt'
#                             try:
#                                 print("Handle: app_id", app_id, " codebookFile:",codebookFile, " scenarioID:",scenarioID," beamformingInterval:",beamformingInterval, " runNumber: ",runNumber, " Power:",power)
                           
#                                 # Load only the MCS column from the UE statistics file
#                                 UEStatsData = pd.read_csv(UEStatsFile, sep=',', usecols=['MCS'])

#                                 # Remove rows where MCS is NaN or 0
#                                 UEStatsData = UEStatsData[UEStatsData['MCS'].notna() & (UEStatsData['MCS'] != 0)]

#                                 # Add to the dictionary
#                                 mcs_values_dict[(app_id, codebookFile, scenarioID, beamformingInterval, power, runNumber)] = UEStatsData
#                             except FileNotFoundError:
#                                 print(f"UE statistics file not found: {UEStatsFile}")
#                                 continue

#     return mcs_values_dict



mcs_folder = os.path.join(figures_folder, 'MCS')
detailed_mcs_folder = os.path.join(mcs_folder, 'Detailed')
os.makedirs(detailed_mcs_folder, exist_ok=True)
general_folder = os.path.join(mcs_folder, 'General')
os.makedirs(general_folder, exist_ok=True)
txPower_effect_folder = os.path.join(general_folder, 'PowerEffect')
os.makedirs(txPower_effect_folder, exist_ok=True)
codebook_effect_folder = os.path.join(general_folder, 'CodebookEffect')
os.makedirs(codebook_effect_folder, exist_ok=True)
beamInterval_effect_folder = os.path.join(general_folder, 'BIntervalEffect')
os.makedirs(beamInterval_effect_folder, exist_ok=True)
if ENABLE_MCS_GENERAL:
    

    print("MCS: Collect MCS Data")
    mcs_data = collect_mcs_data(txPower, app_ids, codebookFiles, scenarioIDs, beamformingIntervals, runNumbers, base_folder)
    print("MCS: Data Collected")

    trends_folder = os.path.join(beamInterval_effect_folder, 'Trends')
    os.makedirs(trends_folder, exist_ok=True)


    global_min_mcs = min(df['MCS'].min() for df in mcs_data.values())
    global_max_mcs = max(df['MCS'].max() for df in mcs_data.values())

    # global_min_mcs = min(mcs_series.min() for mcs_series in mcs_data.values())
    # global_max_mcs = max(mcs_series.max() for mcs_series in mcs_data.values())


    # ######################################################################################
    # #                                                                                    #
    # #                            MCS BEAM INTERVAL EFFECT                              #
    # #                                                                                    #
    # ######################################################################################
    # Begin of the main loop to generate plots
    print("MCS: Create figures for Beam Interval Effect")
    for power in tqdm(txPower, desc='Tx Power Levels', ncols=100):
        for app_id in tqdm(app_ids, desc='App IDs', leave=False, ncols=100):
            for codebookFile in tqdm(codebookFiles, desc='Codebook Files', leave=False, ncols=100):
                # We will reuse this dictionary to keep the proportions of MCS for each scenario and beamforming interval
                mcs_proportion_per_scenario_beamf = {
                    (scenarioID, beamformingInterval): []  # This will now hold a structure with MCS proportions
                    for scenarioID in scenarioIDs
                    for beamformingInterval in beamformingIntervals
                }
                
                for scenarioID in tqdm(scenarioIDs, desc='Scenario IDs', leave=False, ncols=100):
                    for beamformingInterval in tqdm(beamformingIntervals, desc='Beamforming Intervals', leave=False, ncols=100):
                 
                        # Initialize an empty dictionary to store MCS counts for the current scenario and beamforming interval
                        mcs_counts = {mcs: 0 for mcs in range(global_min_mcs, global_max_mcs + 1)}

                        # Iterate through the runs and accumulate the MCS counts
                        for runNumber in runNumbers:
                            key = (app_id, codebookFile, scenarioID, beamformingInterval, power, runNumber)
                            if key in mcs_data:
                                mcs_data_frame = mcs_data[key]

                                # Count the occurrences of each MCS in the data frame
                                mcs_counts_run = mcs_data_frame['MCS'].value_counts().to_dict()
                                
                                # Add the counts from this run to the total counts
                                for mcs, count in mcs_counts_run.items():
                                    mcs_counts[mcs] += count

                        # Calculate total MCS occurrences to determine proportions
                        total_mcs_occurrences = sum(mcs_counts.values())

                        # If we have no MCS occurrences for this combination, continue to the next one
                        if total_mcs_occurrences == 0:
                            continue

                        # Calculate the proportion of each MCS value
                        mcs_proportions = {mcs: count / total_mcs_occurrences for mcs, count in mcs_counts.items()}
                        
                        # Store the proportions in the mcs_proportion_per_scenario_beamf dictionary
                        mcs_proportion_per_scenario_beamf[(scenarioID, beamformingInterval)] = mcs_proportions

                # The rest of the code for data collection remains unchanged
                # ... existing data aggregation logic ...

                # Now we have the data, let's plot the stacked MCS bar chart
                # Generate the new labels for the X-axis
                labels = [f'Scen : {scenarioID} BI: {beamformingInterval}s' 
                        for scenarioID in scenarioIDs 
                        for beamformingInterval in beamformingIntervals]

                # The number of ticks on the X-axis will be the number of scenarios times the number of beamforming intervals
                x = np.arange(len(labels))  # the label locations

                fig, ax = plt.subplots(figsize=(15, 10))

                # Create a colormap for MCS values using RdYlGn
                mcs_colors = plt.cm.RdYlGn(np.linspace(0, 1, global_max_mcs - global_min_mcs + 1))

                # We only need one bar per scenario and beamforming interval combination, so we set the width accordingly
                width = 0.8  # Width of each bar

                # Bar chart plotting for MCS stacked representation

                sceneLabels = sorted(scenarioIDs)
                ### ICI
                # Boxplot plotting
                # positions = np.array(range(len(labels))) * (len(beamformingIntervals) + 1)  # Set starting position for each group of boxplots
             
                n_scenarios = len(sceneLabels)  # Assuming labels corresponds to scenario IDs
                n_beamformingIntervals = len(beamformingIntervals)

                # Calculate the positions
                positions = np.array(range(n_scenarios * n_beamformingIntervals)) + np.repeat(np.arange(n_scenarios), n_beamformingIntervals) * (n_beamformingIntervals - 1)
               
               
               


                bottom = np.zeros(len(labels))  # Reset bottom for the stacked bars
                for mcs in range(global_min_mcs, global_max_mcs + 1):
                    mcs_proportions = []
                    # Retrieve the proportions for each MCS and combination of scenarioID and beamformingInterval
                    for scenarioID in scenarioIDs:
                        for beamformingInterval in beamformingIntervals:
                            mcs_proportions.append(
                                mcs_proportion_per_scenario_beamf[(scenarioID, beamformingInterval)].get(mcs, 0)
                            )
                    
                    
                    # Plotting a stacked bar for each MCS value

                    ax.bar(positions, mcs_proportions, width, bottom=bottom, label=f'{mcs}', 
                        color=mcs_colors[mcs - global_min_mcs], edgecolor='black')
                    bottom += np.array(mcs_proportions)

                ax.set_ylabel('MCS Proportion')
                ax.set_xlabel('Scenario ID and Beamforming Interval (s)')
                throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                codebook_value = codebook_text_map.get(codebookFile, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                ax.set_title(f'MCS Proportion per Scenario and Beamforming Interval for Tx Power: {power} dBm, App ID: {app_id} (~{throughput_value}Mbps), Codebook File: {codebook_value}')
                
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=90)  # Rotate labels to prevent overlap
                

                # Create a custom legend for MCS values with two columns, outside the loop
                handles = [Patch(facecolor=mcs_colors[mcs - global_min_mcs], edgecolor='black', label=f'{mcs}') 
                        for mcs in range(global_min_mcs, global_max_mcs + 1)]
                ax.legend(handles=handles, title="MCS Values", loc='upper left', bbox_to_anchor=(1, 1), ncol=2)

                fig.tight_layout(pad=2.0)

                # Save the stacked MCS bar chart figure
                plt.savefig(os.path.join(trends_folder, f'mcs_proportion_barchart_app{app_id}_TxPower{power}_codebookFile{codebookFile}.png'), bbox_inches='tight')
                plt.close(fig)




    trends_folder = os.path.join(codebook_effect_folder, 'Trends')
    os.makedirs(trends_folder, exist_ok=True)
    # ######################################################################################
    # #                                                                                    #
    # #                            MCS   CODEBOOK  EFFECT                                  #
    # #                                                                                    #
    # ######################################################################################
    print("MCS: Create figures for Codebook Effect")
    for power in tqdm(txPower, desc='Tx Power Levels', ncols=100):
        # Wrap the next loop for application IDs
        for app_id in tqdm(app_ids, desc='App IDs', leave=False, ncols=100):
            # Wrap the loop for beamforming intervals
            for beamformingInterval in tqdm(beamformingIntervals, desc='Beamforming Intervals', leave=False, ncols=100):
                # We will reuse this dictionary to keep the proportions of MCS for each scenario and codebook
                mcs_proportion_per_scenario_codebook = {
                    (scenarioID, codebookFile): {}
                    for scenarioID in scenarioIDs
                    for codebookFile in codebookFiles
                }

                # Iterate through the data to calculate MCS proportions
                for scenarioID in tqdm(scenarioIDs, desc='Scenario IDs', leave=False, ncols=100):
                    # Wrap the loop for codebook files
                    for codebookFile in tqdm(codebookFiles, desc='Codebook Files', leave=False, ncols=100):
                        # Initialize an empty dictionary to store MCS counts for the current scenario and codebook
                        mcs_counts = {mcs: 0 for mcs in range(global_min_mcs, global_max_mcs + 1)}

                        # Iterate through the runs and accumulate the MCS counts
                        for runNumber in runNumbers:
                            
                            key = (app_id, codebookFile, scenarioID, beamformingInterval, power, runNumber)
                            if key in mcs_data:
                                mcs_data_frame = mcs_data[key]
                            
                                mcs_counts_run = mcs_data_frame['MCS'].value_counts().to_dict()

                                # Add the counts from this run to the total counts
                                for mcs, count in mcs_counts_run.items():
                                    mcs_counts[mcs] += count

                        # Calculate total MCS occurrences to determine proportions
                        total_mcs_occurrences = sum(mcs_counts.values())
                        # If we have no MCS occurrences for this combination, skip it
                        if total_mcs_occurrences == 0:
                            continue

                        # Calculate the proportion of each MCS value
                        mcs_proportions = {mcs: count / total_mcs_occurrences for mcs, count in mcs_counts.items()}
                        mcs_proportion_per_scenario_codebook[(scenarioID, codebookFile)] = mcs_proportions

                # Now we have the data, let's plot the stacked MCS bar chart
                # Generate the new labels for the X-axis
                labels = [f'Scen: {scenarioID} CB: {codebook_labels_gen[codebook_index]}'
                for scenarioID in scenarioIDs
                    for codebook_index in range(len(codebook_labels_gen))]


                       

                # The number of ticks on the X-axis will be the number of scenarios times the number of codebooks
                x = np.arange(len(labels))  # the label locations

                sceneLabels = sorted(scenarioIDs)
                # Boxplot plotting
                fig, ax = plt.subplots(figsize=(15, 10))
              
                n_scenarios = len(sceneLabels)  # Assuming labels corresponds to scenario IDs
                n_codebooks = len(codebookFiles)

                # Calculate the positions
                positions = np.array(range(n_scenarios * n_codebooks)) + np.repeat(np.arange(n_scenarios), n_codebooks) * (n_codebooks - 1)
               
               


                # Create a colormap for MCS values using RdYlGn
                mcs_colors = plt.cm.RdYlGn(np.linspace(0, 1, global_max_mcs - global_min_mcs + 1))

                # We only need one bar per scenario and codebook combination, so we set the width accordingly
                width = 0.8  # Width of each bar

                # Bar chart plotting for MCS stacked representation
                bottom = np.zeros(len(labels))  # Reset bottom for the stacked bars
                for mcs in range(global_min_mcs, global_max_mcs + 1):
                    mcs_proportions = []
                    # Retrieve the proportions for each MCS and combination of scenarioID and codebookFile
                    for scenarioID in scenarioIDs:
                        for codebookFile in codebookFiles:
                            mcs_proportions.append(
                                mcs_proportion_per_scenario_codebook[(scenarioID, codebookFile)].get(mcs, 0)
                            )

                    # Plotting a stacked bar for each MCS value
                    ax.bar(positions, mcs_proportions, width, bottom=bottom, label=f'{mcs}',
                        color=mcs_colors[mcs - global_min_mcs], edgecolor='black')
                    bottom += np.array(mcs_proportions)

                ax.set_ylabel('MCS Proportion')
                ax.set_xlabel('Scenario and Codebook')
                throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                codebook_value = codebook_text_map.get(codebookFile, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                ax.set_title(f'MCS Proportion per Scenario and Codebook for Tx Power: {power} dBm, App ID: {app_id} (~{throughput_value}Mbps), Beam Itvl: {beamformingInterval}s')
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=90, ha="right")
                ax.legend(title="MCS Values", bbox_to_anchor=(1.05, 1), loc='upper left')

                # Save the figure
                plt.savefig(os.path.join(trends_folder, f'MCS_proportion_scenario_codebook_TxPower{power}_App{app_id}_Beamforming{beamformingInterval}.png'), bbox_inches='tight')
                plt.close(fig)


    # ######################################################################################
    # #                                                                                    #
    # #                            MCS   TX POWER  EFFECT                                  #
    # #                                                                                    #
    # ######################################################################################
    trends_folder = os.path.join(txPower_effect_folder, 'Trends')
    os.makedirs(trends_folder, exist_ok=True)

    # Loop through all combinations of codebook files, beamforming intervals, and app IDs
    print("MCS: Create figures for Tx Power Effect")
    for codebookFile in tqdm(codebookFiles, desc="Processing codebook files", ncols=100):
        for beamformingInterval in tqdm(beamformingIntervals, desc="Beamforming Intervals", leave=False, ncols=100):
            for app_id in tqdm(app_ids, desc="App IDs", leave=False, ncols=100):

                # We will reuse this dictionary to keep the proportions of MCS for each scenario and power level
                mcs_proportion_per_scenario_power = {
                    (scenarioID, power): {} for scenarioID in scenarioIDs for power in txPower
                }

                # Iterate through the data to calculate MCS proportions
                for scenarioID in tqdm(scenarioIDs, desc="Scenarios", leave=False, ncols=100):
                    for power in tqdm(txPower, desc="Tx Power", leave=False, ncols=100):
                        # Initialize an empty dictionary to store MCS counts for the current scenario and power
                        mcs_counts = {mcs: 0 for mcs in range(global_min_mcs, global_max_mcs + 1)}

                        # Iterate through the runs and accumulate the MCS counts
                        for runNumber in runNumbers:
                            key = (app_id, codebookFile, scenarioID, beamformingInterval, power, runNumber)
                            if key in mcs_data:
                                mcs_data_frame = mcs_data[key]
                                mcs_counts_run = mcs_data_frame['MCS'].value_counts().to_dict()

                                # Add the counts from this run to the total counts
                                for mcs, count in mcs_counts_run.items():
                                    mcs_counts[mcs] += count

                        # Calculate total MCS occurrences to determine proportions
                        total_mcs_occurrences = sum(mcs_counts.values())

                        # If we have no MCS occurrences for this combination, skip it
                        if total_mcs_occurrences == 0:
                            continue

                        # Calculate the proportion of each MCS value
                        mcs_proportions = {mcs: count / total_mcs_occurrences for mcs, count in mcs_counts.items()}
                        mcs_proportion_per_scenario_power[(scenarioID, power)] = mcs_proportions

                # Now we have the data, let's plot the stacked MCS bar chart
                # Generate the new labels for the X-axis
                labels = [f'Scenario {scenarioID} Power: {power}' for scenarioID in scenarioIDs for power in txPower]

                n_scenarios = len(sceneLabels)  # Assuming labels corresponds to scenario IDs
                n_txPower = len(txPower)

                # Calculate the positions
                positions = np.array(range(n_scenarios * n_txPower)) + np.repeat(np.arange(n_scenarios), n_txPower) * (n_txPower - 1)
               

                # The number of ticks on the X-axis will be the number of scenarios times the number of Tx Powers
                x = np.arange(len(labels))  # the label locations

                fig, ax = plt.subplots(figsize=(15, 10))

                # Create a colormap for MCS values
                mcs_colors = plt.cm.RdYlGn(np.linspace(0, 1, global_max_mcs - global_min_mcs + 1))

                # Width of each bar
                width = 0.8

                # Bar chart plotting for MCS stacked representation
                bottom = np.zeros(len(labels))  # Reset bottom for the stacked bars
                for mcs in range(global_min_mcs, global_max_mcs + 1):
                    mcs_proportions = []
                    # Retrieve the proportions for each MCS and combination of scenarioID and Tx Power
                    for scenarioID in scenarioIDs:
                        for power in txPower:
                            mcs_proportions.append(
                                mcs_proportion_per_scenario_power[(scenarioID, power)].get(mcs, 0)
                            )

                    # Plotting a stacked bar for each MCS value
                    ax.bar(positions, mcs_proportions, width, bottom=bottom, label=f'{mcs}',
                        color=mcs_colors[mcs - global_min_mcs], edgecolor='black')
                    bottom += np.array(mcs_proportions)

                ax.set_ylabel('MCS Proportion')
                ax.set_xlabel('Scenario and Tx Power (dBm)')
                throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                codebook_value = codebook_text_map.get(codebookFile, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
               
                ax.set_title(f'MCS Proportion per Scenario and Tx Power for App ID {app_id} (~{throughput_value}Mbps), Codebook: {codebook_value}, Beam Itvl: {beamformingInterval}s')
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=90, ha="right")
                ax.legend(title="MCS Values", bbox_to_anchor=(1.05, 1), loc='upper left')

                # Save the figure
                plt.savefig(os.path.join(trends_folder, f'MCS_proportion_scenario_power_App{app_id}_Codebook{codebookFile}_Beamforming{beamformingInterval}.png'), bbox_inches='tight')
                plt.close(fig)







########################################################################################
########################################################################################
#                                                                                      #
#                                                                                      #
#                                 DELAY                                                #
#                                                                                      #
# ######################################################################################
########################################################################################


def collect_delay_data(txPower, app_ids, codebookFiles, scenarioIDs, beamformingIntervals, runNumbers, base_folder, rlc_mode):
    delay_data = {}

    # Total iterations for the progress bar
    total_iterations = len(txPower) * len(app_ids) * len(codebookFiles) * len(scenarioIDs) * len(beamformingIntervals) * len(runNumbers)
    progress_bar = tqdm(total=total_iterations, desc='Collecting delay data')

    for power in txPower:
        for app_id in app_ids:
            for codebookFile in codebookFiles:
                for scenarioID in scenarioIDs:
                    for beamformingInterval in beamformingIntervals:
                        for runNumber in runNumbers:
                            folder_path = os.path.join(base_folder, codebookFile, f"BeamFInterval{beamformingInterval}")
                            delay_file = os.path.join(
                                folder_path, f'SurgeryFeedbackPos{scenarioID}_delay_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt'
                            )
                            try:
                                # Read the delay data file
                                data = pd.read_csv(delay_file, sep='\s+', header=0)
                                # Ensure that the 'delay(ms)' column is numeric
                                # data['delay(ms)'] = pd.to_numeric(data['delay(ms)'], errors='coerce')
                                # I did a mistake during the export so the delay corresponds to the seqnum column
                                data['delay(ms)'] = pd.to_numeric(data['delay(ms)'], errors='coerce')
                                # Drop any rows that have NaN after conversion (should be none if the file is correct)
                                data.dropna(subset=['delay(ms)'], inplace=True)
                                # Convert delay to milliseconds
                               
                                data['delay(ms)'] = data['delay(ms)'].astype(float) / 1000000
                                # data['delay(ms)'] = data['delay(ms)'].astype(float) / 1000000
                               
                                # Aggregate the delay data
                                key = (power, app_id, codebookFile, scenarioID, beamformingInterval, runNumber)
                                # delay_data[key] = data['delay(ms)'].tolist()
                                delay_data[key] = data['delay(ms)'].tolist()
                                progress_bar.set_description(f"Processing power {power}, app_id {app_id}")
                           

                            except FileNotFoundError:
                                print(f"Delay file not found: {delay_file}")
                            except pd.errors.EmptyDataError:
                                print(f"No data in file: {delay_file}")
                            except Exception as e:
                                print(f"An error occurred while processing {delay_file}: {e}")
                            finally:
                                # Update progress bar at each iteration
                                progress_bar.update(1)

    progress_bar.close()
    return delay_data


delay_folder = os.path.join(figures_folder, 'Delay')
detailed_delay_folder = os.path.join(delay_folder, 'Detailed')
os.makedirs(detailed_delay_folder, exist_ok=True)
general_folder = os.path.join(delay_folder, 'General')
os.makedirs(general_folder, exist_ok=True)
txPower_effect_folder = os.path.join(general_folder, 'PowerEffect')
os.makedirs(txPower_effect_folder, exist_ok=True)
codebook_effect_folder = os.path.join(general_folder, 'CodebookEffect')
os.makedirs(codebook_effect_folder, exist_ok=True)
beamInterval_effect_folder = os.path.join(general_folder, 'BIntervalEffect')
os.makedirs(beamInterval_effect_folder, exist_ok=True)

if ENABLE_DELAY_PER_SCENARIO:
    delay_scenario_folder = os.path.join(general_folder, 'PerScenario')
    os.makedirs(delay_scenario_folder, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(codebookFiles)))
   
    edge_colors = ['black', 'red', 'blue']  # Distinct edge colors for 3 different TxPower values


    

    print("Delay Per Scenario Figures creation")
    print("Collect The Delay for a ScenarioID / App ID")
    for scenarioID in tqdm(scenarioIDs, desc='Scenario File Progress'):
        for app_id in  tqdm(app_ids, desc='App  Progress', leave=False):
            delay_data_collection = {
                (scenarioID, app_id): []
                for scenarioID in scenarioIDs
                for app_id in app_ids
            }
            # This list will hold dictionaries for all throughput values for the current scenarioID and app_id
            scenario_app_delay = []
            for power in  tqdm(txPower, desc='codebookFile  Progress', leave=False):
            # for power in txPower:
                for codebookFile in  tqdm(codebookFiles, desc='codebookFile  Progress', leave=False):
                # for codebookFile in codebookFiles:
                    for beamformingInterval in  tqdm(beamformingIntervals, desc='BI  Progress', leave=False):
                    # for beamformingInterval in beamformingIntervals:
                        all_delays = []
                        for runNumber in runNumbers:
                            # Extract the throughput values for the current combination
                            file_name = f'SurgeryFeedbackPos{scenarioID}_delay_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt'
                            file_path = os.path.join(base_folder, codebookFile, f"BeamFInterval{beamformingInterval}", file_name)
                            
                            # Attempt to read the delay data
                            try:
                                run_delays = pd.read_csv(file_path, sep='\s+', header=0)['delay(ms)'].tolist()
                                run_delays = [x / 1000000 for x in run_delays]  # Convert from ns to ms
                                all_delays.extend(run_delays)
                            except FileNotFoundError:
                                print(f"File not found: {file_path}")
                            except pd.errors.EmptyDataError:
                                print(f"No data in file: {file_path}")
                            except Exception as e:
                                print(f"Error processing file {file_path}: {e}")

                       
                            if all_delays:  # Ensure that there is data
                                # Store each throughput value along with its parameters
                                # for value in all_delays:
                                    scenario_app_delay.append({
                                        'delay': all_delays,
                                        'power': power,
                                        'beamformingInterval': beamformingInterval,
                                        'codebookFile': codebookFile
                                    })
                                
                                # scenario_app_delay.append({
                                #         'delay': all_delays,
                                #         'power': power,
                                #         'beamformingInterval': beamformingInterval,
                                #         'codebookFile': codebookFile
                                # })
                                
                            
            print("Data Collected for an App ID")
            # Once all values are collected for a scenarioID and app_id, add them to the dictionary
            delay_data_collection[(scenarioID, app_id)] = scenario_app_delay
            # pdr_data_collection[(scenarioID, app_id)] = scenario_app_pdr
            # print(delay_data_collection.keys())
            # print(scenario_app_delay)
            # exit()
            throughput_info_list = scenario_app_delay
            # for (scenarioID, app_id), throughput_info_list in delay_data_collection.items():
                
            fig, ax = plt.subplots(figsize=(15, 10))

            # Data structure for plotting
            plot_data = {}
            print("Plot Data Preparation")
            for power in  tqdm(txPower, desc='Power  Progress', leave=False):
            # for power in txPower:
                for beamIdx, beamformingInterval in enumerate(beamformingIntervals):
                    for codeIdx, codebookFile in enumerate(codebookFiles):
                        # Nested keys for power and beamforming interval
                        if (power, beamformingInterval) not in plot_data:
                            plot_data[(power, beamformingInterval)] = []
                        
                        # Filter and append throughput data
                        current_data = [info['delay'] for info in throughput_info_list if info['power'] == power
                                        and info['beamformingInterval'] == beamformingInterval 
                                        and info['codebookFile'] == codebookFile]
                        current_data = [item for sublist in current_data for item in sublist]
                        # exit()
                        
                        if current_data:
                            # print("DATA")
                            # print("\tCONFIG:", scenarioID, app_id, power,beamformingInterval,codebookFile)
                            plot_data[(power, beamformingInterval)].append(current_data)
                        # else:
                        #     print("NO DATA")
                        #     print("\tCONFIG:", scenarioID,app_id, power,beamformingInterval,codebookFile)

            # Boxplot positions and labels
            positions = []
            labels = []
            tick_positions = []  # For major ticks (TxPower levels)
            power_labels = []

            # Spacing configurations
            box_width = 0.1  # Width of each boxplot
            interval_spacing = 0.15  # Space between groups of boxplots for each beamformingInterval
            power_spacing = 0.7  # Space between major TxPower groups

            # Current position tracker
            current_position = 0
            print("Plot  Preparation Done")
        # Create grouped boxplot data
            for power_idx, power in enumerate(txPower):
                # Calculate the center position for the TxPower label
                power_center_position = current_position + (len(beamformingIntervals) * len(codebookFiles) * box_width + (len(beamformingIntervals) - 1) * interval_spacing) / 2
                tick_positions.append(power_center_position)
                power_labels.append(f"{power}")

                for beamIdx, beamformingInterval in enumerate(beamformingIntervals):
                    for codeIdx, _ in enumerate(codebookFiles):
                        codebook_data = plot_data.get((power, beamformingInterval), [])[codeIdx] if plot_data.get((power, beamformingInterval)) else []
                        positions.append(current_position)
                        current_position += box_width
                        labels.append(f"BI{beamformingInterval}s CB{codeIdx+1}")
                    # Add the interval spacing after each group of codebookFiles, but not after the last group in the beamformingInterval
                    if beamIdx < len(beamformingIntervals) - 1:
                        current_position += interval_spacing

                # Add the power spacing after each group of beamformingIntervals, but not after the last power level
                if power_idx < len(txPower) - 1:
                    current_position += power_spacing

            bar_width = 0.1
            
            # Plotting all boxplots with unique color and hatch patterns
            # Plotting all boxplots with unique color, hatch patterns, and edge colors
            for idx, (key, value) in enumerate(plot_data.items()):
                power, beamformingInterval = key
                power_idx = txPower.index(power)  # Get the index of the power value
                edge_color = edge_colors[power_idx]  # Direct index for edge colors
                for codeIdx, codebook_data in enumerate(value):
                    print("avant boxplot")
                    codebook_data_np = np.array(codebook_data)
                    box = ax.boxplot(codebook_data_np,
                                    positions=[positions[idx * len(codebookFiles) + codeIdx]],
                                    widths=box_width,
                                    meanprops=meanpointprops,
                                    showmeans=True,
                                    showfliers=False,
                                    medianprops=medianColor,
                                    whiskerprops={'color': colors[codeIdx], 'linewidth': 1.5},
                                    flierprops={'markerfacecolor': colors[codeIdx], 'markeredgecolor': colors[codeIdx], 'markersize': 10},
                                    capprops={'color': colors[codeIdx], 'linewidth': 1.5},
                                    patch_artist=True)
                    print("apres boxplot")
                    for patch in box['boxes']:
                        patch.set_facecolor(colors[codeIdx])  # Apply color
                        patch.set_edgecolor(edge_color)  # Set edge color
            print("Going to create the graph")
            # Formatting the plot
            # Set the TxPower labels at the bottom
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(power_labels, ha='center')
            ax.set_xlabel('Tx Power (dBm)')

            

            # Add another x-axis at the top for the BI and CB labels
            secondary_ax = ax.secondary_xaxis('top')
            # positions = [pos + 1 for pos in positions]
            secondary_ax.set_xticks(positions)
            secondary_ax.set_xticklabels(labels, rotation=90, ha="center")
            secondary_ax.set_xlabel('Beamforming Interval and Codebook')

        

            # Legend for Codebook Files and TxPower
            # Assuming 'colors' is a list of colors for each codebookFile
            # and 'edge_colors' is a list of colors for each TxPower
            # Create legend handles for the codebook colors
            codebook_legend_handles = [Patch(facecolor=color, label=f'{codebook_labels_gen[idx]}') for idx, color in enumerate(colors)]

        
            # Add the first legend to the plot for codebooks. Save the legend object in a variable.
            codebook_legend = ax.legend(handles=codebook_legend_handles, loc='upper left', bbox_to_anchor=(1, 0.5), title="Codebook")
            # Adjust layout to make room for the legend
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            ax.set_ylabel('Delay (ms)')
            throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                
            ax.set_title(f'Delay Boxplot Grouped by TxPower, Beamforming Interval, and Codebook for Scenario: {scenarioID} and App ID: {app_id} (~{throughput_value} Mbps)')

            # Draw vertical lines to separate major TxPower groups
            # for tick_position in tick_positions:
            #     ax.axvline(x=tick_position, color='gray', linestyle='--')

            plt.tight_layout()
            # plt.savefig(os.path.join(trends_folder_pdr, f'avg_pdr_barchart_app{app_id}_codebook{codebookFile}_beamf{beamformingInterval}.png'), bbox_inches='tight')
                
            plt.savefig(os.path.join(delay_scenario_folder, f'grouped_barchart_delay_scenario{scenarioID}_app{app_id}.png'))
            plt.close(fig)
            del fig, ax, codebook_data_np, box, codebook_legend, plot_data, throughput_info_list,delay_data_collection,scenario_app_delay

            # Optionally, you can also call the garbage collector manually
            gc.collect()



if ENABLE_DELAY_GENERAL:
    print("Delay: Collect Data")
    # Collect the delay data
   

    trends_folder = os.path.join(beamInterval_effect_folder, 'Trends')
    os.makedirs(trends_folder, exist_ok=True)
    # ######################################################################################
    # #                                                                                    #
    # #                            DELAY BEAM INTERVAL EFFECT                              #
    # #                                                                                    #
    # ######################################################################################
    # print("Delay: Create Figure for Beam Interval Effect")
    # for power in tqdm(txPower, desc='TxPower Progress'):
    #     for app_id in tqdm(app_ids, desc='App ID Progress', leave=False):
    #         for codebookFile in tqdm(codebookFiles, desc='Codebook File Progress', leave=False):
    #             avg_delay_per_scenario_beamf = {(scenarioID, beamformingInterval): []
    #                                             for scenarioID in scenarioIDs
    #                                             for beamformingInterval in beamformingIntervals}
    #             delay_data_per_scenario_beamf = {
    #                 (scenarioID, beamformingInterval): []
    #                 for scenarioID in scenarioIDs
    #                 for beamformingInterval in beamformingIntervals
    #             }

    #             # Aggregate data for each scenario and beamforming interval for average delay and boxplot
    #             for scenarioID in scenarioIDs:
    #                 for beamformingInterval in beamformingIntervals:
    #                     delays = []
    #                     for runNumber in runNumbers:
    #                         run_delays = delay_data[(power, app_id, codebookFile, scenarioID, beamformingInterval, runNumber)]
    #                         delays.extend(run_delays)
    #                         delay_data_per_scenario_beamf[(scenarioID, beamformingInterval)].extend(run_delays)

    #                     if delays:
    #                         avg_delay = np.mean(delays)
    #                         avg_delay_per_scenario_beamf[(scenarioID, beamformingInterval)].append(avg_delay)

    #             # Now we have the data, let's plot the bar chart
    #             labels = sorted(scenarioIDs)
    #             x = np.arange(len(labels))  # the label locations
    #             num_beamf_intervals = len(beamformingIntervals)
    #             total_width = 0.8  # Total width for all bars for one scenario
    #             width = total_width / num_beamf_intervals  # Width of each bar

    #             fig, ax = plt.subplots(figsize=(15, 10))
    #             colors = plt.cm.viridis(np.linspace(0, 1, num_beamf_intervals))

    #             # Bar chart plotting
    #             for idx, beamformingInterval in enumerate(beamformingIntervals):
    #                 avg_delays = [np.mean(avg_delay_per_scenario_beamf[(scenarioID, beamformingInterval)])
    #                             for scenarioID in labels]
    #                 rects = ax.bar(x + idx * width - (total_width - width) / 2, avg_delays, width, label=f'Beamforming Interval {beamformingInterval}',
    #                             color=colors[idx], edgecolor='black')

    #             ax.set_ylabel('Average Delay (ms)')
    #             ax.set_xlabel('Scenario ID')
    #             ax.set_title(f'Average Delay per Scenario for Tx Power {power}, App ID {app_id}, Codebook File {codebookFile}')
    #             ax.set_xticks(x)
    #             ax.set_xticklabels(labels)
    #             ax.legend()

    #             plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    #             fig.tight_layout(pad=2.0)

    #             # Save the bar chart figure
    #             plt.savefig(os.path.join(trends_folder, f'avg_delay_barchart_app{app_id}_TxPower{power}_codebookFile{codebookFile}.png'), bbox_inches='tight')
    #             plt.close(fig)

    #             # Boxplot plotting
    #             fig, ax = plt.subplots(figsize=(15, 10))
    #             positions = np.array(range(len(labels))) * (num_beamf_intervals + 1)  # Set starting position for each group of boxplots
    #             for idx, beamformingInterval in enumerate(beamformingIntervals):
    #                 data_to_plot = [delay_data_per_scenario_beamf[(scenarioID, beamformingInterval)] for scenarioID in labels]
    #                 ax.boxplot(data_to_plot, positions=positions + idx, widths=0.6, patch_artist=True,
    #                         boxprops=dict(facecolor=colors[idx], color=colors[idx]),
    #                         medianprops=dict(color='white'), whiskerprops=dict(color=colors[idx]),
    #                         capprops=dict(color=colors[idx]), flierprops=dict(markeredgecolor=colors[idx]))

    #             ax.set_xticks(positions + num_beamf_intervals / 2)
    #             ax.set_xticklabels(labels)

    #             # Adding legend for beamforming intervals
    #             legend_elements = [Patch(facecolor=colors[idx], edgecolor=colors[idx],
    #                                     label=f'Beamforming Interval {beamformingIntervals[idx]}') for idx in range(num_beamf_intervals)]
    #             ax.legend(handles=legend_elements, title="Beamforming Intervals")

    #             ax.set_ylabel('Delay (ms)')
    #             ax.set_xlabel('Scenario ID')
    #             ax.set_title(f'Boxplot of Delay per Scenario for Tx Power {power}, App ID {app_id}, Codebook File {codebookFile}')
    #             fig.tight_layout(pad=2.0)

    #             # Save the boxplot figure
    #             plt.savefig(os.path.join(trends_folder, f'boxplot_delay_app{app_id}_TxPower{power}_codebookFile{codebookFile}.png'), bbox_inches='tight')
    #             plt.close(fig)


    # print("Delay: Create Figure for Beam Interval Effect")
    # for power in tqdm(txPower, desc='TxPower Progress'):
    #     for app_id in tqdm(app_ids, desc='App ID Progress', leave=False):
    #         for codebookFile in tqdm(codebookFiles, desc='Codebook File Progress', leave=False):
    #             avg_delay_per_scenario_beamf = {(scenarioID, beamformingInterval): []
    #                                             for scenarioID in scenarioIDs
    #                                             for beamformingInterval in beamformingIntervals}
    #             delay_data_per_scenario_beamf = {
    #                 (scenarioID, beamformingInterval): []
    #                 for scenarioID in scenarioIDs
    #                 for beamformingInterval in beamformingIntervals
    #             }

    #             # Read data files and compute averages
    #             for scenarioID in scenarioIDs:
    #                 for beamformingInterval in beamformingIntervals:
    #                     all_delays = []
    #                     for runNumber in runNumbers:
    #                         file_name = f'SurgeryFeedbackPos{scenarioID}_delay_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt'
    #                         file_path = os.path.join(base_folder, codebookFile, f"BeamFInterval{beamformingInterval}", file_name)
    #                         try:
    #                             run_delays = pd.read_csv(file_path, sep='\s+', header=0)['delay(ms)'].tolist()
    #                             run_delays = [x / 1000000 for x in run_delays]  # Convert from ns to ms
    #                             all_delays.extend(run_delays)
    #                         except FileNotFoundError:
    #                             print(f"File not found: {file_path}")
    #                         except pd.errors.EmptyDataError:
    #                             print(f"No data in file: {file_path}")
    #                         except Exception as e:
    #                             print(f"Error processing file {file_path}: {e}")

    #                     # Update delay data for the current scenario and beamforming interval
    #                     delay_data_per_scenario_beamf[(scenarioID, beamformingInterval)].extend(all_delays)

    #                     # Calculate the average delay if there are any delays recorded
    #                     if all_delays:
    #                         avg_delay = np.mean(all_delays)
    #                         avg_delay_per_scenario_beamf[(scenarioID, beamformingInterval)].append(avg_delay)

    #             # Now we have the data, let's plot the bar chart
    #             labels = sorted(scenarioIDs)
    #             x = np.arange(len(labels))  # the label locations
    #             num_beamf_intervals = len(beamformingIntervals)
    #             total_width = 0.8  # Total width for all bars for one scenario
    #             width = total_width / num_beamf_intervals  # Width of each bar

    #             fig, ax = plt.subplots(figsize=(15, 10))
    #             colors = plt.cm.viridis(np.linspace(0, 1, num_beamf_intervals))

    #             # Bar chart plotting
    #             for idx, beamformingInterval in enumerate(beamformingIntervals):
    #                 avg_delays = [np.mean(avg_delay_per_scenario_beamf[(scenarioID, beamformingInterval)]) if avg_delay_per_scenario_beamf[(scenarioID, beamformingInterval)] else 0
    #                             for scenarioID in labels]
    #                 rects = ax.bar(x + idx * width - (total_width - width) / 2, avg_delays, width, label=f'{beamformingInterval}s',
    #                             color=colors[idx], edgecolor='black')

    #             ax.set_ylabel('Average Delay (ms)')
    #             ax.set_xlabel('Scenario ID')
    #             throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
    #             codebook_value = codebook_text_map.get(codebookFile, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
               
    #             ax.set_title(f'Average Delay per Scenario for Tx Power: {power} dBm, App ID {app_id} (~{throughput_value} Mbps), Codebook: {codebook_value}')
    #             ax.set_xticks(x)
    #             ax.set_xticklabels(labels)
    #             # ax.legend()
    #             ax.legend(title="Beamforming Interval", bbox_to_anchor=(1.05, 1), loc='upper left')

    #             plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    #             fig.tight_layout(pad=2.0)

    #             # Save the bar chart figure
    #             plt.savefig(os.path.join(trends_folder, f'avg_delay_barchart_app{app_id}_TxPower{power}_codebookFile{codebookFile}.png'), bbox_inches='tight')
    #             plt.close(fig)

    #             # Boxplot plotting
    #             fig, ax = plt.subplots(figsize=(15, 10))
    #             positions = np.array(range(len(labels))) * (num_beamf_intervals + 1)  # Set starting position for each group of boxplots
    #             for idx, beamformingInterval in enumerate(beamformingIntervals):
    #                 data_to_plot = [delay_data_per_scenario_beamf[(scenarioID, beamformingInterval)] for scenarioID in labels]
                    
    #                 ax.boxplot(data_to_plot, positions=positions + idx, widths=0.6, patch_artist=True,
    #                         boxprops=dict(facecolor=colors[idx], color=colors[idx]),
    #                         meanprops=meanpointprops,
    #                         showmeans=True,
    #                         medianprops=medianColor, whiskerprops=dict(color=colors[idx]),
    #                         showfliers=False,
    #                         capprops=dict(color=colors[idx]), flierprops=dict(markeredgecolor=colors[idx]))

                
    #             ax.set_xticks(positions + num_beamf_intervals // 2 -0.5)
    #             ax.set_xticklabels(labels)
    #             # exit()

    #             # Adding legend for beamforming intervals
    #             legend_elements = [Patch(facecolor=colors[idx], edgecolor=colors[idx],
    #                                     label=f'{beamformingIntervals[idx]}s') for idx in range(num_beamf_intervals)]
    #             ax.legend(handles=legend_elements, title="Beamforming Intervals",bbox_to_anchor=(1.05, 1), loc='upper left')

    #             ax.set_ylabel('Delay (ms)')
    #             ax.set_xlabel('Scenario ID')
    #             throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
    #             codebook_value = codebook_text_map.get(codebookFile, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
               
    #             ax.set_title(f'Boxplot of Delay per Scenario for Tx Power: {power} dBm, App ID: {app_id} (~{throughput_value} Mbps), Codebook File {codebook_value}')
    #             fig.tight_layout(pad=2.0)

    #             # Save the boxplot figure
               
    #             plt.savefig(os.path.join(trends_folder, f'boxplot_delay_app{app_id}_TxPower{power}_codebookFile{codebookFile}.png'), bbox_inches='tight')
    #             plt.close(fig)

    # trends_folder = os.path.join(codebook_effect_folder, 'Trends')
    # os.makedirs(trends_folder, exist_ok=True)
    # ######################################################################################
    # #                                                                                    #
    # #                            DELAY CODEBOOK  EFFECT                                  #
    # #                                                                                    #
    # ######################################################################################
    # print("Delay: Create Figure for Codebook Effect")
    # for power in tqdm(txPower, desc='TxPower Progress'):
    #     for app_id in tqdm(app_ids, desc='App ID Progress', leave=False):
    #         for beamformingInterval in tqdm(beamformingIntervals, desc='Beam Interval Progress', leave=False):
    #             delay_data_per_scenario_codebook = {
    #                 (scenarioID, codebookFile): []
    #                 for scenarioID in scenarioIDs
    #                 for codebookFile in codebookFiles
    #             }
                
                
    #             for scenarioID in scenarioIDs:
    #                 for codebookFile in codebookFiles:
    #                     all_delays = []
    #                     for runNumber in runNumbers:
                           
    #                         file_name = f'SurgeryFeedbackPos{scenarioID}_delay_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt'
    #                         file_path = os.path.join(base_folder, codebookFile, f"BeamFInterval{beamformingInterval}", file_name)
    #                         try:
    #                             run_delays = pd.read_csv(file_path, sep='\s+', header=0)['delay(ms)'].tolist()
    #                             run_delays = [x / 1000000 for x in run_delays]  # Convert from ns to ms
    #                             all_delays.extend(run_delays)
    #                         except FileNotFoundError:
    #                             print(f"File not found: {file_path}")
    #                         except pd.errors.EmptyDataError:
    #                             print(f"No data in file: {file_path}")
    #                         except Exception as e:
    #                             print(f"Error processing file {file_path}: {e}")

    #                     # Update delay data for the current scenario and beamforming interval
    #                     delay_data_per_scenario_codebook[(scenarioID, codebookFile)].extend(all_delays)

    #                     # Calculate the average delay if there are any delays recorded
    #                     if all_delays:
    #                         avg_delay = np.mean(all_delays)
    #                         delay_data_per_scenario_codebook[(scenarioID, codebookFile)].append(avg_delay)


    #             # Now that we have all the data, we can create a boxplot
    #             fig, ax = plt.subplots(figsize=(15, 10))
    #             boxplot_data = []
    #             boxplot_positions = []
    #             boxplot_colors = []
    #             colors = plt.cm.viridis(np.linspace(0, 1, len(codebookFiles)))
    #             color_map = dict(zip(codebookFiles, colors))
                
    #             # Define positions for each set of boxplots for each scenario
    #             scenario_positions = np.arange(len(scenarioIDs)) * (len(codebookFiles) + 1)
                
    #             for idx, scenarioID in enumerate(sorted(scenarioIDs)):
    #                 for jdx, codebookFile in enumerate(codebookFiles):
    #                     data = delay_data_per_scenario_codebook.get((scenarioID, codebookFile), [])
    #                     if data:
    #                         boxplot_data.append(data)
    #                         boxplot_positions.append(scenario_positions[idx] + jdx)
    #                         boxplot_colors.append(color_map[codebookFile])

    #             # Create the boxplot
    #             bplot = ax.boxplot(boxplot_data, positions=boxplot_positions, patch_artist=True, showfliers=False,meanprops=meanpointprops,
    #                         showmeans=True)
                
    #             # Color each boxplot by codebook
    #             for patch, color in zip(bplot['boxes'], boxplot_colors):
    #                 patch.set_facecolor(color)
                
    #             # Add legend for codebooks
    #             # legend_patches = [mpatches.Patch(color=color_map[cb], label=cb) for cb in codebookFiles]

    #             legend_elements = [Patch(facecolor=colors[idx], edgecolor=colors[idx],
    #                                     label=f'{codebook_labels_gen[idx]}') for idx in range(len(codebook_labels_gen))]

    #             ############ TOYO
    #             ax.legend(handles=legend_elements, title="Codebook",bbox_to_anchor=(1.05, 1), loc='upper left')
                
    #             # Set x-axis labels and title
    #             ax.set_xticks(np.mean(np.reshape(boxplot_positions, (len(scenarioIDs), len(codebookFiles))), axis=1))
    #             ax.set_xticklabels(sorted(scenarioIDs))
    #             throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                
    #             ax.set_title(f'Boxplot of Delays per Scenario for Tx Power: {power} dBm, App ID: {app_id} (~{throughput_value} Mbps), Beamforming Interval: {beamformingInterval}s')
    #             ax.set_xlabel('Scenario ID')
    #             ax.set_ylabel('Delay (ms)')
                
    #             # Save the figure
                
    #             plt.savefig(os.path.join(trends_folder, f'boxplot_delay_per_scenario_codebook_app{app_id}_TxPower{power}_beamf{beamformingInterval}.png'), bbox_inches='tight')
    #             plt.close(fig)
                
    #             # Now we prepare the data for the bar chart
    #             scenario_labels = sorted(scenarioIDs)
    #             codebook_labels = codebookFiles
    #             bar_width = 0.35  # the width of the bars
    #             n_codebooks = len(codebookFiles)

    #             # Set up the subplot
    #             fig, ax = plt.subplots(figsize=(10, 7))

    #             # Calculate bar positions
    #             index = np.arange(len(scenario_labels))
    #             bar_positions = dict(zip(codebook_labels, [index + bar_width * i for i in range(n_codebooks)]))

    #             # We will use the viridis colormap
    #             cmap = plt.get_cmap('viridis')
    #             colors = cmap(np.linspace(0, 1, n_codebooks))

    #             # Create bars for each codebook
    #             for idx, codebookFile in enumerate(codebook_labels):
    #                 delays = [np.mean(delay_data_per_scenario_codebook[(sid, codebookFile)]) for sid in scenario_labels]
    #                 ax.bar(bar_positions[codebookFile], delays, bar_width,  label=codebook_labels_gen[idx], color=colors[idx], edgecolor='black')

                   

    #             # Add some text for labels, title, and custom x-axis tick labels, etc.
    #             ax.set_xlabel('Scenario ID')
    #             ax.set_ylabel('Delay (ms)')
    #             throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                
    #             ax.set_title(f'Bar Chart of Delays per Scenario for Tx Power: {power} dBm, App ID: {app_id} (~{throughput_value} Mbps), Beamforming Interval: {beamformingInterval}s')
    #             ax.set_xticks(index + bar_width * (n_codebooks - 1) / 2)
    #             ax.set_xticklabels(scenario_labels)
    #             ax.legend(title = "Codebook",bbox_to_anchor=(1.05, 1), loc='upper left')

    #             # Save the figure
    #             plt.savefig(os.path.join(trends_folder, f'barchart_delay_per_scenario_codebook_app{app_id}_TxPower{power}_beamf{beamformingInterval}.png'), bbox_inches='tight')
    #             plt.close(fig)

                
    # ######################################################################################
    # #                                                                                    #
    # #                            DELAY TX POWER  EFFECT                                  #
    # #                                                                                    #
    # ######################################################################################
    print("Delay: Create Figure for Tx Power Effect")
    trends_folder = os.path.join(txPower_effect_folder, 'Trends')
    os.makedirs(trends_folder, exist_ok=True)
    # Loop through all combinations of codebook files, beamforming intervals, and app IDs
    for codebookFile in tqdm(codebookFiles, desc='Codebook File Progress'):
        for beamformingInterval in tqdm(beamformingIntervals, desc='Beamforming Interval Progress', leave=False):
            for app_id in tqdm(app_ids, desc='Ap ID progress', leave=False):
                print(f"\nAverage Delay Summary for App ID {app_id}, Codebook: {codebookFile}, Beamforming Interval: {beamformingInterval}s")
    # for codebookFile in tqdm(codebookFiles, desc="Processing codebook files", ncols=100):
    #     for beamformingInterval in beamformingIntervals:
    #         for app_id in app_ids:
                # Prepare data structure for average delay per scenario and Tx Power
                avg_delay_per_scenario_power = {(scenarioID, power): []
                                                for scenarioID in scenarioIDs
                                                for power in txPower}

                delay_data_per_scenario_power = {(scenarioID, power): []
                                                for scenarioID in scenarioIDs
                                                for power in txPower}

                # Prepare data structure for all delay values per scenario and Tx Power
              

                # Collect delay data for each scenario and Tx Power, filtered by app ID, codebook, and beamforming interval
                for scenarioID in scenarioIDs:
                    for power in txPower:
                        all_delays = []
                        for runNumber in runNumbers:
                           
                            file_name = f'SurgeryFeedbackPos{scenarioID}_delay_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt'
                            file_path = os.path.join(base_folder, codebookFile, f"BeamFInterval{beamformingInterval}", file_name)
                            try:
                                run_delays = pd.read_csv(file_path, sep='\s+', header=0)['delay(ms)'].tolist()
                                run_delays = [x / 1000000 for x in run_delays]  # Convert from ns to ms
                                all_delays.extend(run_delays)
                            except FileNotFoundError:
                                print(f"File not found: {file_path}")
                            except pd.errors.EmptyDataError:
                                print(f"No data in file: {file_path}")
                            except Exception as e:
                                print(f"Error processing file {file_path}: {e}")

                        # Update delay data for the current scenario and beamforming interval
                        delay_data_per_scenario_power[(scenarioID, power)].extend(all_delays)

                        # Calculate the average delay if there are any delays recorded
                        if all_delays:
                            avg_delay = np.mean(all_delays)
                            print(f"    Scenario {scenarioID}, Tx Power {power} dBm: Average Delay = {avg_delay:.2f} ms")

                            delay_data_per_scenario_power[(scenarioID, power)].append(avg_delay)



                       

                # Plotting bar charts for average delay
                labels = sorted(scenarioIDs)
                x = np.arange(len(labels))  # the label locations
                total_width = 0.8  # total width for all bars in a group
                num_bars = len(txPower)
                width = total_width / num_bars  # the width of each bar

                fig, ax = plt.subplots(figsize=(15, 10))
                colors = plt.cm.viridis(np.linspace(0, 1, num_bars))

                # Offset calculation to center bars
                offset = (total_width - width) / 2

                for idx, power in enumerate(txPower):
                    power_avg_delay = [
                        np.mean(delay_data_per_scenario_power.get((scenarioID, power), [0]))
                        for scenarioID in labels
                    ]
                    
                    
                    bar_positions = x - offset + idx * width
                    ax.bar(bar_positions, power_avg_delay, width, label=f'{power}',
                        color=colors[idx], edgecolor='black')

                ax.set_ylabel('Average Delay (ms)')
                ax.set_xlabel('Scenario ID')
                throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                codebook_value = codebook_text_map.get(codebookFile, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
               
                ax.set_title(f'Average Delay by Scenario and Tx Power for App ID: {app_id} (~{throughput_value} Mbps), Codebook: {codebook_value}, Beamforming Interval: {beamformingInterval}s')
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                # ax.legend(title = "Tx Power (dBm)",bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.legend(title = "Tx Power (dBm)", loc='upper right')
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                fig.tight_layout(pad=2.0)
                # plt.savefig(os.path.join(trends_folder, f'avg_delay_barchart_app{app_id}_codebook{codebookFile}_beamf{beamformingInterval}.png'), bbox_inches='tight')
                plt.savefig(os.path.join(trends_folder, f'avg_delay_barchart_app{app_id}_codebook{codebookFile}_beamf{beamformingInterval}.pdf'), bbox_inches='tight')
                plt.close(fig)

                # Plotting box plots for delay distribution
                boxplot_data = []
                for scenarioID in sorted(scenarioIDs):
                    scenario_data = []
                    for power in sorted(txPower):
                        scenario_data.append(delay_data_per_scenario_power[(scenarioID, power)])
                    boxplot_data.append(scenario_data)

                fig, ax = plt.subplots(figsize=(15, 10))
                colors = plt.cm.viridis(np.linspace(0, 1, len(txPower)))
                positions = np.array(range(len(scenarioIDs))) * (len(txPower) + 1)
                show_outliers = False
                legend_elements = []
                show_outliers = False
                for idx, power in enumerate(sorted(txPower)):
                    bp = ax.boxplot(
                        [d[idx] for d in boxplot_data if len(d[idx]) > 0],
                        positions=positions + idx,
                        widths=0.6,
                        patch_artist=True,
                        boxprops=dict(facecolor=colors[idx], color=colors[idx]),
                        meanprops=meanpointprops,
                        showmeans=True,
                        medianprops=medianColor,
                        whiskerprops=dict(color=colors[idx]),
                        capprops=dict(color=colors[idx]),
                        flierprops=dict(markerfacecolor=colors[idx], marker='o', markersize=6) if show_outliers else dict(marker=''),
                        showfliers=False,  # Correct usage of showfliers parameter
                        labels=scenarioIDs if idx == 0 else ['']*len(scenarioIDs)
                    )
                    legend_elements.append(bp['boxes'][0])

                # ax.legend(handles=legend_elements, labels=[f'{p}' for p in sorted(txPower)], title="Tx Power (dBm)", loc='upper right')
                # ax.set_ylabel('Delay (ms)')
                # ax.set_xlabel('Scenario ID')
                # Increase font size for the legend
                ax.legend(handles=legend_elements, labels=[f'{p}' for p in sorted(txPower)], 
                        title="Tx Power (dBm)", loc='upper right', fontsize=30, title_fontsize=30)

                # Increase font size for x and y ticks
                ax.tick_params(axis='x', labelsize=30)
                ax.tick_params(axis='y', labelsize=30)

                # Increase font size for x and y labels
                ax.set_ylabel('Delay (ms)', fontsize=32)
                ax.set_xlabel('BS Position', fontsize=32)

        
                # plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
                plt.grid(True, axis='y')
                throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                codebook_value = codebook_text_map.get(codebookFile, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                # ax.set_title(f'Boxplot of Delay by Scenario and Tx Power for App ID: {app_id} (~{throughput_value} Mbps), Codebook: {codebook_value}, Beamforming Interval: {beamformingInterval}s')
                ax.set_xticks(positions + len(txPower) // 2)
                ax.set_xticklabels(sorted(scenarioIDs))
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                fig.tight_layout(pad=2.0)
                # plt.savefig(os.path.join(trends_folder, f'ATEMPboxplot_delay_scenario_app{app_id}_codebook{codebookFile}_beamf{beamformingInterval}.png'), bbox_inches='tight')
                plt.savefig(os.path.join(trends_folder, f'ATEMPboxplot_delay_scenario_app{app_id}_codebook{codebookFile}_beamf{beamformingInterval}.pdf'), bbox_inches='tight')
                plt.close(fig)







# exit(1)



########################################################################################
########################################################################################
#                                                                                      #
#                                                                                      #
#                                 THROUGHPUT                                           #
#                                                                                      #
# ######################################################################################
########################################################################################


def collect_throughput_and_pdr_data(txPower, app_ids, codebookFiles, scenarioIDs, beamformingIntervals, runNumbers, base_folder, rlc_mode):
    global throughputCollected
    throughputCollected = True
    
    # Dictionary to store throughput data
    throughput_data = {
        (power, app_id, codebookFile, scenarioID, beamformingInterval, runNumber): []
        for power in txPower
        for app_id in app_ids
        for codebookFile in codebookFiles
        for scenarioID in scenarioIDs
        for beamformingInterval in beamformingIntervals
        for runNumber in runNumbers
    }

    # Dictionary to store Packet Delivery Ratio (PDR) data
    pdr_data = {
        (power, app_id, codebookFile, scenarioID, beamformingInterval, runNumber): None
        for power in txPower
        for app_id in app_ids
        for codebookFile in codebookFiles
        for scenarioID in scenarioIDs
        for beamformingInterval in beamformingIntervals
        for runNumber in runNumbers
    }

    # Loop over all the combinations to collect data
    total_iterations = len(txPower) * len(app_ids) * len(codebookFiles) * len(scenarioIDs) * len(beamformingIntervals) * len(runNumbers)
    with tqdm(total=total_iterations, desc="Collecting data", unit="iteration") as pbar:
        for power in txPower:
            for app_id in tqdm(app_ids, desc="App IDs", leave=False):
                for codebookFile in tqdm(codebookFiles, desc="Codebook Files", leave=False):
                    for scenarioID in scenarioIDs:
                        for beamformingInterval in beamformingIntervals:
                            for runNumber in runNumbers:
                                # Define the path for the throughput files
                                folder_path = os.path.join(base_folder, codebookFile, f"BeamFInterval{beamformingInterval}")
                                throughput_file = os.path.join(
                                    folder_path,
                                    f'SurgeryFeedbackPos{scenarioID}_throughput_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt'
                                )
                                
                                # Read the data if the file exists
                                if os.path.exists(throughput_file):
                                    throughput_data_content = pd.read_csv(throughput_file, sep=',', header=None)
                                    # Get throughput values filtering out the undesired rows
                                    filtered_throughput = throughput_data_content[1].drop(index=[0] + list(range(300, 310)), errors='ignore')
                                    throughput_data[(power, app_id, codebookFile, scenarioID, beamformingInterval, runNumber)].extend(filtered_throughput.values)
                                    
                                    # Assuming PDR is in the last row, last column of the file
                                    pdr_value = throughput_data_content.iloc[-1, -1]
                                    pdr_data[(power, app_id, codebookFile, scenarioID, beamformingInterval, runNumber)] = pdr_value
                            
    return throughput_data, pdr_data




throughput_folder = os.path.join(figures_folder, 'Throughput')
detailed_throughput_folder = os.path.join(throughput_folder, 'Detailed')
os.makedirs(detailed_throughput_folder, exist_ok=True)

detailed_mcs_folder = os.path.join(mcs_folder, 'Detailed')
os.makedirs(detailed_mcs_folder, exist_ok=True)

general_folder = os.path.join(throughput_folder, 'General')
os.makedirs(general_folder, exist_ok=True)
txPower_effect_folder = os.path.join(general_folder, 'PowerEffect')
os.makedirs(txPower_effect_folder, exist_ok=True)
codebook_effect_folder = os.path.join(general_folder, 'CodebookEffect')
os.makedirs(codebook_effect_folder, exist_ok=True)
beamInterval_effect_folder = os.path.join(general_folder, 'BIntervalEffect')
os.makedirs(beamInterval_effect_folder, exist_ok=True)


if ENABLE_THROUGHPUT_PER_SCENARIO:
    throuhgput_scenario_folder = os.path.join(general_folder, 'PerScenario')
    os.makedirs(throuhgput_scenario_folder, exist_ok=True)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(codebookFiles)))
   
    edge_colors = ['black', 'red', 'blue']  # Distinct edge colors for 3 different TxPower values

    # Initialize a dictionary to collect throughput data for each scenarioID and app_id combination
    throughput_data_collection = {
        (scenarioID, app_id): []
        for scenarioID in scenarioIDs
        for app_id in app_ids
    }

    pdr_data_collection = {
        (scenarioID, app_id): []
        for scenarioID in scenarioIDs
        for app_id in app_ids
    }

    


    ######################################################################################
    #                            PREPARE DATA                                            #
    ######################################################################################
 
    # Collect the throughput data if needed
    if not throughputCollected:
        print("Throughput per scenario: Collect Data")
        throughput_data, pdr_data = collect_throughput_and_pdr_data(txPower, app_ids, codebookFiles, scenarioIDs, beamformingIntervals, runNumbers, base_folder, rlc_mode)
    else:
        print("Throughput and PDR: Data already Collected")

    ######################################################################################
    #                            THROUGHPUT AND PDR                                      #
    ######################################################################################
    # Loop over the data and collect throughput values along with their associated parameters
    for scenarioID in scenarioIDs:
        for app_id in app_ids:
            # This list will hold dictionaries for all throughput values for the current scenarioID and app_id
            scenario_app_throughputs = []
            scenario_app_pdr = []

            for power in txPower:
                for codebookFile in codebookFiles:
                    for beamformingInterval in beamformingIntervals:
                        for runNumber in runNumbers:
                            # Extract the throughput values for the current combination
                            key = (power, app_id, codebookFile, scenarioID, beamformingInterval, runNumber)
                            throughput_values = throughput_data.get(key, [])
                            if throughput_values:  # Ensure that there is data
                                # Store each throughput value along with its parameters
                                for value in throughput_values:
                                    scenario_app_throughputs.append({
                                        'throughput': value,
                                        'power': power,
                                        'beamformingInterval': beamformingInterval,
                                        'codebookFile': codebookFile
                                    })
                            pdr_values = pdr_data.get(key, [])
                            if pdr_values:  # Ensure that there is data
                                
                               
                                scenario_app_pdr.append({
                                    'pdr': pdr_values,
                                    'power': power,
                                    'beamformingInterval': beamformingInterval,
                                    'codebookFile': codebookFile
                                })

            # Once all values are collected for a scenarioID and app_id, add them to the dictionary
            throughput_data_collection[(scenarioID, app_id)] = scenario_app_throughputs
            pdr_data_collection[(scenarioID, app_id)] = scenario_app_pdr



          

    


    # ######################################################################################
    # #                            PDR                                                     #
    # ######################################################################################
    for (scenarioID, app_id), throughput_info_list in pdr_data_collection.items():
        fig, ax = plt.subplots(figsize=(15, 10))

        # Data structure for plotting
        plot_data = {}
        for power in txPower:
            for beamIdx, beamformingInterval in enumerate(beamformingIntervals):
                for codeIdx, codebookFile in enumerate(codebookFiles):
                    # Nested keys for power and beamforming interval
                    if (power, beamformingInterval) not in plot_data:
                        plot_data[(power, beamformingInterval)] = []
                    
                    # Filter and append throughput data
                    current_data = [info['pdr'] for info in throughput_info_list if info['power'] == power
                                    and info['beamformingInterval'] == beamformingInterval 
                                    and info['codebookFile'] == codebookFile]
                    if current_data:
                        plot_data[(power, beamformingInterval)].append(current_data)

        # Boxplot positions and labels
        positions = []
        labels = []
        tick_positions = []  # For major ticks (TxPower levels)
        power_labels = []

        # Spacing configurations
        box_width = 0.1  # Width of each boxplot
        interval_spacing = 0.15  # Space between groups of boxplots for each beamformingInterval
        power_spacing = 0.7  # Space between major TxPower groups

        # Current position tracker
        current_position = 0

       # Create grouped boxplot data
        for power_idx, power in enumerate(txPower):
            # Calculate the center position for the TxPower label
            power_center_position = current_position + (len(beamformingIntervals) * len(codebookFiles) * box_width + (len(beamformingIntervals) - 1) * interval_spacing) / 2
            tick_positions.append(power_center_position)
            power_labels.append(f"{power}")

            for beamIdx, beamformingInterval in enumerate(beamformingIntervals):
                for codeIdx, _ in enumerate(codebookFiles):
                    codebook_data = plot_data.get((power, beamformingInterval), [])[codeIdx] if plot_data.get((power, beamformingInterval)) else []
                    positions.append(current_position)
                    current_position += box_width
                    labels.append(f"BI{beamformingInterval}s CB{codeIdx+1}")
                # Add the interval spacing after each group of codebookFiles, but not after the last group in the beamformingInterval
                if beamIdx < len(beamformingIntervals) - 1:
                    current_position += interval_spacing

            # Add the power spacing after each group of beamformingIntervals, but not after the last power level
            if power_idx < len(txPower) - 1:
                current_position += power_spacing

        bar_width = 0.1
        # Plotting all boxplots with unique color and hatch patterns
        # Plotting all boxplots with unique color, hatch patterns, and edge colors
        for idx, (key, value) in enumerate(plot_data.items()):
            power, beamformingInterval = key
            power_idx = txPower.index(power)  # Get the index of the power value
            edge_color = edge_colors[power_idx]  # Direct index for edge colors
            for codeIdx, codebook_data in enumerate(value):
                ax.bar(positions[idx * len(codebookFiles) + codeIdx], codebook_data,
                       width=bar_width,
                       color=colors[codeIdx],
                       label=f'{power}' if idx == 0 and codeIdx == 0 else "")
                # box = ax.boxplot(codebook_data,
                #                 positions=[positions[idx * len(codebookFiles) + codeIdx]],
                #                 widths=box_width,
                #                 meanprops=meanpointprops,
                #                 showmeans=True,
                #                 medianprops=medianColor,
                #                 whiskerprops={'color': colors[codeIdx], 'linewidth': 1.5},
                #                 flierprops={'markerfacecolor': colors[codeIdx], 'markeredgecolor': colors[codeIdx], 'markersize': 10},
                #                 capprops={'color': colors[codeIdx], 'linewidth': 1.5},
                #                 patch_artist=True)
                # for patch in box['boxes']:
                #     patch.set_facecolor(colors[codeIdx])  # Apply color
                #     patch.set_edgecolor(edge_color)  # Set edge color

        # Formatting the plot
        # Set the TxPower labels at the bottom
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(power_labels, ha='center')
        ax.set_xlabel('Tx Power (dBm)')

        

        # Add another x-axis at the top for the BI and CB labels
        secondary_ax = ax.secondary_xaxis('top')
        # positions = [pos + 1 for pos in positions]
        secondary_ax.set_xticks(positions)
        secondary_ax.set_xticklabels(labels, rotation=90, ha="center")
        secondary_ax.set_xlabel('Beamforming Interval and Codebook')

       

        # Legend for Codebook Files and TxPower
        # Assuming 'colors' is a list of colors for each codebookFile
        # and 'edge_colors' is a list of colors for each TxPower
        # Create legend handles for the codebook colors
        codebook_legend_handles = [Patch(facecolor=color, label=f'{codebook_labels_gen[idx]}') for idx, color in enumerate(colors)]

      
        # Add the first legend to the plot for codebooks. Save the legend object in a variable.
        codebook_legend = ax.legend(handles=codebook_legend_handles, loc='upper left', bbox_to_anchor=(1, 0.5), title="Codebook")
        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        ax.set_ylabel('PDR')
        throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
               
        ax.set_title(f'PDR Grouped by TxPower, Beamforming Interval, and Codebook for Scenario: {scenarioID} and App ID: {app_id} (~{throughput_value} Mbps)')

        # Draw vertical lines to separate major TxPower groups
        # for tick_position in tick_positions:
        #     ax.axvline(x=tick_position, color='gray', linestyle='--')

        plt.tight_layout()
        # plt.savefig(os.path.join(trends_folder_pdr, f'avg_pdr_barchart_app{app_id}_codebook{codebookFile}_beamf{beamformingInterval}.png'), bbox_inches='tight')
              
        plt.savefig(os.path.join(throuhgput_scenario_folder, f'grouped_barchart_PDR_scenario{scenarioID}_app{app_id}.png'))
        plt.close(fig)

    ######################################################################################
    #                            THROUGHPUT                                              #
    ######################################################################################
    for (scenarioID, app_id), throughput_info_list in throughput_data_collection.items():
        fig, ax = plt.subplots(figsize=(15, 10))

        # Data structure for plotting
        plot_data = {}
        for power in txPower:
            for beamIdx, beamformingInterval in enumerate(beamformingIntervals):
                for codeIdx, codebookFile in enumerate(codebookFiles):
                    # Nested keys for power and beamforming interval
                    if (power, beamformingInterval) not in plot_data:
                        plot_data[(power, beamformingInterval)] = []
                    
                    # Filter and append throughput data
                    current_data = [info['throughput'] for info in throughput_info_list if info['power'] == power
                                    and info['beamformingInterval'] == beamformingInterval 
                                    and info['codebookFile'] == codebookFile]
                    if current_data:
                        plot_data[(power, beamformingInterval)].append(current_data)

        # Boxplot positions and labels
        positions = []
        labels = []
        tick_positions = []  # For major ticks (TxPower levels)
        power_labels = []

        # Spacing configurations
        box_width = 0.1  # Width of each boxplot
        interval_spacing = 0.15  # Space between groups of boxplots for each beamformingInterval
        power_spacing = 0.7  # Space between major TxPower groups

        # Current position tracker
        current_position = 0

       # Create grouped boxplot data
        for power_idx, power in enumerate(txPower):
            # Calculate the center position for the TxPower label
            power_center_position = current_position + (len(beamformingIntervals) * len(codebookFiles) * box_width + (len(beamformingIntervals) - 1) * interval_spacing) / 2
            tick_positions.append(power_center_position)
            power_labels.append(f"{power}")

            for beamIdx, beamformingInterval in enumerate(beamformingIntervals):
                for codeIdx, _ in enumerate(codebookFiles):
                    codebook_data = plot_data.get((power, beamformingInterval), [])[codeIdx] if plot_data.get((power, beamformingInterval)) else []
                    # print(len(codebook_data))
                    # exit()
                    positions.append(current_position)
                    current_position += box_width
                    labels.append(f"BI{beamformingInterval}s CB{codeIdx+1}")
                # Add the interval spacing after each group of codebookFiles, but not after the last group in the beamformingInterval
                if beamIdx < len(beamformingIntervals) - 1:
                    current_position += interval_spacing

            # Add the power spacing after each group of beamformingIntervals, but not after the last power level
            if power_idx < len(txPower) - 1:
                current_position += power_spacing

        # Plotting all boxplots with unique color and hatch patterns
        # Plotting all boxplots with unique color, hatch patterns, and edge colors
        for idx, (key, value) in enumerate(plot_data.items()):
            power, beamformingInterval = key
            power_idx = txPower.index(power)  # Get the index of the power value
            edge_color = edge_colors[power_idx]  # Direct index for edge colors
            for codeIdx, codebook_data in enumerate(value):
                box = ax.boxplot(codebook_data,
                                positions=[positions[idx * len(codebookFiles) + codeIdx]],
                                widths=box_width,
                                meanprops=meanpointprops,
                                showmeans=True,
                                medianprops=medianColor,
                                whiskerprops={'color': colors[codeIdx], 'linewidth': 1.5},
                                flierprops={'markerfacecolor': colors[codeIdx], 'markeredgecolor': colors[codeIdx], 'markersize': 10},
                                capprops={'color': colors[codeIdx], 'linewidth': 1.5},
                                patch_artist=True)
                for patch in box['boxes']:
                    patch.set_facecolor(colors[codeIdx])  # Apply color
                    # patch.set_edgecolor(edge_color)  # Set edge color

        # Formatting the plot
        # Set the TxPower labels at the bottom
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(power_labels, ha='center')
        ax.set_xlabel('Tx Power (dBm)')

        # Add another x-axis at the top for the BI and CB labels
        secondary_ax = ax.secondary_xaxis('top')
        # positions = [pos + 1 for pos in positions]
        secondary_ax.set_xticks(positions)
        secondary_ax.set_xticklabels(labels, rotation=90, ha="center")
        secondary_ax.set_xlabel('Beamforming Interval and Codebook')

        # Legend for Codebook Files and TxPower
        # Assuming 'colors' is a list of colors for each codebookFile
        # and 'edge_colors' is a list of colors for each TxPower
        # Create legend handles for the codebook colors
        codebook_legend_handles = [Patch(facecolor=color, label=f'{codebook_labels_gen[idx]}') for idx, color in enumerate(colors)]

       
        # Add the first legend to the plot for codebooks. Save the legend object in a variable.
        codebook_legend = ax.legend(handles=codebook_legend_handles, loc='upper left', bbox_to_anchor=(1, 0.5), title="Codebook")

    
        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        ax.set_ylabel('Throughput (Mbps)')
        throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
        ax.set_title(f'Throughput Boxplot Grouped by TxPower, Beamforming Interval, and Codebook for Scenario: {scenarioID} and App ID {app_id} (~{throughput_value} Mbps)')

        # Draw vertical lines to separate major TxPower groups
        # for tick_position in tick_positions:
        #     ax.axvline(x=tick_position, color='gray', linestyle='--')

        plt.tight_layout()
        # plt.savefig(os.path.join(trends_folder_pdr, f'avg_pdr_barchart_app{app_id}_codebook{codebookFile}_beamf{beamformingInterval}.png'), bbox_inches='tight')
              
        plt.savefig(os.path.join(throuhgput_scenario_folder, f'grouped_boxplot_throughput_scenario{scenarioID}_app{app_id}.png'))
        plt.close(fig)



if ENABLE_THROUGHPUT_GENERAL:


    PDR_folder = os.path.join(figures_folder, 'PDR')
    detailed_PDR_folder = os.path.join(PDR_folder, 'Detailed')
    os.makedirs(detailed_PDR_folder, exist_ok=True)
    general_PDR_folder = os.path.join(PDR_folder, 'General')
    os.makedirs(general_PDR_folder, exist_ok=True)
    txPower_PDR_effect_folder = os.path.join(general_PDR_folder, 'PowerEffect')
    os.makedirs(txPower_PDR_effect_folder, exist_ok=True)
    codebook_PDR_effect_folder = os.path.join(general_PDR_folder, 'CodebookEffect')
    os.makedirs(codebook_effect_folder, exist_ok=True)
    beamInterval_PDR_effect_folder = os.path.join(general_PDR_folder, 'BIntervalEffect')
    os.makedirs(beamInterval_effect_folder, exist_ok=True)
    trends_folder_pdr = os.path.join(txPower_PDR_effect_folder, 'Trends')
    os.makedirs(trends_folder_pdr, exist_ok=True)

    
    def display_statistics(throughput_data, pdr_data, txPower, app_ids, codebookFiles, scenarioIDs, beamformingIntervals):
    # Loop through each combination
        for app_id in app_ids:
            print(f"App ID: {app_id}")
            for scenarioID in scenarioIDs:
                print(f"  Scenario ID: {scenarioID}")
                for codebookFile in codebookFiles:
                    print(f"    Codebook File: {codebookFile}")
                    for beamformingInterval in beamformingIntervals:
                        print(f"      Beamforming Interval: {beamformingInterval}s")
                        for power in txPower:
                            # Collect throughput and PDR data for this combination
                            key_filter = lambda key: (key[0] == power and key[1] == app_id and key[2] == codebookFile and key[3] == scenarioID and key[4] == beamformingInterval)
                            throughput_values = [value for key, value in throughput_data.items() if key_filter(key)]
                            pdr_values = [value for key, value in pdr_data.items() if key_filter(key)]

                            # Calculate average throughput for each run
                            avg_throughputs = [np.mean(sublist) for sublist in throughput_values if sublist]

                            # Now calculate the standard deviation of these averages
                            if avg_throughputs:
                                stddev_throughput = np.std(avg_throughputs)

                                # Similarly calculate for PDR if needed
                                filtered_pdr = [pdr for pdr in pdr_values if pdr is not None]
                                avg_pdr = np.mean(filtered_pdr)
                                stddev_pdr = np.std(filtered_pdr)

                                # Print the statistics
                                print(f"        Tx Power: {power}")
                                print(f"          Average Throughput: {np.mean(avg_throughputs):.2f} Mbps, StdDev of Averages: {stddev_throughput:.2f}")
                                print(f"          Average PDR: {avg_pdr:.2f}, StdDev: {stddev_pdr:.2e}")

                            else:
                                print(f"        Tx Power: {power} - No data available")


    # Collect the throughput data if needed
    if not throughputCollected:
        print("Throughput and PDR: Collect Data")
        throughput_data, pdr_data = collect_throughput_and_pdr_data(txPower, app_ids, codebookFiles, scenarioIDs, beamformingIntervals, runNumbers, base_folder, rlc_mode)
    else:
        print("Throughput and PDR: Data already Collected")

    display_statistics(throughput_data, pdr_data, txPower, app_ids, codebookFiles, scenarioIDs, beamformingIntervals)
    exit()
    ######################################################################################
    #                                                                                    #
    #                            TX POWER  EFFECT                                        #
    #                                                                                    #
    ######################################################################################
    trends_folder = os.path.join(txPower_effect_folder, 'Trends')
    os.makedirs(trends_folder, exist_ok=True)
    ######################################################################################
    #                            ALL RUNS                                                #
    ######################################################################################
    # Loop through each codebookFile, beamformingInterval, and app_id
    print("Throughput and PDR: Tx Power Effect")
  

    codebook_text_map = {
    "1x16.txt": 'ULA - 1x16',  
    "2x16.txt": 'URA - 2x16', 
}

    for codebookFile in tqdm(codebookFiles, desc='Codebook File Progress'):
        for beamformingInterval in tqdm(beamformingIntervals, desc='Beamforming Interval Progress', leave=False):
            for app_id in tqdm(app_ids, desc='Ap ID progress', leave=False):
                throughput_data_per_scenario_power = {(scenarioID, power): []
                                                    for scenarioID in scenarioIDs
                                                    for power in txPower}
                
                # Prepare data structure for all throughput values per scenario and Tx Power
                all_throughputs_per_scenario_power = {
                    (scenarioID, power): []
                    for scenarioID in scenarioIDs
                    for power in txPower
                }
                
                # Prepare data structure for average PDR per scenario and Tx Power
                pdr_data_per_scenario_power = {(scenarioID, power): []
                                            for scenarioID in scenarioIDs
                                            for power in txPower}


                # Collect data for each scenario and Tx Power, filtered by app ID, codebook, and beamforming interval
                for scenarioID in scenarioIDs:
                    for power in txPower:
                        throughputs = []  # Store throughputs for this scenario, power, and app_id
                        for runNumber in runNumbers:
                            
                            key = (power, app_id, codebookFile, scenarioID, beamformingInterval, runNumber)
                            
                            throughputs.extend(throughput_data.get(key, []))
                            all_throughputs_per_scenario_power[(scenarioID, power)].extend(throughput_data.get(key, []))
                            
                        
                            pdr_value = pdr_data.get(key)
                            if pdr_value is not None:
                                pdr_data_per_scenario_power[(scenarioID, power)].append(pdr_value)

                        
                        # Calculate the average throughput for this scenario and Tx Power
                        if throughputs:
                            avg_throughput = np.mean(throughputs)
                            throughput_data_per_scenario_power[(scenarioID, power)].append(avg_throughput)
                        else:
                            print("Problem")
                            
                            
                ################################
                #     PLOT PDR                 #
                ################################
                # Now we have the PDR data, we can plot it
                labels = sorted(scenarioIDs)
                x = np.arange(len(labels))  # the label locations
                
                total_width = 0.8  # Total width for all bars in a group
                num_bars = len(txPower)
                width = total_width / num_bars  # the width of each bar
                
                fig, ax = plt.subplots(figsize=(15, 10))  # Set figure size
                colors = plt.cm.viridis(np.linspace(0, 1, num_bars))  # Use a colormap
                
                offset = (total_width - width) / 2  # Offset to center bars
                
                for idx, power in enumerate(txPower):
                    # Compute the average PDR for this Tx Power across all scenarios
                    power_avg_pdr = [
                        np.mean(pdr_data_per_scenario_power.get((scenarioID, power), [0]))
                        for scenarioID in labels
                    ]
                    
                    bar_positions = x - offset + idx * width  # Adjust bar positions
                    
                    ax.bar(bar_positions, power_avg_pdr, width, label=power_labels_gen[idx],
                        color=colors[idx], edgecolor='black')

                # Set chart details
                ax.set_ylabel('Average Packet Delivery Ratio')
                ax.set_xlabel('Scenario ID')
                # Get the throughput value from the map using the current app_id
                throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                codebook_value = codebook_text_map.get(codebookFile, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                ax.set_title(f'Average PDR by Scenario and Tx Power for App ID: {app_id} (~{throughput_value}Mbps), Codebook: {codebook_value}, Beam Itvl:{beamformingInterval}s')
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                # ax.legend()
                ax.legend(title="Tx Power (dBm)",loc='upper left', bbox_to_anchor=(1,1))

                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                fig.tight_layout(pad=2.0)

                # Save the figure
                plt.savefig(os.path.join(trends_folder_pdr, f'avg_pdr_barchart_app{app_id}_codebook{codebookFile}_beamf{beamformingInterval}.png'), bbox_inches='tight')
                plt.close(fig)            
                            
                            
                ################################
                #     PLOT THROUGHPUT          #
                ################################
                # Now we have the data, we can plot it
                labels = sorted(scenarioIDs)
                x = np.arange(len(labels))  # the label locations

                # Calculate the total width needed for all bars in a group
                total_width = 0.8  # This can be adjusted to fit as desired
                num_bars = len(txPower)
                width = total_width / num_bars  # the width of each bar

                fig, ax = plt.subplots(figsize=(15, 10))  # Set figure size
                colors = plt.cm.viridis(np.linspace(0, 1, num_bars))  # Use a colormap

                # Offset calculation to center bars
                offset = (total_width - width) / 2

                for idx, power in enumerate(txPower):
                    # Compute the average throughput for this Tx Power across all scenarios
                    power_avg_throughput = [
                        np.mean(throughput_data_per_scenario_power.get((scenarioID, power), [0]))
                        for scenarioID in labels
                    ]

                    # Adjust the position to center the bars
                    bar_positions = x - offset + idx * width

                    # Plot with black edges
                    rects = ax.bar(bar_positions, power_avg_throughput, width, label=power_labels_gen[idx],
                                color=colors[idx], edgecolor='black')

                # Set chart details
                ax.set_ylabel('Average Throughput (Mbps)')
                ax.set_xlabel('Scenario ID')
                throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                codebook_value = codebook_text_map.get(codebookFile, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                ax.set_title(f'Average Throughput by Scenario and Tx Power for App ID: {app_id} (~{throughput_value}Mbps), Codebook: {codebook_value}, Beam Itvl:{beamformingInterval}s')
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                # ax.legend()
                ax.legend(title="Tx Power (dBm)",loc='upper left', bbox_to_anchor=(1,1))


                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                fig.tight_layout(pad=2.0)

                # Save the figure
                plt.savefig(os.path.join(trends_folder, f'avg_throughput_barchart_app{app_id}_codebook{codebookFile}_beamf{beamformingInterval}.png'), bbox_inches='tight')
                plt.close(fig)
                
                
                
                # Create boxplot data
                boxplot_data = []
                for scenarioID in sorted(scenarioIDs):
                    scenario_data = []
                    for power in sorted(txPower):
                        scenario_data.append(all_throughputs_per_scenario_power[(scenarioID, power)])
                    boxplot_data.append(scenario_data)

                # Plot setup
                fig, ax = plt.subplots(figsize=(15, 10))  # Set figure size
                colors = plt.cm.viridis(np.linspace(0, 1, len(txPower)))  # Use a colormap
                positions = np.array(range(len(scenarioIDs))) * (len(txPower) + 1)
                # Plot boxplots for each Tx Power and collect legend info
                legend_elements = []
               
                
                
                for idx, power in enumerate(sorted(txPower)):
                    # This line actually generates the boxplot for each power setting.
                    bp = ax.boxplot(
                        [d[idx] for d in boxplot_data if len(d[idx]) > 0],  # filter out empty data lists
                        positions=positions + idx,
                        widths=0.6,
                        patch_artist=True,
                        boxprops=dict(facecolor=colors[idx], color=colors[idx]),
                        meanprops=meanpointprops,
                        showmeans=True,
                        medianprops=medianColor,
                        whiskerprops=dict(color=colors[idx]),
                        capprops=dict(color=colors[idx]),
                        flierprops=dict(markerfacecolor=colors[idx], marker='o', markersize=6, linestyle='none'),
                        labels=scenarioIDs if idx == 0 else ['']*len(scenarioIDs)  # Label only the first group
                    )
                    # Collect patches for the legend
                    legend_elements.append(bp['boxes'][0])

                # Add legend to the plot
                # ax.legend(handles=legend_elements, labels=[f'Tx Power {p}' for p in sorted(txPower)], title="Tx Power")

                ax.legend(handles=legend_elements, labels=[f'{p}' for p in sorted(txPower)], title="Tx Power (dBm)",loc='upper left', bbox_to_anchor=(1,1))
 
                # Set chart details
                ax.set_ylabel('Throughput (Mbps)')
                ax.set_xlabel('Scenario ID')
                throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                codebook_value = codebook_text_map.get(codebookFile, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                ax.set_title(f'Boxplot of Throughput by Scenario and Tx Power for App ID: {app_id} (~{throughput_value}Mbps), Codebook: {codebook_value}, Beam Itvl:{beamformingInterval}s')
                ax.set_xticks(positions + len(txPower) // 2)
                ax.set_xticklabels(sorted(scenarioIDs))
                # ax.legend([bp["boxes"][0] for bp in bp], sorted(txPower), title="Tx Power")

                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                fig.tight_layout(pad=2.0)

                # Save the figure
                plt.savefig(os.path.join(trends_folder, f'boxplot_throughput_scenario_app{app_id}_codebook{codebookFile}_beamf{beamformingInterval}.png'), bbox_inches='tight')
                plt.close(fig)



    
    ######################################################################################
    #                                                                                    #
    #                            CODEBOOK  EFFECT                                        #
    #                                                                                    #
    ######################################################################################
    print("Throughput and PDR: Codebook Effect Figures Creation")
    trends_folder = os.path.join(codebook_effect_folder, 'Trends')
    os.makedirs(trends_folder, exist_ok=True)
    trends_folder_pdr = os.path.join(codebook_PDR_effect_folder, 'Trends')
    os.makedirs(trends_folder_pdr, exist_ok=True)
    # ######################################################################################
    # #                            ALL RUNS                                                #
    # ######################################################################################
    for power in tqdm(txPower, desc='TxPower Progress'):
        for app_id in tqdm(app_ids, desc='App ID Progress', leave=False):
            for beamformingInterval in tqdm(beamformingIntervals, desc='Beam Interval Progress', leave=False):
    # for power in txPower:
    #     for app_id in app_ids:
    #         for beamformingInterval in beamformingIntervals:
                # Prepare structure for average throughput per scenario and codebook
                avg_throughput_per_scenario_codebook = {
                    (scenarioID, codebookFile): []
                    for scenarioID in scenarioIDs
                    for codebookFile in codebookFiles
                }
                
                # Prepare structure for throughput data per scenario and codebook
                throughput_data_per_scenario_codebook = {
                    (scenarioID, codebookFile): []
                    for scenarioID in scenarioIDs
                    for codebookFile in codebookFiles
                }
                
                avg_pdr_per_scenario_codebook = {
                    (scenarioID, codebookFile): []
                    for scenarioID in scenarioIDs
                    for codebookFile in codebookFiles
                }
                
                # Prepare structure for PDR data per scenario and codebook
                pdr_data_per_scenario_codebook = {
                    (scenarioID, codebookFile): []
                    for scenarioID in scenarioIDs
                    for codebookFile in codebookFiles
                }
                
                # Use collected data instead of reading the files again
                for scenarioID in scenarioIDs:
                    for codebookFile in codebookFiles:
                        throughputs = []  # Store throughputs for this scenario and codebook file
                        pd_ratios = []  # Store PDRs for this scenario and codebook file
                    
                        for runNumber in runNumbers:
                            # Adjust the key to match the one used when storing the data
                            key = (power, app_id, codebookFile, scenarioID, beamformingInterval, runNumber)
                            throughputs.extend(throughput_data.get(key, []))
                            data = throughput_data.get(key, [])
                            if data:
                                throughput_data_per_scenario_codebook[(scenarioID, codebookFile)].extend(data)
                                pdr_data_per_scenario_codebook[(scenarioID, codebookFile)].extend(data)
                            
                                
                            data_pdr = pdr_data.get(key)
                            if data_pdr is not None:
                                pd_ratios.append(data_pdr)  # Append the single value to the list
                                pdr_data_per_scenario_codebook[(scenarioID, codebookFile)].append(data)


                        
                        # Calculate the average throughput for this scenario and codebook file
                        if throughputs:
                            avg_throughput = np.mean(throughputs)
                            avg_throughput_per_scenario_codebook[(scenarioID, codebookFile)].append(avg_throughput)

                        if pd_ratios:
                            avg_pdr = np.mean(pd_ratios)
                            avg_pdr_per_scenario_codebook[(scenarioID, codebookFile)].append(avg_pdr)
                            
                ###################
                # PLOT PDR        #
                ###################
                # Now that we have all the data, we can create a bar chart
                fig, ax = plt.subplots(figsize=(15, 10))
                bar_width = 0.35
                bar_positions = np.arange(len(scenarioIDs))
                colors = plt.cm.viridis(np.linspace(0, 1, len(codebookFiles)))

                # Plot a bar for each codebook file per scenario
                for idx, codebookFile in enumerate(codebookFiles):
                    avg_pdrs = [
                        np.mean(avg_pdr_per_scenario_codebook.get((scenarioID, codebookFile), [0]))
                        for scenarioID in scenarioIDs
                    ]
                    ax.bar(bar_positions + idx * bar_width, avg_pdrs, bar_width, label=codebook_labels_gen[idx], color=colors[idx])
              
                # Set chart details
                ax.set_ylabel('Average PDR (%)')
                ax.set_xlabel('Scenario ID')
                throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                ax.set_title(f'Average PDR by Scenario for Tx Power: {power} dBm, App ID: {app_id} (~{throughput_value}Mbps), Beam Itvl: {beamformingInterval}s')
                ax.set_xticks(bar_positions + bar_width * (len(codebookFiles) - 1) / 2)
                ax.set_xticklabels(scenarioIDs)
                ax.legend(title="Codebook",loc='upper left', bbox_to_anchor=(1,1))

                

                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                fig.tight_layout(pad=2.0)

                # Save the figure
                plt.savefig(os.path.join(trends_folder_pdr, f'avg_pdr_barchart_scenario_codebook_app{app_id}_TxPower{power}_beamf{beamformingInterval}.png'), bbox_inches='tight')
                plt.close(fig)
                
                ###################
                # PLOT THROUGHPUT #
                ###################
                labels = sorted(scenarioIDs)
                ### ICI
                # Boxplot plotting
                fig, ax = plt.subplots(figsize=(15, 10))
                # positions = np.array(range(len(labels))) * (len(beamformingIntervals) + 1)  # Set starting position for each group of boxplots
             
                n_scenarios = len(labels)  # Assuming labels corresponds to scenario IDs
                n_codebooks = len(codebookFiles)

                # Calculate the positions
                # For each scenario, we want 2 positions (for 2 codebooks), hence we use `range(n_scenarios * n_codebooks)`
                # We space them by the number of codebooks (e.g., `* n_codebooks`), and add an offset to group them
                positions = np.array(range(n_scenarios * n_codebooks)) + np.repeat(np.arange(n_scenarios), n_codebooks) * (n_codebooks - 1)
                
               
               

                for idx, codebookFile in enumerate(codebookFiles):
                    data_to_plot = [throughput_data_per_scenario_codebook[(scenarioID, codebookFile)] for scenarioID in labels]
   
                    # Collect patches for the legend
                    # legend_elements.append(bp['boxes'][0])
                    

                   
                    ax.boxplot(data_to_plot, positions = positions[idx::n_codebooks], widths=0.6, patch_artist=True,
                            boxprops=dict(facecolor=colors[idx], color=colors[idx]),
                            meanprops=meanpointprops,
                            showmeans=True,
                            medianprops=medianColor,
                            flierprops=dict(markerfacecolor=colors[idx], marker='o', markersize=6, linestyle='none'),
                            whiskerprops=dict(color=colors[idx]),
                            capprops=dict(color=colors[idx]))
                
                # ax.set_xticks(positions + len(codebookFiles) / 2)
                # print(positions + len(codebookFiles) / 2)
                group_spacing = 1
                start_positions = np.arange(n_scenarios) * (n_codebooks + group_spacing)
                # print("START POS:",start_positions)
                tick_positions = start_positions + (n_codebooks - 1) / 2.0
                # print("TICKPOS:",tick_positions)
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(labels)  # Assuming labels is a list of scenario IDs

                # Adding legend for beamforming intervals
                legend_elements = [Patch(facecolor=colors[idx], edgecolor=colors[idx],
                                        label=f'{codebook_labels_gen[idx]}') for idx in range(len(codebook_labels_gen))]
               

                ax.legend(handles=legend_elements, title="Codebooks",loc='upper left', bbox_to_anchor=(1,1))

                ax.set_ylabel('Throughput (Mbps)')
                ax.set_xlabel('Scenario ID')
                throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                ax.set_title(f'Boxplot of Throughput per Scenario for Tx Power: {power} dBm, App ID: {app_id} (~{throughput_value}Mbps), Beam Itvl: {beamformingInterval}s')
                fig.tight_layout(pad=2.0)
                # Save the figure
                plt.savefig(os.path.join(trends_folder, f'boxplot_throughput_per_scenario_codebook_app{app_id}_TxPower{power}_beamf{beamformingInterval}.png'), bbox_inches='tight')
                plt.close(fig)
               

                ### END ICI
                        
                # Now that we have all the data, we can create a boxplot
                # fig, ax = plt.subplots(figsize=(15, 10))
                # boxplot_data = []
                # boxplot_positions = []
                # boxplot_colors = []
                # colors = plt.cm.viridis(np.linspace(0, 1, len(codebookFiles)))
                # color_map = dict(zip(codebookFiles, colors))
                
                # # Define positions for each set of boxplots for each scenario
                # scenario_positions = np.arange(len(scenarioIDs)) * (len(codebookFiles) + 1)  # Spacing between sets of boxplots
                
                # for idx, scenarioID in enumerate(sorted(scenarioIDs)):
                #     for jdx, codebookFile in enumerate(codebookFiles):
                #         data = throughput_data_per_scenario_codebook.get((scenarioID, codebookFile), [])
                #         if data:  # If there's data, add it to the list for plotting
                #             boxplot_data.append(data)
                #             boxplot_positions.append(scenario_positions[idx] + jdx)
                #             boxplot_colors.append(color_map[codebookFile])

                # # Create the boxplot
                # bplot = ax.boxplot(boxplot_data, positions=boxplot_positions, patch_artist=True,
                #                    meanprops=meanpointprops,
                #                     showmeans=True,
                #                     medianprops=medianColor,)
                
                # # Color each boxplot by codebook
                # for patch, color in zip(bplot['boxes'], boxplot_colors):
                #     patch.set_facecolor(color)
                
                # # Add legend by creating custom patches
                # import matplotlib.patches as mpatches
                # legend_patches = [mpatches.Patch(color=color_map[cb], label=cb) for cb in codebookFiles]
                # ax.legend(handles=legend_patches, title="Codebook")
                
                # # Set x-axis labels and title
                # ax.set_xticks(np.mean(np.reshape(boxplot_positions, (len(scenarioIDs), len(codebookFiles))), axis=1))
                # ax.set_xticklabels(sorted(scenarioIDs))
                # ax.set_title(f'Boxplot of Throughputs per Scenario for Tx Power {power}, App ID {app_id}, Beamforming Interval {beamformingInterval}')
                # ax.set_xlabel('Scenario ID')
                # ax.set_ylabel('Throughput (Mbps)')
                
                # # Save the figure
                # plt.savefig(os.path.join(trends_folder, f'boxplot_throughput_per_scenario_codebook_app{app_id}_TxPower{power}_beamf{beamformingInterval}.png'), bbox_inches='tight')
                # plt.close(fig)


    ######################################################################################
    #                                                                                    #
    #                            BEAMFORMING INTERVAL EFFECT                             #
    #                                                                                    #
    ######################################################################################
    # Ensure the 'Trends' subdirectory exists within the 'beamInterval_effect_folder'
    trends_folder = os.path.join(beamInterval_effect_folder, 'Trends')
    os.makedirs(trends_folder, exist_ok=True)
    trends_folder_pdr = os.path.join(beamInterval_PDR_effect_folder, 'Trends')
    os.makedirs(trends_folder_pdr, exist_ok=True)
            
    ######################################################################################
    #                            PER RUN                                                 #
    ######################################################################################
    # for power in txPower:
    #     for app_id in app_ids:
    #         for codebookFile in codebookFiles:
    #             for runNumber in runNumbers:
    #                 # Plot the boxplot
    #                 labels = sorted(scenarioIDs)
    #                 fig, ax = plt.subplots(figsize=(15, 10))
    #                 colors = plt.cm.viridis(np.linspace(0, 1, len(beamformingIntervals)))
    #                 positions = np.array(range(len(labels))) * (len(beamformingIntervals) + 1)
                    
    #                 for idx, beamformingInterval in enumerate(beamformingIntervals):
    #                     data_to_plot = [
    #                         throughput_data.get((power, app_id, codebookFile, scenarioID, beamformingInterval, runNumber), [])
    #                         for scenarioID in labels
    #                     ]
    #                     ax.boxplot(data_to_plot, positions=positions + idx, widths=0.6, patch_artist=True,
    #                             boxprops=dict(facecolor=colors[idx], color=colors[idx]),
    #                             medianprops=dict(color='white'), whiskerprops=dict(color=colors[idx]),
    #                             capprops=dict(color=colors[idx]), flierprops=dict(markeredgecolor=colors[idx]))
                    
    #                 ax.set_xticks(positions + len(beamformingIntervals) / 2)
    #                 ax.set_xticklabels(labels)
    #                 legend_elements = [Patch(facecolor=colors[idx], edgecolor=colors[idx], label=f'Beamforming Interval {bi}') 
    #                                 for idx, bi in enumerate(beamformingIntervals)]
    #                 ax.legend(handles=legend_elements, title="Beamforming Intervals")
    #                 ax.set_ylabel('Throughput (Mbps)')
    #                 ax.set_xlabel('Scenario ID')
    #                 ax.set_title(f'Boxplot of Throughput per Scenario for Run {runNumber} - Tx Power {power}, App ID {app_id}, Codebook File {codebookFile}')
    #                 fig.tight_layout(pad=2.0)

    #                 run_folder = os.path.join(beamInterval_effect_folder, f'runNumber{runNumber}')
    #                 os.makedirs(run_folder, exist_ok=True)
    #                 plt.savefig(os.path.join(run_folder, f'boxplot_throughput_app{app_id}_TxPower{power}_codebookFile{codebookFile}_run{runNumber}.png'), bbox_inches='tight')
    #                 plt.close(fig)

    #             # Plot the barchart
    #             fig, ax = plt.subplots(figsize=(15, 10))
    #             x = np.arange(len(labels))
    #             total_width = 0.8
    #             width = total_width / len(beamformingIntervals)

    #             for idx, beamformingInterval in enumerate(beamformingIntervals):
    #                 avg_throughputs = [
    #                     np.mean(throughput_data.get((power, app_id, codebookFile, scenarioID, beamformingInterval, runNumber), [0]))
    #                     for scenarioID in labels
    #                 ]
    #                 rects = ax.bar(
    #                     x + idx * width - (total_width - width) / 2, avg_throughputs, width,
    #                     label=f'Beamforming Interval {beamformingInterval}',
    #                     color=colors[idx], edgecolor='black'
    #                 )

    #             ax.set_ylabel('Average Throughput (Mbps)')
    #             ax.set_xlabel('Scenario ID')
    #             ax.set_title(f'Run {runNumber} - Average Throughput per Scenario for Tx Power {power}, App ID {app_id}, Codebook File {codebookFile}')
    #             ax.set_xticks(x)
    #             ax.set_xticklabels(labels)
    #             ax.legend()
    #             plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    #             fig.tight_layout(pad=2.0)

    #             plt.savefig(os.path.join(run_folder, f'avg_throughput_barchart_app{app_id}_TxPower{power}_codebookFile{codebookFile}_run{runNumber}.png'), bbox_inches='tight')
    #             plt.close(fig)
                
    ######################################################################################
    #                            ALL RUNS                                                #
    ######################################################################################
    print("Throughput and PDR: Beamforming Interval")
    for power in tqdm(txPower, desc='TxPower Progress'):
        for app_id in tqdm(app_ids, desc='App ID Progress', leave=False):
            for codebookFile in tqdm(codebookFiles, desc='Codebook File Progress', leave=False):
    # for power in txPower:
    #     for app_id in app_ids:
    #         for codebookFile in codebookFiles:
                avg_throughput_per_scenario_beamf = {(scenarioID, beamformingInterval): []
                                                    for scenarioID in scenarioIDs
                                                    for beamformingInterval in beamformingIntervals}
                throughput_data_per_scenario_beamf = {
                    (scenarioID, beamformingInterval): []
                    for scenarioID in scenarioIDs
                    for beamformingInterval in beamformingIntervals
                }
                
                avg_pdr_per_scenario_beamf = {(scenarioID, beamformingInterval): []
                                            for scenarioID in scenarioIDs
                                            for beamformingInterval in beamformingIntervals}
                

                # Aggregate data for each scenario and beamforming interval for average throughput and boxplot
                for scenarioID in scenarioIDs:
                    for beamformingInterval in beamformingIntervals:
                        throughputs = []
                        pdrs = []
                        for runNumber in runNumbers:
                            run_throughputs = throughput_data[(power, app_id, codebookFile, scenarioID, beamformingInterval, runNumber)]
                            throughputs.extend(run_throughputs)
                            throughput_data_per_scenario_beamf[(scenarioID, beamformingInterval)].extend(run_throughputs)
                            
                            run_pdrs = pdr_data[(power, app_id, codebookFile, scenarioID, beamformingInterval, runNumber)]
                            pdrs.append(run_pdrs) 

                        if throughputs:
                            avg_throughput = np.mean(throughputs)
                            avg_throughput_per_scenario_beamf[(scenarioID, beamformingInterval)].append(avg_throughput)
                            
                        if pdrs:
                            avg_pdr = np.mean(pdrs)
                            avg_pdr_per_scenario_beamf[(scenarioID, beamformingInterval)].append(avg_pdr)

                #######################
                # PLOT PDR            #
                #######################
                # Now we have the data, let's plot the bar chart for PDR
                labels = sorted(scenarioIDs)
                x = np.arange(len(labels))  # the label locations
                num_beamf_intervals = len(beamformingIntervals)
                total_width = 0.8  # Total width for all bars for one scenario
                width = total_width / num_beamf_intervals  # Width of each bar

                fig, ax = plt.subplots(figsize=(15, 10))
                colors = plt.cm.viridis(np.linspace(0, 1, num_beamf_intervals))

                # Bar chart plotting for PDR
                for idx, beamformingInterval in enumerate(beamformingIntervals):
                    avg_pdrs = [np.mean(avg_pdr_per_scenario_beamf[(scenarioID, beamformingInterval)])
                                for scenarioID in labels]
                    rects = ax.bar(x + idx * width - (total_width - width) / 2, avg_pdrs, width, label=f'{beamformingInterval}s',
                                color=colors[idx], edgecolor='black')

                ax.set_ylabel('Average PDR (%)')
                ax.set_xlabel('Scenario ID')
                throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                codebook_value = codebook_text_map.get(codebookFile, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                ax.set_title(f'Average PDR per Scenario for Tx Power: {power} dBm, App ID: {app_id} (~{throughput_value}Mbps), Codebook File: {codebook_value}')
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                # ax.legend()
                ax.legend(title="Beamforming Interval",loc='upper left', bbox_to_anchor=(1,1))

                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                fig.tight_layout(pad=2.0)

                # Save the bar chart figure for PDR
                
                plt.savefig(os.path.join(trends_folder_pdr, f'avg_pdr_barchart_app{app_id}_TxPower{power}_codebookFile{codebookFile}.png'), bbox_inches='tight')
                plt.close(fig)

                #######################
                # PLOT THROUGHPUT     #
                #######################
                # Now we have the data, let's plot the bar chart and boxplot
                labels = sorted(scenarioIDs)
                x = np.arange(len(labels))  # the label locations
                num_beamf_intervals = len(beamformingIntervals)
                total_width = 0.8  # Total width for all bars for one scenario
                width = total_width / num_beamf_intervals  # Width of each bar

                fig, ax = plt.subplots(figsize=(15, 10))
                colors = plt.cm.viridis(np.linspace(0, 1, num_beamf_intervals))

                # Bar chart plotting
                for idx, beamformingInterval in enumerate(beamformingIntervals):
                    avg_throughputs = [np.mean(avg_throughput_per_scenario_beamf[(scenarioID, beamformingInterval)])
                                    for scenarioID in labels]
                    rects = ax.bar(x + idx * width - (total_width - width) / 2, avg_throughputs, width, label=f'{beamformingInterval}s',
                                color=colors[idx], edgecolor='black')

                ax.set_ylabel('Average Throughput (Mbps)')
                ax.set_xlabel('Scenario ID')
                throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                codebook_value = codebook_text_map.get(codebookFile, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                ax.set_title(f'Average Throughput per Scenario for Tx Power: {power} dBm, App ID: {app_id} (~{throughput_value}Mbps), Codebook File: {codebook_value}')
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                # ax.legend()
                ax.legend(title="Beamforming Interval",loc='upper left', bbox_to_anchor=(1,1))

                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                fig.tight_layout(pad=2.0)

                # Save the bar chart figure
                plt.savefig(os.path.join(trends_folder, f'avg_throughput_barchart_app{app_id}_TxPower{power}_codebookFile{codebookFile}.png'), bbox_inches='tight')
                plt.close(fig)

                # Boxplot plotting
                fig, ax = plt.subplots(figsize=(15, 10))
                positions = np.array(range(len(labels))) * (len(beamformingIntervals) + 1)  # Set starting position for each group of boxplots
                for idx, beamformingInterval in enumerate(beamformingIntervals):
                    data_to_plot = [throughput_data_per_scenario_beamf[(scenarioID, beamformingInterval)] for scenarioID in labels]
                    ax.boxplot(data_to_plot, positions=positions + idx, widths=0.6, patch_artist=True,
                            boxprops=dict(facecolor=colors[idx], color=colors[idx]),
                            meanprops=meanpointprops,
                            showmeans=True,
                            medianprops=medianColor,
                            flierprops=dict(markerfacecolor=colors[idx], marker='o', markersize=6, linestyle='none'),
                            whiskerprops=dict(color=colors[idx]),
                            capprops=dict(color=colors[idx]))

                ax.set_xticks(positions + len(beamformingIntervals) / 2)
                ax.set_xticklabels(labels)

                # Adding legend for beamforming intervals
                legend_elements = [Patch(facecolor=colors[idx], edgecolor=colors[idx],
                                        label=f'{beamformingIntervals[idx]}s') for idx in range(len(beamformingIntervals))]
                # ax.legend(handles=legend_elements, title="Beamforming Intervals")

                ax.legend(handles=legend_elements,title="Beamforming Interval",loc='upper left', bbox_to_anchor=(1,1))

                ax.set_ylabel('Throughput (Mbps)')
                ax.set_xlabel('Scenario ID')
                throughput_value = app_throughput_map.get(app_id, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                codebook_value = codebook_text_map.get(codebookFile, 'Unknown')  # Default to 'Unknown' if app_id is not in the map
                ax.set_title(f'Boxplot of Throughput per Scenario for Tx Power: {power} dBm, App ID {app_id} (~{throughput_value}Mbps), Codebook File: {codebook_value}')
                fig.tight_layout(pad=2.0)

                # Save the boxplot figure
                plt.savefig(os.path.join(trends_folder, f'boxplot_throughput_app{app_id}_TxPower{power}_codebookFile{codebookFile}.png'), bbox_inches='tight')
                plt.close(fig)
                        
            


######################### END BEAMFORMING INTERVAL EFFECT ###################################
# Assuming `codebookFiles`, `beamformingIntervals`, `app_ids`, `txPower`, `scenarioIDs`, and other necessary data have been defined

# Loop over each Tx Power, App ID, and Beamforming Interval
# for power in txPower:
#     for app_id in app_ids:
#         for beamformingInterval in beamformingIntervals:
#             avg_throughput_per_scenario_codebook = {(scenarioID, codebookFile): []
#                                                     for scenarioID in scenarioIDs
#                                                     for codebookFile in codebookFiles}
            
#             # Collect data for each scenario and codebook file
#             for scenarioID in scenarioIDs:
#                 for codebookFile in codebookFiles:
#                     throughputs = []  # Store throughputs for this scenario and codebook file
#                     for runNumber in runNumbers:
#                         # Define the path for the throughput files
#                         folder_path = os.path.join(base_folder, codebookFile, f"BeamFInterval{beamformingInterval}")
#                         throughput_file = os.path.join(folder_path, f'SurgeryFeedbackPos{scenarioID}_throughput_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt')
                        
#                         # Read the data if the file exists
#                         if os.path.exists(throughput_file):
#                             throughput_data = pd.read_csv(throughput_file, sep=',', header=None)
#                             filtered_throughput = throughput_data[1].drop(index=[0] + list(range(300, 310)), errors='ignore')
#                             throughputs.extend(filtered_throughput.values)
                
#                     # Calculate the average throughput for this scenario and codebook file
#                     if throughputs:
#                         avg_throughput = np.mean(throughputs)
#                         avg_throughput_per_scenario_codebook[(scenarioID, codebookFile)].append(avg_throughput)

#             # Now we have the data, we can plot it
#             labels = sorted(scenarioIDs)
#             x = np.arange(len(labels))  # the label locations
#             num_codebooks = len(codebookFiles)
#             total_width = 0.8  # Total width for all bars for one scenario
#             width = total_width / num_codebooks  # Width of each bar

#             fig, ax = plt.subplots(figsize=(15, 10))
#             colors = plt.cm.viridis(np.linspace(0, 1, num_codebooks))

#             for idx, codebookFile in enumerate(codebookFiles):
#                 avg_throughputs = [np.mean(avg_throughput_per_scenario_codebook[(scenarioID, codebookFile)])
#                                    for scenarioID in labels]

#                 # Plot
#                 rects = ax.bar(x + idx * width - (total_width - width) / 2, avg_throughputs, width, label=f'Codebook {codebookFile}',
#                                color=colors[idx], edgecolor='black')

#             ax.set_ylabel('Average Throughput (Mbps)')
#             ax.set_xlabel('Scenario ID')
#             ax.set_title(f'Average Throughput per Scenario for Tx Power {power}, App ID {app_id}, Beamforming Interval {beamformingInterval}')
#             ax.set_xticks(x)
#             ax.set_xticklabels(labels)
#             ax.legend()

#             plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#             fig.tight_layout(pad=2.0)

#             # Save the figure
#             plt.savefig(os.path.join(codebook_effect_folder, f'avg_throughput_barchart_app{app_id}_TxPower{power}_beamf{beamformingInterval}.png'), bbox_inches='tight')
#             plt.close(fig)





# # Loop through each codebookFile, beamformingInterval, and app_id
# for codebookFile in tqdm(codebookFiles, desc="Processing codebook files", ncols=100):
#     for beamformingInterval in beamformingIntervals:
#         for app_id in app_ids:
#             # Prepare data structure for average throughput per scenario and Tx Power
#             avg_throughput_per_scenario_power = {(scenarioID, power): []
#                                                  for scenarioID in scenarioIDs
#                                                  for power in txPower}

#             # Collect data for each scenario and Tx Power, filtered by app ID
#             for scenarioID in scenarioIDs:
#                 for power in txPower:
#                     throughputs = []  # Store throughputs for this scenario, power, and app_id
#                     for runNumber in runNumbers:
#                         # Define the path for the throughput files
#                         folder_path = os.path.join(base_folder, codebookFile, f"BeamFInterval{beamformingInterval}")
#                         throughput_file = os.path.join(folder_path, f'SurgeryFeedbackPos{scenarioID}_throughput_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt')
                        
#                         # Read the data if the file exists
#                         if os.path.exists(throughput_file):
#                             throughput_data = pd.read_csv(throughput_file, sep=',', header=None)
#                             filtered_throughput = throughput_data[1].drop(index=[0] + list(range(300, 310)), errors='ignore')
#                             throughputs.extend(filtered_throughput.values)
                
#                     # Calculate the average throughput for this scenario and Tx Power
#                     if throughputs:
#                         avg_throughput = np.mean(throughputs)
#                         avg_throughput_per_scenario_power[(scenarioID, power)].append(avg_throughput)
            
#              # Now we have the data, we can plot it
#             labels = sorted(set(scenarioID for scenarioID, _ in avg_throughput_per_scenario_power.keys()))
#             x = np.arange(len(labels))  # the label locations

#             # Calculate the total width needed for all bars in a group
#             total_width = 0.8  # This can be adjusted to fit as desired
#             num_bars = len(txPower)
#             width = total_width / num_bars  # the width of each bar

#             fig, ax = plt.subplots(figsize=(15, 10))  # Set figure size to make it bigger
#             colors = plt.cm.viridis(np.linspace(0, 1, num_bars))  # Use a colormap

#             # Offset calculation to center bars
#             offset = (total_width - width) / 2

#             for idx, power in enumerate(sorted(txPower)):
#                 # Compute the average throughput for this Tx Power across all scenarios
#                 power_avg_throughput = [np.mean(avg_throughput_per_scenario_power[(scenarioID, power)])
#                                         for scenarioID in labels]

#                 # Adjust the position to center the bars
#                 bar_positions = x - offset + idx * width

#                 # Plot with black edges
#                 rects = ax.bar(bar_positions, power_avg_throughput, width, label=f'Tx Power {power}',
#                                color=colors[idx], edgecolor='black')

#             # Add some text for labels, title and custom x-axis tick labels, etc.
#             ax.set_ylabel('Average Throughput (Mbps)')
#             ax.set_xlabel('Scenario')  # Label for the X-axis
#             ax.set_title(f'Average Throughput by Scenario and Tx Power for App ID {app_id}, Codebook {codebookFile}, Beamforming Interval {beamformingInterval}')
#             ax.set_xticks(x)
#             ax.set_xticklabels(labels)
#             ax.legend()

#             # Rotate the tick labels for better readability
#             plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

#             # Make the figure layout fit the current axis and labels. Pad it a bit.
#             fig.tight_layout(pad=2.0)
#             plt.savefig(os.path.join(txPower_effect_folder, f'avg_throughput_barchart_app{app_id}_codebook{codebookFile}_beamf{beamformingInterval}.png'), bbox_inches='tight')
#             plt.close(fig)  # Close the figure to avoid memory issues

############# END BARCHART
# exit(1)
# Before the loop, create dictionaries to store the max throughput and SINR for each scenario

print("Individual Delay: Create all the figures")
if ENABLE_DELAY_INDIVIDUAL:

    # Process and plot delay and SNR for each configuration
    for codebookFile in tqdm(codebookFiles, desc="Processing codebook ", ncols=100):
        for beamformingInterval in tqdm(beamformingIntervals, desc="Processing beamforming", ncols=100):
            beamforming_folder = os.path.join(detailed_delay_folder, codebookFile, f"BeamFInterval{beamformingInterval}")
            os.makedirs(beamforming_folder, exist_ok=True)

            for app_id in tqdm(app_ids, desc="Processing app ids", leave=False, ncols=100):
                for scenarioID in tqdm(scenarioIDs, desc="Processing Scenario ids", leave=False, ncols=100):
                    for runNumber in tqdm(runNumbers, desc="Processing Run Number", leave=False, ncols=100):
                        for power in tqdm(txPower, desc="Processing Power", leave=False, ncols=100):
                            # Define the path for delay and SNR files
                            folder_path = os.path.join(base_folder, codebookFile, f"BeamFInterval{beamformingInterval}")
                            delay_file = os.path.join(folder_path, f'SurgeryFeedbackPos{scenarioID}_delay_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt')
                            snr_file = os.path.join(folder_path, f'SurgeryFeedbackPos{scenarioID}_snr_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt')
                            # print("Read File:",delay_file)
                            # Read and process the delay data
                            if os.path.exists(delay_file) and os.path.exists(snr_file):
                                delay_data = pd.read_csv(delay_file, sep='\t', skiprows=1, header=None)
                                delay_time = delay_data.iloc[:, 0]
                                delay_values = delay_data.iloc[:, 2] / 1e6  # Convert ns to ms
                                
                                snr_data = pd.read_csv(snr_file, sep=',', header=None)
                                snr_data.replace(-np.inf, np.nan, inplace=True)  # Replace -inf with NaN
                                

                                #####################NEW
                                # # SINR VS TIME combined with NUMSYM VS TIME
                                # fig, ax1 = plt.subplots(figsize=(15, 10))  # Adjust figure size as needed
                                # # ax1.plot(mcs_data['TIME'], mcs_data['NUMSYM'], label=f'NUMSYM, txPower={power}')
                                # ax1.scatter(mcs_data['TIME'], mcs_data['NUMSYM'], label=f'Number of Symbol', color='b', s=1)
                                # ax1.set_xlabel('Time (s)')
                                # ax1.set_ylabel('Number of Symbol')
                                # ax1.tick_params(axis='y')

                                # # Set major and minor ticks for MCS axis
                                # # ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
                                # # ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))

                                # ax2 = ax1.twinx()
                                # ax2.plot(snr_data[0], snr_data[1], 'r--', label=f'SINR')
                                # ax2.set_ylabel('SINR (dB)', color='r')
                                # ax2.tick_params(axis='y', labelcolor='r')

                                
                                # plt.title(f'Combined Number of Symbols and SINR vs Time for rfh-app-{app_id}, txPower={power}, ScenarioID={scenarioID}, Run={runNumber}')
                                # lines, labels = ax1.get_legend_handles_labels()
                                # lines2, labels2 = ax2.get_legend_handles_labels()
                                # ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside

                                # # Add grid to the plot
                                # ax1.grid(True, which='both')

                                # # Save the figure in the corresponding folder
                                # plt.savefig(os.path.join(beamforming_folder, f'combined_numsym_snr_rfh-app-{app_id}_ScenarioID{scenarioID}_txPower{power}_Run{runNumber}.png'), bbox_inches='tight')
                                # plt.close(fig)


                                ############# NEW END


                                # DELAY VS TIME combined with SINR VS TIME
                                # DELAY VS TIME combined with SINR VS TIME
                                fig, ax1 = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed

                                # Scatter plot for delay vs time
                                ax1.scatter(delay_time, delay_values, color='b', s=0.1, marker="x", label='Delay')
                                ax1.set_xlabel('Time (s)',fontsize=26)
                                ax1.set_ylabel('Delay (ms)', color='b',fontsize=26)
                                ax1.tick_params(axis='y', labelcolor='b')
                                ax1.grid(True, which='major')
                                ax1.tick_params(axis='x', labelsize=22)  # Set x-axis tick label font size
                                ax1.tick_params(axis='y', labelsize=22)  # Set y-axis tick label font size

                                ax1.set_xlim(0,310)

                                # Set major and minor ticks for time (x-axis)
                                ax1.xaxis.set_major_locator(MultipleLocator(50))  # Major ticks every 10
                                ax1.xaxis.set_minor_locator(MultipleLocator(10))  # Minor ticks every 1

                                # Plot SNR vs time on a secondary y-axis
                                ax2 = ax1.twinx()
                                ax2.plot(snr_data.iloc[:, 0], snr_data.iloc[:, 1], 'r--', label='SINR')
                                ax2.set_ylabel('SINR (dB)', color='r',fontsize=26)
                                ax2.tick_params(axis='y', labelcolor='r')
                                
                                ax2.tick_params(axis='y', labelsize=22)  # Set y-axis tick label font size

                                # Set minor ticks for delay (y-axis of ax1)
                                ax1.yaxis.set_minor_locator(MultipleLocator(0.1))  # Minor ticks every 0.1

                                # Create custom legend handles
                                delay_handle = mlines.Line2D([], [], color='blue', marker='x', linestyle='None', label='Delay')
                                sinr_handle = mlines.Line2D([], [], color='red', linestyle='--', label='SINR')

                                # Combine the handles for the legend and position it inside the upper right of the graph
                                plt.legend(handles=[delay_handle, sinr_handle], loc='upper right',fontsize=26)

                                # plt.xlabel('SINR (dB)',fontsize=16)
                                # plt.ylabel('Cumulative Probability', fontsize=16)
                                # ax.tick_params(axis='x', labelsize=14)  # Set x-axis tick label font size
                                # ax.tick_params(axis='y', labelsize=14)  # Set y-axis tick label font size

                                # legend = plt.legend(handles=scenario_legend_handles, loc='upper left',title='BS position',fontsize=11.5,ncol=2)
                                # legend.get_title().set_fontsize('14')  # Set font size for the legend title


                                # Save the figure in the corresponding folder
                                toto = os.path.join(beamforming_folder, f'combined_delay_snr_rfh-app-{app_id}_ScenarioID{scenarioID}_txPower{power}_Run{runNumber}.pdf')
                                print("savefig",toto)
                                plt.savefig(os.path.join(beamforming_folder, f'combined_delay_snr_rfh-app-{app_id}_ScenarioID{scenarioID}_txPower{power}_Run{runNumber}.pdf'), bbox_inches='tight')
                                plt.close(fig)  # Close the figure to avoid memory issues
                                                              
    


print("Individual Throughput: Create all the figures")
if ENABLE_THROUGHPUT_INDIVIDUAL:
    max_throughput_per_scenario = {}
    max_snr_per_scenario = {}
    min_snr_per_scenario = {}

    # Loop through each file to populate the dictionaries with the max values
    for codebookFile in codebookFiles:
        for beamformingInterval in beamformingIntervals:
            for app_id in app_ids:
                for scenarioID in scenarioIDs:
                    max_throughput = 0
                    max_snr = -100  # Assuming -100 dB is the minimum possible SINR
                    min_snr = 100   # Assuming 100 dB is a high enough value to be reduced to the actual min SINR

                    for runNumber in runNumbers:
                        for power in txPower:
                            # Define the path for throughput and SNR files
                            folder_path = os.path.join(base_folder, codebookFile, f"BeamFInterval{beamformingInterval}")
                            throughput_file = os.path.join(folder_path, f'SurgeryFeedbackPos{scenarioID}_throughput_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt')
                            snr_file = os.path.join(folder_path, f'SurgeryFeedbackPos{scenarioID}_snr_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt')
                            
                            # Read the data
                            if os.path.exists(throughput_file):
                                throughput_data = pd.read_csv(throughput_file, sep=',', header=None)
                                current_max_throughput = throughput_data[1].drop(index=[0] + list(range(300, 310)), errors='ignore').max()
                                max_throughput = max(max_throughput, current_max_throughput)

                            # Read the data
                            if os.path.exists(snr_file):
                                snr_data = pd.read_csv(snr_file, sep=',', header=None)
                                snr_data.replace(-np.inf, np.nan, inplace=True)
                                current_max_snr = snr_data[1].max()
                                current_min_snr = snr_data[1].min()
                                max_snr = max(max_snr, current_max_snr)
                                min_snr = min(min_snr, current_min_snr)

                    # Store the max and min values with a 10% wiggle room for max
                    max_snr_per_scenario[(app_id, scenarioID, beamformingInterval, codebookFile)] = max_snr * 1.1
                    min_snr_per_scenario[(app_id, scenarioID, beamformingInterval, codebookFile)] = min_snr

                    # Store the max values with a 10% wiggle room
                    max_throughput_per_scenario[(app_id, scenarioID, beamformingInterval, codebookFile)] = max_throughput * 1.1
                    max_snr_per_scenario[(app_id, scenarioID, beamformingInterval, codebookFile)] = max_snr * 1.1

    for codebookFile in tqdm(codebookFiles, desc="Processing codebook ", ncols=100):
        for beamformingInterval in tqdm(beamformingIntervals, desc="Processing beamforming", ncols=100):
        # for beamformingInterval in beamformingIntervals:
            codebook_folder = os.path.join(detailed_throughput_folder, codebookFile)
            beamforming_folder = os.path.join(codebook_folder, f"BeamFInterval{beamformingInterval}")

            # Create the codebook and beamforming folders if they don't exist
            os.makedirs(beamforming_folder, exist_ok=True)

            for app_id in tqdm(app_ids, desc="Processing app ids", leave=False, ncols=100):
                for scenarioID in tqdm(scenarioIDs, desc="Processing Scenario ids", leave=False, ncols=100):
                # for scenarioID in scenarioIDs:
                    for runNumber in tqdm(runNumbers, desc="Processing Run Number", leave=False, ncols=100):
                    # for runNumber in runNumbers:
                        for power in tqdm(txPower, desc="Processing Power", leave=False, ncols=100):
                        # for power in txPower:
                            # Define the path for throughput and SNR files
                            folder_path = os.path.join(base_folder, codebookFile, f"BeamFInterval{beamformingInterval}")
                            throughput_file = os.path.join(folder_path, f'SurgeryFeedbackPos{scenarioID}_throughput_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt')
                            snr_file = os.path.join(folder_path, f'SurgeryFeedbackPos{scenarioID}_snr_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt')
                            
                            # Read the data
                            if os.path.exists(throughput_file) and os.path.exists(snr_file):
                                throughput_data = pd.read_csv(throughput_file, sep=',', header=None)
                                filtered_throughput = throughput_data[1].drop(index=[0] + list(range(300, 310)), errors='ignore')
                                
                                throughput_values_per_combination[(app_id, scenarioID, runNumber, power, beamformingInterval, codebookFile)].extend(filtered_throughput.values)
                                
                                snr_data = pd.read_csv(snr_file, sep=',', header=None)
                                snr_data.replace(-np.inf, np.nan, inplace=True)  # Replace -inf with NaN

                                # THROUGHPUT VS TIME combined with SINR VS TIME
                                fig, ax1 = plt.subplots()
                                ax1.plot(throughput_data[0], throughput_data[1], label=f'Throughput, txPower={power}')
                                ax1.set_xlabel('Time (s)')
                                ax1.set_ylabel('Throughput (Mbps)', color='b')
                                ax1.tick_params(axis='y', labelcolor='b')

                                # Plot average throughput as a horizontal line
                                average_throughput = np.nanmean(filtered_throughput)
                                ax1.axhline(y=average_throughput, color='g', linestyle='-', label=f'Average Throughput ({average_throughput:.2f} Mbps)')

                                ax2 = ax1.twinx()
                                ax2.plot(snr_data[0], snr_data[1], 'r--', label=f'SINR')
                                ax2.set_ylabel('SINR (dB)', color='r')
                                ax2.tick_params(axis='y', labelcolor='r')
                                # ax2.set_ylim(-100, snr_data[1].max() + 5)  # Set the y-axis limits
                                
                                ax1.set_ylim(0, max_throughput_per_scenario[(app_id, scenarioID, beamformingInterval, codebookFile)])
                                ax2.set_ylim(min_snr_per_scenario[(app_id, scenarioID, beamformingInterval, codebookFile)], max_snr_per_scenario[(app_id, scenarioID, beamformingInterval, codebookFile)])

                                fig.tight_layout()
                                plt.title(f'Combined Throughput and SINR vs Time for rfh-app-{app_id}, txPower={power}, ScenarioID={scenarioID}, Run={runNumber}')
                                lines, labels = ax1.get_legend_handles_labels()
                                lines2, labels2 = ax2.get_legend_handles_labels()
                                ax1.legend(lines + lines2, labels + labels2, loc=0)

                                # Save the figure in the corresponding folder
                                plt.savefig(os.path.join(beamforming_folder, f'combined_throughput_snr_rfh-app-{app_id}_ScenarioID{scenarioID}_txPower{power}_Run{runNumber}.png'))
                                plt.close(fig)  # Close the figure to avoid memory issues

if ENABLE_MCS_SINR_INDIVIDUAL:
    for codebookFile in tqdm(codebookFiles, desc="Processing codebook ", ncols=100):
        for beamformingInterval in tqdm(beamformingIntervals, desc="Processing beamforming", ncols=100):
            codebook_folder = os.path.join(detailed_mcs_folder, codebookFile)
            beamforming_folder = os.path.join(codebook_folder, f"BeamFInterval{beamformingInterval}")

            # Create the codebook and beamforming folders if they don't exist
            os.makedirs(beamforming_folder, exist_ok=True)

            for app_id in tqdm(app_ids, desc="Processing app ids", leave=False, ncols=100):
                for scenarioID in tqdm(scenarioIDs, desc="Processing Scenario ids", leave=False, ncols=100):
                    for runNumber in tqdm(runNumbers, desc="Processing Run Number", leave=False, ncols=100):
                        for power in tqdm(txPower, desc="Processing Power", leave=False, ncols=100):
                            # Define the path for SNR and MCS files
                            folder_path = os.path.join(base_folder, codebookFile, f"BeamFInterval{beamformingInterval}")
                            snr_file = os.path.join(folder_path, f'SurgeryFeedbackPos{scenarioID}_snr_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt')
                            mcs_file = os.path.join(folder_path, f'SurgeryFeedbackPos{scenarioID}_UE_rlcAmEnabledrfh-app-{app_id}txPower{power}runNumber{runNumber}.txt')

                            if os.path.exists(snr_file) and os.path.exists(mcs_file):
                                snr_data = pd.read_csv(snr_file, sep=',', header=None)
                                snr_data.replace(-np.inf, np.nan, inplace=True)  # Replace -inf with NaN

                                mcs_data = pd.read_csv(mcs_file, sep=',', usecols=['TIME', 'MCS', 'Corrupt','RV','NUMSYM', 'TBSIZE'])
                                # mcs_data = mcs_data[mcs_data['MCS'] != 0]  # Filter out MCS values of 0

                                # corrup_data = pd.read_csv(mcs_file, sep=',', usecols=['TIME', 'Corrupt'])

                                # retrans_data = pd.read_csv(mcs_file, sep=',', usecols=['TIME', 'RV'])

                                # numSym_data = pd.read_csv(mcs_file, sep=',', usecols=['TIME', 'NUMSYM'])

                                # tbSize_data = pd.read_csv(mcs_file, sep=',', usecols=['TIME', 'TBSIZE'])
                              

                                ############################# SINR VS TIME combined with MCS VS TIME
                                fig, ax1 = plt.subplots(figsize=(15, 10))  # Adjust figure size as needed
                                # ax1.plot(mcs_data['TIME'], mcs_data['MCS'], label=f'MCS, txPower={power}')
                                ax1.scatter(mcs_data['TIME'], mcs_data['MCS'], label=f'MCS',marker='+', color='b', s=1,)
                                ax1.set_xlabel('Time (s)')
                                ax1.set_ylabel('MCS')
                                ax1.tick_params(axis='y')

                                # Set major and minor ticks for MCS axis
                                ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
                                ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))

                                ax2 = ax1.twinx()
                                ax2.plot(snr_data[0], snr_data[1], 'r--', label=f'SINR')
                                ax2.set_ylabel('SINR (dB)', color='r')
                                ax2.tick_params(axis='y', labelcolor='r')

                                plt.subplots_adjust(top=0.97)  # Adjust the top spacing to make room for title
                                plt.title(f'Combined MCS and SINR vs Time for rfh-app-{app_id}, txPower={power}, ScenarioID={scenarioID}, Run={runNumber}')
                                lines, labels = ax1.get_legend_handles_labels()
                                lines2, labels2 = ax2.get_legend_handles_labels()
                                ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside

                                # Add grid to the plot
                                ax1.grid(True, which='both')

                                # Save the figure in the corresponding folder
                                plt.savefig(os.path.join(beamforming_folder, f'combined_mcs_snr_rfh-app-{app_id}_ScenarioID{scenarioID}_txPower{power}_Run{runNumber}.png'), bbox_inches='tight')
                                plt.close(fig)


                                # SINR VS TIME combined with Coorup VS TIME
                                fig, ax1 = plt.subplots(figsize=(15, 10))  # Adjust figure size as needed
                                ax1.scatter(mcs_data['TIME'], mcs_data['Corrupt'], label=f'Corrupted', color='b', s=1)
                                # ax1.plot(mcs_data['TIME'], mcs_data['Corrupt'], label=f'Corrupt, txPower={power}')
                                ax1.set_xlabel('Time (s)')
                                ax1.set_ylabel('Corrupt')
                                ax1.tick_params(axis='y')

                                # Set major and minor ticks for MCS axis
                                ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
                                ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))

                                ax2 = ax1.twinx()
                                ax2.plot(snr_data[0], snr_data[1], 'r--', label=f'SINR')
                                
                                ax2.set_ylabel('SINR (dB)', color='r')
                                ax2.tick_params(axis='y', labelcolor='r')

                            
                                plt.title(f'Combined Corrupted Packet and SINR vs Time for rfh-app-{app_id}, txPower={power}, ScenarioID={scenarioID}, Run={runNumber}')
                                lines, labels = ax1.get_legend_handles_labels()
                                lines2, labels2 = ax2.get_legend_handles_labels()
                                ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside

                                # Add grid to the plot
                                ax1.grid(True, which='both')

                                # Save the figure in the corresponding folder
                                plt.savefig(os.path.join(beamforming_folder, f'combined_corrupt_snr_rfh-app-{app_id}_ScenarioID{scenarioID}_txPower{power}_Run{runNumber}.png'), bbox_inches='tight')
                                plt.close(fig)

                                # SINR VS TIME combined with RV VS TIME
                                fig, ax1 = plt.subplots(figsize=(15, 10))  # Adjust figure size as needed
                                # ax1.plot(mcs_data['TIME'], mcs_data['RV'], label=f'Retransmission, txPower={power}')
                                ax1.scatter(mcs_data['TIME'], mcs_data['RV'], label=f'Retransmission', color='b', s=5)
                                ax1.set_xlabel('Time (s)')
                                ax1.set_ylabel('Retransmission')
                                ax1.tick_params(axis='y')

                                # Set major and minor ticks for MCS axis
                                ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
                                ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))

                                ax2 = ax1.twinx()
                                ax2.plot(snr_data[0], snr_data[1], 'r--', label=f'SINR')
                                ax2.set_ylabel('SINR (dB)', color='r')
                                ax2.tick_params(axis='y', labelcolor='r')

                                
                                plt.title(f'Combined Retransmission and SINR vs Time for rfh-app-{app_id}, txPower={power}, ScenarioID={scenarioID}, Run={runNumber}')
                                lines, labels = ax1.get_legend_handles_labels()
                                lines2, labels2 = ax2.get_legend_handles_labels()
                                ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside

                                # Add grid to the plot
                                ax1.grid(True, which='both')

                                # Save the figure in the corresponding folder
                                plt.savefig(os.path.join(beamforming_folder, f'combined_rtx_snr_rfh-app-{app_id}_ScenarioID{scenarioID}_txPower{power}_Run{runNumber}.png'), bbox_inches='tight')
                                plt.close(fig)


                                # SINR VS TIME combined with NUMSYM VS TIME
                                fig, ax1 = plt.subplots(figsize=(15, 10))  # Adjust figure size as needed
                                # ax1.plot(mcs_data['TIME'], mcs_data['NUMSYM'], label=f'NUMSYM, txPower={power}')
                                ax1.scatter(mcs_data['TIME'], mcs_data['NUMSYM'], label=f'Number of Symbol', color='b', s=1)
                                ax1.set_xlabel('Time (s)')
                                ax1.set_ylabel('Number of Symbol')
                                ax1.tick_params(axis='y')

                                # Set major and minor ticks for MCS axis
                                # ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
                                # ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))

                                ax2 = ax1.twinx()
                                ax2.plot(snr_data[0], snr_data[1], 'r--', label=f'SINR')
                                ax2.set_ylabel('SINR (dB)', color='r')
                                ax2.tick_params(axis='y', labelcolor='r')

                                
                                plt.title(f'Combined Number of Symbols and SINR vs Time for rfh-app-{app_id}, txPower={power}, ScenarioID={scenarioID}, Run={runNumber}')
                                lines, labels = ax1.get_legend_handles_labels()
                                lines2, labels2 = ax2.get_legend_handles_labels()
                                ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside

                                # Add grid to the plot
                                ax1.grid(True, which='both')

                                # Save the figure in the corresponding folder
                                plt.savefig(os.path.join(beamforming_folder, f'combined_numsym_snr_rfh-app-{app_id}_ScenarioID{scenarioID}_txPower{power}_Run{runNumber}.png'), bbox_inches='tight')
                                plt.close(fig)

                                # SINR VS TIME combined with TBSIZE VS TIME
                                fig, ax1 = plt.subplots(figsize=(15, 10))  # Adjust figure size as needed
                                # ax1.plot(mcs_data['TIME'], mcs_data['TBSIZE'], label=f'NUMSYM, txPower={power}')
                                ax1.scatter(mcs_data['TIME'], mcs_data['TBSIZE'], label=f'Transport Block Size', color='b', s=1)
                                ax1.set_xlabel('Time (s)')
                                ax1.set_ylabel('Transport Block Size')
                                ax1.tick_params(axis='y')

                                # Set major and minor ticks for MCS axis
                                # ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
                                # ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))

                                ax2 = ax1.twinx()
                                ax2.plot(snr_data[0], snr_data[1], 'r--', label=f'SINR')
                                ax2.set_ylabel('SINR (dB)', color='r')
                                ax2.tick_params(axis='y', labelcolor='r')

                                
                                plt.title(f'Combined Transport Block Size and SINR vs Time for rfh-app-{app_id}, txPower={power}, ScenarioID={scenarioID}, Run={runNumber}')
                                lines, labels = ax1.get_legend_handles_labels()
                                lines2, labels2 = ax2.get_legend_handles_labels()
                                ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside

                                # Add grid to the plot
                                ax1.grid(True, which='both')

                                # Save the figure in the corresponding folder
                                plt.savefig(os.path.join(beamforming_folder, f'combined_tbsize_snr_rfh-app-{app_id}_ScenarioID{scenarioID}_txPower{power}_Run{runNumber}.png'), bbox_inches='tight')
                                plt.close(fig)

# max_throughput_per_scenario = defaultdict(float)

# # Determine the max throughput for each combination, for each scenario
# for params, values in throughput_values_per_combination.items():
#     app_id, scenarioID, runNumber, power, beamformingInterval, codebookFile = params
#     key = (app_id, scenarioID, beamformingInterval, codebookFile)
#     current_max_throughput = max(values)
#     max_throughput_per_scenario[key] = max(max_throughput_per_scenario[key], current_max_throughput)

# # Apply a 10% wiggle room to the maximum throughput
# max_throughput_per_scenario_with_wiggle = {k: v * 1.1 for k, v in max_throughput_per_scenario.items()}

# # Generate and save the boxplots for throughput
# for params, values in throughput_values_per_combination.items():
#     app_id, scenarioID, runNumber, power, beamformingInterval, codebookFile = params
#     plt.figure()
#     plt.boxplot(values)
#     plt.title(f'Boxplot of Throughput for AppID {app_id}, ScenarioID {scenarioID}, Run {runNumber}, txPower {power}, BeamFInterval {beamformingInterval}, Codebook {codebookFile}')
#     plt.xlabel('Samples')
#     plt.ylabel('Throughput (Mbps)')

#     # Set y-axis limits based on the calculated maximum throughput for the scenario with a 10% wiggle room
#     max_throughput = max_throughput_per_scenario_with_wiggle[(app_id, scenarioID, beamformingInterval, codebookFile)]
#     plt.ylim(0, max_throughput)  # Setting the y-axis to start at 0 and max to the calculated max throughput with wiggle

#     # Define the folder path based on parameters
#     boxplot_folder = os.path.join(detailed_throughput_folder, codebookFile, f"BeamFInterval{beamformingInterval}", f"ScenarioID{scenarioID}")
#     os.makedirs(boxplot_folder, exist_ok=True)  # Ensure the directory exists

#     # Save the figure in the corresponding folder
#     boxplot_filename = f'boxplot_throughput_appID{app_id}_scenarioID{scenarioID}_runNumber{runNumber}_txPower{power}_beamFInterval{beamformingInterval}_codebook{codebookFile}.png'
#     plt.savefig(os.path.join(boxplot_folder, boxplot_filename))
#     plt.close()  # Close the figure to free up memory



from itertools import cycle
# Load and process the rotation data
rotation_data = pd.read_csv('UERotations.txt', header=None, names=['rotation_x', 'rotation_y', 'rotation_z'])
rotation_data['angular_speed'] = np.sqrt(rotation_data['rotation_x']**2 + 
                                         rotation_data['rotation_y']**2 + 
                                         rotation_data['rotation_z']**2)

# Assuming each row in rotation_data represents one second
time_data = range(0, len(rotation_data))

# Phase information
phase_durations = [100, 10, 30, 10]
phase_names = ["Surgeon facing the patient", "Surgeon rotates for imagery", "Surgeon observes imagery", "Surgeon rotates to patient"]
phase_colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
cycle_duration = sum(phase_durations)

if ENABLE_SINR_BEAMINTERVAL_INDIVIDUAL:
    line_styles = cycle(['-', '--', '-.', ':'])  # Define different line styles

    for codebookFile in tqdm(codebookFiles, desc="Processing codebook ", ncols=100):
        codebook_folder = os.path.join(detailed_sinr_folder, codebookFile)

        for app_id in tqdm(app_ids, desc="Processing app ids", leave=False, ncols=100):
            for scenarioID in tqdm(scenarioIDs, desc="Processing Scenario ids", leave=False, ncols=100):
                for runNumber in tqdm(runNumbers, desc="Processing Run Number", leave=False, ncols=100):
                    for power in tqdm(txPower, desc="Processing Power", leave=False, ncols=100):
                        # Create the figure and axis for SINR and angular speed
                        fig, ax1 = plt.subplots(figsize=(10, 6))
                        ax2 = ax1.twinx()

                        # Plot SINR for each beamInterval
                        for beamformingInterval, line_style in zip(beamformingIntervals, line_styles):
                            folder_path = os.path.join(base_folder, codebookFile, f"BeamFInterval{beamformingInterval}")
                            snr_file = os.path.join(folder_path, f'SurgeryFeedbackPos{scenarioID}_snr_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt')

                            if os.path.exists(snr_file):
                                snr_data = pd.read_csv(snr_file, sep=',', header=None)
                                snr_data.replace(-np.inf, np.nan, inplace=True)
                                ax1.plot(snr_data[0], snr_data[1], line_style, color='blue', label=f'Beam Interval {beamformingInterval}')

                        # Plot the angular speed data
                        ax2.plot(time_data, rotation_data['angular_speed'], label='Angular Speed', color='purple')

                        # Add vertical bands for phases
                        for t in range(0, len(rotation_data), cycle_duration):
                            start_time = t
                            for i, duration in enumerate(phase_durations):
                                end_time = start_time + duration
                                ax2.axvspan(start_time, end_time, facecolor=phase_colors[i], alpha=0.5, label=phase_names[i] if t == 0 else "")
                                start_time = end_time

                        # Set labels and title
                        ax1.set_xlabel('Time (seconds)')
                        ax1.set_ylabel('SINR')
                        ax2.set_ylabel('Angular Speed', color='purple')
                        ax1.set_title(f'SINR and Angular Speed Evolution for rfh-app-{app_id}, txPower={power}, ScenarioID={scenarioID}, Run={runNumber}')

                        # Legend and aesthetics
                        handles1, labels1 = ax1.get_legend_handles_labels()
                        handles2, labels2 = ax2.get_legend_handles_labels()
                        ax1.legend(handles1, labels1, loc='lower left')
                        ax2.legend(handles2, labels2, loc='upper right')

                        ax1.spines['top'].set_visible(False)
                        ax2.spines['top'].set_visible(False)

                        # Adjusting the layout to accommodate the legend
                        fig.tight_layout()

                        # Save the figure
                        plt.xlim(0, 305)
                        fig.savefig(os.path.join(detailed_sinr_folder, f'SINR_angular_speed_evolution_rfhcodebook{codebookFile}-app-{app_id}_ScenarioID{scenarioID}_txPower{power}_Run{runNumber}.png'), dpi=300, bbox_inches='tight')
                        plt.close(fig)

                        # Creating a zoomed-in figure
                       # Creating a zoomed-in figure
                        fig_zoom, ax_zoom = plt.subplots(figsize=(10, 6))
                        ax_zoom2 = ax_zoom.twinx()

                        # Plot SINR for each beamInterval (zoomed)
                        for beamformingInterval, line_style in zip(beamformingIntervals, line_styles):
                            zoom_snr_file = os.path.join(base_folder, codebookFile, f"BeamFInterval{beamformingInterval}", f'SurgeryFeedbackPos{scenarioID}_snr_{rlc_mode}rfh-app-{app_id}txPower{power}runNumber{runNumber}.txt')
                            
                            if os.path.exists(zoom_snr_file):
                                zoom_snr_data = pd.read_csv(zoom_snr_file, sep=',', header=None)
                                zoom_snr_data.replace(-np.inf, np.nan, inplace=True)
                                ax_zoom.plot(zoom_snr_data[0], zoom_snr_data[1], line_style, color='blue', label=f'Beam Interval {beamformingInterval}')

                        # Plot angular speed (zoomed)
                        ax_zoom2.plot(time_data, rotation_data['angular_speed'], label='Angular Speed', color='purple')

                        # Adjust the start time for phases in the zoomed plot
                        zoom_start = 100
                        for i, duration in enumerate(phase_durations):
                            start_time = zoom_start + i * sum(phase_durations)
                            while start_time < 160:
                                end_time = start_time + duration
                                ax_zoom2.axvspan(max(start_time, 100), min(end_time, 160), facecolor=phase_colors[i], alpha=0.5, label=phase_names[i] if start_time == zoom_start else "")
                                start_time += cycle_duration

                        # Zoom range
                        ax_zoom.set_xlim(150, 250)
                        ax_zoom2.set_xlim(150, 250)
                        # ax_zoom2.set_xlim(100, 160)
                        # ax_zoom2.set_xlim(100, 160)

                        # Set labels and title for zoomed figure
                        ax_zoom.set_xlabel('Time (seconds)')
                        ax_zoom.set_ylabel('SINR')
                        ax_zoom2.set_ylabel('Angular Speed', color='purple')
                        ax_zoom.set_title(f'Zoomed SINR and Angular Speed (100-160s) for rfh-app-{app_id}, txPower={power}, ScenarioID{scenarioID}, Run={runNumber}')

                        # Legend for zoomed figure
                        handles_zoom1, labels_zoom1 = ax_zoom.get_legend_handles_labels()
                        handles_zoom2, labels_zoom2 = ax_zoom2.get_legend_handles_labels()
                        unique_zoom = [(h, l) for i, (h, l) in enumerate(zip(handles_zoom1 + handles_zoom2, labels_zoom1 + labels_zoom2)) if l not in labels_zoom1[:i] + labels_zoom2[:i]]
                        ax_zoom.legend(*zip(*unique_zoom), loc='upper left')

                        # Adjusting the layout for the zoomed figure
                        fig_zoom.tight_layout()

                        # Save the zoomed figure
                        fig_zoom.savefig(os.path.join(detailed_sinr_folder, f'Zoomed_SINR_angular_speed_evolution_rfh-app-{app_id}_ScenarioID{scenarioID}_txPower{power}_Run{runNumber}.png'), dpi=300, bbox_inches='tight')
                        plt.close(fig_zoom)