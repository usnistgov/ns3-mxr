import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_delay_data(file_path):
    try:
        # Load delay data
        delay_data = pd.read_csv(file_path, sep='\s+', header=0)
        
        # Convert delay from microseconds to milliseconds
        delay_data['delay(ms)'] = delay_data['delay(ms)'] / 1000.0
        
        return delay_data['delay(ms)']
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None

# Specify the file paths
file_paths = [
    "Results/SurgeryFeedbackPos10_delay_rlcAmEnabledrfh-app-6100txPower0.0runNumber1.txt",
    "Results/SurgeryFeedbackPos10_delay_rlcAmEnabledrfh-app-6100txPower10.0runNumber1.txt"
]

# Load delay data
delays = []
labels = []
for file_path in file_paths:
    delay_data = load_delay_data(file_path)
    if delay_data is not None:
        delays.append(delay_data)
        labels.append(f"TxPower={file_path.split('txPower')[1].split('runNumber')[0]}")

# Create a CDF plot for the delay values
plt.figure(figsize=(10, 6))
for delay_data, label in zip(delays, labels):
    sorted_data = np.sort(delay_data)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data, yvals, label=label)

plt.xlabel('Delay (ms)')
plt.ylabel('Cumulative Probability')
plt.title('CDF of Delay Values')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('Results/delay_distribution_cdf.png')
plt.show()
