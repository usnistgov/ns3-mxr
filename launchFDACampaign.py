import os
from multiprocessing import Pool
from tqdm import tqdm

# Check if the Results folder exists and create it if not
if not os.path.exists("Results"):
    os.mkdir("Results")

scenarioPrefix = "SurgeryFeedbackPos"
scenarioToTest = [11,12,13]
# scenarioToTest = [4]
rclAmEnabled = ['true']
rfhAppId = [1, 2, 3, 4, 5, 6, 7, 610,650,6100,6500]
rfhTraffic = ['true']
txPower = [i for i in range(0, 21, 10)]

# Define the number of RngRun values you want to loop over
RngRunCount = 1  # For example, 10 runs
RngRuns = [i for i in range(1, RngRunCount + 1)]

def start_simulation(params):
    rclAmEnabled, rfhTraffic, rfhAppId, txPower, scenario, rngRun = params
    print(f'Starting simulation with rclAmEnabled={rclAmEnabled}, rfhTraffic={rfhTraffic}, rfhAppId={rfhAppId}, txPower={txPower}, scenario={scenario}, RngRun={rngRun}')
    os.system('./ns3 run "CDFSurgeryNorm --rlcAmEnabled=%s --rfhTraffic=%s --rfhAppId=%d --txPower=%d --scenario=%s --runNumber=%d" > Results/log_%s_%s_%d_%d_%s_%d.txt 2>&1' % (rclAmEnabled, rfhTraffic, rfhAppId, txPower, scenario, rngRun, rclAmEnabled, rfhTraffic, rfhAppId, txPower, scenario, rngRun))

processes = 35

params = []
for r in rclAmEnabled:
    for t in rfhTraffic:
        for rf in rfhAppId:
            for p in txPower:
                for s in scenarioToTest:
                    for rng in RngRuns:
                        scenario = scenarioPrefix + str(s)
                        params.append((r, t, rf, p, scenario, rng))

print(f'Total number of simulations: {len(params)}')

pool = Pool(processes=processes)
for _ in tqdm(pool.imap_unordered(start_simulation, params), total=len(params)):
    pass

pool.close()
pool.join()