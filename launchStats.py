import os
from multiprocessing import Pool
from tqdm import tqdm

runProcess = True  # Set to True to enable launching the simulations

scenarioPrefix = "SurgeryFeedbackPos"
# scenarioToTest =list(range(1, 14))
scenarioToTest = [1,2,3,4,5,6,7,8,9,10]
rclAmEnabled = ['true']
rfhAppId = [1,2,3,4,5,6,7,610,650,6100,6250]
# rfhAppId = [1,4,6,650]
rfhTraffic = ['true']
# txPower = [i for i in range(0, 21, 10)]
txPower = [0.0,10.0,20.0]
# beamformingIntervals = [1]
beamformingIntervals = [1,2,5,10]
# beamformingIntervals = [1]
codebookFiles = ["1x16.txt","2x16.txt"]
# codebookFiles = ["1x16.txt"]


RngRunCount = 10  # There are 8 runs from 3 to 10
RngRuns = [i for i in range(1, RngRunCount + 1)]  # This will generate a list from 1 to 10

processes = 100

def create_folder_structure(codebook_file, beamforming_interval):
    folder_name = f"BeamFInterval{beamforming_interval}"
    folder_path = os.path.join("ResultsWithTraces", codebook_file, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def start_simulation(params):
    rclAmEnabled, rfhTraffic, rfhAppId, txPower, scenario, rngRun, codebookFile, beamformingInterval = params
    folder_name = f"BeamFInterval{beamformingInterval}"
    folder_path = os.path.join("ResultsWithTraces", codebookFile, folder_name)
    output_file = os.path.join(folder_path, f'log_{rclAmEnabled}_{rfhTraffic}_{rfhAppId}_{txPower}_{scenario}_{rngRun}.txt')
    print(f'Starting simulation with rclAmEnabled={rclAmEnabled}, rfhTraffic={rfhTraffic}, rfhAppId={rfhAppId}, txPower={txPower}, scenario={scenario}, RngRun={rngRun}, codebookFile={codebookFile}, beamformingInterval={beamformingInterval}')
    os.system(f'./ns3 run "CDFSurgeryStats --rlcAmEnabled={rclAmEnabled} --rfhTraffic={rfhTraffic} --rfhAppId={rfhAppId} --txPower={txPower} --scenario={scenario} --runNumber={rngRun} --codebookFile={codebookFile} --beamformingInterval={beamformingInterval}" > {output_file} 2>&1')

# Create folder structure before starting simulations
for cbf in codebookFiles:
    for bi in beamformingIntervals:
        create_folder_structure(cbf, bi)

params = []
for r in rclAmEnabled:
    for t in rfhTraffic:
        for rf in rfhAppId:
            for p in txPower:
                for s in scenarioToTest:
                    for rng in RngRuns:
                        for cbf in codebookFiles:
                            for bi in beamformingIntervals:
                                scenario = scenarioPrefix + str(s)
                                params.append((r, t, rf, p, scenario, rng, cbf, bi))

print(f'Total number of simulations: {len(params)}')

if runProcess:
    pool = Pool(processes=processes)
    for _ in tqdm(pool.imap_unordered(start_simulation, params), total=len(params)):
        pass
    pool.close()
    pool.join()
else:
    print("Simulation process is disabled. Set runProcess to True to enable it.")
