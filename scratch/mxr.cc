/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 SIGNET Lab, Department of Information Engineering,
 * University of Padova
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

/**
 * This example shows how to simulate a MXR scenario using full stack simulation using the
 * QdChannelModel.
 * The simulation involves two nodes (1 AP and one STA). The STA is the only node moving.
 * They communicates through a wireless channel at 60 GHz with a bandwidth
 * of about 200 MHz.
 */

#include "ns3/applications-module.h"
#include "ns3/config-store.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/isotropic-antenna-model.h"
#include "ns3/mmwave-helper.h"
#include "ns3/mmwave-net-device.h"
#include "ns3/mmwave-phy.h"
#include "ns3/mmwave-point-to-point-epc-helper.h"
#include "ns3/network-module.h"
#include "ns3/node-container.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/psc-module.h"
#include "ns3/qd-channel-model.h"
#include "ns3/simple-net-device.h"
#include "ns3/three-gpp-spectrum-propagation-loss-model.h"

#include <cfloat>
#include <fstream>

using namespace ns3;
using namespace mmwave;
using namespace psc;

#include <iomanip>


/*
 * Structure to keep track of the transmission time of the packets at the
 * application layer. Used to calculate packet delay.
 */
struct PacketWithRxTimestamp
{
    Ptr<const Packet> p;
    Time txTimestamp;
};

/*
 * Map to store received packets and reception timestamps at the application
 * layer. Used to calculate packet delay at the application layer.
 */
std::map<std::string, PacketWithRxTimestamp> g_rxPacketsForDelayCalc;

/*
 * Extract the geometry of the Phased Antenna Array
 */
void
ExtractRowsCols(const std::string& codebookFile, int& numRows, int& numCols)
{
    std::stringstream ss(codebookFile);
    char x;
    if (ss >> numRows >> x >> numCols >> x && x == '.')
    {
        // Successfully extracted numRows, 'x', numCols, and '.'
        // 'numRows' and 'numCols' now hold the extracted values
    }
    else
    {
        // Extraction failed, handle error accordingly
        std::cerr << "Error: Unable to extract numRows and numCols from codebookFile." << std::endl;
    }
}

uint64_t totalBytesSent; // Store the total number of bytes sent by the AP

/*
 * Callback function: Initialize the structure for the delay whenever a new application packet is generated
 */
void
TxPacketTraceForDelay(const Address& localAddrs,
                      Ptr<const Packet> p,
                      const Address& srcAddrs,
                      const Address& dstAddrs,
                      const SeqTsSizeHeader& seqTsSizeHeader,
                      const uint32_t& bytesSent)
{
    std::ostringstream oss;
    oss << Ipv4Address::ConvertFrom(localAddrs) << "->"
        << InetSocketAddress::ConvertFrom(dstAddrs).GetIpv4() << "(" << seqTsSizeHeader.GetSeq()
        << ")";
    std::string mapKey = oss.str();

    PacketWithRxTimestamp mapValue;
    mapValue.p = p;
    mapValue.txTimestamp = Simulator::Now();
    g_rxPacketsForDelayCalc[mapKey] = mapValue; // Overwrite any existing entry
}

/*
 *  Callback function: Store the delay when a new application packet is received
 */
void
RxPacketTraceForDelay(Ptr<OutputStreamWrapper> stream,
                      Ptr<Node> node,
                      const Address& localAddrs,
                      Ptr<const Packet> p,
                      const Address& srcAddrs,
                      const Address& dstAddrs,
                      const SeqTsSizeHeader& seqTsSizeHeader)
{
    double delay = 0.0;
    std::ostringstream oss;
    oss << InetSocketAddress::ConvertFrom(srcAddrs).GetIpv4() << "->"
        << Ipv4Address::ConvertFrom(localAddrs) << "(" << seqTsSizeHeader.GetSeq() << ")";
    std::string mapKey = oss.str();
    auto it = g_rxPacketsForDelayCalc.find(mapKey);
    if (it == g_rxPacketsForDelayCalc.end())
    {
        // Entry Not Found - Just print it for log purpose
        std::cout << InetSocketAddress::ConvertFrom(srcAddrs).GetIpv4() << "->"
                  << Ipv4Address::ConvertFrom(localAddrs) << "(" << seqTsSizeHeader.GetSeq()
                  << ")\n";
    }
    else
    {
        // Entry found
        auto range = g_rxPacketsForDelayCalc.equal_range(mapKey);
        auto it = range.first;
        delay = Simulator::Now().GetNanoSeconds() - it->second.txTimestamp.GetNanoSeconds();
        g_rxPacketsForDelayCalc.erase(it);
    }
    // Store the delay
    *stream->GetStream() << Simulator::Now().GetSeconds() << "\t" << seqTsSizeHeader.GetSeq()
                         << "\t" << delay << "\t" << g_rxPacketsForDelayCalc.size() << std::endl;
}

/*
 * Used to manage the structure of the files saved depending on the parameters used
 */
#include <filesystem>
#include <sstream>
#include <string>

std::string GenerateFileName(const std::string& scenario,
                             const std::string& resultsFolder,
                             const std::string& prefix,
                             bool rlcAmEnabled,
                             const std::string& distribution,
                             const std::string& txPower,
                             const std::string& runNumber,
                             double beamformingInterval,
                             const std::string& codebookFile)
{
    std::stringstream pathStream;
    // Generate the directory path
    pathStream << resultsFolder << "/" << codebookFile << "/BeamFInterval"
               << static_cast<int>(beamformingInterval) << "/";
    std::string directoryPath = pathStream.str();
    
    // Ensure the directory exists
    std::filesystem::create_directories(directoryPath);
    
    // Append the file name to the path
    std::stringstream fileNameStream;
    fileNameStream << directoryPath << scenario << "_" << prefix
                   << "_"  << distribution
                   << "_"  << "txPower" << txPower 
                   << "_"  << "runNumber" 
                   << runNumber << ".txt";
    return fileNameStream.str();
}

/*
 * Callback function: Used to increment the total number of bytes sent whenever an application packet is transmitted
 */
void
TxPacketTrace( Ptr<const Packet> p)
{
    totalBytesSent += p->GetSize();
}


uint64_t lastTotalRx = 0; //!< The value of the last total received bytes

/*
 * Callback function: Print the throughput observed every second
 */
void
PrintThroughput(Ptr<OutputStreamWrapper> streamWrapper, Ptr<PacketSink> sink)
{
    double timeNow = Simulator::Now().GetSeconds();
    double throughput =
        (sink->GetTotalRx() - lastTotalRx) * (8.0 / 1e6) / (1); // Calculate the throughput
    lastTotalRx = sink->GetTotalRx(); // TR++
    double pdr;
    if (totalBytesSent)
        pdr = static_cast<double>(lastTotalRx) / totalBytesSent;
    else
        pdr = 0;
    std::cout << "Throughput at " << timeNow << " seconds: " << throughput << " Mbps"
              << std::endl; // TR++;

    *streamWrapper->GetStream() << timeNow << "," << throughput << "," << throughput << "," << pdr
                                << std::endl;
    Simulator::Schedule(MilliSeconds(1000), &PrintThroughput, streamWrapper, sink);
}

/*
 * Callback function: Log PHY stats whenever a packet is received at the PHY
 */
void
LogReceivedAtUe(Ptr<OutputStreamWrapper> streamWrapper, RxPacketTraceParams params)
{
    *streamWrapper->GetStream() << Simulator::Now().GetSeconds() << ","
                                << static_cast<unsigned int>(params.m_mcs) << ","
                                << static_cast<unsigned int>(params.m_rv) << "," << params.m_tbler
                                << "," << params.m_corrupt << ","
                                << static_cast<unsigned int>(params.m_numSym) << ","
                                << static_cast<unsigned int>(params.m_tbSize) << std::endl;
}

/*
 * Callback function: Log and print beamforming results whenever a beamforming event is complete
 */
void
LogBeamforming(Ptr<OutputStreamWrapper> streamWrapper,
               uint32_t initiator,
               uint32_t responder,
               uint32_t cbTx,
               uint32_t cbRx)
{
    *streamWrapper->GetStream() << Simulator::Now().GetSeconds() << "," << +initiator << ","
                                << +responder << "," << +cbTx << "," << +cbRx << std::endl;
}

/*
 * Callback function: Log the average Signal-to-Noise Ratio (SNR) of received packets.
 * This function is invoked in response to the "ReportCurrentCellRsrpSinr" trace source, 
 * which reports the current Reference Signal Received Power (RSRP) and Signal-to-Interference-plus-Noise Ratio (SINR)
 * for the cell. The function calculates the average SINR across all subcarriers of the received signal
 * and logs it to the specified output stream, but only if at least 100ms have passed since the last log entry.
 * This ensures that the log is updated at a maximum frequency of 10Hz to prevent excessive logging.
 */
void
LogReceivedPacketSNR(Ptr<OutputStreamWrapper> streamWrapper,
                     uint64_t t,
                     SpectrumValue& sinr,
                     SpectrumValue& y)
{
    // Get the current simulation time
    double currentTime = ns3::Simulator::Now().GetSeconds();

    // Static variable to store the last logged time
    static double lastLoggedTime = 0.0;

    // Check if at least 100ms have passed since the last logged time
    if (currentTime - lastLoggedTime >= 0.1)
    {
        // Update the last logged time
        lastLoggedTime = currentTime;

        // Get the SpectrumModel pointer
        Ptr<const ns3::SpectrumModel> spectrumModel = sinr.GetSpectrumModel();

        // Calculate the average SINR in linear scale
        double sinrAvgLin = 0.0;
        uint32_t numSubcarriers = spectrumModel->GetNumBands();
        for (uint32_t i = 0; i < numSubcarriers; ++i)
        {
            sinrAvgLin += sinr[i] / numSubcarriers;
        }

        // Convert the average SINR to dB
        double sinrAvgDb = 10 * std::log10(sinrAvgLin);
        *streamWrapper->GetStream() << currentTime << "," << sinrAvgDb << std::endl;
    }
}

int
main(int argc, char* argv[])
{
    std::string qdFilesPath =
        "contrib/qd-channel/model/QD/"; // The path of the folder with the QD scenarios
    std::string scenario = "MXRPosition1";   // The name of the scenario
    std::string resultsFolder = "ResultsMXR";; // Folder where the simulations results are saved
    double txPower = 10.0; // Transmitted power for both eNB and UE [dBm]
    double noiseFigure = 12.0;   // Noise figure for both eNB and UE [dB]
    unsigned int simTime = 310;    // simulation time [s]
    unsigned int appStart = 1;  // application start time [s]
    unsigned int appEnd = simTime - 10; // application end time [s]
    bool rfhTraffic = true; // Is RFH MXR Traffic enabled
    bool rlcAmEnabled = true; // Enable RFC Automatic
    bool harqEnabled = true; // Enable ARQ
    std::string rfhPrefix = "rfh-app-"; // The prefix used for the RFH app traffic name
    std::string rfhAppId = "1"; // The ID for the RFH app 
    uint32_t boostLength = 0; // Boost parameters: We don't use boost in our case
    double boostPercentile = 90; // Boost not used so non-relevant
    double beamformingInterval = 1; // How often is performed the beamforming [s]
    std::string codebookPath = "src/mmwave/model/Codebooks/"; // Folder where are located the codebooks
    std::string codebookFile = "1x16.txt"; // Name of the codebook
    uint16_t seedNumber = 1;
    uint16_t runNumber = 1;

    // Initialization of command-line arguments for simulation configuration
    CommandLine cmd;
    cmd.AddValue("qdFilesPath", "The path of the folder with the QD scenarios", qdFilesPath);
    cmd.AddValue("scenario", "The name of the scenario", scenario);
    cmd.AddValue("txPower", "Transmitted power for both eNB and UE [dBm]", txPower);
    cmd.AddValue("noiseFigure", "Noise figure for both eNB and UE [dB]", noiseFigure);
    cmd.AddValue("appStart", "application start time [s]", appStart);
    cmd.AddValue("appEnd", "application end time [s]", appEnd);
    cmd.AddValue("resultsFolder", "Where to save the output traces", resultsFolder);
    cmd.AddValue("rfhTraffic", "Use RFH Traffic or Not", rfhTraffic);
    cmd.AddValue("rfhAppId", "RFH Traffic ID", rfhAppId);
    cmd.AddValue("rlcAmEnabled", "RFH Traffic ID", rlcAmEnabled);
    cmd.AddValue("runNumber", "Run Number", runNumber);
    cmd.AddValue("beamformingInterval", "Beamforming Interval", beamformingInterval);
    cmd.AddValue("codebookFile", "The file name of the codebook to use", codebookFile);

    cmd.Parse(argc, argv);
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(1) << txPower;
    std::string txPowerStr = stream.str();
    std::string distribution;
    if (rfhTraffic)
        distribution = rfhPrefix + rfhAppId;
    RngSeedManager::SetSeed(seedNumber);
    RngSeedManager::SetRun(runNumber);

    // Configuration of simulation parameters based on input
    Config::SetDefault("ns3::MmWaveBeamformingModel::UpdatePeriod",
                       TimeValue(MilliSeconds(beamformingInterval * 1000)));

    Config::SetDefault("ns3::MmWaveHelper::RlcAmEnabled", BooleanValue(rlcAmEnabled));
    Config::SetDefault("ns3::MmWaveHelper::HarqEnabled", BooleanValue(harqEnabled));
    Config::SetDefault("ns3::MmWaveFlexTtiMacScheduler::HarqEnabled", BooleanValue(harqEnabled));
    Config::SetDefault("ns3::MmWaveFlexTtiMacScheduler::CqiTimerThreshold", UintegerValue(1000));
    Config::SetDefault("ns3::MmWaveHelper::UseIdealRrc", BooleanValue(false));

    // Node creation for UE (User Equipment) and eNB (Evolved Node B, base station)
    NodeContainer ueNodes;
    NodeContainer enbNodes;
    enbNodes.Create(1);
    ueNodes.Create(1);

    // initial positions of the nodes in the ray tracer
    Ptr<MobilityModel> ueRefMob = CreateObject<ConstantPositionMobilityModel>();
    ueRefMob->SetPosition(Vector(5, 0.1, 1.5));
    Ptr<MobilityModel> enb1Mob = CreateObject<ConstantPositionMobilityModel>();
    enb1Mob->SetPosition(Vector(5, 0.1, 2.9));

    // Aggregate mobility models to the nodes
    enbNodes.Get(0)->AggregateObject(enb1Mob);
    ueNodes.Get(0)->AggregateObject(ueRefMob);

    // Configure the channel
    Config::SetDefault("ns3::MmWaveHelper::PathlossModel", StringValue(""));
    Config::SetDefault("ns3::MmWaveHelper::ChannelModel",
                       StringValue("ns3::ThreeGppSpectrumPropagationLossModel"));
    Ptr<QdChannelModel> qdModel = CreateObject<QdChannelModel>(qdFilesPath, scenario);
    Config::SetDefault("ns3::ThreeGppSpectrumPropagationLossModel::ChannelModel",
                       PointerValue(qdModel));

    // Set power and noise figure
    Config::SetDefault("ns3::MmWavePhyMacCommon::Bandwidth", DoubleValue(200e6));
    Config::SetDefault("ns3::MmWaveEnbPhy::TxPower", DoubleValue(txPower));
    Config::SetDefault("ns3::MmWaveEnbPhy::NoiseFigure", DoubleValue(noiseFigure));
    Config::SetDefault("ns3::MmWaveUePhy::TxPower", DoubleValue(txPower));
    Config::SetDefault("ns3::MmWaveUePhy::NoiseFigure", DoubleValue(noiseFigure));

    // Setup antenna configuration
    Config::SetDefault("ns3::PhasedArrayModel::AntennaElement",
                       PointerValue(CreateObject<IsotropicAntennaModel>()));

    // Create the MmWave helper
    Ptr<MmWaveHelper> mmwaveHelper = CreateObject<MmWaveHelper>();

    // select the beamforming model
    mmwaveHelper->SetBeamformingModelType("ns3::MmWaveCodebookBeamforming");

    // configure the UE antennas:
    // 1. specify the path of the file containing the codebook
    mmwaveHelper->SetUeBeamformingCodebookAttribute("CodebookFilename",
                                                    StringValue(codebookPath + codebookFile));

    // 2. Get and set the antenna dimensions
    int numRows, numCols;
    ExtractRowsCols(codebookFile, numRows, numCols);

    mmwaveHelper->SetUePhasedArrayModelAttribute("NumRows", UintegerValue(numRows));
    mmwaveHelper->SetUePhasedArrayModelAttribute("NumColumns", UintegerValue(numCols));

    // configure the BS antennas:
    // 1. specify the path of the file containing the codebook
    mmwaveHelper->SetEnbBeamformingCodebookAttribute("CodebookFilename",
                                                     StringValue(codebookPath + codebookFile));
    mmwaveHelper->SetEnbPhasedArrayModelAttribute("NumRows", UintegerValue(numRows));
    // 2. Get and set the antenna dimensions
    mmwaveHelper->SetEnbPhasedArrayModelAttribute("NumColumns", UintegerValue(numCols));

    mmwaveHelper->SetSchedulerType("ns3::MmWaveFlexTtiMacScheduler");
    Ptr<MmWavePointToPointEpcHelper> epcHelper = CreateObject<MmWavePointToPointEpcHelper>();
    mmwaveHelper->SetEpcHelper(epcHelper);
    mmwaveHelper->SetHarqEnabled(harqEnabled);

    // Create a single RemoteHost
    Ptr<Node> pgw = epcHelper->GetPgwNode();
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);
    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    // Create the Internet
    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(1500));
    p2ph.SetChannelAttribute("Delay", TimeValue(Seconds(0.00)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);
    // Interface 0 is localhost, 1 is the p2p device
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    // Create the tx and rx devices
    NetDeviceContainer enbMmWaveDevs = mmwaveHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueMmWaveDevs = mmwaveHelper->InstallUeDevice(ueNodes);

    // Install the IP stack on the UEs
    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIface;
    ueIpIface = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueMmWaveDevs));
    // Assign IP address to UEs, and install applications
    Ptr<Node> ueNode = ueNodes.Get(0);
    // Set the default gateway for the UE
    Ptr<Ipv4StaticRouting> ueStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
    ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);

    // This performs the attachment of each UE to a specific eNB
    mmwaveHelper->AttachToEnbWithIndex(ueMmWaveDevs.Get(0), enbMmWaveDevs, 0);

    // Create the application from the RH to the UE using MXR RFH App
    Ptr<Ipv4> ueIpv4 = ueNodes.Get(0)->GetObject<Ipv4>();
    Ipv4InterfaceAddress ueIpAddr = ueIpv4->GetAddress(1, 0);

    ApplicationContainer apps;
    Config::SetDefault("ns3::PacketSink::EnableSeqTsSizeHeader", BooleanValue(true));

    Ptr<PscVideoStreaming> streamingServer = CreateObject<PscVideoStreaming>();
    streamingServer->SetAttribute("ReceiverAddress", AddressValue(ueIpAddr.GetLocal()));
    streamingServer->SetAttribute("ReceiverPort", UintegerValue(5554));
    std::string packetSizeFilePath = "contrib/psc/examples/mxr_cdf/cdfPacketSize-" + distribution + ".txt";
    std::string interarrivalFilePath = "contrib/psc/examples/mxr_cdf/cdfInterarrival-" + distribution + ".txt";
    streamingServer->ReadCustomDistribution(packetSizeFilePath, interarrivalFilePath);
    streamingServer->SetAttribute("BoostLengthPacketCount", UintegerValue(boostLength));
    streamingServer->SetAttribute("BoostPercentile", DoubleValue(boostPercentile));

    apps.Add(streamingServer);
    remoteHostContainer.Get(0)->AddApplication(streamingServer);
    PacketSinkHelper packetSinkHelper("ns3::UdpSocketFactory",
                                      InetSocketAddress(Ipv4Address::GetAny(), 5554));
    apps.Add(packetSinkHelper.Install(ueNodes.Get(0)));

    apps.Start(Seconds(appStart));
    apps.Stop(Seconds(appEnd));

    // Callbacks
    Ptr<NetDevice> ueNetDevice = ueMmWaveDevs.Get(0);
    Ptr<mmwave::MmWaveUeNetDevice> mmWaveUeNetDevice =
        DynamicCast<mmwave::MmWaveUeNetDevice>(ueNetDevice);

    // Get the MmWaveUePhy instance and connect the custom callback function
    Ptr<mmwave::MmWaveUePhy> mmWaveUePhy = mmWaveUeNetDevice->GetPhy();
    
    // Callback for SNR Reporting 
    AsciiTraceHelper asciiTraceHelper;
    Ptr<OutputStreamWrapper> snrStreamWrapper =
        asciiTraceHelper.CreateFileStream(GenerateFileName(scenario,
                                                           resultsFolder,
                                                           "snr",
                                                           rlcAmEnabled,
                                                           distribution,
                                                           txPowerStr,
                                                           std::to_string(runNumber),
                                                           beamformingInterval,
                                                           codebookFile));
    mmWaveUePhy->TraceConnectWithoutContext(
        "ReportCurrentCellRsrpSinr",
        MakeBoundCallback(&LogReceivedPacketSNR, snrStreamWrapper));

    // Callback for PHY stats at the UE Reporting 
    Ptr<OutputStreamWrapper> ueStats =
        asciiTraceHelper.CreateFileStream(GenerateFileName(scenario,
                                                           resultsFolder,
                                                           "UE",
                                                           rlcAmEnabled,
                                                           distribution,
                                                           txPowerStr,
                                                           std::to_string(runNumber),
                                                           beamformingInterval,
                                                           codebookFile));
    Ptr<MmWaveSpectrumPhy> ueSpectrumPhy = mmWaveUePhy->GetDlSpectrumPhy();
    Ptr<MmWaveBeamformingModel> ueBeamformingModel = ueSpectrumPhy->GetBeamformingModel();
    if (ueSpectrumPhy != nullptr)
    {
        ueSpectrumPhy->TraceConnectWithoutContext("RxPacketTraceUe",
                                                  MakeBoundCallback(&LogReceivedAtUe, ueStats));
    }
    else
    {
        std::cout << "Failed to get MmWaveSpectrumPhy from mmWaveUePhy";
    }
    *ueStats->GetStream() << "TIME,MCS,RV,BLER,Corrupt,NUMSYM,TBSIZE" << std::endl;

    // Callback for beamforming reporting
    Ptr<OutputStreamWrapper> beamformingStats =
        asciiTraceHelper.CreateFileStream(GenerateFileName(scenario,
                                                           resultsFolder,
                                                           "Beamforming",
                                                           rlcAmEnabled,
                                                           distribution,
                                                           txPowerStr,
                                                           std::to_string(runNumber),
                                                           beamformingInterval,
                                                           codebookFile));
    ueBeamformingModel->TraceConnectWithoutContext(
        "BeamformingTrace",
        MakeBoundCallback(&LogBeamforming, beamformingStats));


    
    // Callback for throughput reporting 
    Ptr<Application> appStream = apps.Get(1);
    Ptr<PacketSink> sink = appStream->GetObject<PacketSink>();
    AsciiTraceHelper asciiTraceHelperThroughput;
    Ptr<OutputStreamWrapper> throughputStreamWrapper =
        asciiTraceHelper.CreateFileStream(GenerateFileName(scenario,
                                                           resultsFolder,
                                                           "throughput",
                                                           rlcAmEnabled,
                                                           distribution,
                                                           txPowerStr,
                                                           std::to_string(runNumber),
                                                           beamformingInterval,
                                                           codebookFile));
    Simulator::Schedule(MilliSeconds(1000), &PrintThroughput, throughputStreamWrapper, sink);
    // Callback used for PDR
    apps.Get(0)->TraceConnectWithoutContext("Tx", MakeBoundCallback(&TxPacketTrace));
    
    // Callback for delay reporting 
    Ipv4Address localAddrs =
        apps.Get(0)->GetNode()->GetObject<Ipv4L3Protocol>()->GetAddress(1, 0).GetLocal();
    apps.Get(0)->TraceConnectWithoutContext("TxWithSeqTsSize",                                     MakeBoundCallback(&TxPacketTraceForDelay, localAddrs));
    Ptr<OutputStreamWrapper> delayTraceStream =
        asciiTraceHelper.CreateFileStream((GenerateFileName(scenario,
                                                 resultsFolder,
                                                 "delay",
                                                 rlcAmEnabled,
                                                 distribution,
                                                 txPowerStr,
                                                 std::to_string(runNumber),
                                                 beamformingInterval,
                                                 codebookFile)));
    *delayTraceStream->GetStream() << "time(s)\tseqNum\tdelay(ms)\tQUEUESIZE" << std::endl;
    Ipv4Address localAddrSink =
        apps.Get(1)->GetNode()->GetObject<Ipv4L3Protocol>()->GetAddress(1, 0).GetLocal();
    apps.Get(1)->TraceConnectWithoutContext("RxWithSeqTsSize",
                                            MakeBoundCallback(&RxPacketTraceForDelay,
                                                              delayTraceStream,
                                                              apps.Get(1)->GetNode(),
                                                              localAddrSink));


    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
