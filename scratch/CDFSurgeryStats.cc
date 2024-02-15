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
 * This example shows how to configure a full stack simulation using the
 * QdChannelModel.
 * The simulation involves two nodes moving in an empty rectangular room
 * and communicates through a wireless channel at 60 GHz with a bandwidth
 * of about 400 MHz.
 */

// #include "simulation-config/simulation-config.h"
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

void
PrintIpAddresses(Ptr<Node> node)
{
    Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
    for (uint32_t i = 0; i < ipv4->GetNInterfaces(); ++i)
    {
        for (uint32_t j = 0; j < ipv4->GetNAddresses(i); ++j)
        {
            Ipv4InterfaceAddress ifaceAddr = ipv4->GetAddress(i, j);
            std::cout << "Device " << node->GetId() << " Interface " << i
                      << " IP Address: " << ifaceAddr.GetLocal() << std::endl;
        }
    }
}
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
// std::map<std::string, PacketWithRxTimestamp> g_rxPacketsForDelayCalc;
// std::multimap<std::string, PacketWithRxTimestamp> g_rxPacketsForDelayCalc;
std::map <std::string, PacketWithRxTimestamp> g_rxPacketsForDelayCalc;
void ExtractRowsCols(const std::string& codebookFile, int& numRows, int& numCols) {
    std::stringstream ss(codebookFile);
    char x;
    if (ss >> numRows >> x >> numCols >> x && x == '.') {
        // Successfully extracted numRows, 'x', numCols, and '.'
        // 'numRows' and 'numCols' now hold the extracted values
    } else {
        // Extraction failed, handle error accordingly
        std::cerr << "Error: Unable to extract numRows and numCols from codebookFile." << std::endl;
    }
}


uint64_t totalBytesSent;
void
TxPacketTraceForDelay (const Address &localAddrs, Ptr<const Packet> p, const Address &srcAddrs,
                       const Address &dstAddrs, const SeqTsSizeHeader &seqTsSizeHeader, const uint32_t &bytesSent)
{
  std::ostringstream  oss;
  oss << Ipv4Address::ConvertFrom (localAddrs)
      << "->"
      << InetSocketAddress::ConvertFrom (dstAddrs).GetIpv4 ()
      << "("
      << seqTsSizeHeader.GetSeq ()
      << ")";
  std::string mapKey = oss.str ();

//   PacketWithRxTimestamp mapValue;
//   mapValue.p = p;
//   mapValue.txTimestamp = Simulator::Now ();
//   g_rxPacketsForDelayCalc.emplace(mapKey, mapValue);
    PacketWithRxTimestamp mapValue;
    mapValue.p = p;
    mapValue.txTimestamp = Simulator::Now ();
    g_rxPacketsForDelayCalc[mapKey] = mapValue; // Overwrite any existing entry

}


void
RxPacketTraceForDelay (Ptr<OutputStreamWrapper> stream, Ptr<Node> node, const Address &localAddrs,
                       Ptr<const Packet> p, const Address &srcAddrs,
                       const Address &dstAddrs, const SeqTsSizeHeader &seqTsSizeHeader)
{




  double delay = 0.0;
  std::ostringstream  oss;
  oss << InetSocketAddress::ConvertFrom (srcAddrs).GetIpv4 ()
      << "->"
      << Ipv4Address::ConvertFrom (localAddrs)
      << "("
      << seqTsSizeHeader.GetSeq ()
      << ")";
  std::string mapKey = oss.str ();
//     auto range = g_rxPacketsForDelayCalc.equal_range(mapKey);
//   if (range.first == range.second) {
//     std::cout << "Not FOUND\n";
//     std::cout << InetSocketAddress::ConvertFrom(srcAddrs).GetIpv4()
//               << "->"
//               << Ipv4Address::ConvertFrom(localAddrs)
//               << "("
//               << seqTsSizeHeader.GetSeq()
//               << ")\n";
//     std::cout << "END Not FOUND\n";
    
//     NS_FATAL_ERROR("Rx packet not found?!"); // Uncomment if needed
//   } else {
//     auto it = range.first;
//     delay = Simulator::Now().GetNanoSeconds() - it->second.txTimestamp.GetNanoSeconds();
//     g_rxPacketsForDelayCalc.erase(it);
//   }


   auto it = g_rxPacketsForDelayCalc.find(mapKey);
    if (it == g_rxPacketsForDelayCalc.end()) {
        std::cout << "Not FOUND\n";
    std::cout << InetSocketAddress::ConvertFrom(srcAddrs).GetIpv4()
              << "->"
              << Ipv4Address::ConvertFrom(localAddrs)
              << "("
              << seqTsSizeHeader.GetSeq()
              << ")\n";
    std::cout << "END Not FOUND\n";
    } else {
         auto range = g_rxPacketsForDelayCalc.equal_range(mapKey);
         auto it = range.first;
    delay = Simulator::Now().GetNanoSeconds() - it->second.txTimestamp.GetNanoSeconds();
    g_rxPacketsForDelayCalc.erase(it);
    }

    // std::cout << "DELAY: " << Simulator::Now ().GetSeconds ()
    //                     << "\t" << node->GetId ()
    //                     << "\t" << InetSocketAddress::ConvertFrom (srcAddrs).GetIpv4 ()
    //                     << "\t" << Ipv4Address::ConvertFrom (localAddrs)
    //                     << "\t" << seqTsSizeHeader.GetSeq ()
    //                     << "\t" << delay
    //                     << std::endl;  


  *stream->GetStream () << Simulator::Now ().GetSeconds ()
                        << "\t" << seqTsSizeHeader.GetSeq ()
                        << "\t" << delay
                        << "\t" << g_rxPacketsForDelayCalc.size()
                        << std::endl;  
}

// void TimeoutOldEntries(double timeoutSeconds, Ptr<OutputStreamWrapper> stream) {
//     auto it = g_rxPacketsForDelayCalc.begin();
//     while (it != g_rxPacketsForDelayCalc.end()) {
//         double packetAge = Simulator::Now().GetSeconds() - it->second.txTimestamp.GetSeconds();
//         if (packetAge > timeoutSeconds) {
//             // Parse the key as before
//             std::string mapKey = it->first;

//             // Erase all entries with this key that have timed out
//             auto range = g_rxPacketsForDelayCalc.equal_range(mapKey);
//             for (auto itRange = range.first; itRange != range.second; ) {
//                 if (Simulator::Now().GetSeconds() - itRange->second.txTimestamp.GetSeconds() > timeoutSeconds) {
//                     // Log and erase as before, using itRange
//                     std::cout << Simulator::Now().GetSeconds () << "Delete\n";
//                     itRange = g_rxPacketsForDelayCalc.erase(itRange);
//                 } else {
//                     ++itRange;
//                 }
//             }

//             // After erasing, the initial iterator is invalid; start from the beginning again
//             it = g_rxPacketsForDelayCalc.begin();
//         } else {
//             ++it; // Only increment if we didn't erase
//         }
//     }
//     // Reschedule the function
//     Simulator::Schedule(Seconds(1), &TimeoutOldEntries, timeoutSeconds, stream);
// }

// void TimeoutOldEntries(double timeoutSeconds,Ptr<OutputStreamWrapper> stream) {
//     auto it = g_rxPacketsForDelayCalc.begin();
//     while (it != g_rxPacketsForDelayCalc.end()) {
//         double packetAge = Simulator::Now().GetSeconds() - it->second.txTimestamp.GetSeconds();
//         if (packetAge > timeoutSeconds) {
//             *stream->GetStream () << Simulator::Now().GetSeconds()
//                         << "\t" << "0"
//                         << "\t" << "0"
//                         << "\t" << "0"
//                         << "\t" << "0"
//                         << "\t" << "Timeout" // Delay on timeout is not meaningful
//                         << "\t" << g_rxPacketsForDelayCalc.size()
//                         << std::endl;
//             it = g_rxPacketsForDelayCalc.erase(it); // Erase returns the next iterator
//         } else {
//             ++it; // Only increment if we didn't erase
//         }
//     }
//     // Schedule this function to run again after one second
//     Simulator::Schedule(Seconds(1), &TimeoutOldEntries, timeoutSeconds,stream);
// }


uint64_t lastTotalRx = 0; //!< The value of the last total received bytes

// std::string
// GenerateFileName(const std::string& prefix, bool blockage, const std::string& distribution,const std::string& txPower )
// {
//     return prefix + (blockage ? "Blockage" : "NoBlockage") + distribution + "txPower" + txPower + ".txt";
// }

// std::string GenerateFileName(const std::string& scenario, const std::string& prefix, bool rlcAmEnabled, const std::string& distribution, const std::string& txPower,const std::string& runNumber)
// {
//     return "Results/" + scenario + "_" + prefix + "_" + (rlcAmEnabled ? "rlcAmEnabled" : "rlcAmDisabled") + distribution + "txPower" + txPower + "runNumber" + runNumber + ".txt";
// }

std::string GenerateFileName(const std::string& scenario, const std::string& prefix, bool rlcAmEnabled, 
                             const std::string& distribution, const std::string& txPower, const std::string& runNumber,
                             double beamformingInterval, const std::string& codebookFile)
{
    std::stringstream fileNameStream;
    fileNameStream << "ResultsWithTraces/" << codebookFile << "/BeamFInterval" << static_cast<int>(beamformingInterval) << "/" 
                   << scenario << "_" << prefix << "_" << (rlcAmEnabled ? "rlcAmEnabled" : "rlcAmDisabled") 
                   << distribution << "txPower" << txPower << "runNumber" << runNumber << ".txt";
    return fileNameStream.str();
}

void
TxPacketTrace(std::string context, Ptr<const Packet> p)
{
    // std::ostringstream oss;
    // *stream->GetStream() << Simulator::Now().GetNanoSeconds() << "\t" << context << "\t"
    //                      << p->GetSize() << std::endl;
                         totalBytesSent+=p->GetSize();
                        //   std::cout << "bytes2:" << totalBytesSent << std::endl;

}

void
RxPacketTrace(
              std::string context,
              Ptr<const Packet> p,
              const Address& srcAddrs)
{
    // std::ostringstream oss;
    // std::cout << "RX Callback" << std::endl;
    // *stream->GetStream() << Simulator::Now().GetSeconds() << "\t" << context << "\t" << p->GetSize()
    //                      << std::endl;
}

// Function to print the throughput
void
PrintThroughput(Ptr<OutputStreamWrapper> streamWrapper, Ptr<PacketSink> sink)
{
    double timeNow = Simulator::Now().GetSeconds();
    std::cout << "\tTotal Rx " << sink->GetTotalRx() << "\n";
    std::cout << "\tLast Rx " << lastTotalRx << "\n";
    std::cout << "\tTotal Rx interval " << sink->GetTotalRx() << "\n";
    std::cout << "\tInterval " << timeNow - 0.1 << "\n";
    double throughput =
        (sink->GetTotalRx() * 8.0 / 1e6) / (timeNow - 0.1); // Calculate the throughput
    double throughput2 =
        (sink->GetTotalRx() - lastTotalRx) * (8.0 / 1e6) / (1); // Calculate the throughput
    // std::cout << "Total Rx " << sink->GetTotalRx()-lastTotalRx << " Mbps\n";
    // std::cout << "Throughput at " << timeNow << " seconds: " << throughput << " Mbps\n";
    // std::cout << "bytes sent:" << totalBytesSent << std::endl;
    // std::cout << "bytes received:" << sink->GetTotalRx() << std::endl;
    lastTotalRx = sink->GetTotalRx(); // TR++
    double pdr;
    if (totalBytesSent)
        pdr = static_cast<double>(lastTotalRx) / totalBytesSent;
    else
        pdr=0;
     std::cout << "Throughput2 at " << timeNow << " seconds: " << throughput2 << " Mbps" << "PDR:" << pdr << std::endl; // TR++;

    *streamWrapper->GetStream() << timeNow << "," << throughput2 << "," << throughput <<  "," << pdr << std::endl;
    // lastTotalRx = sink->GetTotalRx(); // TR++
    Simulator::Schedule(MilliSeconds(1000), &PrintThroughput, streamWrapper, sink);
}

void LogReceivedAtUe(Ptr<OutputStreamWrapper> streamWrapper, RxPacketTraceParams params) {
    
    *streamWrapper->GetStream() << Simulator::Now().GetSeconds() << "," << static_cast<unsigned int>(params.m_mcs) << ","
                       << static_cast<unsigned int>(params.m_rv) << ","
                       << params.m_tbler << ","
                       << params.m_corrupt << ","
                       << static_cast<unsigned int>(params.m_numSym) << ","
                       << static_cast<unsigned int>(params.m_tbSize) 
                       << std::endl;

                    //    std::exit(1); // Quit program with a status code of 1
}

void LogBeamforming(Ptr<OutputStreamWrapper> streamWrapper, uint32_t initiator,uint32_t responder, uint32_t cbTx,uint32_t cbRx) {
    
    std::cout << "Beamforming" << +initiator << "," << +responder << "," <<  +cbTx << "," << +cbRx <<  std::endl;       
     *streamWrapper->GetStream() << Simulator::Now().GetSeconds() << "," << +initiator << "," << +responder << "," <<  +cbTx << "," << +cbRx <<  std::endl;         
}




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
        // Log the average SINR value (in dB)
        // std::cout << ns3::Simulator::Now().GetSeconds()<< " Average SINR: " << sinrAvgDb << " dB"
        // << std::endl;
    }
}

int
main(int argc, char* argv[])
{
    std::string qdFilesPath =
        "contrib/qd-channel/model/QD/"; // The path of the folder with the QD scenarios
    // std::string scenario = "FDA"; // The name of the scenario
    // std::string scenario = "FDANewRoom"; // The name of the scenario

    std::string scenario = "FDANewRoom";   // The name of the scenario
    uint32_t interPacketInterval = 1000e3; // App inter packet arrival [us]
    // uint32_t interPacketInterval = 1000000; // App inter packet arrival [us]
    double txPower = -10.0;       // Transmitted power for both eNB anNist673!d UE [dBm]
    // default 24
    double noiseFigure = 12.0;   // Noise figure for both eNB and UE [dB]
    uint16_t enbAntennaNum = 64; // The number of antenna elements for the gNBs antenna arrays,
                                 // assuming a square architecture
    uint16_t ueAntennaNum = 2; // The number of antenna elements for the UE antenna arrays, assuming
                               // a square architecture
    uint32_t appPacketSize = 1460;     // Application packet size [B]
    unsigned int simTime = 310;         // simulation time [s]
    //  unsigned int simTime = 50;         // simulation time [s]
    unsigned int appEnd = simTime - 10; // application end time [s]
    unsigned int minStart = 300;       // application min start time [ms]
    unsigned int maxStart = 400;       // application max start time [ms]
    std::string filePath = "./";       // where to save the traces
    std::string appRate = "50Mbps";    // application data rate
    // TR++
    // bool blockageFDA = false;
    bool rfhTraffic = true;
    bool rlcAmEnabled = true;

    std::string rfhPrefix = "rfh-app-";
    uint16_t rfhAppId = 1;
    uint16_t seedNumber = 1;
    uint16_t  runNumber = 1;
    // TR==

    // Attributes
    // std::string distribution = "720p-bright";
    std::string distribution = "rfh-app-1";

    // uint32_t    boostLength = 50;
    // double      boostPercentile = 90;
    uint32_t boostLength = 0;
    double boostPercentile = 90;
    double beamformingInterval = 1; // In Second but need to be passed in ms
    std::string codebookPath = "src/mmwave/model/Codebooks/";
    std::string codebookFile = "1x16.txt";
   

    CommandLine cmd;
    cmd.AddValue("qdFilesPath", "The path of the folder with the QD scenarios", qdFilesPath);
    cmd.AddValue("scenario", "The name of the scenario", scenario);
    cmd.AddValue("ipi", "App inter packet arrival [us]", interPacketInterval);
    cmd.AddValue("txPower", "Transmitted power for both eNB and UE [dBm]", txPower);
    cmd.AddValue("noiseFigure", "Noise figure for both eNB and UE [dB]", noiseFigure);
    cmd.AddValue("enbAntennaNum",
                 "The number of antenna elements for the gNBs antenna arrays, assuming a square "
                 "architecture",
                 enbAntennaNum);
    cmd.AddValue(
        "ueAntennaNum",
        "The number of antenna elements for the UE antenna arrays, assuming a square architecture",
        ueAntennaNum);
    cmd.AddValue("appPacketSize", "Application packet size [B]", appPacketSize);
    cmd.AddValue("minStart", "application start time [ms]", minStart);
    cmd.AddValue("maxStart", "application max start time [ms]", maxStart);
    cmd.AddValue("appEnd", "application end time [s]", appEnd);
    cmd.AddValue("filePath", "Where to save the output traces", filePath);
    cmd.AddValue("appRate", "The data-rate for the video applications", appRate);
    // TR++
    // cmd.AddValue("blockageFDA", "Use the blockage scenario or not", blockageFDA);
    cmd.AddValue("rfhTraffic", "Use RFH Traffic or Not", rfhTraffic);
    cmd.AddValue("rfhAppId", "RFH Traffic ID", rfhAppId);
    cmd.AddValue("rfhAppId", "RFH Traffic ID", rfhAppId);
    cmd.AddValue("rlcAmEnabled", "RFH Traffic ID", rlcAmEnabled);
    cmd.AddValue("runNumber", "Run Number", runNumber);
    cmd.AddValue("beamformingInterval", "Beamforming Interval", beamformingInterval);
    cmd.AddValue("codebookFile", "The file name of the codebook to use", codebookFile);


   
    // TR--
    
    cmd.Parse(argc, argv);
    // TR++
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(1) << txPower;
    std::string txPowerStr = stream.str();
    if (rfhTraffic)
        distribution = rfhPrefix + std::to_string(rfhAppId);
    RngSeedManager::SetSeed(seedNumber);  
    RngSeedManager::SetRun(runNumber);  
     Config::SetDefault ("ns3::MmWaveBeamformingModel::UpdatePeriod", TimeValue(MilliSeconds(beamformingInterval*1000)));
    // if (blockageFDA)
    //     scenario = "FDANewRoomBlockage";


    // TR--

    bool harqEnabled = true;
    // bool rlcAmEnabled = false;

    Config::SetDefault("ns3::MmWaveHelper::RlcAmEnabled", BooleanValue(rlcAmEnabled));
    Config::SetDefault("ns3::MmWaveHelper::HarqEnabled", BooleanValue(harqEnabled));
    Config::SetDefault("ns3::MmWaveFlexTtiMacScheduler::HarqEnabled", BooleanValue(harqEnabled));
    Config::SetDefault("ns3::MmWaveHelper::UseIdealRrc", BooleanValue(false));

    // TR ++
    /* In order to fix the MCS */
    // Config::SetDefault("ns3::MmWaveFlexTtiMacScheduler::FixedMcsUl", BooleanValue(true));
    // Config::SetDefault("ns3::MmWaveFlexTtiMacScheduler::McsDefaultUl", UintegerValue(1));
    // Config::SetDefault("ns3::MmWaveFlexTtiMacScheduler::FixedMcsDl", BooleanValue(true));
    // Config::SetDefault("ns3::MmWaveFlexTtiMacScheduler::McsDefaultDl", UintegerValue(1));
    // TR--

    // Create the tx and rx nodes
    NodeContainer ueNodes;
    NodeContainer enbNodes;
    enbNodes.Create(1);
    ueNodes.Create(1);

    // initial positions of the nodes in the ray tracer
    Ptr<MobilityModel> ueRefMob = CreateObject<ConstantPositionMobilityModel>();
    ueRefMob->SetPosition(Vector(5, 0.1, 1.5));
    Ptr<MobilityModel> enb1Mob = CreateObject<ConstantPositionMobilityModel>();
    enb1Mob->SetPosition(Vector(5, 0.1, 2.9));

    enbNodes.Get(0)->AggregateObject(enb1Mob);
    ueNodes.Get(0)->AggregateObject(ueRefMob);

    // Configure the channel
    Config::SetDefault("ns3::MmWaveHelper::PathlossModel", StringValue(""));
    Config::SetDefault("ns3::MmWaveHelper::ChannelModel",
                       StringValue("ns3::ThreeGppSpectrumPropagationLossModel"));
    Ptr<QdChannelModel> qdModel = CreateObject<QdChannelModel>(qdFilesPath, scenario);
    // Time simTime = qdModel->GetQdSimTime ();
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
    mmwaveHelper->SetUeBeamformingCodebookAttribute(
        "CodebookFilename",
        StringValue(codebookPath+codebookFile));
    
        
    // 2. set the antenna dimensions
    int numRows, numCols;
    ExtractRowsCols(codebookFile, numRows, numCols);

    mmwaveHelper->SetUePhasedArrayModelAttribute("NumRows", UintegerValue(numRows));
    mmwaveHelper->SetUePhasedArrayModelAttribute("NumColumns", UintegerValue(numCols));

    // configure the BS antennas:
    // 1. specify the path of the file containing the codebook
    mmwaveHelper->SetEnbBeamformingCodebookAttribute(
        "CodebookFilename",
        StringValue(codebookPath+codebookFile));
    mmwaveHelper->SetEnbPhasedArrayModelAttribute("NumRows", UintegerValue(numRows));
    // 2. set the antenna dimensions
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
    // p2ph.EnablePcapAll("my-pcap-file");
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

    // Add apps
    // uint16_t dlPort = 1234;
    // uint16_t ulPort = 2000;
    // uint16_t otherPort = 3000;
    // ApplicationContainer clientApps;
    // ApplicationContainer serverApps;
    // ++ulPort;
    // ++otherPort;
    // PacketSinkHelper dlPacketSinkHelper("ns3::UdpSocketFactory",
    //                                     InetSocketAddress(Ipv4Address::GetAny(), dlPort));
    // serverApps.Add(dlPacketSinkHelper.Install(ueNodes.Get(0)));

    // UdpClientHelper dlClient(ueIpIface.GetAddress(0), dlPort);
    // dlClient.SetAttribute("Interval", TimeValue(MicroSeconds(interPacketInterval)));
    // dlClient.SetAttribute("MaxPackets", UintegerValue(1000000));
    // dlClient.SetAttribute("PacketSize", UintegerValue(appPacketSize));

    // clientApps.Add(dlClient.Install(remoteHost));

    // serverApps.Start(Seconds(0.01));
    // clientApps.Start(Seconds(0.01));
    // mmwaveHelper->EnableTraces();

    // Get the PacketSink object
    // Ptr<Application> app = serverApps.Get(0);
    // Ptr<PacketSink> sink = app->GetObject<PacketSink>();

    // Assume 'ueNodes' is a NodeContainer holding the UE nodes
    Ptr<NetDevice> ueNetDevice = ueMmWaveDevs.Get(0);
    // Assume 'ueNodes' is a NodeContainer holding the UE nodes

    Ptr<mmwave::MmWaveUeNetDevice> mmWaveUeNetDevice =
        DynamicCast<mmwave::MmWaveUeNetDevice>(ueNetDevice);

    // Get the MmWaveUePhy instance and connect the custom callback function
    Ptr<mmwave::MmWaveUePhy> mmWaveUePhy = mmWaveUeNetDevice->GetPhy();
    // Wrap the file in an OutputStreamWrapper

    AsciiTraceHelper asciiTraceHelper;
    Ptr<OutputStreamWrapper> snrStreamWrapper =
    asciiTraceHelper.CreateFileStream(GenerateFileName(scenario, "snr", rlcAmEnabled, distribution, txPowerStr,std::to_string(runNumber),beamformingInterval, codebookFile));

   

    // Connect the trace with the modified callback function
    mmWaveUePhy->TraceConnectWithoutContext(
        "ReportCurrentCellRsrpSinr",
        MakeBoundCallback(&LogReceivedPacketSNR, snrStreamWrapper));

    
    Ptr<OutputStreamWrapper> ueStats = 
    asciiTraceHelper.CreateFileStream(GenerateFileName(scenario, "UE", rlcAmEnabled, distribution, txPowerStr,std::to_string(runNumber),beamformingInterval, codebookFile));
    Ptr<MmWaveSpectrumPhy> ueSpectrumPhy = mmWaveUePhy->GetDlSpectrumPhy();
    Ptr<MmWaveBeamformingModel> ueBeamformingModel = ueSpectrumPhy->GetBeamformingModel();
    if (ueSpectrumPhy != nullptr) {
    ueSpectrumPhy->TraceConnectWithoutContext("RxPacketTraceUe", MakeBoundCallback(&LogReceivedAtUe, ueStats));
    } else {
    std::cout << "Failed to get MmWaveSpectrumPhy from mmWaveUePhy";
    }
        *ueStats->GetStream() << "TIME,MCS,RV,BLER,Corrupt,NUMSYM,TBSIZE" << std::endl;

   

     
    
    Ptr<OutputStreamWrapper> beamformingStats = 
    asciiTraceHelper.CreateFileStream(GenerateFileName(scenario, "Beamforming", rlcAmEnabled, distribution, txPowerStr,std::to_string(runNumber),beamformingInterval, codebookFile));
    ueBeamformingModel->TraceConnectWithoutContext("BeamformingTrace", MakeBoundCallback(&LogBeamforming, beamformingStats));


    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        PrintIpAddresses(ueNodes.Get(i));
    }

    for (uint32_t i = 0; i < enbNodes.GetN(); ++i)
    {
        PrintIpAddresses(enbNodes.Get(i));
    }

    for (uint32_t i = 0; i < remoteHostContainer.GetN(); ++i)
    {
        PrintIpAddresses(remoteHostContainer.Get(i));
    }

  

    Ptr<Ipv4> ueIpv4 = ueNodes.Get(0)->GetObject<Ipv4>();
    Ipv4InterfaceAddress ueIpAddr = ueIpv4->GetAddress(1, 0);
    
    ApplicationContainer apps;
    Config::SetDefault ("ns3::PacketSink::EnableSeqTsSizeHeader", BooleanValue (true));
   
    // Config::SetDefault ("ns3::PscVideoStreaming::EnableSeqTsSizeHeader", BooleanValue (true));

    

    Ptr<PscVideoStreaming> streamingServer = CreateObject<PscVideoStreaming>();
    // streamingServer->SetAttribute("ReceiverAddress", AddressValue(internetIpIfaces.GetAddress(1)));
    streamingServer->SetAttribute("ReceiverAddress",  AddressValue(ueIpAddr.GetLocal()));
    
    streamingServer->SetAttribute("ReceiverPort", UintegerValue(5554));
    streamingServer->SetAttribute("Distribution", StringValue(distribution));
    streamingServer->SetAttribute("BoostLengthPacketCount", UintegerValue(boostLength));
    streamingServer->SetAttribute("BoostPercentile", DoubleValue(boostPercentile));

    apps.Add(streamingServer);
    // ueNodes.Get(0)->AddApplication(streamingServer);

    remoteHostContainer.Get(0)->AddApplication(streamingServer);

    PacketSinkHelper packetSinkHelper("ns3::UdpSocketFactory",
                                      InetSocketAddress(Ipv4Address::GetAny(), 5554));

    // apps.Add(packetSinkHelper.Install(remoteHost));
    apps.Add(packetSinkHelper.Install(ueNodes.Get(0)));

    apps.Start(Seconds(1));
    apps.Stop(Seconds(appEnd));
    
    
    Ptr<Application> appStream = apps.Get(1);
    Ptr<PacketSink> sink = appStream->GetObject<PacketSink>();
    AsciiTraceHelper asciiTraceHelperThroughput;


    Ptr<OutputStreamWrapper> throughputStreamWrapper =
    asciiTraceHelper.CreateFileStream(GenerateFileName(scenario, "throughput", rlcAmEnabled, distribution,txPowerStr,std::to_string(runNumber),beamformingInterval, codebookFile));    
    Simulator::Schedule(MilliSeconds(1000), &PrintThroughput, throughputStreamWrapper, sink);


   
    // Traces
    AsciiTraceHelper ascii;
    std::ostringstream oss;
        Ptr<OutputStreamWrapper> packetOutputStream =
        ascii.CreateFileStream("Whatev");
    *packetOutputStream->GetStream() << "time(sec)\ttx/rx\tPktSize(bytes)" << std::endl;

    oss.str("tx\t");
    apps.Get(0)->TraceConnect("Tx",
                              oss.str(),
                              MakeBoundCallback(&TxPacketTrace));
    oss.str("rx\t");
    apps.Get(1)->TraceConnect("Rx",
                              oss.str(),
                              MakeBoundCallback(&RxPacketTrace));

    // To Handle delay
     Ipv4Address localAddrs =  apps.Get(0)->GetNode ()->GetObject<Ipv4L3Protocol> ()->GetAddress (1,0).GetLocal ();
    apps.Get(0)->TraceConnectWithoutContext ("TxWithSeqTsSize", MakeBoundCallback (&TxPacketTraceForDelay, localAddrs));

    Ptr<OutputStreamWrapper> delayTraceStream = ascii.CreateFileStream((GenerateFileName(scenario, "delay", rlcAmEnabled, distribution, txPowerStr,std::to_string(runNumber),beamformingInterval, codebookFile)));
    *delayTraceStream->GetStream() << "time(s)\tseqNum\tdelay(ms)\tQUEUESIZE" << std::endl;


     Ipv4Address localAddrSink =  apps.Get(1)->GetNode ()->GetObject<Ipv4L3Protocol> ()->GetAddress (1,0).GetLocal ();
      apps.Get(1)->TraceConnectWithoutContext ("RxWithSeqTsSize", MakeBoundCallback (&RxPacketTraceForDelay, delayTraceStream,  apps.Get(1)->GetNode (), localAddrSink));


    //  Simulator::Schedule(Seconds(1), &TimeoutOldEntries, 5.0,delayTraceStream); // Assuming 30 seconds timeout
    // Ptr<FlowMonitor> flowmon;
    // FlowMonitorHelper flowmonHelper;
    // flowmon = flowmonHelper.InstallAll();

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    

    // flowmonHelper.SerializeToXmlFile("stat.flowmon", false, false);
    Simulator::Destroy();
    return 0;
}
