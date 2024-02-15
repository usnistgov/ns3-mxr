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
#include <fstream>
#include "ns3/core-module.h"
#include "ns3/three-gpp-spectrum-propagation-loss-model.h"
#include "ns3/simple-net-device.h"
#include "ns3/node-container.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/qd-channel-model.h"
#include "ns3/mmwave-helper.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/config-store.h"
#include "ns3/mmwave-point-to-point-epc-helper.h"
#include "ns3/isotropic-antenna-model.h"

#include "ns3/mmwave-helper.h"
#include "ns3/mmwave-net-device.h"
#include "ns3/mmwave-phy.h"
#include <cfloat>
#include <fstream>



using namespace ns3;
using namespace mmwave;

// Function to print the throughput
void PrintThroughput(Ptr<PacketSink> sink) {
  double timeNow = Simulator::Now().GetSeconds();
  double throughput = (sink->GetTotalRx() * 8.0/1e6) / (timeNow - 0.1); // Calculate the throughput
  std::cout << "Throughput at " << timeNow << " seconds: " << throughput << " Mbps\n";
  Simulator::Schedule(MilliSeconds(100), &PrintThroughput, sink);
}

void LogReceivedPacketSNR(Ptr<OutputStreamWrapper> streamWrapper,uint64_t t, SpectrumValue& sinr, SpectrumValue& y)
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
    *streamWrapper->GetStream () << currentTime << "," << sinrAvgDb << std::endl;
    // Log the average SINR value (in dB)
    // std::cout << ns3::Simulator::Now().GetSeconds()<< " Average SINR: " << sinrAvgDb << " dB" << std::endl;
  }
}


int
main (int argc, char *argv[])
{
  std::string qdFilesPath = "../../contrib/qd-channel/model/QD/"; // The path of the folder with the QD scenarios
  // std::string scenario = "FDA"; // The name of the scenario
  std::string scenario = "FDANewRoom"; // The name of the scenario
  uint32_t interPacketInterval = 1000e3; // App inter packet arrival [us]
  // uint32_t interPacketInterval = 1000000; // App inter packet arrival [us] 
  double txPower = 24.0; // Transmitted power for both eNB and UE [dBm]
  double noiseFigure = 12.0; // Noise figure for both eNB and UE [dB]
  uint16_t enbAntennaNum = 64; // The number of antenna elements for the gNBs antenna arrays, assuming a square architecture
  uint16_t ueAntennaNum = 2; // The number of antenna elements for the UE antenna arrays, assuming a square architecture
  uint32_t appPacketSize = 1460; // Application packet size [B]
  unsigned int simTime = 595;  		// simulation time [s]
  unsigned int appEnd = simTime - 1;	// application end time [s]	
  unsigned int minStart = 300;		// application min start time [ms]
  unsigned int maxStart = 400; 		// application max start time [ms]
  std::string filePath = "./";		// where to save the traces
  std::string appRate = "50Mbps";	    // application data rate	
  
  CommandLine cmd;
  cmd.AddValue ("qdFilesPath", "The path of the folder with the QD scenarios", qdFilesPath);
  cmd.AddValue ("scenario", "The name of the scenario", scenario);
  cmd.AddValue ("ipi", "App inter packet arrival [us]", interPacketInterval);
  cmd.AddValue ("txPower", "Transmitted power for both eNB and UE [dBm]", txPower);
  cmd.AddValue ("noiseFigure", "Noise figure for both eNB and UE [dB]", noiseFigure);
  cmd.AddValue ("enbAntennaNum", "The number of antenna elements for the gNBs antenna arrays, assuming a square architecture", enbAntennaNum);
  cmd.AddValue ("ueAntennaNum", "The number of antenna elements for the UE antenna arrays, assuming a square architecture", ueAntennaNum);
  cmd.AddValue ("appPacketSize", "Application packet size [B]", appPacketSize);
  cmd.AddValue ("minStart", "application start time [ms]", minStart);
  cmd.AddValue ("maxStart", "application max start time [ms]", maxStart);
  cmd.AddValue ("appEnd", "application end time [s]", appEnd);
  cmd.AddValue ("filePath", "Where to save the output traces", filePath);
  cmd.AddValue ("appRate", "The data-rate for the video applications", appRate);
  cmd.Parse (argc, argv);

  bool harqEnabled = true;
  bool rlcAmEnabled = true;

  Config::SetDefault ("ns3::MmWaveHelper::RlcAmEnabled", BooleanValue (rlcAmEnabled));
  Config::SetDefault ("ns3::MmWaveHelper::HarqEnabled", BooleanValue (harqEnabled));
  Config::SetDefault ("ns3::MmWaveFlexTtiMacScheduler::HarqEnabled", BooleanValue (harqEnabled));

  // Create the tx and rx nodes
  NodeContainer ueNodes;
  NodeContainer enbNodes;
  enbNodes.Create (1);
  ueNodes.Create (1);

    // initial positions of the nodes in the ray tracer
    Ptr<MobilityModel> ueRefMob = CreateObject<ConstantPositionMobilityModel>();
    ueRefMob->SetPosition(Vector(5, 0.1, 1.5));
    Ptr<MobilityModel> enb1Mob = CreateObject<ConstantPositionMobilityModel>();
    enb1Mob->SetPosition(Vector(5, 0.1, 2.9));

        enbNodes.Get(0)->AggregateObject(enb1Mob);
    ueNodes.Get(0)->AggregateObject(ueRefMob);

  // Configure the channel
  Config::SetDefault ("ns3::MmWaveHelper::PathlossModel", StringValue (""));
  Config::SetDefault ("ns3::MmWaveHelper::ChannelModel", StringValue ("ns3::ThreeGppSpectrumPropagationLossModel"));
  Ptr<QdChannelModel> qdModel = CreateObject<QdChannelModel> (qdFilesPath, scenario);
  //Time simTime = qdModel->GetQdSimTime ();
  Config::SetDefault ("ns3::ThreeGppSpectrumPropagationLossModel::ChannelModel", PointerValue (qdModel));

  // Set power and noise figure
  Config::SetDefault ("ns3::MmWavePhyMacCommon::Bandwidth", DoubleValue (200e6));
  Config::SetDefault ("ns3::MmWaveEnbPhy::TxPower", DoubleValue (txPower));
  Config::SetDefault ("ns3::MmWaveEnbPhy::NoiseFigure", DoubleValue (noiseFigure));
  Config::SetDefault ("ns3::MmWaveUePhy::TxPower", DoubleValue (txPower));
  Config::SetDefault ("ns3::MmWaveUePhy::NoiseFigure", DoubleValue (noiseFigure));
  

  // Setup antenna configuration
  Config::SetDefault ("ns3::PhasedArrayModel::AntennaElement", PointerValue (CreateObject<IsotropicAntennaModel> ()));

  // Create the MmWave helper
  Ptr<MmWaveHelper> mmwaveHelper = CreateObject<MmWaveHelper> ();

  // select the beamforming model
  mmwaveHelper->SetBeamformingModelType("ns3::MmWaveCodebookBeamforming");

  // configure the UE antennas:
  // 1. specify the path of the file containing the codebook
  mmwaveHelper->SetUeBeamformingCodebookAttribute ("CodebookFilename", StringValue ("../../src/mmwave/model/Codebooks/1x8.txt"));
  // 2. set the antenna dimensions
  mmwaveHelper->SetUePhasedArrayModelAttribute ("NumRows", UintegerValue (1));
  mmwaveHelper->SetUePhasedArrayModelAttribute ("NumColumns", UintegerValue (8));
  
  // configure the BS antennas:
  // 1. specify the path of the file containing the codebook
  mmwaveHelper->SetEnbBeamformingCodebookAttribute ("CodebookFilename", StringValue ("../../src/mmwave/model/Codebooks/1x8.txt"));
  mmwaveHelper->SetEnbPhasedArrayModelAttribute ("NumRows", UintegerValue (1));
  // 2. set the antenna dimensions
  mmwaveHelper->SetEnbPhasedArrayModelAttribute ("NumColumns", UintegerValue (8));

  mmwaveHelper->SetSchedulerType ("ns3::MmWaveFlexTtiMacScheduler");
  Ptr<MmWavePointToPointEpcHelper>  epcHelper = CreateObject<MmWavePointToPointEpcHelper> ();
  mmwaveHelper->SetEpcHelper (epcHelper);
  mmwaveHelper->SetHarqEnabled (harqEnabled);

  // Create a single RemoteHost
  Ptr<Node> pgw = epcHelper->GetPgwNode ();
  NodeContainer remoteHostContainer;
  remoteHostContainer.Create (1);
  Ptr<Node> remoteHost = remoteHostContainer.Get (0);
  InternetStackHelper internet;
  internet.Install (remoteHostContainer);

  // Create the Internet
  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("100Gb/s")));
  p2ph.SetDeviceAttribute ("Mtu", UintegerValue (1500));
  p2ph.SetChannelAttribute ("Delay", TimeValue (Seconds (0.010)));
  NetDeviceContainer internetDevices = p2ph.Install (pgw, remoteHost);
  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase ("1.0.0.0", "255.0.0.0");
  Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign (internetDevices);
  // Interface 0 is localhost, 1 is the p2p device
  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());
  remoteHostStaticRouting->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"), 1);

  // Create the tx and rx devices
  NetDeviceContainer enbMmWaveDevs = mmwaveHelper->InstallEnbDevice (enbNodes);
  NetDeviceContainer ueMmWaveDevs = mmwaveHelper->InstallUeDevice (ueNodes);

  // Install the IP stack on the UEs
  internet.Install (ueNodes);
  Ipv4InterfaceContainer ueIpIface;
  ueIpIface = epcHelper->AssignUeIpv4Address (NetDeviceContainer (ueMmWaveDevs));
  // Assign IP address to UEs, and install applications
  Ptr<Node> ueNode = ueNodes.Get (0);
  // Set the default gateway for the UE
  Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting (ueNode->GetObject<Ipv4> ());
  ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);

  // This performs the attachment of each UE to a specific eNB
  mmwaveHelper->AttachToEnbWithIndex (ueMmWaveDevs.Get (0), enbMmWaveDevs, 0);
  
     // Add apps
    uint16_t dlPort = 1234;
    uint16_t ulPort = 2000;
    uint16_t otherPort = 3000;
    ApplicationContainer clientApps;
    ApplicationContainer serverApps;
    ++ulPort;
    ++otherPort;
    PacketSinkHelper dlPacketSinkHelper("ns3::UdpSocketFactory",
                                        InetSocketAddress(Ipv4Address::GetAny(), dlPort));
    serverApps.Add(dlPacketSinkHelper.Install(ueNodes.Get(0)));

    UdpClientHelper dlClient(ueIpIface.GetAddress(0), dlPort);
    dlClient.SetAttribute("Interval", TimeValue(MicroSeconds(interPacketInterval)));
    dlClient.SetAttribute("MaxPackets", UintegerValue(1000000));
    dlClient.SetAttribute("PacketSize", UintegerValue(appPacketSize));

    clientApps.Add(dlClient.Install(remoteHost));

    serverApps.Start(Seconds(0.01));
    clientApps.Start(Seconds(0.01));
    mmwaveHelper->EnableTraces();

    
    
    // Get the PacketSink object
Ptr<Application> app = serverApps.Get(0);
Ptr<PacketSink> sink = app->GetObject<PacketSink>();

// Assume 'ueNodes' is a NodeContainer holding the UE nodes
Ptr<NetDevice> ueNetDevice = ueMmWaveDevs.Get(0);
// Assume 'ueNodes' is a NodeContainer holding the UE nodes

Ptr<mmwave::MmWaveUeNetDevice> mmWaveUeNetDevice = DynamicCast<mmwave::MmWaveUeNetDevice> (ueNetDevice);

// Get the MmWaveUePhy instance and connect the custom callback function
Ptr<mmwave::MmWaveUePhy> mmWaveUePhy = mmWaveUeNetDevice->GetPhy();
// Wrap the file in an OutputStreamWrapper

  AsciiTraceHelper asciiTraceHelper;
  Ptr<OutputStreamWrapper> snrStreamWrapper = asciiTraceHelper.CreateFileStream ("snrResults1s48.txt");


  // Your code to set up the simulation

  // Connect the trace with the modified callback function
  mmWaveUePhy->TraceConnectWithoutContext("ReportCurrentCellRsrpSinr", MakeBoundCallback(&LogReceivedPacketSNR, snrStreamWrapper));

// mmWaveUePhy->TraceConnectWithoutContext("ReportCurrentCellRsrpSinr", MakeCallback(&LogReceivedPacketSNR,snrFile));
// mmWaveUePhy->TraceConnectWithoutContext("ReportCurrentCellRsrpSinr", MakeBoundCallback(&LogReceivedPacketSNR, std::ref(snrFile)));



// Schedule a function to print the throughput every 100ms
// Simulator::Schedule(MilliSeconds(100), &PrintThroughput, sink);
//  Simulator::Schedule(MilliSeconds(100), &PrintThroughput, sink);





  Simulator::Stop(Seconds (simTime));
  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}
