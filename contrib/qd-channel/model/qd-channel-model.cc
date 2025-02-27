/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
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
 *
 */

#include "ns3/qd-channel-model.h"

#include "ns3/csv-reader.h"
#include "ns3/double.h"
#include "ns3/integer.h"
#include "ns3/log.h"
#include "ns3/mobility-model.h"
#include "ns3/net-device.h"
#include "ns3/node.h"
#include "ns3/string.h"
#include <ns3/node-list.h>
#include <ns3/simulator.h>

#include <algorithm>
#include <fstream>
#include <glob.h>
#include <random>
#include <sstream>
#include <filesystem>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("QdChannelModel");

NS_OBJECT_ENSURE_REGISTERED(QdChannelModel);

QdChannelModel::QdChannelModel(std::string path, std::string scenario)
{
    NS_LOG_FUNCTION(this);

    SetPath(path);
    SetScenario(scenario);
}

QdChannelModel::~QdChannelModel()
{
    NS_LOG_FUNCTION(this);
}

TypeId
QdChannelModel::GetTypeId(void)
{
    static TypeId tid =
        TypeId("ns3::QdChannelModel")
            .SetParent<MatrixBasedChannelModel>()
            .SetGroupName("Spectrum")
            .AddConstructor<QdChannelModel>()
            .AddAttribute(
                "Frequency",
                "The operating Frequency in Hz. This attribute is here "
                "only for compatibility with ns3::ThreeGppSpectrumPropagationLossModel.",
                DoubleValue(__DBL_MIN__),
                MakeDoubleAccessor(&QdChannelModel::SetFrequency, &QdChannelModel::GetFrequency),
                MakeDoubleChecker<double>());

    return tid;
}

std::vector<std::string>
QdChannelModel::GetQdFilesList(const std::string& pattern)
{
    NS_LOG_FUNCTION(this << pattern);

    glob_t glob_result;
    glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    std::vector<std::string> files;
    for (uint32_t i = 0; i < glob_result.gl_pathc; ++i)
    {
        files.push_back(std::string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return files;
}

std::vector<double>
QdChannelModel::ParseCsv(const std::string& str, bool toRad)
{
    NS_LOG_FUNCTION(this << str);

    std::stringstream ss(str);
    CsvReader csv(ss, ',');
    csv.FetchNextRow();

    std::vector<double> vect{};
    vect.reserve(csv.ColumnCount());

    double value;
    bool ok;
    for (size_t i = 0; i < csv.ColumnCount(); i++)
    {
        ok = csv.GetValue(i, value);
        NS_ABORT_MSG_IF(!ok, "Something went wrong while parsing the line: " << str);

        if (toRad)
        {
            value = DegreesToRadians(value);
        }

        vect.push_back(value);
    }

    return vect;
}

QdChannelModel::RtIdToNs3IdMap_t
QdChannelModel::ReadNodesPosition()
{
    NS_LOG_FUNCTION(this);

    std::string posFileName{m_path + m_scenario + "Output/Ns3/NodesPosition/NodesPosition.csv"};

    uint32_t id{0};
    QdChannelModel::RtIdToNs3IdMap_t rtIdToNs3IdMap;

    CsvReader csv(posFileName, ',');
    while (csv.FetchNextRow())
    {

        // Ignore blank lines
        if (csv.IsBlankRow())
        {
            continue;
        }

        // Expecting cartesian coordinates
        double x, y, z;
        bool ok = csv.GetValue(0, x);
        ok |= csv.GetValue(1, y);
        ok |= csv.GetValue(2, z);

        NS_ABORT_MSG_IF(!ok, "Something went wrong while parsing the file: " << posFileName);
        Vector3D nodePosition{x, y, z};

        NS_LOG_DEBUG("Trying to match position from file: " << nodePosition);
        m_nodePositionList.push_back(nodePosition);
        bool found{false};
        uint32_t matchedNodeId;
        for (NodeList::Iterator nit = NodeList::Begin(); nit != NodeList::End(); ++nit)
        {
            Ptr<MobilityModel> mm = (*nit)->GetObject<MobilityModel>();
            if (mm)
            {
                // TODO automatically import nodes' initial positions to avoid manual setting every
                // time the scenario changes
                Vector3D pos = mm->GetPosition();
                NS_LOG_DEBUG("Checking node with position: " << pos);
                if (nodePosition == pos)
                {
                    found = true;
                    matchedNodeId = (*nit)->GetId();
                    NS_LOG_LOGIC("got a match " << pos << " ID " << matchedNodeId);
                    break;
                }
            }
        }
        if (!found)
        {
            NS_LOG_ERROR("Position not found: " << nodePosition);
        }

        NS_ABORT_MSG_IF(!found,
                        "Position not matched - did you install the mobility model before "
                        "the channel is created");

        rtIdToNs3IdMap.insert(std::make_pair(id, matchedNodeId));
        m_ns3IdToRtIdMap.insert(std::make_pair(matchedNodeId, id));
        NS_LOG_INFO("qdId=" << id << " matches NodeId=" << matchedNodeId
                            << " with position=" << nodePosition);

        ++id;

    } // while FetchNextRow

    for (auto elem : m_nodePositionList)
    {
        NS_LOG_INFO(elem);
    }

    return rtIdToNs3IdMap;
}

void
QdChannelModel::ReadParaCfgFile()
{
    NS_LOG_FUNCTION(this);

    std::string paraCfgCurrentFileName{m_path + m_scenario + "Input/paraCfgCurrent.txt"};
    CsvReader csv(paraCfgCurrentFileName, '\t');
    csv.FetchNextRow(); // ignore first line (header)

    std::string varName, varValue;
    while (csv.FetchNextRow())
    {
        // Ignore blank lines
        if (csv.IsBlankRow())
        {
            continue;
        }

        // Expecting three values
        bool ok = csv.GetValue(0, varName);
        ok |= csv.GetValue(1, varValue);
        NS_ABORT_MSG_IF(!ok,
                        "Something went wrong while parsing the file: " << paraCfgCurrentFileName);

        if (varName.compare("numberOfTimeDivisions") == 0)
        {
            m_totTimesteps = atoi(varValue.c_str());
            NS_LOG_DEBUG("numberOfTimeDivisions (int) = " << m_totTimesteps);
        }
        else if (varName.compare("totalTimeDuration") == 0)
        {
            m_totalTimeDuration = Seconds(atof(varValue.c_str()));
            NS_LOG_DEBUG("m_totalTimeDuration = " << m_totalTimeDuration.GetSeconds() << " s");
        }
        else if (varName.compare("carrierFrequency") == 0)
        {
            m_frequency = atof(varValue.c_str());
            NS_LOG_DEBUG("carrierFrequency (float) = " << m_frequency);
        }

    } // while FetchNextRow
}

void
QdChannelModel::ReadQdFiles(QdChannelModel::RtIdToNs3IdMap_t rtIdToNs3IdMap)
{
    NS_LOG_FUNCTION(this);

    // QdFiles input
    NS_LOG_INFO("m_path + m_scenario = " << m_path + m_scenario);
    auto qdFileList = GetQdFilesList(m_path + m_scenario + "Output/Ns3/QdFiles/*");
    NS_LOG_DEBUG("qdFileList.size ()=" << qdFileList.size());

    std::filesystem::path currentPath = std::filesystem::current_path();

    // Convert the path to a string and store it in a variable
    std::string pathString = currentPath.string();

    for (auto fileName : qdFileList)
    {
        // get the nodes IDs from the file name
        int txIndex = fileName.find("Tx");
        int rxIndex = fileName.find("Rx");
        int txtIndex = fileName.find(".txt");
        
        

        int len{rxIndex - txIndex - 2};
        int id_tx{::atoi(fileName.substr(txIndex + 2, len).c_str())};
        len = txtIndex - rxIndex - 2;
        int id_rx{::atoi(fileName.substr(rxIndex + 2, len).c_str())};
        
        NS_ABORT_MSG_IF(rtIdToNs3IdMap.find(id_tx) == rtIdToNs3IdMap.end(), "ID not found for TX!");
        uint32_t nodeIdTx = rtIdToNs3IdMap.find(id_tx)->second;
        NS_ABORT_MSG_IF(rtIdToNs3IdMap.find(id_rx) == rtIdToNs3IdMap.end(), "ID not found for RX!");
        uint32_t nodeIdRx = rtIdToNs3IdMap.find(id_rx)->second;

        NS_LOG_DEBUG("id_tx: " << id_tx << ", id_rx: " << id_rx);

        uint32_t key = GetKey(nodeIdTx, nodeIdRx);
        // std::pair<Ptr<const MobilityModel>, Ptr<const MobilityModel>> idPair
        // {std::make_pair(tx_mm, rx_mm)};

        std::ifstream qdFile{fileName.c_str()};

        std::string line{};
        std::vector<QdInfo> qdInfoVector;

        //TR++
        bool chop = false;
        if (chop){
        int chopLine = 85;
        int choppedLine = 0;
        while (choppedLine < chopLine)
        {
           
            int nbLineForMPCs=7;
            int nbLineRead = 0;
            int numMPCs = 0;
            std::getline(qdFile, line);
            numMPCs = std::stoul(line, 0, 10);
          

            if (numMPCs > 0){
                while (nbLineRead<nbLineForMPCs){
                    std::getline(qdFile, line);
                    
                    nbLineRead++;
                }
                choppedLine++;
            }
        }
        }
        //TR==
        while (std::getline(qdFile, line))
        {
            QdInfo qdInfo{};
            // the file has a line with the number of multipath components
            qdInfo.numMpcs = std::stoul(line, 0, 10);
            NS_LOG_DEBUG("numMpcs " << qdInfo.numMpcs);

            if (qdInfo.numMpcs > 0)
            {
                // a line with the delays
                std::getline(qdFile, line);
                auto pathDelays = ParseCsv(line);
                NS_ABORT_MSG_IF(pathDelays.size() != qdInfo.numMpcs,
                                "mismatch between number of path delays ("
                                    << pathDelays.size() << ") and number of MPCs ("
                                    << qdInfo.numMpcs << "), timestep=" << qdInfoVector.size() + 1
                                    << ", fileName=" << fileName);
                qdInfo.delay_s = pathDelays;
                // a line with the path gains
                std::getline(qdFile, line);
                auto pathGains = ParseCsv(line);
                NS_ABORT_MSG_IF(pathGains.size() != qdInfo.numMpcs,
                                "mismatch between number of path gains ("
                                    << pathGains.size() << ") and number of MPCs ("
                                    << qdInfo.numMpcs << "), timestep=" << qdInfoVector.size() + 1
                                    << ", fileName=" << fileName);
                qdInfo.pathGain_dbpow = pathGains;
                // a line with the path phases
                std::getline(qdFile, line);
                auto pathPhases = ParseCsv(line);
                NS_ABORT_MSG_IF(pathPhases.size() != qdInfo.numMpcs,
                                "mismatch between number of path phases ("
                                    << pathPhases.size() << ") and number of MPCs ("
                                    << qdInfo.numMpcs << "), timestep=" << qdInfoVector.size() + 1
                                    << ", fileName=" << fileName);
                qdInfo.phase_rad = pathPhases;
                // a line with the elev AoD
                std::getline(qdFile, line);
                auto pathElevAod = ParseCsv(line, true);
                NS_ABORT_MSG_IF(pathElevAod.size() != qdInfo.numMpcs,
                                "mismatch between number of path elev AoDs ("
                                    << pathElevAod.size() << ") and number of MPCs ("
                                    << qdInfo.numMpcs << "), timestep=" << qdInfoVector.size() + 1
                                    << ", fileName=" << fileName);
                qdInfo.elAod_rad = pathElevAod;
                // a line with the azimuth AoD
                std::getline(qdFile, line);
                auto pathAzAod = ParseCsv(line, true);
                NS_ABORT_MSG_IF(pathAzAod.size() != qdInfo.numMpcs,
                                "mismatch between number of path az AoDs ("
                                    << pathAzAod.size() << ") and number of MPCs ("
                                    << qdInfo.numMpcs << "), timestep=" << qdInfoVector.size() + 1
                                    << ", fileName=" << fileName);
                qdInfo.azAod_rad = pathAzAod;
                // a line with the elev AoA
                std::getline(qdFile, line);
                auto pathElevAoa = ParseCsv(line, true);
                NS_ABORT_MSG_IF(pathElevAoa.size() != qdInfo.numMpcs,
                                "mismatch between number of path elev AoAs ("
                                    << pathElevAoa.size() << ") and number of MPCs ("
                                    << qdInfo.numMpcs << "), timestep=" << qdInfoVector.size() + 1
                                    << ", fileName=" << fileName);
                qdInfo.elAoa_rad = pathElevAoa;
                // a line with the azimuth AoA
                std::getline(qdFile, line);
                auto pathAzAoa = ParseCsv(line, true);
                NS_ABORT_MSG_IF(pathAzAoa.size() != qdInfo.numMpcs,
                                "mismatch between number of path az AoAs ("
                                    << pathAzAoa.size() << ") and number of MPCs ("
                                    << qdInfo.numMpcs << "), timestep=" << qdInfoVector.size() + 1
                                    << ", fileName=" << fileName);
                qdInfo.azAoa_rad = pathAzAoa;
            }
            qdInfoVector.push_back(qdInfo);
        }
        NS_LOG_DEBUG("qdInfoVector.size ()=" << qdInfoVector.size());
        m_qdInfoMap.insert(std::make_pair(key, qdInfoVector));
    }

    NS_LOG_INFO("Imported files for " << m_qdInfoMap.size() << " tx/rx pairs");
}

void
QdChannelModel::ReadAllInputFiles()
{
    NS_LOG_FUNCTION(this);
    NS_LOG_INFO("ReadAllInputFiles for scenario " << m_scenario << " path " << m_path);

    m_ns3IdToRtIdMap.clear();
    m_qdInfoMap.clear();

    ReadParaCfgFile();
    QdChannelModel::RtIdToNs3IdMap_t rtIdToNs3IdMap = ReadNodesPosition();
    ReadQdFiles(rtIdToNs3IdMap);
    // Setup simulation timings assuming constant periodicity
    //TR++
    // Dictate how often the channel is updated
    m_updatePeriod = Seconds(1);
    //TR++
    NS_LOG_DEBUG("m_totalTimeDuration=" << m_totalTimeDuration.GetSeconds()
                                        << " s"
                                           ", m_updatePeriod="
                                        << m_updatePeriod.GetNanoSeconds() / 1e6
                                        << " ms"
                                           ", m_totTimesteps="
                                        << m_totTimesteps);
}

Time
QdChannelModel::GetQdSimTime() const
{
    NS_LOG_FUNCTION(this);
    return m_totalTimeDuration;
}

void
QdChannelModel::SetFrequency(double fc)
{
    NS_LOG_FUNCTION(this);
}

double
QdChannelModel::GetFrequency() const
{
    NS_LOG_FUNCTION(this);
    return m_frequency;
}

void
QdChannelModel::TrimFolderName(std::string& folder)
{
    // avoid starting with multiple '/'
    while (folder.front() == '/' && folder.substr(1, folder.size()).front() == '/')
    {
        folder = folder.substr(1, folder.size());
    }

    // avoid ending with multiple '/'
    while (folder.back() == '/')
    {
        folder = folder.substr(0, folder.size() - 1);
    }

    folder += '/';
}

void
QdChannelModel::SetScenario(std::string scenario)
{
    NS_LOG_FUNCTION(this << scenario);
    NS_ABORT_MSG_IF(m_path == "", "m_path empty, use SetPath first");

    TrimFolderName(scenario);

    if (scenario != m_scenario // avoid re-reading input files
        && scenario != "")
    {
        m_scenario = scenario;
        // read the information for this scenario
        ReadAllInputFiles();
    }
}

std::string
QdChannelModel::GetScenario() const
{
    NS_LOG_FUNCTION(this);
    return m_scenario;
}

void
QdChannelModel::SetPath(std::string path)
{
    NS_LOG_FUNCTION(this << path);
    TrimFolderName(path);
    m_path = path;
}

std::string
QdChannelModel::GetPath() const
{
    NS_LOG_FUNCTION(this);
    return m_path;
}

    //TR++ Const Removed
bool
QdChannelModel::ChannelMatrixNeedsUpdate(
    Ptr< MatrixBasedChannelModel::ChannelMatrix> channelMatrix) const
{
    NS_LOG_FUNCTION(this << channelMatrix);

    uint64_t nowTimestep = GetTimestep();
    uint64_t lastChanUpdateTimestep = GetTimestep(channelMatrix->m_generatedTime);

    NS_ASSERT_MSG(nowTimestep >= lastChanUpdateTimestep,
                  "Current timestep=" << nowTimestep << ", last channel update timestep="
                                      << lastChanUpdateTimestep);

    bool update = false;
    // if the coherence time is over the channel has to be updated
    if (lastChanUpdateTimestep < nowTimestep)
    {
        NS_LOG_LOGIC("Generation time " << channelMatrix->m_generatedTime.GetNanoSeconds()
                                        << " now " << Simulator::Now().GetNanoSeconds()
                                        << " update needed");
        update = true;    
    }
    else
    {
        NS_LOG_LOGIC("Generation time " << channelMatrix->m_generatedTime.GetNanoSeconds()
                                        << " now " << Simulator::Now().GetNanoSeconds()
                                        << " update not needed");
    }

    return update;
}

Ptr<const MatrixBasedChannelModel::ChannelMatrix>
QdChannelModel::GetChannel(Ptr<const MobilityModel> aMob,
                           Ptr<const MobilityModel> bMob,
                           Ptr<const PhasedArrayModel> aAntenna,
                           Ptr<const PhasedArrayModel> bAntenna)
{
    NS_LOG_FUNCTION(this << aMob << bMob << aAntenna << bAntenna);

    // Compute the channel keys
    uint32_t aId = aMob->GetObject<Node>()->GetId();
    uint32_t bId = bMob->GetObject<Node>()->GetId();

    uint32_t channelId = GetKey(aId, bId);

    NS_LOG_DEBUG("channelId " << channelId << ", ns-3 aId=" << aId << " bId=" << bId
                              << ", RT sim. aId=" << m_ns3IdToRtIdMap[aId]
                              << " bId=" << m_ns3IdToRtIdMap[bId]);

    // Check if the channel is present in the map and return it, otherwise
    // generate a new channel
    bool update = false;
    bool notFound = false;
    Ptr<MatrixBasedChannelModel::ChannelMatrix> channelMatrix;
    if (m_channelMap.find(channelId) != m_channelMap.end())
    {
        // channel matrix present in the map
        NS_LOG_LOGIC("channel matrix present in the map");
        channelMatrix = m_channelMap[channelId];

        // check if it has to be updated
        update = ChannelMatrixNeedsUpdate(channelMatrix);
    }
    else
    {
        NS_LOG_LOGIC("channel matrix not found");
        notFound = true;
    }

    // If the channel is not present in the map or if it has to be updated
    // generate a new channel
    if (notFound || update)
    {
        NS_LOG_LOGIC("channelMatrix notFound=" << notFound << " || update=" << update);
        channelMatrix = GetNewChannel(aMob, bMob, aAntenna, bAntenna);
        channelMatrix->m_antennaPair =
            std::make_pair(aAntenna->GetId(),
                           bAntenna->GetId()); // save antenna pair, with the exact order of s and u
                                               // antennas at the moment of the channel generation

        
        
        //TR++ Update the last time the channel was generated
        channelMatrix->m_generatedTime = Simulator::Now();
        // store the channel matrix in the channel map
        m_channelMap[channelId] = channelMatrix;
    }

    return channelMatrix;
}

Ptr<MatrixBasedChannelModel::ChannelMatrix>
QdChannelModel::GetNewChannel(Ptr<const MobilityModel> aMob,
                              Ptr<const MobilityModel> bMob,
                              Ptr<const PhasedArrayModel> aAntenna,
                              Ptr<const PhasedArrayModel> bAntenna)
{
    NS_LOG_FUNCTION(this << aMob << bMob << aAntenna << bAntenna);

    Ptr<MatrixBasedChannelModel::ChannelMatrix> channelMatrix =
        Create<MatrixBasedChannelModel::ChannelMatrix>();


    uint32_t timestep = GetTimestep();
    

    uint32_t aId = aMob->GetObject<Node>()->GetId();
    uint32_t bId = bMob->GetObject<Node>()->GetId();
    uint32_t channelId = GetKey(aId, bId);

    QdInfo qdInfo = m_qdInfoMap.at(channelId)[timestep];

    uint64_t bSize = bAntenna->GetNumberOfElements();
    uint64_t aSize = aAntenna->GetNumberOfElements();

    NS_LOG_DEBUG("timestep=" << timestep << ", aId=" << aId << ", bId=" << bId
                             << ", m_ns3IdToRtIdMap[aId]=" << m_ns3IdToRtIdMap.at(aId)
                             << ", m_ns3IdToRtIdMap[bId]=" << m_ns3IdToRtIdMap.at(bId)
                             << ", channelId=" << channelId << ", bSize=" << bSize
                             << ", aSize=" << aSize);

    // channel coffecient H[u][s][n];
    // considering only 1 cluster for retrocompatibility -> n=1
    MatrixBasedChannelModel::Complex3DVector H(bSize, aSize, qdInfo.numMpcs);

    for (uint64_t mpcIndex = 0; mpcIndex < qdInfo.numMpcs; ++mpcIndex)
    {
        double initialPhase =
            -2 * M_PI * qdInfo.delay_s[mpcIndex] * m_frequency + qdInfo.phase_rad[mpcIndex];
        double pathGain = pow(10, qdInfo.pathGain_dbpow[mpcIndex] / 20);

        Angles bAngle = Angles(qdInfo.azAoa_rad[mpcIndex], qdInfo.elAoa_rad[mpcIndex]);
        Angles aAngle = Angles(qdInfo.azAod_rad[mpcIndex], qdInfo.elAod_rad[mpcIndex]);
        NS_LOG_DEBUG("aAngle: " << aAngle << ", bAngle: " << bAngle);

        // ignore polarization
        double bFieldPattH, bFieldPattV, aFieldPattH, aFieldPattV;
        std::tie(bFieldPattH, bFieldPattV) = bAntenna->GetElementFieldPattern(bAngle);
        double bElementGain = std::sqrt(bFieldPattH * bFieldPattH + bFieldPattV * bFieldPattV);
        std::tie(aFieldPattH, aFieldPattV) = aAntenna->GetElementFieldPattern(aAngle);
        double aElementGain = std::sqrt(aFieldPattH * aFieldPattH + aFieldPattV * aFieldPattV);

        double pgTimesGains = pathGain * bElementGain * aElementGain;
        std::complex<double> complexRay = pgTimesGains * std::polar(1.0, initialPhase);

        NS_LOG_DEBUG("qdInfo.delay_s[mpcIndex]="
                     << qdInfo.delay_s[mpcIndex]
                     << ", qdInfo.phase_rad[mpcIndex]=" << qdInfo.phase_rad[mpcIndex]
                     << ", qdInfo.pathGain_dbpow[mpcIndex]=" << qdInfo.pathGain_dbpow[mpcIndex]
                     << ", bAngle=" << bAngle << ", aAngle=" << aAngle
                     << ", initialPhase=" << initialPhase << ", pathGain=" << pathGain
                     << ", bElementGain=" << bElementGain << ", aElementGain=" << aElementGain
                     << ", pgTimesGains=" << pgTimesGains << ", complexRay=" << complexRay);

        for (uint64_t bIndex = 0; bIndex < bSize; ++bIndex)
        {
            Vector uLoc = bAntenna->GetElementLocation(bIndex);
            double bPhaseElementPhase =
                2 * M_PI *
                (sin(qdInfo.elAoa_rad[mpcIndex]) * cos(qdInfo.azAoa_rad[mpcIndex]) * uLoc.x +
                 sin(qdInfo.elAoa_rad[mpcIndex]) * sin(qdInfo.azAoa_rad[mpcIndex]) * uLoc.y +
                 cos(qdInfo.elAoa_rad[mpcIndex]) * uLoc.z);
            std::complex<double> bWeight = std::polar(1.0, bPhaseElementPhase);

            for (uint64_t aIndex = 0; aIndex < aSize; ++aIndex)
            {
                Vector sLoc = aAntenna->GetElementLocation(aIndex);
                // minus sign: complex conjugate for TX steering vector
                double aPhaseElementPhase =
                    2 * M_PI *
                    (sin(qdInfo.elAod_rad[mpcIndex]) * cos(qdInfo.azAod_rad[mpcIndex]) * sLoc.x +
                     sin(qdInfo.elAod_rad[mpcIndex]) * sin(qdInfo.azAod_rad[mpcIndex]) * sLoc.y +
                     cos(qdInfo.elAod_rad[mpcIndex]) * sLoc.z);
                std::complex<double> aWeight = std::polar(1.0, aPhaseElementPhase);

                std::complex<double> ray = complexRay * bWeight * aWeight;
                H(bIndex, aIndex, 0) += ray;
            }
        }
    }

    Ptr<MatrixBasedChannelModel::ChannelParams> channelParams =
        Create<MatrixBasedChannelModel::ChannelParams>();

    channelMatrix->m_channel = H;
    channelParams->m_delay = qdInfo.delay_s;

    channelParams->m_angle.clear();
    channelParams->m_angle.push_back(qdInfo.azAoa_rad);
    channelParams->m_angle.push_back(qdInfo.elAoa_rad);
    channelParams->m_angle.push_back(qdInfo.azAod_rad);
    channelParams->m_angle.push_back(qdInfo.elAod_rad);

    channelParams->m_generatedTime = Simulator::Now();
    channelParams->m_nodeIds = std::make_pair(aId, bId);

    // Compute alpha and D as described in 3GPP TR 37.885 v15.3.0, Sec. 6.2.3
    // These terms account for an additional Doppler contribution due to the
    // presence of moving objects in the surrounding environment, such as in
    // vehicular scenarios.
    // This contribution is applied only to the delayed (reflected) paths and
    // must be properly configured by setting the value of
    // m_vScatt, which is defined as "maximum speed of the vehicle in the
    // layout".
    // By default, m_vScatt is set to 0, so there is no additional Doppler
    // contribution.

    DoubleVector dopplerTermAlpha;
    DoubleVector dopplerTermD;
    for (uint8_t cIndex = 0; cIndex < H.GetNumPages(); cIndex++)
    {
        // Set the alpha and D as described in 3GPP TR 37.885 v15.3.0, Sec. 6.2.3,
        // both to 0, to avoid introducing additional scatter terms
        dopplerTermAlpha.push_back(0.0);
        dopplerTermD.push_back(0.0);
    }
    channelParams->m_alpha = dopplerTermAlpha;
    channelParams->m_D = dopplerTermD;

    // Store channel parameters
    m_channelParamsMap[channelId] = channelParams;

    return channelMatrix;
}

Ptr<const MatrixBasedChannelModel::ChannelParams>
QdChannelModel::GetParams(Ptr<const MobilityModel> aMob, Ptr<const MobilityModel> bMob) const
{
    NS_LOG_FUNCTION(this);

    // Compute the channel key. The key is reciprocal, i.e., key (a, b) = key (b, a)
    uint64_t channelParamsKey =
        GetKey(aMob->GetObject<Node>()->GetId(), bMob->GetObject<Node>()->GetId());

    if (m_channelParamsMap.find(channelParamsKey) != m_channelParamsMap.end())
    {
        return m_channelParamsMap.find(channelParamsKey)->second;
    }
    else
    {
        NS_LOG_WARN("Channel params map not found. Returning a nullptr.");
        return nullptr;
    }
}

uint64_t
QdChannelModel::GetTimestep(void) const
{
    return GetTimestep(Simulator::Now());
}

uint64_t
QdChannelModel::GetTimestep(Time t) const
{
    NS_LOG_FUNCTION(this << t);
    NS_ASSERT_MSG(m_updatePeriod.GetNanoSeconds() > 0.0,
                  "QdChannelModel update period not set correctly");
    uint64_t timestep = t.GetNanoSeconds() / m_updatePeriod.GetNanoSeconds();
    NS_LOG_DEBUG("t = " << t.GetNanoSeconds() << " ns"
                        << ", updatePeriod = " << m_updatePeriod.GetNanoSeconds() << " ns"
                        << ", timestep = " << timestep);

    NS_ASSERT_MSG(timestep <= m_totTimesteps,
                  "Simulator is running for longer that expected: timestep > m_totTimesteps");

    return timestep;
}

} // namespace ns3
