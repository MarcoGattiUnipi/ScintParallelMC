#include "PhotonSimulation.h"
#include "PhotonSimulationCUDA.cuh"
#include "PMTResponseROOT.h"

#include <cuda_runtime.h>

#include <TFile.h>
#include <TH1D.h>
#include <TTree.h>
#include <TRandom3.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

/*
    Full scan workflow for the scintillator photon Monte Carlo simulation.

    This executable performs the complete production chain used in the
    project:

      1. scan the scintillator bar at different z positions;
      2. for each position, simulate several independent events;
      3. for each event, simulate photonsPerEvent optical photons on the GPU;
      4. fill event-by-event photon arrival-time histograms for the two PMT
         faces, z = 0 and z = L;
      5. copy the histograms back to the CPU;
      6. convert the arrival-time histograms into PMT voltage waveforms
         using ROOT;
      7. store event-level quantities and waveforms in a ROOT TTree.

    The division of responsibilities is intentional:

        GPU:
            optical photon transport and arrival-time histogramming

        CPU/ROOT:
            PMT response, waveform construction and data storage

    This keeps ROOT objects such as TH1D and TTree outside CUDA kernels while
    still accelerating the computationally expensive photon propagation.
*/


/*
    Helper function used to check CUDA runtime calls.

    If a CUDA call fails, the function prints the error message and stops the
    program. This makes debugging easier because the program fails immediately
    at the point where the CUDA error occurs.

    The function is used after memory allocations, memory transfers, event
    operations and kernel launches.
*/


static void checkCuda(
    cudaError_t err,
    const char* message
)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at "
                  << message
                  << ": "
                  << cudaGetErrorString(err)
                  << "\n";

        std::exit(1);
    }
}

static std::string makeName(
    const std::string& prefix,
    int positionId,
    int eventId
)
{
    return prefix +
           "_pos_" +
           std::to_string(positionId) +
           "_event_" +
           std::to_string(eventId);
}

/*
    Default scan configur*tion.

    The command-line arguments can override these values:

        argv[1] -> number of events per z position
        argv[2] -> number of photons per event
        ar*v[3] -> first z position of the scan
        argv[4] -> last z position of the scan
        argv[5] -> z scan step
        argv[6] -> output ROOT filename

    Example:

        ./TestScanWaveformsROOT 10 80000 0.5 279.5 10.0 scan_waveforms.root

    This simulates 10 events per position, 80000 photons per event, scanning
    the full scintillator bar every 10 cm.
*/


int main(
    int argc,
    char** argv
)
{
    int eventsPerPosition = 10;
    int photonsPerEvent = 80000;

    double zStart = 0.5;
    double zEnd = 279.5;
    double zStep = 10.0;

    std::string outputRoot =
        "scan_waveforms.root";

    if (argc >= 2)
    {
        eventsPerPosition = std::atoi(argv[1]);
    }

    if (argc >= 3)
    {
        photonsPerEvent = std::atoi(argv[2]);
    }

    if (argc >= 4)
    {
        zStart = std::atof(argv[3]);
    }

    if (argc >= 5)
    {
        zEnd = std::atof(argv[4]);
    }

    if (argc >= 6)
    {
        zStep = std::atof(argv[5]);
    }

    if (argc >= 7)
    {
        outputRoot = argv[6];
    }

    if (eventsPerPosition <= 0)
    {
        std::cerr << "Error: eventsPerPosition must be positive\n";
        return 1;
    }

    if (photonsPerEvent <= 0)
    {
        std::cerr << "Error: photonsPerEvent must be positive\n";
        return 1;
    }

    if (zStep <= 0.0)
    {
        std::cerr << "Error: zStep must be positive\n";
        return 1;
    }

    double zHalfWidth = 3.0;

    double tMin = 0.0;
    double tMax = 500.0;
    int nArrivalBins = 500;

    double waveformWindowNs = 500.0;
    double waveformSamplingNs = 0.5;

    int nWaveformBins =
        static_cast<int>(
            waveformWindowNs / waveformSamplingNs
        );

    unsigned long long baseSeed = 12345ULL;

    //building the box configuration for the scintillator bar and the PMT faces at the ends of the bar

    PhotonConfig config;

    config.reflectivity = 0.98;
    config.refractive_index = 1.57;
    config.attenuation_length = 210.0;

    config.ax = 4.0;
    config.ay = 4.0;
    config.L  = 280.0;

    config.x0 = -2.0;
    config.x1 =  2.0;

    config.y0 = -2.0;
    config.y1 =  2.0;

    config.z0 = zStart - zHalfWidth;
    config.z1 = zStart + zHalfWidth;

    config.max_bounces = 10000;

    std::size_t histSize =
        static_cast<std::size_t>(eventsPerPosition) *
        static_cast<std::size_t>(nArrivalBins);

    std::size_t histBytes =
        histSize * sizeof(unsigned int);

    unsigned int* dHistZ0 = nullptr;
    unsigned int* dHistZL = nullptr;

    //memory allocation on the GPU for the arrival time histograms of the two PMT faces, z=0 and z=L

    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&dHistZ0),
            histBytes
        ),
        "cudaMalloc dHistZ0"
    );

    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&dHistZL),
            histBytes
        ),
        "cudaMalloc dHistZL"
    );

    std::vector<unsigned int> hHistZ0(histSize);
    std::vector<unsigned int> hHistZL(histSize);

    cudaEvent_t gpuStart;
    cudaEvent_t gpuStop;

    //create CUDA events to measure the GPU execution time of the photon transport kernel

    checkCuda(cudaEventCreate(&gpuStart), "cudaEventCreate gpuStart");
    checkCuda(cudaEventCreate(&gpuStop),  "cudaEventCreate gpuStop");

    TFile* fout =
        TFile::Open(
            outputRoot.c_str(),
            "RECREATE"
        );

    if (!fout || fout->IsZombie())
    {
        std::cerr << "Error: cannot create ROOT file "
                  << outputRoot << "\n";

        return 1;
    }

    /*
    Output event tree.

    Each entry of this TTree corresponds to one simulated event at one scan
    position.

    Stored information includes:
      - event and position identifiers;
      - z position of the scintillation region;
      - number of photons detected by each PMT;
      - mean photon arrival times;
      - binned arrival-time histograms;
      - simulated PMT voltage waveforms;

    The tree is designed to be a*alyzed later with ROOT RDataFrame.*/

    TTree tree(
        "Events",
        "GPU scintillator photon MC scan with CPU ROOT PMT waveforms"
    );

    int globalEventId = 0;
    int positionId = 0;
    int eventIdAtPosition = 0;

    int nZ0 = 0;
    int nZL = 0;

    double zCenterBranch = 0.0;
    double meanTimeZ0 = 0.0;
    double meanTimeZL = 0.0;

    std::vector<float> arrivalHistZ0;
    std::vector<float> arrivalHistZL;
    std::vector<float> waveformZ0;
    std::vector<float> waveformZL;

    tree.Branch("global_event_id", &globalEventId);
    tree.Branch("position_id", &positionId);
    tree.Branch("event_id_at_position", &eventIdAtPosition);
    tree.Branch("z_center", &zCenterBranch);

    tree.Branch("n_z0", &nZ0);
    tree.Branch("n_zL", &nZL);

    tree.Branch("mean_time_z0_ns", &meanTimeZ0);
    tree.Branch("mean_time_zL_ns", &meanTimeZL);

    tree.Branch("arrival_hist_z0", &arrivalHistZ0);
    tree.Branch("arrival_hist_zL", &arrivalHistZL);

    tree.Branch("waveform_z0", &waveformZ0);
    tree.Branch("waveform_zL", &waveformZL);

    TRandom3 pmtRng(987654);

    auto wallStart =
        std::chrono::high_resolution_clock::now();

    double totalGpuTransportSeconds = 0.0;

    std::cout << "------ SCAN WAVEFORM ROOT TEST ------\n";
    std::cout << "Events/position    : " << eventsPerPosition << "\n";
    std::cout << "Photons/event      : " << photonsPerEvent << "\n";
    std::cout << "zStart             : " << zStart << " cm\n";
    std::cout << "zEnd               : " << zEnd << " cm\n";
    std::cout << "zStep              : " << zStep << " cm\n";
    std::cout << "Output ROOT file   : " << outputRoot << "\n";
    std::cout << "-------------------------------------\n";

    /*
    Main scan loop.

    For each z position:
      1. reset the GPU histograms;
      2. launch the CUDA event-batch simulation;
      3. copy the event-by-event arrival histograms back to the CPU;
      4. build ROOT histograms for each event;
      5. simulate the PMT waveform;
      6. fill one entry per event in the output TTree.
*/

    for (
        double zCenter = zStart;
        zCenter <= zEnd + 1e-9;
        zCenter += zStep
    )
    {
        zCenterBranch = zCenter;

        std::cout << "[SCAN] positionId = "
                  << positionId
                  << " zCenter = "
                  << zCenter
                  << " cm\n";

        checkCuda( //reset the GPU histograms to zero before launching the photon transport kernel
            cudaMemset(
                dHistZ0,
                0,
                histBytes
            ),
            "cudaMemset dHistZ0"
        );

        checkCuda(
            cudaMemset(
                dHistZL,
                0,
                histBytes
            ),
            "cudaMemset dHistZL"
        );

        unsigned long long positionSeed =
            baseSeed +
            1000003ULL *
            static_cast<unsigned long long>(positionId + 1);

        checkCuda(
            cudaEventRecord(gpuStart),
            "cudaEventRecord gpuStart"
        );

        launchEventBatchArrivalHistogramKernel( //Launch the main GPU kernel
            dHistZ0,
            dHistZL,
            eventsPerPosition,
            photonsPerEvent,
            config,
            zCenter,
            zHalfWidth,
            tMin,
            tMax,
            nArrivalBins,
            positionSeed
        );
        //routine errors check
        checkCuda(
            cudaGetLastError(),
            "launchEventBatchArrivalHistogramKernel"
        );

        checkCuda(
            cudaEventRecord(gpuStop),
            "cudaEventRecord gpuStop"
        );

        checkCuda(
            cudaEventSynchronize(gpuStop),
            "cudaEventSynchronize gpuStop"
        );

        float elapsedMs = 0.0f;

        checkCuda(
            cudaEventElapsedTime(
                &elapsedMs,
                gpuStart,
                gpuStop
            ),
            "cudaEventElapsedTime"
        );

        totalGpuTransportSeconds +=
            static_cast<double>(elapsedMs) * 1e-3;
        checkCuda(
            cudaMemcpy(//copy the GPU histograms back to the CPU for further processing and waveform simulation
                hHistZ0.data(),
                dHistZ0,
                histBytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy hHistZ0"
        );

        checkCuda(
            cudaMemcpy(
                hHistZL.data(),
                dHistZL,
                histBytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy hHistZL"
        );

        double arrivalBinWidth =
            (tMax - tMin) /
            static_cast<double>(nArrivalBins);

        for (
            eventIdAtPosition = 0;
            eventIdAtPosition < eventsPerPosition;
            ++eventIdAtPosition
        )
        {
            TH1D hArrZ0(//create ROOT histograms for the arrival times at the two PMT faces
                makeName(
                    "h_arr_z0",
                    positionId,
                    eventIdAtPosition
                ).c_str(),
                "Arrival times z=0;Time [ns];Photons",
                nArrivalBins,
                tMin,
                tMax
            );

            TH1D hArrZL(
                makeName(
                    "h_arr_zL",
                    positionId,
                    eventIdAtPosition
                ).c_str(),
                "Arrival times z=L;Time [ns];Photons",
                nArrivalBins,
                tMin,
                tMax
            );

            nZ0 = 0;
            nZL = 0;

            double sumTimeZ0 = 0.0;
            double sumTimeZL = 0.0;

            arrivalHistZ0.clear();
            arrivalHistZL.clear();

            arrivalHistZ0.reserve(nArrivalBins);
            arrivalHistZL.reserve(nArrivalBins);

            for (int bin = 0; bin < nArrivalBins; ++bin) //loop over the arrival time bins to fill the ROOT histograms and compute the mean arrival times
            {
                int index =
                    eventIdAtPosition * nArrivalBins + bin;

                unsigned int countZ0 =
                    hHistZ0[index];

                unsigned int countZL =
                    hHistZL[index];

                hArrZ0.SetBinContent(
                    bin + 1,
                    static_cast<double>(countZ0)
                );

                hArrZL.SetBinContent(
                    bin + 1,
                    static_cast<double>(countZL)
                );

                arrivalHistZ0.push_back(
                    static_cast<float>(countZ0)
                );

                arrivalHistZL.push_back(
                    static_cast<float>(countZL)
                );

                double tCenter =
                    tMin +
                    (static_cast<double>(bin) + 0.5) *
                    arrivalBinWidth;

                nZ0 += static_cast<int>(countZ0);
                nZL += static_cast<int>(countZL);

                sumTimeZ0 +=
                    static_cast<double>(countZ0) *
                    tCenter;

                sumTimeZL +=
                    static_cast<double>(countZL) *
                    tCenter;
            }

            meanTimeZ0 =
                nZ0 > 0
                    ? sumTimeZ0 / static_cast<double>(nZ0)
                    : 0.0;

            meanTimeZL =
                nZL > 0
                    ? sumTimeZL / static_cast<double>(nZL)
                    : 0.0;

            std::string waveNameZ0 =
                makeName(
                    "h_wave_z0",
                    positionId,
                    eventIdAtPosition
                );

            std::string waveNameZL =
                makeName(
                    "h_wave_zL",
                    positionId,
                    eventIdAtPosition
                );

            TH1D* hWaveZ0 =
                SimulatePMTWaveformROOT(//simulate the PMT voltage waveform from the arrival time histogram
                    &hArrZ0,
                    waveNameZ0,
                    pmtRng,
                    waveformWindowNs,
                    waveformSamplingNs
                );

            TH1D* hWaveZL =
                SimulatePMTWaveformROOT(
                    &hArrZL,
                    waveNameZL,
                    pmtRng,
                    waveformWindowNs,
                    waveformSamplingNs
                );

            waveformZ0.clear();
            waveformZL.clear();

            waveformZ0.reserve(nWaveformBins);//reserve space in the vectors to avoid reallocations during the loop
            waveformZL.reserve(nWaveformBins);

            for (int bin = 1; bin <= nWaveformBins; ++bin)
            {//loop over the waveform bins to fill the vectors with the simulated PMT voltage waveforms
                waveformZ0.push_back(
                    static_cast<float>(
                        hWaveZ0->GetBinContent(bin)
                    )
                );

                waveformZL.push_back(
                    static_cast<float>(
                        hWaveZL->GetBinContent(bin)
                    )
                );
            }

            tree.Fill();//fill the TTree with the event-level information and the simulated waveforms

            if (
                positionId < 2 &&
                eventIdAtPosition < 2
            )
            {
                hArrZ0.Write();
                hArrZL.Write();
                hWaveZ0->Write();
                hWaveZL->Write();
            }

            delete hWaveZ0;
            delete hWaveZL;

            globalEventId++;
        }

        positionId++;
    }

    tree.Write();

    fout->Close();

    checkCuda(cudaEventDestroy(gpuStart), "cudaEventDestroy gpuStart");
    checkCuda(cudaEventDestroy(gpuStop),  "cudaEventDestroy gpuStop");

    checkCuda(cudaFree(dHistZ0), "cudaFree dHistZ0");
    checkCuda(cudaFree(dHistZL), "cudaFree dHistZL");

    auto wallStop =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> wallElapsed =
        wallStop - wallStart;

    std::cout << "------ SCAN COMPLETED ------\n";
    std::cout << "Total events written      : " << globalEventId << "\n";
    std::cout << "Number of positions       : " << positionId << "\n";
    std::cout << "Total GPU transport time  : "
              << totalGpuTransportSeconds
              << " s\n";
    std::cout << "Total wall time           : "
              << wallElapsed.count()
              << " s\n";
    std::cout << "Output ROOT file          : "
              << outputRoot
              << "\n";
    std::cout << "Tree name                 : Events\n";
    std::cout << "----------------------------\n";

    return 0;
}