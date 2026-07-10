
#include "PhotonSimulation.h"
#include "PhotonSimulationCUDA.cuh"
#include "PMTResponseROOT.h"

#include <cuda_runtime.h>

#include <TFile.h>
#include <TH1D.h>
#include <TTree.h>
#include <TRandom3.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

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

int main(
    int argc,
    char** argv
)
{
    int nEvents = 2;
    int photonsPerEvent = 80000;
    double zCenter = 30.0;

    std::string outputRoot =
        "event_waveforms.root";

    if (argc >= 2)
    {
        nEvents = std::atoi(argv[1]);
    }

    if (argc >= 3)
    {
        photonsPerEvent = std::atoi(argv[2]);
    }

    if (argc >= 4)
    {
        zCenter = std::atof(argv[3]);
    }

    if (argc >= 5)
    {
        outputRoot = argv[4];
    }

    double zHalfWidth = 3.0;

    double tMin = 0.0;
    double tMax = 500.0;
    int nArrivalBins = 500;

    double waveformWindowNs = 500.0;
    double waveformSamplingNs = 0.5;

    unsigned long long baseSeed = 12345ULL;

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

    config.z0 = zCenter - zHalfWidth;
    config.z1 = zCenter + zHalfWidth;

    config.max_bounces = 10000;

    std::size_t histSize =
        static_cast<std::size_t>(nEvents) *
        static_cast<std::size_t>(nArrivalBins);

    std::size_t histBytes =
        histSize * sizeof(unsigned int);

    unsigned int* dHistZ0 = nullptr;
    unsigned int* dHistZL = nullptr;

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

    checkCuda(
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

    cudaEvent_t start;
    cudaEvent_t stop;

    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop),  "cudaEventCreate stop");

    checkCuda(cudaEventRecord(start), "cudaEventRecord start");

    launchEventBatchArrivalHistogramKernel(
        dHistZ0,
        dHistZL,
        nEvents,
        photonsPerEvent,
        config,
        zCenter,
        zHalfWidth,
        tMin,
        tMax,
        nArrivalBins,
        baseSeed
    );

    checkCuda(
        cudaGetLastError(),
        "launchEventBatchArrivalHistogramKernel"
    );

    checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float elapsedMs = 0.0f;

    checkCuda(
        cudaEventElapsedTime(
            &elapsedMs,
            start,
            stop
        ),
        "cudaEventElapsedTime"
    );

    std::vector<unsigned int> hHistZ0(histSize);
    std::vector<unsigned int> hHistZL(histSize);

    checkCuda(
        cudaMemcpy(
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

    checkCuda(cudaFree(dHistZ0), "cudaFree dHistZ0");
    checkCuda(cudaFree(dHistZL), "cudaFree dHistZL");

    checkCuda(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCuda(cudaEventDestroy(stop),  "cudaEventDestroy stop");

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

    TTree tree(
        "Events",
        "GPU photon MC events with CPU ROOT PMT waveforms"
    );

    int eventId = 0;
    int nZ0 = 0;
    int nZL = 0;

    double zCenterBranch = zCenter;
    double meanTimeZ0 = 0.0;
    double meanTimeZL = 0.0;

    std::vector<float> arrivalHistZ0;
    std::vector<float> arrivalHistZL;
    std::vector<float> waveformZ0;
    std::vector<float> waveformZL;

    tree.Branch("event_id", &eventId);
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

    int nWaveformBins =
        static_cast<int>(
            waveformWindowNs / waveformSamplingNs
        );

    for (eventId = 0; eventId < nEvents; ++eventId)
    {
        TH1D hArrZ0(
            Form("h_arr_z0_event_%d", eventId),
            "Arrival times z=0;Time [ns];Photons",
            nArrivalBins,
            tMin,
            tMax
        );

        TH1D hArrZL(
            Form("h_arr_zL_event_%d", eventId),
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

        double arrivalBinWidth =
            (tMax - tMin) /
            static_cast<double>(nArrivalBins);

        for (int bin = 0; bin < nArrivalBins; ++bin)
        {
            int index =
                eventId * nArrivalBins + bin;

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

        TH1D* hWaveZ0 =
            SimulatePMTWaveformROOT(
                &hArrZ0,
                Form("h_wave_z0_event_%d", eventId),
                pmtRng,
                waveformWindowNs,
                waveformSamplingNs
            );

        TH1D* hWaveZL =
            SimulatePMTWaveformROOT(
                &hArrZL,
                Form("h_wave_zL_event_%d", eventId),
                pmtRng,
                waveformWindowNs,
                waveformSamplingNs
            );

        waveformZ0.clear();
        waveformZL.clear();

        waveformZ0.reserve(nWaveformBins);
        waveformZL.reserve(nWaveformBins);

        for (int bin = 1; bin <= nWaveformBins; ++bin)
        {
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

        tree.Fill();

        if (eventId < 5)
        {
            hArrZ0.Write();
            hArrZL.Write();
            hWaveZ0->Write();
            hWaveZL->Write();
        }

        delete hWaveZ0;
        delete hWaveZL;
    }

    tree.Write();

    fout->Close();

    double elapsedSeconds =
        static_cast<double>(elapsedMs) * 1e-3;

    unsigned long long totalPhotons =
        static_cast<unsigned long long>(nEvents) *
        static_cast<unsigned long long>(photonsPerEvent);

    std::cout << "------ EVENT WAVEFORM ROOT TEST ------\n";
    std::cout << "Events              : " << nEvents << "\n";
    std::cout << "Photons/event       : " << photonsPerEvent << "\n";
    std::cout << "Total photons       : " << totalPhotons << "\n";
    std::cout << "z center            : " << zCenter << " cm\n";
    std::cout << "GPU transport time  : " << elapsedSeconds << " s\n";
    std::cout << "Output ROOT file    : " << outputRoot << "\n";
    std::cout << "Tree name           : Events\n";
    std::cout << "--------------------------------------\n";

    return 0;
}
