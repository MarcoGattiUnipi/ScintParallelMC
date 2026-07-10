#include "PhotonSimulation.h"
#include "PhotonSimulationCUDA.cuh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <fstream>
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
    int nEvents = 10;
    int photonsPerEvent = 80000;
    double zCenter = 30.0;

    std::string outputFile =
        "event_batch_gpu_summary.csv";

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
        outputFile = argv[4];
    }

    double zHalfWidth = 3.0;

    double tMin = 0.0;
    double tMax = 500.0;
    int nBins = 500;

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
        static_cast<std::size_t>(nBins);

    std::size_t histBytes =
        histSize * sizeof(unsigned int);

    unsigned int* d_histZ0 = nullptr;
    unsigned int* d_histZL = nullptr;

    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&d_histZ0),
            histBytes
        ),
        "cudaMalloc d_histZ0"
    );

    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&d_histZL),
            histBytes
        ),
        "cudaMalloc d_histZL"
    );

    checkCuda(
        cudaMemset(
            d_histZ0,
            0,
            histBytes
        ),
        "cudaMemset d_histZ0"
    );

    checkCuda(
        cudaMemset(
            d_histZL,
            0,
            histBytes
        ),
        "cudaMemset d_histZL"
    );

    cudaEvent_t start;
    cudaEvent_t stop;

    checkCuda(
        cudaEventCreate(&start),
        "cudaEventCreate start"
    );

    checkCuda(
        cudaEventCreate(&stop),
        "cudaEventCreate stop"
    );

    checkCuda(
        cudaEventRecord(start),
        "cudaEventRecord start"
    );

    launchEventBatchArrivalHistogramKernel(
        d_histZ0,
        d_histZL,
        nEvents,
        photonsPerEvent,
        config,
        zCenter,
        zHalfWidth,
        tMin,
        tMax,
        nBins,
        baseSeed
    );

    checkCuda(
        cudaGetLastError(),
        "launchEventBatchArrivalHistogramKernel"
    );

    checkCuda(
        cudaEventRecord(stop),
        "cudaEventRecord stop"
    );

    checkCuda(
        cudaEventSynchronize(stop),
        "cudaEventSynchronize stop"
    );

    float elapsedMs = 0.0f;

    checkCuda(
        cudaEventElapsedTime(
            &elapsedMs,
            start,
            stop
        ),
        "cudaEventElapsedTime"
    );

    std::vector<unsigned int> h_histZ0(histSize);
    std::vector<unsigned int> h_histZL(histSize);

    checkCuda(
        cudaMemcpy(
            h_histZ0.data(),
            d_histZ0,
            histBytes,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy histZ0 DeviceToHost"
    );

    checkCuda(
        cudaMemcpy(
            h_histZL.data(),
            d_histZL,
            histBytes,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy histZL DeviceToHost"
    );

    checkCuda(
        cudaFree(d_histZ0),
        "cudaFree d_histZ0"
    );

    checkCuda(
        cudaFree(d_histZL),
        "cudaFree d_histZL"
    );

    checkCuda(
        cudaEventDestroy(start),
        "cudaEventDestroy start"
    );

    checkCuda(
        cudaEventDestroy(stop),
        "cudaEventDestroy stop"
    );

    std::vector<unsigned long long> nZ0(nEvents, 0ULL);
    std::vector<unsigned long long> nZL(nEvents, 0ULL);

    std::vector<double> meanTimeZ0(nEvents, 0.0);
    std::vector<double> meanTimeZL(nEvents, 0.0);

    double binWidth =
        (tMax - tMin) / static_cast<double>(nBins);

    unsigned long long totalZ0 = 0ULL;
    unsigned long long totalZL = 0ULL;

    for (int eventId = 0; eventId < nEvents; ++eventId)
    {
        double sumTimeZ0 = 0.0;
        double sumTimeZL = 0.0;

        for (int bin = 0; bin < nBins; ++bin)
        {
            int index =
                eventId * nBins + bin;

            unsigned int countZ0 =
                h_histZ0[index];

            unsigned int countZL =
                h_histZL[index];

            double tCenter =
                tMin +
                (static_cast<double>(bin) + 0.5) *
                binWidth;

            nZ0[eventId] += countZ0;
            nZL[eventId] += countZL;

            sumTimeZ0 +=
                static_cast<double>(countZ0) *
                tCenter;

            sumTimeZL +=
                static_cast<double>(countZL) *
                tCenter;
        }

        totalZ0 += nZ0[eventId];
        totalZL += nZL[eventId];

        if (nZ0[eventId] > 0ULL)
        {
            meanTimeZ0[eventId] =
                sumTimeZ0 /
                static_cast<double>(nZ0[eventId]);
        }

        if (nZL[eventId] > 0ULL)
        {
            meanTimeZL[eventId] =
                sumTimeZL /
                static_cast<double>(nZL[eventId]);
        }
    }

    double elapsedSeconds =
        static_cast<double>(elapsedMs) * 1e-3;

    unsigned long long totalPhotons =
        static_cast<unsigned long long>(nEvents) *
        static_cast<unsigned long long>(photonsPerEvent);

    double photonsPerSecond = 0.0;

    if (elapsedSeconds > 0.0)
    {
        photonsPerSecond =
            static_cast<double>(totalPhotons) /
            elapsedSeconds;
    }

    std::cout << "------ EVENT BATCH GPU TEST ------\n";
    std::cout << "Events              : " << nEvents << "\n";
    std::cout << "Photons/event       : " << photonsPerEvent << "\n";
    std::cout << "Total photons       : " << totalPhotons << "\n";
    std::cout << "z center            : " << zCenter << " cm\n";
    std::cout << "z half width        : " << zHalfWidth << " cm\n";
    std::cout << "Time window         : "
              << tMin << " - " << tMax << " ns\n";
    std::cout << "Bins                : " << nBins << "\n";
    std::cout << "Total detected z=0  : " << totalZ0 << "\n";
    std::cout << "Total detected z=L  : " << totalZL << "\n";
    std::cout << "GPU time            : " << elapsedSeconds << " s\n";
    std::cout << "Photons/s           : " << photonsPerSecond << "\n";
    std::cout << "Output file         : " << outputFile << "\n";
    std::cout << "----------------------------------\n";

    for (int eventId = 0; eventId < nEvents; ++eventId)
    {
        std::cout << "event "
                  << eventId
                  << " z0 = "
                  << nZ0[eventId]
                  << " zL = "
                  << nZL[eventId]
                  << " mean_t_z0 = "
                  << meanTimeZ0[eventId]
                  << " mean_t_zL = "
                  << meanTimeZL[eventId]
                  << "\n";
    }

    std::ofstream fout(outputFile);

    if (!fout)
    {
        std::cerr << "Error: cannot open output file "
                  << outputFile << "\n";

        return 1;
    }

    fout << "event_id,z_center,n_z0,n_zL,mean_time_z0_ns,mean_time_zL_ns\n";

    for (int eventId = 0; eventId < nEvents; ++eventId)
    {
        fout << eventId << ","
             << zCenter << ","
             << nZ0[eventId] << ","
             << nZL[eventId] << ","
             << meanTimeZ0[eventId] << ","
             << meanTimeZL[eventId] << "\n";
    }

    fout.close();

    if (totalZ0 + totalZL == 0ULL)
    {
        std::cerr << "Error: no detected photons in event batch\n";
        return 1;
    }

    return 0;
}