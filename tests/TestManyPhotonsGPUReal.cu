#include "PhotonSimulation.h"
#include "PhotonSimulationCUDA.cuh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
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
    unsigned long long nPhotons = 100000ULL;

    if (argc >= 2)
    {
        nPhotons =
            std::strtoull(
                argv[1],
                nullptr,
                10
            );
    }

    std::cout << "------ MANY PHOTONS GPU TEST ------\n";
    std::cout << "N photons: " << nPhotons << "\n";

    PhotonResult* d_results = nullptr;

    std::size_t bytes =
        static_cast<std::size_t>(nPhotons) *
        sizeof(PhotonResult);

    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&d_results),
            bytes
        ),
        "cudaMalloc d_results"
    );

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

    config.z0 = 27.0;
    config.z1 = 33.0;

    config.max_bounces = 10000;

    const unsigned long long baseSeed = 12345ULL;

    launchPhotonSimulationKernel(
        d_results,
        nPhotons,
        config,
        baseSeed
    );

    checkCuda(
        cudaGetLastError(),
        "kernel launch"
    );

    checkCuda(
        cudaDeviceSynchronize(),
        "cudaDeviceSynchronize"
    );

    std::vector<PhotonResult> h_results(nPhotons);

    checkCuda(
        cudaMemcpy(
            h_results.data(),
            d_results,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy DeviceToHost"
    );

    checkCuda(
        cudaFree(d_results),
        "cudaFree d_results"
    );

    unsigned long long nDetected = 0ULL;
    unsigned long long nDetectedZ0 = 0ULL;
    unsigned long long nDetectedZL = 0ULL;
    unsigned long long nAbsorbedWall = 0ULL;
    unsigned long long nAbsorbedVolume = 0ULL;
    unsigned long long nMaxBounces = 0ULL;
    unsigned long long nNumericalError = 0ULL;

    for (unsigned long long i = 0ULL; i < nPhotons; ++i)
    {
        const PhotonResult& r = h_results[i];

        if (r.status == PHOTON_DETECTED_Z0)
        {
            nDetected++;
            nDetectedZ0++;
        }
        else if (r.status == PHOTON_DETECTED_ZL)
        {
            nDetected++;
            nDetectedZL++;
        }
    
        else if (r.status == PHOTON_ABSORBED_WALL)
        {
            nAbsorbedWall++;
        }
        else if (r.status == PHOTON_ABSORBED_VOLUME)
        {
            nAbsorbedVolume++;
        }
        else if (r.status == PHOTON_MAX_BOUNCES)
        {
            nMaxBounces++;
        }
        else if (r.status == PHOTON_NUMERICAL_ERROR)
        {
        nNumericalError++;
        }
    }

    std::cout << "------ MANY PHOTONS GPU REAL TEST ------\n";
    std::cout << "N photons        : " << nPhotons << "\n";
    std::cout << "Detected         : " << nDetected << "\n";
    std::cout << "Detected z=0     : " << nDetectedZ0 << "\n";
    std::cout << "Detected z=L     : " << nDetectedZL << "\n";
    std::cout << "Absorbed wall    : " << nAbsorbedWall << "\n";
    std::cout << "Absorbed volume  : " << nAbsorbedVolume << "\n";
    std::cout << "Max bounces      : " << nMaxBounces << "\n";
    std::cout << "Numerical errors : " << nNumericalError << "\n";
    std::cout << "----------------------------------------\n";

    if (nNumericalError == nPhotons)
    {
        std::cerr << "Error: all photons ended with numerical error\n";
        return 1;
    }
}