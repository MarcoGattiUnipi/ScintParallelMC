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

    launchDummyPhotonKernel(
        d_results,
        nPhotons
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

    unsigned long long nNumericalError = 0ULL;

    for (unsigned long long i = 0ULL; i < nPhotons; ++i)
    {
        if (h_results[i].status == PHOTON_NUMERICAL_ERROR)
        {
            nNumericalError++;
        }
    }

    std::cout << "Dummy numerical-error results: "
              << nNumericalError
              << "\n";

    std::cout << "-----------------------------------\n";

    if (nNumericalError != nPhotons)
    {
        std::cerr << "Error: dummy kernel did not fill all results\n";
        return 1;
    }

    return 0;
}