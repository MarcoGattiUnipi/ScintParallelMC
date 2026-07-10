#include "PhotonSimulationCUDA.cuh"

#include <cuda_runtime.h>

__global__
void dummyPhotonKernel(
    PhotonResult* results,
    unsigned long long nPhotons
)
{
    unsigned long long tid =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= nPhotons)
    {
        return;
    }

    PhotonResult result;

    result.status = PHOTON_NUMERICAL_ERROR;
    result.detected = false;
    result.detectorFace = Z_NEG;

    result.birthPos = {0.0, 0.0, 0.0};
    result.finalPos = {0.0, 0.0, 0.0};

    result.pathLength = 0.0;
    result.arrivalTime = 0.0;
    result.nBounces = 0;

    results[tid] = result;
}

void launchDummyPhotonKernel(
    PhotonResult* d_results,
    unsigned long long nPhotons
)
{
    constexpr int threadsPerBlock = 256;

    int blocks =
        static_cast<int>(
            (nPhotons + threadsPerBlock - 1) /
            threadsPerBlock
        );

    dummyPhotonKernel<<<blocks, threadsPerBlock>>>(
        d_results,
        nPhotons
    );
}