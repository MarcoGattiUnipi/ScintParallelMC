#pragma once

#include "PhotonSimulation.h"

void launchDummyPhotonKernel(
    PhotonResult* d_results,
    unsigned long long nPhotons
);

void launchPhotonSimulationKernel(
    PhotonResult* d_results,
    unsigned long long nPhotons,
    PhotonConfig config,
    unsigned long long baseSeed
);

void launchEventBatchArrivalHistogramKernel(
    unsigned int* d_histZ0,
    unsigned int* d_histZL,
    int nEvents,
    int photonsPerEvent,
    PhotonConfig config,
    double zCenter,
    double zHalfWidth,
    double tMin,
    double tMax,
    int nBins,
    unsigned long long baseSeed
);