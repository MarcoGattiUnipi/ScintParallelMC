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