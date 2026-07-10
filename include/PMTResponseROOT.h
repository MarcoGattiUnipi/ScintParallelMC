#pragma once

#include <TH1D.h>
#include <TRandom3.h>

#include <string>

TH1D* SimulatePMTWaveformROOT(
    const TH1D* hPhotons,
    const std::string& waveformName,
    TRandom3& rng,
    double timeWindowNs = 500.0,
    double samplingNs = 0.5
);