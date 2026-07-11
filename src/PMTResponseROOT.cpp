#include "PMTResponseROOT.h"

#include <TMath.h>

#include <cmath>

namespace
{
    constexpr double E_CHARGE     = 1.602e-19; // elementary charge in Coulombs
    constexpr double GAIN         = 5.0e5;// typical PMT gain, unitless
    constexpr double Q_MEAN_PC    = GAIN * E_CHARGE * 1.0e12;// mean charge per photoelectron in picoCoulombs

    constexpr double TAU_RISE     = 3.5;
    constexpr double IMPEDANCE    = 50.0;

    constexpr double QE           = 0.28;//quantity of photoelectrons produced per incident photon  
    constexpr double GAIN_RES     = 0.20;//relative standard deviation of the PMT gain

    constexpr double TRANSIT_TIME = 35.0;//transit time of the PMT in nanoseconds
    constexpr double TTS_SIGMA    = 1.5;
}


/*
    Convert a photon arrival-time histogram into a PMT voltage waveform.

    For each time bin of the input histogram:
      1. read the number of photons arriving at the photocathode;
      2. loop over the photons in that bin;
      3. apply the quantum efficiency;
      4. sample the PMT gain fluctuation;
      5. sample transit time and transit time spread;
      6. add the corresponding single-photoelectron response to the
         output waveform. Exp(-t^2/(2*tau^2)) is used for the single-photoelectron response.
*/


TH1D* SimulatePMTWaveformROOT(
    const TH1D* hPhotons,
    const std::string& waveformName,
    TRandom3& rng,
    double timeWindowNs,
    double samplingNs
)
{
    int nBinsOut =
        static_cast<int>(
            timeWindowNs / samplingNs
        );

    TH1D* hWaveform =
        new TH1D(
            waveformName.c_str(),
            "PMT waveform;Time [ns];Voltage [mV]",
            nBinsOut,
            0.0,
            timeWindowNs
        );

    double gainSigma =
        GAIN_RES * Q_MEAN_PC;

    int nPhotonBins =
        hPhotons->GetNbinsX();

    for (int b = 1; b <= nPhotonBins; ++b)
    {
        double tArrival =
            hPhotons->GetBinCenter(b);

        int nPhotons =
            static_cast<int>(
                hPhotons->GetBinContent(b)
            );

        if (nPhotons <= 0)
        {
            continue;
        }

        for (int p = 0; p < nPhotons; ++p)
        {
            if (rng.Uniform() > QE * 0.8)
            {
                continue;
            }

            double qActual =
                rng.Gaus(
                    Q_MEAN_PC,
                    gainSigma
                );

            if (qActual < 0.0)
            {
                qActual = 0.0;
            }

            double tAnode =
                rng.Gaus(
                    tArrival + TRANSIT_TIME,
                    TTS_SIGMA
                );

            if (tAnode >= timeWindowNs)
            {
                continue;
            }

            double A =
                qActual /
                (
                    TAU_RISE *
                    std::sqrt(2.0 * TMath::Pi())
                );

            for (int i = 1; i <= nBinsOut; ++i)
            {
                double t =
                    hWaveform->GetBinCenter(i);

                if (t > tAnode)
                {
                    double dt =
                        t - tAnode;

                    double currentMA =
                        -A *
                        std::exp(
                            -dt * dt /
                            (2.0 * TAU_RISE * TAU_RISE)
                        );

                    double voltageMV =
                        currentMA * IMPEDANCE;

                    hWaveform->AddBinContent(
                        i,
                        voltageMV
                    );
                }
            }
        }
    }

    return hWaveform;
}