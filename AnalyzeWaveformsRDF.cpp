#include <ROOT/RDataFrame.hxx>

#include <TCanvas.h>
#include <TFile.h>
#include <TF1.h>
#include <TGraphErrors.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

/*
    RDataFrame analysis of the simulated PMT waveforms.

    This file performs the final physics analysis of the project.

    The input ROOT file is produced by TestScanWaveformsROOT.cpp and contains
    one TTree entry per simulated event. Each event stores the PMT waveforms
    for the two ends of the scintillator bar:

        waveform_z0  -> PMT at z = 0
        waveform_zL  -> PMT at z = L

    The goal of this analysis is to estimate the effective propagation
    velocity of light inside the scintillator bar.

    For each event:
      1. extract a timing value from waveform_z0;
      2. extract a timing value from waveform_zL;
      3. compute DeltaT = tL - t0.

    The dependence of DeltaT on the scintillation position z is then fitted
    with a straight line:

        DeltaT(z) = q + m z

    Assuming PMTs at z = 0 and z = L:

        t0(z) = offset0 + z / v
        tL(z) = offsetL + (L - z) / v

    therefore:

        DeltaT(z) = tL - t0
                  = q - 2 z / v

    and the effective propagation velocity is obtained as:

        v = -2 / m

    where z is expressed in cm and time in ns, giving v in cm/ns.
*/

using Waveform_t = std::vector<float>;

static double gSamplingNs = 0.5; // sampling interval of the PMT waveform in nanoseconds
static double gFraction = 0.20; // fraction of the peak amplitude to use for timing

double FindFractionalTimeNegativePulse( //extract a timing value from a negative-going pulse waveform using fractional threshold
    const Waveform_t& waveform,
    double samplingNs,
    double fraction
)
{
    if (waveform.empty())
    {
        return NAN;
    }

    double maxAbs = 0.0;

    for (float y : waveform)
    {
        double amp = -static_cast<double>(y);

        if (amp > maxAbs)
        {
            maxAbs = amp;
        }
    }

    if (maxAbs <= 0.0)
    {
        return NAN;
    }

    double threshold = fraction * maxAbs;

    for (std::size_t i = 1; i < waveform.size(); ++i)
    {
        double y0 = -static_cast<double>(waveform[i - 1]);
        double y1 = -static_cast<double>(waveform[i]);

        if (y0 < threshold && y1 >= threshold)
        {
            double t0 =
                (static_cast<double>(i - 1) + 0.5) *
                samplingNs;

            double t1 =
                (static_cast<double>(i) + 0.5) *
                samplingNs;

            if (y1 != y0)
            {
                return t0 +
                       (threshold - y0) *
                       (t1 - t0) /
                       (y1 - y0);
            }

            return t1;
        }
    }

    return NAN;
}
/*Various helper for RDF becasue inline functions wouldn't work*/
double RDF_T0Fractional(
    const Waveform_t& waveform
)
{
    return FindFractionalTimeNegativePulse(
        waveform,
        gSamplingNs,
        gFraction
    );
}

double RDF_TLFractional(
    const Waveform_t& waveform
)
{
    return FindFractionalTimeNegativePulse(
        waveform,
        gSamplingNs,
        gFraction
    );
}

double RDF_DeltaT(
    double tL,
    double t0
)
{
    return tL - t0;
}

bool RDF_IsFinite2(
    double a,
    double b
)
{
    return
        std::isfinite(a) &&
        std::isfinite(b);
}

/*
    Main analysis function.

    inputs:
      - filename:
          ROOT file containing the Events TT*ee produced by the simulation
          workflow.

      - barLengthCm:
          length of the scintillator bar in cm.

      - samplingNs:
          sampling step of the stored waveforms in ns.

      - fraction:
          fraction of the maximum pulse amplitude used for timing extraction.

    The function:
      1. opens the Events TTree with ROOT::RDataFrame;
      2. defines new columns t0_frac, tL_frac and delta_t;
      3. fills a TGraphErrors with DeltaT as a function of z_center;
      4. fits the graph with a linear function;
      5. extracts the effective light propagation velocity;
      6. stores the analysis results in a ROOT file and saves a PDF plot.
*/

void AnalyzePropagationVelocityRDF(
    const char* filename = "scan_waveforms.root",
    double barLengthCm = 280.0,
    double samplingNs = 0.5,
    double fraction = 0.20
)
{
    gSamplingNs = samplingNs;
    gFraction = fraction;
/*
    Create an RDataFrame connected to the simulated event tree.

    Each row corresponds to one simulated event at a given z position.
    The waveform branches are processed event by event.
*/
    ROOT::RDataFrame df(
        "Events",
        filename
    );

    auto dfTiming =
        df.Define(
              "t0_frac",
              RDF_T0Fractional,
              {"waveform_z0"}
          )
          .Define(
              "tL_frac",
              RDF_TLFractional,
              {"waveform_zL"}
          )
          .Define(
              "delta_t",
              RDF_DeltaT,
              {"tL_frac", "t0_frac"}
          )
          .Filter(
              RDF_IsFinite2,
              {"t0_frac", "tL_frac"}
          );
/*
    Materialize the z positions and DeltaT values.

    These arrays are used to build a TGraphErrors, which is then fitted with
    a straight line.
*/
    auto zValues =
        dfTiming.Take<double>(
            "z_center"
        );

    auto deltaTValues =
        dfTiming.Take<double>(
            "delta_t"
        );

    const auto& zVec = *zValues;
    const auto& dtVec = *deltaTValues;

    if (zVec.empty())
    {
        std::cerr
            << "Errore: nessun evento valido per il fit DeltaT(z)."
            << std::endl;

        return;
    }
//creating graphs, canvas and saving the results to a ROOT file and a PDF plot
    TGraphErrors* gDeltaT =
        new TGraphErrors();

    for (std::size_t i = 0; i < zVec.size(); ++i)
    {
        gDeltaT->SetPoint(
            static_cast<int>(i),
            zVec[i],
            dtVec[i]
        );

        gDeltaT->SetPointError(
            static_cast<int>(i),
            0.0,
            0.0
        );
    }

    TCanvas* c1 =
        new TCanvas(
            "c_delta_t_vs_z",
            "DeltaT vs z",
            900,
            700
        );

    gDeltaT->SetTitle(
        "#Delta t = t_{L} - t_{0} vs z;z [cm];#Delta t [ns]"
    );

    gDeltaT->SetMarkerStyle(20);
    gDeltaT->SetMarkerSize(0.8);
    gDeltaT->Draw("AP");

    TF1* fit =
        new TF1(
            "fit_delta_t",
            "[0] + [1]*x",
            0.0,
            barLengthCm
        );

    gDeltaT->Fit(
        fit,
        "Q"
    );

    double intercept =
        fit->GetParameter(0);

    double slope =
        fit->GetParameter(1);

    double interceptErr =
        fit->GetParError(0);

    double slopeErr =
        fit->GetParError(1);

    double velocity =
        -2.0 / slope;

    double velocityErr =
        std::abs(
            2.0 * slopeErr /
            (slope * slope)
        );

    std::cout << "\n";
    std::cout << "------ PROPAGATION VELOCITY ------\n";
    std::cout << "Input file       : " << filename << "\n";
    std::cout << "Sampling         : " << samplingNs << " ns\n";
    std::cout << "Fraction         : " << fraction << "\n";
    std::cout << "Fit model        : DeltaT = q + m z\n";
    std::cout << "q                : "
              << intercept
              << " +/- "
              << interceptErr
              << " ns\n";
    std::cout << "m                : "
              << slope
              << " +/- "
              << slopeErr
              << " ns/cm\n";
    std::cout << "v = -2/m         : "
              << velocity
              << " +/- "
              << velocityErr
              << " cm/ns\n";
    std::cout << "c/n naive n=1.57 : "
              << 29.9792458 / 1.57
              << " cm/ns\n";
    std::cout << "----------------------------------\n";
    std::cout << "\n";

    c1->SaveAs(
        "analysis/delta_t_vs_z.pdf"
    );

    std::vector<std::string> snapshotColumns = {
        "global_event_id",
        "position_id",
        "event_id_at_position",
        "z_center",
        "n_z0",
        "n_zL",
        "mean_time_z0_ns",
        "mean_time_zL_ns",
        "t0_frac",
        "tL_frac",
        "delta_t"
    };
    /*
    Store the derived analysis quantities in a separate ROOT file.

    This file contains one entry per event after the timing selection and can
    be used for further checks without recomputing the waveform timing.

    mainly used for debugging and validation of the analysis workflow.
    */
*/
    dfTiming.Snapshot(
        "VelocityAnalysis",
        "analysis/propagation_velocity_analysis.root",
        snapshotColumns
    );

    TFile* fout =
        TFile::Open(
            "analysis/propagation_velocity_plots.root",
            "RECREATE"
        );

    if (fout && !fout->IsZombie())
    {
        gDeltaT->Write(
            "g_delta_t_vs_z"
        );

        fit->Write(
            "fit_delta_t"
        );

        c1->Write();

        fout->Close();
    }

    std::cout << "Output files:\n";
    std::cout << "  delta_t_vs_z.pdf\n";
    std::cout << "  propagation_velocity_analysis.root\n";
    std::cout << "  propagation_velocity_plots.root\n";
}