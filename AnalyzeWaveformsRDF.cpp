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

using Waveform_t = std::vector<float>;

static double gSamplingNs = 0.5;
static double gFraction = 0.20;

double FindFractionalTimeNegativePulse(
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

void AnalyzePropagationVelocityRDF(
    const char* filename = "scan_waveforms.root",
    double barLengthCm = 280.0,
    double samplingNs = 0.5,
    double fraction = 0.20
)
{
    gSamplingNs = samplingNs;
    gFraction = fraction;

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