#include "PhotonSimulation.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

static unsigned long long makePhotonSeed(
    unsigned long long baseSeed,
    unsigned long long photonId
)
{
    unsigned long long x =
        baseSeed + 0x9E3779B97F4A7C15ULL * (photonId + 1ULL);

    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);

    if (x == 0ULL)
    {
        x = 1ULL;
    }

    return x;
}

int main(
    int argc,
    char** argv
)
{
    unsigned long long nPhotons = 100000ULL;

    std::string outputFile = "many_photons_cpu_summary.csv";

    if (argc >= 2)
    {
        nPhotons = std::strtoull(
            argv[1],
            nullptr,
            10
        );
    }

    if (argc >= 3)
    {
        outputFile = argv[2];
    }

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

    std::array<unsigned long long, 6> statusCounts;
    statusCounts.fill(0ULL);

    unsigned long long nDetected = 0ULL;
    unsigned long long nDetectedZ0 = 0ULL;
    unsigned long long nDetectedZL = 0ULL;

    double sumArrivalTime = 0.0;
    double sumArrivalTimeZ0 = 0.0;
    double sumArrivalTimeZL = 0.0;

    double sumPathAll = 0.0;
    double sumBouncesAll = 0.0;

    double sumPathDetected = 0.0;
    double sumBouncesDetected = 0.0;

    auto start = std::chrono::high_resolution_clock::now();

    for (unsigned long long i = 0ULL; i < nPhotons; ++i)
    {
        RNGState rng;
        rng.state = makePhotonSeed(baseSeed, i);

        PhotonResult result = simulatePhotonCPU(
            config,
            rng
        );

        int statusIndex = static_cast<int>(result.status);

        if (
            statusIndex >= 0 &&
            statusIndex < static_cast<int>(statusCounts.size())
        )
        {
            statusCounts[statusIndex]++;
        }

        sumPathAll += result.pathLength;
        sumBouncesAll += static_cast<double>(result.nBounces);

        if (result.detected)
        {
            nDetected++;

            sumArrivalTime += result.arrivalTime;
            sumPathDetected += result.pathLength;
            sumBouncesDetected += static_cast<double>(result.nBounces);

            if (result.detectorFace == Z_NEG)
            {
                nDetectedZ0++;
                sumArrivalTimeZ0 += result.arrivalTime;
            }
            else if (result.detectorFace == Z_POS)
            {
                nDetectedZL++;
                sumArrivalTimeZL += result.arrivalTime;
            }
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = stop - start;

    double elapsedSeconds = elapsed.count();

    double photonsPerSecond = 0.0;

    if (elapsedSeconds > 0.0)
    {
        photonsPerSecond =
            static_cast<double>(nPhotons) / elapsedSeconds;
    }

    double efficiency = NAN;
    double meanArrivalTime = NAN;
    double meanArrivalTimeZ0 = NAN;
    double meanArrivalTimeZL = NAN;
    double meanPathAll = NAN;
    double meanBouncesAll = NAN;
    double meanPathDetected = NAN;
    double meanBouncesDetected = NAN;

    if (nPhotons > 0ULL)
    {
        efficiency =
            static_cast<double>(nDetected) /
            static_cast<double>(nPhotons);

        meanPathAll =
            sumPathAll /
            static_cast<double>(nPhotons);

        meanBouncesAll =
            sumBouncesAll /
            static_cast<double>(nPhotons);
    }

    if (nDetected > 0ULL)
    {
        meanArrivalTime =
            sumArrivalTime /
            static_cast<double>(nDetected);

        meanPathDetected =
            sumPathDetected /
            static_cast<double>(nDetected);

        meanBouncesDetected =
            sumBouncesDetected /
            static_cast<double>(nDetected);
    }

    if (nDetectedZ0 > 0ULL)
    {
        meanArrivalTimeZ0 =
            sumArrivalTimeZ0 /
            static_cast<double>(nDetectedZ0);
    }

    if (nDetectedZL > 0ULL)
    {
        meanArrivalTimeZL =
            sumArrivalTimeZL /
            static_cast<double>(nDetectedZL);
    }

    std::cout << "------ MANY PHOTONS CPU TEST ------\n";

    std::cout << "N photons             : "
              << nPhotons << "\n";

    std::cout << "Detected              : "
              << nDetected << "\n";

    std::cout << "Detected z=0          : "
              << nDetectedZ0 << "\n";

    std::cout << "Detected z=L          : "
              << nDetectedZL << "\n";

    std::cout << "Absorbed wall         : "
              << statusCounts[PHOTON_ABSORBED_WALL] << "\n";

    std::cout << "Absorbed volume       : "
              << statusCounts[PHOTON_ABSORBED_VOLUME] << "\n";

    std::cout << "Max bounces           : "
              << statusCounts[PHOTON_MAX_BOUNCES] << "\n";

    std::cout << "Numerical errors      : "
              << statusCounts[PHOTON_NUMERICAL_ERROR] << "\n";

    std::cout << "Efficiency            : "
              << efficiency << "\n";

    std::cout << "Mean arrival time     : "
              << meanArrivalTime << " ns\n";

    std::cout << "Mean arrival time z=0 : "
              << meanArrivalTimeZ0 << " ns\n";

    std::cout << "Mean arrival time z=L : "
              << meanArrivalTimeZL << " ns\n";

    std::cout << "Mean path all         : "
              << meanPathAll << " cm\n";

    std::cout << "Mean bounces all      : "
              << meanBouncesAll << "\n";

    std::cout << "Mean path detected    : "
              << meanPathDetected << " cm\n";

    std::cout << "Mean bounces detected : "
              << meanBouncesDetected << "\n";

    std::cout << "Elapsed time          : "
              << elapsedSeconds << " s\n";

    std::cout << "Photons per second    : "
              << photonsPerSecond << "\n";

    std::cout << "Output file           : "
              << outputFile << "\n";

    std::cout << "-----------------------------------\n";

    std::ofstream fout(outputFile);

    if (!fout)
    {
        std::cerr << "Error: cannot open output file "
                  << outputFile << "\n";

        return 1;
    }

    fout << "quantity,value\n";

    fout << "n_photons," << nPhotons << "\n";
    fout << "n_detected," << nDetected << "\n";
    fout << "n_detected_z0," << nDetectedZ0 << "\n";
    fout << "n_detected_zL," << nDetectedZL << "\n";

    fout << "n_detected_status_z0,"
         << statusCounts[PHOTON_DETECTED_Z0] << "\n";

    fout << "n_detected_status_zL,"
         << statusCounts[PHOTON_DETECTED_ZL] << "\n";

    fout << "n_absorbed_wall,"
         << statusCounts[PHOTON_ABSORBED_WALL] << "\n";

    fout << "n_absorbed_volume,"
         << statusCounts[PHOTON_ABSORBED_VOLUME] << "\n";

    fout << "n_max_bounces,"
         << statusCounts[PHOTON_MAX_BOUNCES] << "\n";

    fout << "n_numerical_error,"
         << statusCounts[PHOTON_NUMERICAL_ERROR] << "\n";

    fout << "efficiency," << efficiency << "\n";
    fout << "mean_arrival_time_ns," << meanArrivalTime << "\n";
    fout << "mean_arrival_time_z0_ns," << meanArrivalTimeZ0 << "\n";
    fout << "mean_arrival_time_zL_ns," << meanArrivalTimeZL << "\n";

    fout << "mean_path_all_cm," << meanPathAll << "\n";
    fout << "mean_bounces_all," << meanBouncesAll << "\n";

    fout << "mean_path_detected_cm," << meanPathDetected << "\n";
    fout << "mean_bounces_detected," << meanBouncesDetected << "\n";

    fout << "elapsed_time_s," << elapsedSeconds << "\n";
    fout << "photons_per_second," << photonsPerSecond << "\n";

    fout.close();

    return 0;
}