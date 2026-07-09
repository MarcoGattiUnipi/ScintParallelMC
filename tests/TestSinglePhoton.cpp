#include "PhotonSimulation.h"

#include <iostream>
#include <cmath>

int main()
{
    PhotonConfig config;

    config.reflectivity = 0.98;
    config.refractive_index = 1.57;
    config.attenuation_length = 210.0;

    config.ax = 4.0;
    config.ay = 4.0;
    config.L = 280.0;

    config.x0 = -2.0;
    config.x1 =  2.0;

    config.y0 = -2.0;
    config.y1 =  2.0;

    config.z0 = 27.0;
    config.z1 = 33.0;

    config.max_bounces = 10000;

    
    RNGState rng;
    rng.state = 12345ULL;


    PhotonResult result =
        simulatePhotonCPU(
            config,
            rng
        );

    std::cout << "Status       : " << result.status << "\n";
    std::cout << "Detected     : " << result.detected << "\n";
    std::cout << "Detector face: " << result.detectorFace << "\n";
    std::cout << "Birth pos    : "
              << result.birthPos.x << " "
              << result.birthPos.y << " "
              << result.birthPos.z << "\n";

    std::cout << "Final pos    : "
              << result.finalPos.x << " "
              << result.finalPos.y << " "
              << result.finalPos.z << "\n";

    std::cout << "Path length  : " << result.pathLength << " cm\n";
    std::cout << "Arrival time : " << result.arrivalTime << " ns\n";
    std::cout << "Bounces      : " << result.nBounces << "\n";

    if (result.pathLength < 0.0)
    {
        std::cerr << "Error: negative path length\n";
        return 1;
    }

    if (result.detected)
    {
        if (
            result.detectorFace != Z_NEG &&
            result.detectorFace != Z_POS
        )
        {
            std::cerr << "Error: detected photon on non-z face\n";
            return 1;
        }

        if (!std::isfinite(result.arrivalTime))
        {
            std::cerr << "Error: non-finite arrival time\n";
            return 1;
        }
    }

    return 0;
}