#pragma once

#include "Geometry.h"

enum PhotonStatus
{
    PHOTON_DETECTED_Z0 = 0,
    PHOTON_DETECTED_ZL = 1,
    PHOTON_ABSORBED_WALL = 2,
    PHOTON_ABSORBED_VOLUME = 3,
    PHOTON_MAX_BOUNCES = 4,
    PHOTON_NUMERICAL_ERROR = 5
};

struct PhotonConfig
{
    double reflectivity = 0.98;
    double refractive_index = 1.57;
    double attenuation_length = 210.0;

    double ax = 4.0;
    double ay = 4.0;
    double L  = 280.0;

    double x0 = -2.0;
    double x1 =  2.0;

    double y0 = -2.0;
    double y1 =  2.0;

    double z0 = 27.0;
    double z1 = 33.0;

    int max_bounces = 10000;
};

struct PhotonResult
{
    PhotonStatus status;

    bool detected;

    Face detectorFace;

    Vec3 birthPos;
    Vec3 finalPos;

    double pathLength;
    double arrivalTime;

    int nBounces;
};

struct RNGState
{
    unsigned long long state;
};

double uniform01(
    RNGState& rng
);

double uniform(
    RNGState& rng,
    double a,
    double b
);

PhotonResult simulatePhotonCPU(
    const PhotonConfig& config,
    RNGState& rng
);