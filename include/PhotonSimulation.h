#pragma once

#include "Geometry.h"


/*
    Possible final states of a simulated optical photon.

    These status codes are used both in the CPU and GPU implementations
    to classify the outcome of the photon transport.

    A photon can:
      - be detected on one of the two PMT faces;
      - be absorbed on a lateral wall;
      - be absorbed inside the scintillator bulk;
      - exceed the maximum allowed number of reflections;
      - end in a numerical error state.
*/


enum PhotonStatus
{
    PHOTON_DETECTED_Z0 = 0,
    PHOTON_DETECTED_ZL = 1,
    PHOTON_ABSORBED_WALL = 2,
    PHOTON_ABSORBED_VOLUME = 3,
    PHOTON_MAX_BOUNCES = 4,
    PHOTON_NUMERICAL_ERROR = 5
};



/*Configuration parameters used in the simulation, borrowed form th eold code use for LAB foundamental ineractions. 
It's the BICRON BC - 408 plstic scintillator paired with 2 fast PMT at the ends with typical response feaures.
The reflectivity is the fraction of light that get sent back in the medium if the angle is not critical: only 2% of th ephoton that reach a face
and is not critical angle gets absorbed by the wrapping around the material.*/
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


/*
    Result of the propagation of a single optical photon.

    This structure stores the relevant output quantities for one photon
    after the transport simulation has finished.

    It is used in both CPU and GPU tests. In the full event-based workflow,
    the GPU does not necessarily copy all PhotonResult objects back to the
    CPU; instead, detected photons are accumulated directly into
    event-by-event arrival-time histograms. However, this structure remains
    useful for validation and CPU/GPU comparison tests.

    Stored information:
      - status:
          final classification of the photon;
      - detected:
          true if the photon reached one of the PMT faces;
      - detectorFace:
          Z_NEG for the PMT at z = 0, Z_POS for the PMT at z = L;
      - birthPos:
          initial photon position;
      - finalPos:
          final photon position when the transport terminates;
      - pathLength:
          total optical path length travelled by the photon;
      - arrivalTime:
          arrival time at the detector face, computed as path / (c / n);
      - nBounces:
          number of reflections before termination.
*/


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


/*
    Minimal random-number-generator state.

    The original ROOT macro used TRandom3. For the CUDA implementation,
    ROOT classes cannot be used inside kernels. Therefore a lightweight
    custom RNG state is introduced.

    The same RNG state type is used by the CPU reference implementation
    and by the CUDA device implementation. This makes the CPU/GPU tests
    easier to compare and keeps the simulation logic independent from ROOT.

    The state is updated by a simple deterministic generator implemented
    in PhotonSimulation.cpp for CPU and in PhotonSimulationCUDA.cu for GPU.
*/



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

/*

Simulate the transport of a single optical photon on the CPU.

    This function is the CPU reference version of the photon propagation
    algorithm. It is used for validation, debugging and comparison against
    the CUDA implementation
*/

PhotonResult simulatePhotonCPU(
    const PhotonConfig& config,
    RNGState& rng
);