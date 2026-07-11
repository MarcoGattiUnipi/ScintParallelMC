#include "PhotonSimulationCUDA.cuh"

#include <cuda_runtime.h>

/*
    CUDA implementation of the optical photon transport.

    This file contains the GPU-side version of the photon propagation
    algorithm. The main design choice is:

        one CUDA thread = one optical photon

    Each thread independently samples the photon initial conditions,
    propagates the photon inside the scintillator box, and either detects
    it on one of the PMT faces or terminates it because of absorption or
    numerical limits.

    Two GPU workflows are implemented:

      1. PhotonResult workflow:
           each thread writes one PhotonResult object.
           This is mainly used for validation and CPU/GPU comparison.

      2. Event-batch histogram workflow:
           many events are simulated at the same z position.
           Each event contains photonsPerEvent photons.
           Detected photons are accumulated directly into arrival-time
           histograms indexed by event and time bin.

    ROOT classes are intentionally not used in this file. ROOT is only used
    on the CPU side to build TH1D objects, PMT waveforms and TTrees.
*/


//this is a dummy kernel that can be used to test the GPU setup and the memory transfer of the PhotonResult array
__global__
void dummyPhotonKernel(
    PhotonResult* results,
    unsigned long long nPhotons
)
{
    unsigned long long tid =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= nPhotons)
    {
        return;
    }

    PhotonResult result;

    result.status = PHOTON_NUMERICAL_ERROR;
    result.detected = false;
    result.detectorFace = Z_NEG;

    result.birthPos = {0.0, 0.0, 0.0};
    result.finalPos = {0.0, 0.0, 0.0};

    result.pathLength = 0.0;
    result.arrivalTime = 0.0;
    result.nBounces = 0;

    results[tid] = result;
}

void launchDummyPhotonKernel(
    PhotonResult* d_results,
    unsigned long long nPhotons
)
{
    constexpr int threadsPerBlock = 256;

    int blocks =
        static_cast<int>(
            (nPhotons + threadsPerBlock - 1) /
            threadsPerBlock
        );

    dummyPhotonKernel<<<blocks, threadsPerBlock>>>(
        d_results,
        nPhotons
    );
}
//this function is used to generate a unique seed for each photon based on the base seed and the photon id. 
//It uses a simple hash function to ensure that the seeds are well distributed and not correlated. 
//The result is guaranteed to be non-zero, as a zero seed would lead to a degenerate RNG state.
__device__
unsigned long long makePhotonSeedDevice(
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
//this function is used to generate a uniform random number in the range [0, 1) using a simple linear congruential generator (LCG).
//The LCG parameters are chosen to have a long period and good statistical properties.
__device__
double uniform01Device(
    RNGState& rng
)
{
    rng.state =
        2862933555777941757ULL * rng.state
        + 3037000493ULL;

    unsigned long long x =
        rng.state >> 11;

    return static_cast<double>(x) *
           (1.0 / 9007199254740992.0);
}

__device__
double uniformDevice(
    RNGState& rng,
    double a,
    double b
)
{
    return a + (b - a) * uniform01Device(rng);
}
/*
Compute then nxt intersection of a ray and the edge of the box.
the ray is defined by its position and direction, and the box is defined by its minimum and maximum coordinates.
The function returns a StepResult struct that contains the distance t to the intersection point and the face of the box that was hit. 
If the ray does not intersect the box, t is set to a large value (1e300) and the face is set to Z_POS by default.
For each axis the function computes the distance to the correspong face of the box. Finds the smallest distance for the direction it has 
and returns the pathLenght and the face of the box that was hit. 

*/
__device__
StepResult nextIntersectionDevice(
    const Vec3& pos,
    const Vec3& dir,
    const Box& b
)
{
    const double INF = 1e300;

    double tmin = INF;
    Face face = Z_POS;

    if (dir.x > 0.0)
    {
        double t = (b.xmax - pos.x) / dir.x;

        if (t > 1e-12 && t < tmin)
        {
            tmin = t;
            face = X_POS;
        }
    }
    else if (dir.x < 0.0)
    {
        double t = (b.xmin - pos.x) / dir.x;

        if (t > 1e-12 && t < tmin)
        {
            tmin = t;
            face = X_NEG;
        }
    }

    if (dir.y > 0.0)
    {
        double t = (b.ymax - pos.y) / dir.y;

        if (t > 1e-12 && t < tmin)
        {
            tmin = t;
            face = Y_POS;
        }
    }
    else if (dir.y < 0.0)
    {
        double t = (b.ymin - pos.y) / dir.y;

        if (t > 1e-12 && t < tmin)
        {
            tmin = t;
            face = Y_NEG;
        }
    }

    if (dir.z > 0.0)
    {
        double t = (b.zmax - pos.z) / dir.z;

        if (t > 1e-12 && t < tmin)
        {
            tmin = t;
            face = Z_POS;
        }
    }
    else if (dir.z < 0.0)
    {
        double t = (b.zmin - pos.z) / dir.z;

        if (t > 1e-12 && t < tmin)
        {
            tmin = t;
            face = Z_NEG;
        }
    }

    StepResult result;
    result.t = tmin;
    result.face = face;

    return result;
}


/*
    Compute the specular reflection of a photon direction on a box face.

    The reflected direction is computed using:

        r = d - 2 (d · n) n

    where:
      - d is the normalized incoming direction;
      - n is the outward normal of the hit face.

    The result is normalized before being returned. This function is used
    for reflections on lateral walls.
*/


__device__
Vec3 reflectDevice(
    const Vec3& dir,
    Face f
)
{
    Vec3 n;
    n.x = 0.0;
    n.y = 0.0;
    n.z = 0.0;

    switch (f)
    {
        case X_NEG:
            n.x = -1.0;
            break;

        case X_POS:
            n.x = 1.0;
            break;

        case Y_NEG:
            n.y = -1.0;
            break;

        case Y_POS:
            n.y = 1.0;
            break;

        case Z_NEG:
            n.z = -1.0;
            break;

        case Z_POS:
            n.z = 1.0;
            break;
    }

    Vec3 d = Normalize(dir);

    double proj = Dot(d, n);

    Vec3 r;
    r.x = d.x - 2.0 * proj * n.x;
    r.y = d.y - 2.0 * proj * n.y;
    r.z = d.z - 2.0 * proj * n.z;

    return Normalize(r);
}
//just a simple device function to generate a random isotropic direction in 3D space.
__device__
Vec3 isotropicDirDevice(
    double cosTheta,
    double phi
)
{
    double s =
        sqrt(fmax(0.0, 1.0 - cosTheta * cosTheta));

    Vec3 dir;
    dir.x = s * cos(phi);
    dir.y = s * sin(phi);
    dir.z = cosTheta;

    return dir;
}


/*
    Compute the incidence angle between the photon direction and a box face.

    The angle is measured with respect to the normal of the face and returned
    in degrees.

    The absolute value of the dot product is used because the relevant
    quantity is the angle between the direction and the surface normal,
    independent of the sign convention of the normal.

    This angle is used to decide whether total internal reflection should
    occur.
*/\

__device__
double incidenceAngleDegDevice(
    const Vec3& dir,
    Face f
)
{
    Vec3 n;
    n.x = 0.0;
    n.y = 0.0;
    n.z = 0.0;

    switch (f)
    {
        case X_NEG:
            n.x = -1.0;
            break;

        case X_POS:
            n.x = 1.0;
            break;

        case Y_NEG:
            n.y = -1.0;
            break;

        case Y_POS:
            n.y = 1.0;
            break;

        case Z_NEG:
            n.z = -1.0;
            break;

        case Z_POS:
            n.z = 1.0;
            break;
    }

    Vec3 d = Normalize(dir);

    double c = fabs(Dot(d, n));

    c = fmax(-1.0, fmin(1.0, c));

    return acos(c) * 180.0 / PI;
}

/*
    Compute the critical angle*for total internal reflection.

    The formula is:

        theta_c * asin(n_out / n_in)

    for n_in * n_out.

    If n_in <= n_out, tot*l internal reflection cannot occur*and the
    function returns 90 de*rees.

    The returned angle is e*pressed in degrees.
*/

__device__
double criticalAngleDegDevice(
    double n_in,
    double n_out
)
{
    if (n_in <= n_out)
    {
        return 90.0;
    }

    double ratio = n_out / n_in;

    ratio = fmax(0.0, fmin(1.0, ratio));

    return asin(ratio) * 180.0 / PI;
}

/*Main function of this simulation Does amny things:
    1. Build the scintillator box from PhotonConfig parameters
    2. Sample the initial position and direction of the photon
    3. Propagate the photon until it is detected or absorbed
    4. Return a PhotonResult struct with the final state of the photon
*/

__device__
PhotonResult simulatePhotonDevice(
    const PhotonConfig& config,
    RNGState& rng
)
{
    PhotonResult result;

    result.status = PHOTON_NUMERICAL_ERROR;
    result.detected = false;
    result.detectorFace = Z_NEG;

    result.birthPos = {0.0, 0.0, 0.0};
    result.finalPos = {0.0, 0.0, 0.0};

    result.pathLength = 0.0;
    result.arrivalTime = 0.0;
    result.nBounces = 0;

    Box box; //building the box

    box.xmin = -config.ax / 2.0;
    box.xmax =  config.ax / 2.0;

    box.ymin = -config.ay / 2.0;
    box.ymax =  config.ay / 2.0;

    box.zmin = 0.0;
    box.zmax = config.L;

    double x0 = fmax(config.x0, box.xmin);
    double x1 = fmin(config.x1, box.xmax);

    double y0 = fmax(config.y0, box.ymin);
    double y1 = fmin(config.y1, box.ymax);

    double z0 = fmax(config.z0, box.zmin + 1e-9);
    double z1 = fmin(config.z1, box.zmax - 1e-9);

    if (x0 >= x1 || y0 >= y1 || z0 >= z1)
    {
        result.status = PHOTON_NUMERICAL_ERROR;
        return result;
    }

    Vec3 pos;

    pos.x = uniformDevice(rng, x0, x1);
    pos.y = uniformDevice(rng, y0, y1); // uniform sampling of the initial position within the allowed ranges
    pos.z = uniformDevice(rng, z0, z1);

    result.birthPos = pos;

    double cosTheta = uniformDevice(rng, -1.0, 1.0);
    double phi = uniformDevice(rng, 0.0, 2.0 * PI);

    Vec3 dir = isotropicDirDevice(cosTheta, phi);

    const double n_out_lateral = 1.0;
    const double n_out_end = 1.0;
    //calculate the critical angles for total internal reflection at the lateral and end faces of the scintillator box.
    //the end and lateral index are kept differetn for coupling factors with the PMT and the wrapping here i assumed to be air in both cases. The critical angle is used to determine whether a photon will be reflected or transmitted when it hits a face of the box.
    double angle_limit_lateral_deg =
        criticalAngleDegDevice(
            config.refractive_index,
            n_out_lateral
        );

    double angle_limit_end_deg =
        criticalAngleDegDevice(
            config.refractive_index,
            n_out_end
        );

    bool ignore_next_z_tir = false;

    int nb = 0;
    double path = 0.0;

    while (true)
    {
        StepResult step =
            nextIntersectionDevice( //calculate next intersection of the photon with the box
                pos,
                dir,
                box
            );

        if (!(step.t < 1e290)) //check if the step length is finite, if not return a numerical error
        {
            result.status = PHOTON_NUMERICAL_ERROR;
            result.detected = false;
            result.pathLength = path;
            result.nBounces = nb;
            result.finalPos = pos;

            return result;
        }

        Vec3 newpos = //calculate the new position of the photon after the step
            Add(
                pos,
                Scale(dir, step.t)
            );

        path += step.t;

        if (config.attenuation_length > 0.0)    //if the attenuation length is positive, calculate the probability of survival of the photon after the step and check if it is absorbed in the volume
        {
            double surviveProb =
                exp(
                    -step.t / config.attenuation_length
                );

            if (uniform01Device(rng) > surviveProb)
            {
                result.status = PHOTON_ABSORBED_VOLUME;
                result.detected = false;
                result.pathLength = path;
                result.nBounces = nb;
                result.finalPos = newpos;

                return result;
            }
        }

        if (step.face == Z_NEG || step.face == Z_POS)
        {
            double incAngle =
                incidenceAngleDegDevice(    //calculate the incidence angle of the photon on the face it hit
                    dir,
                    step.face
                );

            bool forceReflect =
                !ignore_next_z_tir &&
                incAngle > angle_limit_end_deg;

            if (forceReflect)
            {
                dir.z = -dir.z;

                pos =
                    Add(
                        newpos,
                        Scale(dir, 1e-9)
                    );

                nb++;

                ignore_next_z_tir = true;

                if (nb >= config.max_bounces) //check if the number of bounces exceeds the maximum allowed, if so return a max bounces status
                {
                    result.status = PHOTON_MAX_BOUNCES;
                    result.detected = false;
                    result.pathLength = path;
                    result.nBounces = nb;
                    result.finalPos = pos;

                    return result;
                }
            }
            else //if the photon is not forced to reflect, it is detected and the result is filled with the appropriate values
            {
                const double c_cm_ns = 29.9792458;

                double c_material =
                    c_cm_ns / config.refractive_index;

                result.detected = true;
                result.detectorFace = step.face;
                result.finalPos = newpos;
                result.pathLength = path;
                result.arrivalTime = path / c_material;
                result.nBounces = nb;

                if (step.face == Z_NEG)
                {
                    result.status = PHOTON_DETECTED_Z0;
                }
                else
                {
                    result.status = PHOTON_DETECTED_ZL;
                }

                return result;
            }
        }
        else//if the photon hit a lateral face, check if it is reflected or absorbed based on the incidence angle and the reflectivity of the material
        {
            double incAngle =
                incidenceAngleDegDevice(
                    dir,
                    step.face
                );

            bool forceReflect =
                angle_limit_lateral_deg > 0.0 &&
                incAngle > angle_limit_lateral_deg;

            if (
                forceReflect ||
                uniform01Device(rng) <= config.reflectivity
            )
            {
                dir =
                    reflectDevice(
                        dir,
                        step.face
                    );

                pos =
                    Add(
                        newpos,
                        Scale(dir, 1e-9)
                    );

                nb++;

                if (nb >= config.max_bounces)
                {
                    result.status = PHOTON_MAX_BOUNCES;
                    result.detected = false;
                    result.pathLength = path;
                    result.nBounces = nb;
                    result.finalPos = pos;

                    return result;
                }
            }
            else
            {
                result.status = PHOTON_ABSORBED_WALL;
                result.detected = false;
                result.pathLength = path;
                result.nBounces = nb;
                result.finalPos = newpos;

                return result;
            }
        }
    }
}

/*
    Kernel that simulates many independent photons.

    Mapping:

        one CUDA thread = one photon

    Each thread:
      - computes its global photon index;
      - initializes an independent RNG state;
      - calls simulatePhotonDevice();
      - stores the resulting PhotonResult in global memory.

    This kernel is mainly used for validation, debugging and direct CPU/GPU
    comparison, because it copies one PhotonResult per photon back to the CPU.
*/
__global__
void simulatePhotonsKernel(
    PhotonResult* results,
    unsigned long long nPhotons,
    PhotonConfig config,
    unsigned long long baseSeed
)
{
    unsigned long long tid =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= nPhotons)
    {
        return;
    }

    RNGState rng;
    rng.state =
        makePhotonSeedDevice(
            baseSeed,
            tid
        );

    PhotonResult result =
        simulatePhotonDevice(
            config,
            rng
        );

    results[tid] = result;
}
/*
    Host-side launcher for simulatePhotonsKernel.

    The launcher computes the CUDA grid size from the requested number of
    photons and starts the kernel.

    This workflow returns one PhotonResult per photon and is useful for
    validation tests. For the final event-based simulation, the histogram
    workflow is more efficient because it avoids copying all photon results
    back to the CPU.
*/
void launchPhotonSimulationKernel(
    PhotonResult* d_results,
    unsigned long long nPhotons,
    PhotonConfig config,
    unsigned long long baseSeed
)
{
    constexpr int threadsPerBlock = 256;

    int blocks =
        static_cast<int>(
            (nPhotons + threadsPerBlock - 1ULL) /
            threadsPerBlock
        );

    simulatePhotonsKernel<<<blocks, threadsPerBlock>>>(
        d_results,
        nPhotons,
        config,
        baseSeed
    );
}

/*
    Simulate a batch of events and fill arrival-time histograms on the GPU.

    This kernel implements the event-based workflow used in the final
    simulation.

    Mapping:

        one CUDA thread = one photon

    The global photon index is converted into:

        eventId  = globalPhotonId / photonsPerEvent
        photonId = globalPhotonId % photonsPerEvent

    Each event corresponds to photons generated around the same z position.
    The generation interval is:

        z in [zCenter - zHalfWidth, zCenter + zHalfWidth]

    For each detected photon, the arrival time is converted into a time bin.
    The detected photon is then accumulated into the corresponding event
    histogram:

        histZ0[eventId * nBins + bin]
        histZL[eventId * nBins + bin]

    atomicAdd is required because many photon threads belonging to the same
    event may try to increment the same histogram bin at the same time.

    This kernel avoids copying one PhotonResult per photon back to the CPU.
    Instead, it produces compact event-by-event arrival-time histograms.
*/

__global__
void eventBatchArrivalHistogramKernel(
    unsigned int* histZ0,
    unsigned int* histZL,
    int nEvents,
    int photonsPerEvent,
    PhotonConfig config,
    double zCenter,
    double zHalfWidth,
    double tMin,
    double tMax,
    int nBins,
    unsigned long long baseSeed
)
{
    unsigned long long globalPhotonId =
        blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long totalPhotons =
        static_cast<unsigned long long>(nEvents) *
        static_cast<unsigned long long>(photonsPerEvent);

    if (globalPhotonId >= totalPhotons)
    {
        return;
    }

    /*
    Convert the flat CUDA thread index into an event index and a photon index
    inside that event. This layout makes it possible to simulate many events
    in a single kernel launch.
*/

    int eventId =
        static_cast<int>(
            globalPhotonId /
            static_cast<unsigned long long>(photonsPerEvent)
        );

    int photonId =
        static_cast<int>(
            globalPhotonId %
            static_cast<unsigned long long>(photonsPerEvent)
        );

    PhotonConfig localConfig = config;

    localConfig.z0 = zCenter - zHalfWidth;
    localConfig.z1 = zCenter + zHalfWidth;

    unsigned long long photonSeedId =
        static_cast<unsigned long long>(eventId) *
        static_cast<unsigned long long>(photonsPerEvent)
        +
        static_cast<unsigned long long>(photonId);

    RNGState rng;

    rng.state =
        makePhotonSeedDevice(
            baseSeed,
            photonSeedId
        );

    PhotonResult result =
        simulatePhotonDevice(
            localConfig,
            rng
        );

    if (!result.detected)
    {
        return;
    }

    if (result.arrivalTime < tMin || result.arrivalTime >= tMax)
    {
        return;
    }

    double binWidth =
        (tMax - tMin) / static_cast<double>(nBins);

    int bin =
        static_cast<int>(
            (result.arrivalTime - tMin) / binWidth
        );

    if (bin < 0 || bin >= nBins)
    {
        return;
    }

    /*
    Several photon threads from the same event can fall into the same time
    bin. atomicAdd prevents race conditions and guarantees that no counts
    are lost.
*/

    int index =
        eventId * nBins + bin;

    if (result.detectorFace == Z_NEG)
    {
        atomicAdd(
            &histZ0[index],
            1u
        );
    }
    else if (result.detectorFace == Z_POS)
    {
        atomicAdd(
            &histZL[index],
            1u
        );
    }
}
    
/*
    Host-side launcher for the event-batch arrival histogram kernel.

    Inputs:
      - d_histZ0, d_histZL:
          device arrays storing event-by-event arrival-time histograms;
      - nEvents:
          number of simulated events in the batch;
      - photonsPerEvent:
          number of scintillation photons simulated per event;
      - config:
          photon transport configuration;
      - zCenter, zHalfWidth:
          define the photon generation region along the bar;
      - tMin, tMax, nBins:
          define the arrival-time histogram range and binning;
      - baseSeed:
          base seed used to generate deterministic per-photon RNG states.

    The histograms must be allocated and initialized to zero by the caller.
*/

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
)
{
    constexpr int threadsPerBlock = 256;

    unsigned long long totalPhotons =
        static_cast<unsigned long long>(nEvents) *
        static_cast<unsigned long long>(photonsPerEvent);

    int blocks =
        static_cast<int>(
            (totalPhotons + threadsPerBlock - 1ULL) /
            threadsPerBlock
        );

    eventBatchArrivalHistogramKernel<<<blocks, threadsPerBlock>>>(
        d_histZ0,
        d_histZL,
        nEvents,
        photonsPerEvent,
        config,
        zCenter,
        zHalfWidth,
        tMin,
        tMax,
        nBins,
        baseSeed
    );
}