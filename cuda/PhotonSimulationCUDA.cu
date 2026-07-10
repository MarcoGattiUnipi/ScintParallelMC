#include "PhotonSimulationCUDA.cuh"

#include <cuda_runtime.h>

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

    Box box;

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
    pos.y = uniformDevice(rng, y0, y1);
    pos.z = uniformDevice(rng, z0, z1);

    result.birthPos = pos;

    double cosTheta = uniformDevice(rng, -1.0, 1.0);
    double phi = uniformDevice(rng, 0.0, 2.0 * PI);

    Vec3 dir = isotropicDirDevice(cosTheta, phi);

    const double n_out_lateral = 1.0;
    const double n_out_end = 1.0;

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
            nextIntersectionDevice(
                pos,
                dir,
                box
            );

        if (!(step.t < 1e290))
        {
            result.status = PHOTON_NUMERICAL_ERROR;
            result.detected = false;
            result.pathLength = path;
            result.nBounces = nb;
            result.finalPos = pos;

            return result;
        }

        Vec3 newpos =
            Add(
                pos,
                Scale(dir, step.t)
            );

        path += step.t;

        if (config.attenuation_length > 0.0)
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
                incidenceAngleDegDevice(
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
        else
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