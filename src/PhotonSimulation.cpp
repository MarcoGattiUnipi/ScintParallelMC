#include "PhotonSimulation.h"
#include "PhotonTransport.h"

#include <cmath>
#include <algorithm>
#include <limits>

double uniform01(
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

double uniform(
    RNGState& rng,
    double a,
    double b
)
{
    return a + (b - a) * uniform01(rng);
}

PhotonResult simulatePhotonCPU(
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
    result.arrivalTime = std::numeric_limits<double>::quiet_NaN();

    result.nBounces = 0;

    Box box;

    box.xmin = -config.ax / 2.0;
    box.xmax =  config.ax / 2.0;

    box.ymin = -config.ay / 2.0;
    box.ymax =  config.ay / 2.0;

    box.zmin = 0.0;
    box.zmax = config.L;

    double x0 = std::max(config.x0, box.xmin);
    double x1 = std::min(config.x1, box.xmax);

    double y0 = std::max(config.y0, box.ymin);
    double y1 = std::min(config.y1, box.ymax);

    double z0 = std::max(config.z0, box.zmin + 1e-9);
    double z1 = std::min(config.z1, box.zmax - 1e-9);

    if (x0 >= x1 || y0 >= y1 || z0 >= z1)
    {
        result.status = PHOTON_NUMERICAL_ERROR;
        return result;
    }

    Vec3 pos;

    pos.x = uniform(rng, x0, x1);
    pos.y = uniform(rng, y0, y1);
    pos.z = uniform(rng, z0, z1);

    result.birthPos = pos;

    double cosTheta = uniform(rng, -1.0, 1.0);
    double phi      = uniform(rng, 0.0, 2.0 * PI);

    Vec3 dir = isotropicDir(cosTheta, phi);

    const double n_out_lateral = 1.0;
    const double n_out_end     = 1.0;

    double angle_limit_lateral_deg =
        criticalAngleDeg(
            config.refractive_index,
            n_out_lateral
        );

    double angle_limit_end_deg =
        criticalAngleDeg(
            config.refractive_index,
            n_out_end
        );

    bool ignore_next_z_tir = false;

    int nb = 0;
    double path = 0.0;

    while (true)
    {
        StepResult step =
            nextIntersection(
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
                std::exp(
                    -step.t / config.attenuation_length
                );

            if (uniform01(rng) > surviveProb)
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
                incidenceAngleDeg(
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
                static constexpr double c_cm_ns = 29.9792458;

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
                incidenceAngleDeg(
                    dir,
                    step.face
                );

            bool forceReflect =
                angle_limit_lateral_deg > 0.0 &&
                incAngle > angle_limit_lateral_deg;

            if (
                forceReflect ||
                uniform01(rng) <= config.reflectivity
            )
            {
                dir =
                    reflect(
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