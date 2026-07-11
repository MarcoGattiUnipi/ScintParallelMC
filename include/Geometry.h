#pragma once


/*
    Macro used to make selected functions available both on CPU and GPU.

    When this file is compiled by nvcc, the __CUDACC__ macro is defined.
    In that case, HD expands to __host__ __device__, meaning that the
    function can be called both from host code and from CUDA device code.

    When the file is compiled by a standard C++ compiler, such as g++,
    HD expands to nothing and the code remains standard C++.

    This allows the same basic geometry utilities to be reused in both
    the CPU implementation and the CUDA kernels.
*/


#include <cmath>
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

//M_PI avoided because i had problems in GPU calculations, so I defined my own constant

constexpr double PI = 3.14159265358979323846;
/*Custom 3D vector structure my original macro used TVector3 but couldn't use it in CUDA
and in theory was really expensive to copy it in VRAM so i created the minimal functions i needed*/
struct Vec3
{
    double x;
    double y;
    double z;
};

/*All the Vec3 functions are inline cause small and frequently called both by GPU and CPU so i decided not to
use a separate file for the implementation of those functions. It should also be better for CUDA as i read online. 
They are standard geometric functions */

HD inline double Dot(const Vec3& a, const Vec3& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

HD inline double Norm(const Vec3& v)
{
    return std::sqrt(Dot(v, v));
}

HD inline Vec3 Normalize(const Vec3& v)
{
    double n = Norm(v);

    if (n == 0.0) {
        return {0.0, 0.0, 0.0};
    }

    return {
        v.x / n,
        v.y / n,
        v.z / n
    };
}

HD inline Vec3 Add(const Vec3& a, const Vec3& b)
{
    return {
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    };
}

HD inline Vec3 Scale(const Vec3& v, double s)
{
    return {
        s * v.x,
        s * v.y,
        s * v.z
    };
}

/*
    Identifiers for the six faces of the scintillator box.

    The X and Y faces correspond to the lateral walls of the scintillator.
    The Z_NEG and Z_POS faces correspond to the two end faces of the bar,
    where the two PMTs are located.
*/

enum Face
{
    X_NEG = 0,
    X_POS = 1,
    Y_NEG = 2,
    Y_POS = 3,
    Z_NEG = 4,
    Z_POS = 5
};
/*

  Rectangular box geometry used to represent the scintillator bar.

    In the standard configuration:

        x in [-ax/2, ax/2]
        y in [-ay/2, ay/2]
        z in [0, L]

    All lengths are expressed in centimeters.
*/


struct Box
{
    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;
};


/*
    Result of the intersection calculation between a photon trajectory
    and the box boundaries.

    The variable t is the distance, measured along the current photon
    direction, to the next intersected plane.

    The variable face identifies which face of the box is reached.
*/


struct StepResult
{
    double t;
    Face face;
};