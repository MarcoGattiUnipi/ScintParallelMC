#pragma once

#include <cmath>
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

constexpr double PI = 3.14159265358979323846;

struct Vec3
{
    double x;
    double y;
    double z;
};

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

enum Face
{
    X_NEG = 0,
    X_POS = 1,
    Y_NEG = 2,
    Y_POS = 3,
    Z_NEG = 4,
    Z_POS = 5
};

struct Box
{
    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;
};

struct StepResult
{
    double t;
    Face face;
};