
#include "Geometry.h"

double Dot(const Vec3& a, const Vec3& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

double Norm(const Vec3& v)
{
    return std::sqrt(Dot(v,v));
}

Vec3 Normalize(const Vec3& v)
{
    double n = Norm(v);

    return {
        v.x/n,
        v.y/n,
        v.z/n
    };
}

Vec3 Add(const Vec3& a,const Vec3& b)
{
    return {
        a.x+b.x,
        a.y+b.y,
        a.z+b.z
    };
}

Vec3 Scale(const Vec3& v,double s)
{
    return {
        s*v.x,
        s*v.y,
        s*v.z
    };
}
