#include "PhotonTransport.h"
#include <cmath>
#include <algorithm>

StepResult nextIntersection(
    const Vec3& pos,
    const Vec3& dir,
    const Box& b
)
{
    const double INF = 1e300;

    double tmin = INF;
    Face face = Z_POS;

    auto update = [&](double t, Face f)
    {
        if(t > 1e-12 && t < tmin)
        {
            tmin = t;
            face = f;
        }
    };

    if(dir.x > 0)
        update((b.xmax-pos.x)/dir.x,X_POS);
    else if(dir.x < 0)
        update((b.xmin-pos.x)/dir.x,X_NEG);

    if(dir.y > 0)
        update((b.ymax-pos.y)/dir.y,Y_POS);
    else if(dir.y < 0)
        update((b.ymin-pos.y)/dir.y,Y_NEG);

    if(dir.z > 0)
        update((b.zmax-pos.z)/dir.z,Z_POS);
    else if(dir.z < 0)
        update((b.zmin-pos.z)/dir.z,Z_NEG);

    return {tmin,face};
}

Vec3 reflect(
    const Vec3& dir,
    Face f
)
{
    Vec3 n{0,0,0};

    switch(f)
    {
        case X_NEG: n={-1,0,0}; break;
        case X_POS: n={+1,0,0}; break;
        case Y_NEG: n={0,-1,0}; break;
        case Y_POS: n={0,+1,0}; break;
        case Z_NEG: n={0,0,-1}; break;
        case Z_POS: n={0,0,+1}; break;
    }

    Vec3 d = Normalize(dir);

    double proj = Dot(d,n);

    Vec3 r{
        d.x - 2.0*proj*n.x,
        d.y - 2.0*proj*n.y,
        d.z - 2.0*proj*n.z
    };

    return Normalize(r);
}

double incidenceAngleDeg(
    const Vec3& dir,
    Face f
)
{
    Vec3 n{0,0,0};

    switch(f)
    {
        case X_NEG: n={-1,0,0}; break;
        case X_POS: n={+1,0,0}; break;
        case Y_NEG: n={0,-1,0}; break;
        case Y_POS: n={0,+1,0}; break;
        case Z_NEG: n={0,0,-1}; break;
        case Z_POS: n={0,0,+1}; break;
    }

    double c =
        std::abs(
            Dot(
                Normalize(dir),
                n
            )
        );

    c = std::clamp(c,-1.0,1.0);

    return std::acos(c)*180.0/M_PI;
}

Vec3 isotropicDir(
    double cosTheta,
    double phi
)
{
    double s = std::sqrt(
        std::max(0.0, 1.0 - cosTheta * cosTheta)
    );

    return {
        s * std::cos(phi),
        s * std::sin(phi),
        cosTheta
    };
}

double criticalAngleDeg(
    double n_in,
    double n_out
)
{
    if (n_in <= n_out)
    {
        return 90.0;
    }

    double ratio = n_out / n_in;

    ratio = std::clamp(
        ratio,
        0.0,
        1.0
    );

    return std::asin(ratio) * 180.0 / PI;
}