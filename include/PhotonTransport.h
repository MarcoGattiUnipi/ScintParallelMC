#pragma once

#include "Geometry.h"

Vec3 reflect(
    const Vec3& dir,
    Face face
);

Vec3 isotropicDir(
    double u,
    double phi
);

double incidenceAngleDeg(
    const Vec3& dir,
    Face face
);

StepResult nextIntersection(
    const Vec3& pos,
    const Vec3& dir,
    const Box& box
);

Vec3 isotropicDir(
    double cosTheta,
    double phi
);


double criticalAngleDeg(
    double n_in,
    double n_out
);