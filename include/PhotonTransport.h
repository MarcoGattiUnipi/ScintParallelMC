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