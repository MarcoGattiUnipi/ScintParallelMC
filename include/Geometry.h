#pragma once
struct Vec3 {
    double x, y, z;
};

enum Face {
    X_NEG=0,
    X_POS=1,
    Y_NEG=2,
    Y_POS=3,
    Z_NEG=4,
    Z_POS=5
};

struct Box {
    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;
};

struct StepResult {
    double t;
    Face face;
};