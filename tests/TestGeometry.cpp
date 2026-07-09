#include <iostream>

#include "PhotonTransport.h"

int main()
{
    Box box;

    box.xmin = -2;
    box.xmax = 2;

    box.ymin = -2;
    box.ymax = 2;

    box.zmin = 0;
    box.zmax = 280;

    Vec3 pos{0,0,140};

    Vec3 dir{0,0,1};

    StepResult s =
        nextIntersection(
            pos,
            dir,
            box
        );

    std::cout
        << "t = "
        << s.t
        << std::endl;

    std::cout
        << "face = "
        << s.face
        << std::endl;
}