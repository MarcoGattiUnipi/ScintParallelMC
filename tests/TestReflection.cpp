#include <iostream>

#include "PhotonTransport.h"

int main()
{
    Vec3 dir{1,0,0};

    Vec3 r =
        reflect(
            dir,
            X_POS
        );

    std::cout
        << r.x << " "
        << r.y << " "
        << r.z
        << std::endl;
}