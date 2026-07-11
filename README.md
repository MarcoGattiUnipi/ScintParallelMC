# ScintParallelMC

CUDA/ROOT C++ project for GPU-accelerated optical photon simulation in a scintillator bar.

This README currently describes only the software requirements and the compilation procedure.

---

## Requirements

The project requires a Linux environment with the following tools installed:

C++ standard:     C++17
CMake:            >= 3.20
CUDA Toolkit:     >= 11.0
ROOT:             >= 6.26
NVIDIA driver:    compatible with the installed CUDA Toolkit

CC:              13.3.0
CUDA architecture: sm_61
ROOT:             ROOT 6 compatible
CMake:            CMake 3.20 or newer

This project currently compiles CUDA with sm_61 for compiling problems with my machine (GTX 1080) 
The architecture 61 is set in the CMakeLists.txt in Root folder and tests folder. The user maybe has to change both of them to the correct version to be able to compile the project.
Common example:
GTX 10xx      sm_61
RTX 20xx      sm_75
RTX 30xx      sm_86
RTX 40xx      sm_89

## Build Instruction
From the root directory of the repo:
mkdir build
cd build
cmake ..
make -j

the executable are all created inside /build/tests

The user can type "make help" to visualize all the tests name.

from root folder the test will always be executed as ./tests/TestName
