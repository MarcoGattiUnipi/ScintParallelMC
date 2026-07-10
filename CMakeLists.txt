cmake_minimum_required(VERSION 3.20)

project(ScintMC LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 61-real)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


include_directories(${PROJECT_SOURCE_DIR}/include)


find_package(CUDAToolkit REQUIRED)
find_package(ROOT REQUIRED COMPONENTS Core Hist RIO Tree)


add_library(PhotonTransport
    src/PhotonTransport.cpp
    src/PhotonSimulation.cpp
)

target_include_directories(PhotonTransport
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)


add_library(PhotonCUDA
    cuda/PhotonSimulationCUDA.cu
)

target_include_directories(PhotonCUDA
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/cuda
)


add_library(PMTResponseROOT
    src/PMTResponseROOT.cpp
)

target_include_directories(PMTResponseROOT
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(PMTResponseROOT
    PUBLIC
    ROOT::Core
    ROOT::Hist
    ROOT::RIO
    ROOT::Tree
)





enable_testing()

add_subdirectory(tests)