cmake_minimum_required(VERSION 3.20)

project(ScintMC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(PhotonTransport
    src/PhotonTransport.cpp
)

target_include_directories(PhotonTransport
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

add_subdirectory(tests)