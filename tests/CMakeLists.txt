add_executable(TestGeometry
    TestGeometry.cpp
)

target_link_libraries(TestGeometry
    PhotonTransport
)

add_executable(TestReflection
    TestReflection.cpp
)

target_link_libraries(TestReflection
    PhotonTransport
)