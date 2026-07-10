add_executable(TestGeometry
    TestGeometry.cpp
)

target_link_libraries(TestGeometry
    PhotonTransport
)

add_test(
    NAME GeometryTest
    COMMAND TestGeometry
)

add_executable(TestReflection
    TestReflection.cpp
)

target_link_libraries(TestReflection
    PhotonTransport
)

add_test(
    NAME ReflectionTest
    COMMAND TestReflection
)

add_executable(TestSinglePhoton
    TestSinglePhoton.cpp
)

target_link_libraries(TestSinglePhoton
    PhotonTransport
)

add_test(
    NAME SinglePhotonTest
    COMMAND TestSinglePhoton
)

add_executable(TestManyPhotonsCPU
    TestManyPhotonsCPU.cpp
)

target_link_libraries(TestManyPhotonsCPU
    PhotonTransport
)

add_test(
    NAME ManyPhotonsCPUTest
    COMMAND TestManyPhotonsCPU 10000 many_photons_cpu_summary.csv
)