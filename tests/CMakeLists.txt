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

add_executable(TestManyPhotonsGPU
    TestManyPhotonsGPU.cu
)

target_link_libraries(TestManyPhotonsGPU
    PhotonCUDA
)


set_target_properties(TestManyPhotonsGPU PROPERTIES
    CUDA_ARCHITECTURES 61-real
)


add_test(
    NAME ManyPhotonsGPUTest
    COMMAND TestManyPhotonsGPU 10000
)

add_executable(TestManyPhotonsGPUReal
    TestManyPhotonsGPUReal.cu
)

target_link_libraries(TestManyPhotonsGPUReal
    PhotonCUDA
)

set_target_properties(TestManyPhotonsGPUReal PROPERTIES
    CUDA_ARCHITECTURES 61
)

add_test(
    NAME ManyPhotonsGPURealTest
    COMMAND TestManyPhotonsGPUReal 10000
)

add_executable(CompareCPUvsGPU
    CompareCPUvsGPU.cpp
)

target_link_libraries(CompareCPUvsGPU
    PhotonTransport
)

add_test(
    NAME CompareCPUvsGPUTest
    COMMAND CompareCPUvsGPU
            many_photons_cpu_summary.csv
            many_photons_gpu_summary.csv
)

add_executable(TestEventBatchGPU
    TestEventBatchGPU.cu
)

target_link_libraries(TestEventBatchGPU
    PhotonCUDA
)

set_target_properties(TestEventBatchGPU PROPERTIES
    CUDA_ARCHITECTURES 61
)

add_test(
    NAME EventBatchGPUTest
    COMMAND TestEventBatchGPU 2 10000 30.0 event_batch_gpu_summary.csv
)

add_executable(TestEventWaveformsROOT
    TestEventWaveformsROOT.cpp
)

target_link_libraries(TestEventWaveformsROOT
    PhotonCUDA
    PMTResponseROOT
    CUDA::cudart
    ROOT::Core
    ROOT::Hist
    ROOT::RIO
    ROOT::Tree
)

add_test(
    NAME EventWaveformsROOTTest
    COMMAND TestEventWaveformsROOT 2 10000 30.0 event_waveforms.root
)

add_executable(TestScanWaveformsROOT
    TestScanWaveformsROOT.cpp
)

target_link_libraries(TestScanWaveformsROOT
    PhotonCUDA
    PMTResponseROOT
    CUDA::cudart
    ROOT::Core
    ROOT::Hist
    ROOT::RIO
    ROOT::Tree
)

add_test(
    NAME ScanWaveformsROOTTest
    COMMAND TestScanWaveformsROOT 1 10000 0.5 20.5 10.0 scan_waveforms_test.root
)