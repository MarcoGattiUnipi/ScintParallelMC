#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static std::map<std::string, double> readSummaryCSV(
    const std::string& filename
)
{
    std::ifstream fin(filename);

    if (!fin)
    {
        std::cerr << "Error: cannot open file "
                  << filename << "\n";

        std::exit(1);
    }

    std::map<std::string, double> values;

    std::string line;

    bool firstLine = true;

    while (std::getline(fin, line))
    {
        if (line.empty())
        {
            continue;
        }

        if (firstLine)
        {
            firstLine = false;

            if (line == "quantity,value")
            {
                continue;
            }
        }

        std::stringstream ss(line);

        std::string key;
        std::string valueString;

        if (!std::getline(ss, key, ','))
        {
            continue;
        }

        if (!std::getline(ss, valueString, ','))
        {
            continue;
        }

        double value = std::strtod(
            valueString.c_str(),
            nullptr
        );

        values[key] = value;
    }

    return values;
}

static bool compareValue(
    const std::string& name,
    double cpuValue,
    double gpuValue,
    double absTolerance,
    double relTolerance
)
{
    double diff = std::fabs(cpuValue - gpuValue);

    double scale =
        std::max(
            std::fabs(cpuValue),
            std::fabs(gpuValue)
        );

    double allowed =
        absTolerance + relTolerance * scale;

    bool pass = diff <= allowed;

    std::cout << name << "\n";
    std::cout << "  CPU      = " << cpuValue << "\n";
    std::cout << "  GPU      = " << gpuValue << "\n";
    std::cout << "  diff     = " << diff << "\n";
    std::cout << "  allowed  = " << allowed << "\n";
    std::cout << "  result   = " << (pass ? "PASS" : "FAIL") << "\n";

    return pass;
}

int main(
    int argc,
    char** argv
)
{
    if (argc < 3)
    {
        std::cerr << "Usage:\n";
        std::cerr << "  " << argv[0]
                  << " cpu_summary.csv gpu_summary.csv\n";

        return 1;
    }

    std::string cpuFile = argv[1];
    std::string gpuFile = argv[2];

    auto cpu = readSummaryCSV(cpuFile);
    auto gpu = readSummaryCSV(gpuFile);

    std::vector<std::string> requiredKeys = {
        "n_photons",
        "n_detected",
        "n_detected_z0",
        "n_detected_zL",
        "n_absorbed_wall",
        "n_absorbed_volume",
        "n_max_bounces",
        "n_numerical_error",
        "efficiency"
    };

    bool allKeysPresent = true;

    for (const std::string& key : requiredKeys)
    {
        if (cpu.find(key) == cpu.end())
        {
            std::cerr << "Missing key in CPU file: "
                      << key << "\n";

            allKeysPresent = false;
        }

        if (gpu.find(key) == gpu.end())
        {
            std::cerr << "Missing key in GPU file: "
                      << key << "\n";

            allKeysPresent = false;
        }
    }

    if (!allKeysPresent)
    {
        return 1;
    }

    std::cout << "------ CPU vs GPU COMPARISON ------\n";
    std::cout << "CPU file: " << cpuFile << "\n";
    std::cout << "GPU file: " << gpuFile << "\n";
    std::cout << "-----------------------------------\n";

    bool pass = true;

    pass &= compareValue(
        "n_photons",
        cpu["n_photons"],
        gpu["n_photons"],
        0.0,
        0.0
    );

    pass &= compareValue(
        "n_detected",
        cpu["n_detected"],
        gpu["n_detected"],
        5.0,
        0.01
    );

    pass &= compareValue(
        "n_detected_z0",
        cpu["n_detected_z0"],
        gpu["n_detected_z0"],
        5.0,
        0.01
    );

    pass &= compareValue(
        "n_detected_zL",
        cpu["n_detected_zL"],
        gpu["n_detected_zL"],
        5.0,
        0.01
    );

    pass &= compareValue(
        "n_absorbed_wall",
        cpu["n_absorbed_wall"],
        gpu["n_absorbed_wall"],
        5.0,
        0.01
    );

    pass &= compareValue(
        "n_absorbed_volume",
        cpu["n_absorbed_volume"],
        gpu["n_absorbed_volume"],
        5.0,
        0.01
    );

    pass &= compareValue(
        "efficiency",
        cpu["efficiency"],
        gpu["efficiency"],
        1e-4,
        1e-2
    );

    if (
        cpu.find("mean_arrival_time_ns") != cpu.end() &&
        gpu.find("mean_arrival_time_ns") != gpu.end()
    )
    {
        pass &= compareValue(
            "mean_arrival_time_ns",
            cpu["mean_arrival_time_ns"],
            gpu["mean_arrival_time_ns"],
            1e-3,
            1e-2
        );
    }

    if (
        cpu.find("mean_path_all_cm") != cpu.end() &&
        gpu.find("mean_path_all_cm") != gpu.end()
    )
    {
        pass &= compareValue(
            "mean_path_all_cm",
            cpu["mean_path_all_cm"],
            gpu["mean_path_all_cm"],
            1e-3,
            1e-2
        );
    }

    if (
        cpu.find("mean_bounces_all") != cpu.end() &&
        gpu.find("mean_bounces_all") != gpu.end()
    )
    {
        pass &= compareValue(
            "mean_bounces_all",
            cpu["mean_bounces_all"],
            gpu["mean_bounces_all"],
            1e-3,
            1e-2
        );
    }

    std::cout << "-----------------------------------\n";

    if (pass)
    {
        std::cout << "CPU/GPU comparison: PASS\n";
        return 0;
    }

    std::cout << "CPU/GPU comparison: FAIL\n";
    return 1;
}