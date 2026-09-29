#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <mc_rtc/log/FlatLog.h>
#include <vector>
#include "hpp/controllers_host.hpp"

int main(int argc, char **argv)
{
    assert(argc == 2);
    char       pattern[] = "/tmp/mc_mjlab_host_logging_XXXXXX";
    const auto directory = ::mkdtemp(pattern);
    assert(directory);
    const std::filesystem::path root(directory);

    for (bool reset : {false, true})
    {
        const auto log_dir = root / (reset ? "reset" : "initialize");
        std::filesystem::create_directory(log_dir);
        mc_rtc::Configuration config(argv[1]);
        config.add("Log", true);
        config.add("LogPolicy", std::string("non-threaded"));
        config.add("LogDirectory", log_dir.string());
        config.add("LogTemplate", std::string("probe"));
        config.add("ProbeBaseReset", true);
        const auto config_path = log_dir / "mc_rtc.yaml";
        config.save(config_path.string());

        const GlobalConfiguration configuration(config_path.string());
        IoLayout                  layout;
        layout.set_joint_order(configuration.main_robot_module->ref_joint_order());
        std::vector<double> input(layout.input_size(), 0.0);
        std::vector<double> output(layout.output_size());
        input[layout.input.root_offset()]     = 0.2;
        input[layout.input.root_offset() + 1] = -0.3;
        input[layout.input.root_offset() + 2] = 0.8;
        input[layout.input.root_offset() + 6] = 1.0;
        {
            ControllersHost host(config_path.string(), 1, layout);
            host.initialize(input, output);
            if (reset) host.reset();
            host.step();
            host.step();
            assert(output[layout.output.status_offset()] == static_cast<double>(OutputLayout::OK));
        }

        const mc_rtc::log::FlatLog log((log_dir / "probe-InstanceProbe-latest.bin").string());
        assert(log.size() == 2);
        const auto positions = log.getRaw<Eigen::Vector3d>("probe_position");
        assert(positions.size() == log.size());
        for (const auto *position : positions)
        {
            assert(position);
            assert(position->isApprox(Eigen::Vector3d(0.2, -0.3, 0.8), 1e-12));
        }
        const auto timings = log.getRaw<double>("perf_UpdateContacts");
        assert(timings.size() == log.size());
        for (const auto *timing : timings) assert(timing && std::isfinite(*timing));
    }

    std::filesystem::remove_all(root);
}
