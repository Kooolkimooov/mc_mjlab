#include <cassert>
#include <cmath>
#include <exception>
#include <vector>
#include "hpp/controllers_host.hpp"
#include "probe_control.hpp"

namespace
{
    IoLayout base_layout(const GlobalConfiguration &configuration)
    {
        IoLayout layout;
        layout.set_joint_order(configuration.main_robot_module->ref_joint_order());
        return layout;
    }

    void seed(std::vector<double> &inputs, const IoLayout &layout)
    {
        inputs[layout.input.root_offset() + 2] = 0.8;
        inputs[layout.input.root_offset() + 6] = 1.0;
    }

    bool initialization_throws(const char *path, const IoLayout &layout)
    {
        std::vector<double> inputs(layout.input_size(), 0.0);
        std::vector<double> outputs(layout.output_size(), 0.0);
        seed(inputs, layout);
        try
        {
            ControllersHost host(path, 1, layout);
            host.initialize(inputs, outputs);
        }
        catch (const std::exception &)
        {
            return true;
        }
        return false;
    }
} // namespace

int main(int argc, char **argv)
{
    assert(argc == 2);
    const GlobalConfiguration configuration(argv[1]);

    auto layout                     = base_layout(configuration);
    layout.input.datastore_scalar   = {"set_support_foot"};
    layout.output.datastore_scalar  = {"mc_mjlab::support_foot"};
    layout.output.datastore_vector3 = {"mc_mjlab::planned_zmp", "mc_mjlab::control_com", "mc_mjlab::control_com_vel"};

    std::vector<double> inputs(layout.input_size(), 0.0);
    std::vector<double> outputs(layout.output_size(), 0.0);
    seed(inputs, layout);

    const auto vectors      = layout.output.datastore_vector3_offset();
    const auto support_foot = layout.output.datastore_scalar_offset();
    const auto command      = layout.input.datastore_scalar_offset();

    ControllersHost host(argv[1], 1, layout);
    host.initialize(inputs, outputs);

    assert(outputs[vectors + 5] > 0.1);
    assert(outputs[vectors + 2] == 0.0);
    for (std::size_t axis = 0; axis < 2; ++axis)
        assert(std::abs(outputs[vectors + axis] - outputs[vectors + 3 + axis]) < 1e-12);
    for (std::size_t axis = 0; axis < 3; ++axis) assert(std::abs(outputs[vectors + 6 + axis]) < 1e-9);
    assert(outputs[support_foot] == 1.0);

    inputs[command] = 0.0;
    host.step();
    assert(outputs[support_foot] == 0.0);
    inputs[command] = 1.0;
    host.step();
    assert(outputs[support_foot] == 1.0);

    // The controller is rebuilt on reset, so the entries have to be registered again.
    host.reset();
    inputs[command] = 0.0;
    host.step();
    assert(outputs[support_foot] == 0.0);
    assert(outputs[layout.output.status_offset()] == static_cast<double>(OutputLayout::OK));

    inputs[command] = -1.0;
    host.step();
    assert(outputs[layout.output.status_offset()] == static_cast<double>(OutputLayout::WORKER_FAILED));

    auto unknown                     = base_layout(configuration);
    unknown.output.datastore_vector3 = {"mc_mjlab::not_a_function"};
    assert(initialization_throws(argv[1], unknown));

    auto conflict                            = base_layout(configuration);
    conflict.output.datastore_vector3        = {"mc_mjlab::control_com"};
    ProbeControl::instance().decoy_datastore = true;
    assert(initialization_throws(argv[1], conflict));
    ProbeControl::instance().decoy_datastore = false;
}
