#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>
#include "hpp/controllers_host.hpp"

int main(int argc, char **argv)
{
    assert(argc == 2);
    const GlobalConfiguration configuration(argv[1]);
    IoLayout                  layout;
    layout.set_joint_order(configuration.main_robot_module->ref_joint_order());
    layout.input.datastore_scalar   = {"set_throw"};
    layout.output.datastore_scalar  = {"get_scalar"};
    layout.output.datastore_vector3 = {"reset_position"};

    constexpr std::size_t rows = 2;
    std::vector<double>   inputs(rows * layout.input_size(), 0.0);
    std::vector<double>   outputs(rows * layout.output_size());
    const auto            joints = layout.input.joint_order.size();
    assert(joints > 0);
    const auto &mb      = configuration.main_robot_module->mb;
    const auto &indices = mb.jointIndexByName();
    const auto  joint   = std::find_if(
        layout.input.joint_order.begin(),
        layout.input.joint_order.end(),
        [&](const auto &name) { return indices.contains(name) && mb.joint(indices.at(name)).dof() == 1; });
    assert(joint != layout.input.joint_order.end());
    const auto joint_index = static_cast<std::size_t>(joint - layout.input.joint_order.begin());

    auto prepare = [&](double position)
    {
        std::fill(outputs.begin(), outputs.end(), std::numeric_limits<double>::quiet_NaN());
        for (std::size_t row = 0; row < rows; ++row)
        {
            auto input         = IoOutput(inputs).subspan(row * layout.input_size(), layout.input_size());
            input[joint_index] = position + 0.01 * row;
            input[layout.input.qd_offset() + joint_index]  = 0.4;
            input[layout.input.tau_offset() + joint_index] = 2.0;
            input[layout.input.root_offset()]              = position;
            input[layout.input.root_offset() + 2]          = 0.8;
            input[layout.input.root_offset() + 6]          = 1.0;
        }
    };
    auto check = [&]()
    {
        for (std::size_t row = 0; row < rows; ++row)
        {
            const auto input  = IoInput(inputs).subspan(row * layout.input_size(), layout.input_size());
            const auto output = IoInput(outputs).subspan(row * layout.output_size(), layout.output_size());
            for (double value : output) assert(std::isfinite(value));
            assert(output[layout.output.status_offset()] == static_cast<double>(OutputLayout::OK));
            assert(std::abs(output[joint_index] - input[joint_index]) < 1e-12);
            assert(output[layout.output.qd_offset() + joint_index] == 0.0);
            assert(output[layout.output.tau_offset() + joint_index] == 0.0);
            assert(output[layout.output.datastore_scalar_offset()] == 0.0);
            for (std::size_t axis = 0; axis < 3; ++axis)
                assert(
                    output[layout.output.datastore_vector3_offset() + axis] ==
                    input[layout.input.root_offset() + axis]);
        }
    };

    ControllersHost host(argv[1], rows, layout);
    prepare(0.1);
    host.initialize(inputs, outputs);
    check();

    host.step();
    assert(outputs[layout.output.qd_offset() + joint_index] == 0.4);
    assert(outputs[layout.output.tau_offset() + joint_index] == 2.0);
    prepare(0.2);
    host.reset();
    check();

    inputs[layout.input.datastore_scalar_offset()] = 1.0;
    host.step();
    assert(outputs[layout.output.status_offset()] == static_cast<double>(OutputLayout::WORKER_FAILED));
    inputs[layout.input.datastore_scalar_offset()] = 0.0;
    prepare(0.3);
    host.reset();
    check();
}
