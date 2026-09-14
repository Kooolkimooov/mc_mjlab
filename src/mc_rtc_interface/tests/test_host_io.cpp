#include <algorithm>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include "hpp/controllers_host.hpp"

int main(int argc, char **argv)
{
    assert(argc == 2);
    const GlobalConfiguration configuration(argv[1]);
    for (const std::string suffix : {"", "_value"})
    {
        IoLayout layout;
        layout.set_joint_order(configuration.main_robot_module->ref_joint_order());
        layout.input.datastore_scalar   = {"set_scalar"};
        layout.input.datastore_vector3  = {"set_vector" + suffix};
        layout.output.datastore_scalar  = {"get_scalar", "encoder_q"};
        layout.output.datastore_vector3 = {"get_vector" + suffix, "reset_position", "reset_axis"};
        for (const auto &sensor : configuration.main_robot_module->bodySensors())
        {
            layout.input.body_sensors.push_back(sensor.name());
            for (const std::string field : {"position", "orientation", "velocity", "gyro", "accel"})
                layout.output.datastore_vector3.push_back(sensor.name() + "/" + field);
        }
        for (const auto &sensor : configuration.main_robot_module->forceSensors())
        {
            layout.input.force_sensors.push_back(sensor.name());
            layout.output.datastore_vector3.push_back(sensor.name() + "/force");
            layout.output.datastore_vector3.push_back(sensor.name() + "/torque");
        }
        constexpr size_t    rows = 2;
        std::vector<double> inputs(rows * layout.input_size(), 0.0);
        std::vector<double> outputs(rows * layout.output_size());
        for (size_t row = 0; row < rows; ++row)
        {
            auto input = IoOutput(inputs).subspan(row * layout.input_size(), layout.input_size());
            for (size_t joint = 0; joint < 17; ++joint)
            {
                input[joint]                             = 0.01 * (joint + row);
                input[layout.input.qd_offset() + joint]  = 0.1 + 0.01 * (joint + row);
                input[layout.input.tau_offset() + joint] = 1.0 + 0.1 * (joint + row);
            }
            const std::vector<double> root{0.2, -0.3, 0.8, 0.0, 0.0, 0.6, 0.8, 1.0, 2.0, 3.0};
            std::copy(root.begin(), root.end(), input.begin() + layout.input.root_offset());
            for (size_t col = layout.input.body_sensors_offset(); col < layout.input.datastore_scalar_offset(); ++col)
                input[col] = 10.0 + col + row;
            input[layout.input.datastore_scalar_offset()] = 2.0 + row;
            for (size_t axis = 0; axis < 3; ++axis)
                input[layout.input.datastore_vector3_offset() + axis] = 0.1 * (axis + row + 1);
        }
        ControllersHost host(argv[1], rows, layout);
        host.initialize(inputs, outputs);
        host.step();
        for (size_t row = 0; row < rows; ++row)
        {
            const auto input  = IoInput(inputs).subspan(row * layout.input_size(), layout.input_size());
            const auto output = IoInput(outputs).subspan(row * layout.output_size(), layout.output_size());
            const auto vector = [&](const std::string &name, size_t axis)
            {
                const auto &names = layout.output.datastore_vector3;
                const auto  found = std::find(names.begin(), names.end(), name);
                assert(found != names.end());
                return output[layout.output.datastore_vector3_offset() + 3 * (found - names.begin()) + axis];
            };
            for (size_t offset : {layout.input.q_offset(), layout.input.qd_offset(), layout.input.tau_offset()})
                for (size_t joint = 0; joint < 17; ++joint)
                    assert(std::abs(output[offset + joint] - input[offset + joint]) < 1e-12);
            assert(output[layout.output.status_offset()] == static_cast<double>(OutputLayout::OK));
            assert(output[layout.output.datastore_scalar_offset()] == 2.0 + row);
            assert(output[layout.output.datastore_scalar_offset() + 1] == input[0]);
            for (size_t axis = 0; axis < 3; ++axis)
            {
                assert(vector("get_vector" + suffix, axis) == input[layout.input.datastore_vector3_offset() + axis]);
                assert(std::abs(vector("reset_position", axis) - input[layout.input.root_offset() + axis]) < 1e-12);
                assert(vector("FloatingBase/position", axis) == input[layout.input.root_offset() + axis]);
                assert(vector("FloatingBase/velocity", axis) == input[layout.input.root_offset() + 7 + axis]);
                assert(std::abs(vector("FloatingBase/orientation", axis) - (axis == 2 ? -0.6 : 0.0)) < 1e-12);
                for (size_t sensor = 0; sensor < layout.input.body_sensors.size(); ++sensor)
                {
                    const auto  offset = layout.input.body_sensors_offset() + 6 * sensor;
                    const auto &name   = layout.input.body_sensors[sensor];
                    assert(vector(name + "/gyro", axis) == input[offset + axis]);
                    assert(vector(name + "/accel", axis) == input[offset + 3 + axis]);
                }
                for (size_t sensor = 0; sensor < layout.input.force_sensors.size(); ++sensor)
                {
                    const auto  offset = layout.input.force_sensors_offset() + 6 * sensor;
                    const auto &name   = layout.input.force_sensors[sensor];
                    assert(vector(name + "/force", axis) == -input[offset + axis]);
                    assert(vector(name + "/torque", axis) == -input[offset + 3 + axis]);
                }
            }
            assert(std::abs(vector("reset_axis", 0) - 0.28) < 1e-12);
            assert(std::abs(vector("reset_axis", 1) - 0.96) < 1e-12);
            assert(std::abs(vector("reset_axis", 2)) < 1e-12);
        }
    }
}
