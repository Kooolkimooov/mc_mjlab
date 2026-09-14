#include <algorithm>
#include <cassert>
#include <exception>
#include <string>
#include <vector>
#include "hpp/controllers_host.hpp"
#include "probe_control.hpp"

namespace
{
    template <typename Operation> void rejects(Operation operation, const std::string &message)
    {
        bool rejected = false;
        try
        {
            operation();
        }
        catch (const std::exception &error)
        {
            rejected = std::string(error.what()).find(message) != std::string::npos;
        }
        assert(rejected);
    }
} // namespace

int main(int argc, char **argv)
{
    assert(argc == 2);
    const GlobalConfiguration configuration(argv[1]);
    IoLayout                  layout;
    layout.set_joint_order(configuration.main_robot_module->ref_joint_order());
    layout.input.datastore_scalar  = {"set_scalar", "set_throw", "set_throw_output"};
    layout.output.datastore_scalar = {"get_scalar"};
    constexpr size_t    rows       = 2;
    std::vector<double> inputs(rows * layout.input_size(), 0.0);
    std::vector<double> outputs(rows * layout.output_size());
    for (size_t row = 0; row < rows; ++row)
    {
        inputs[row * layout.input_size() + layout.input.root_offset() + 2] = 0.8;
        inputs[row * layout.input_size() + layout.input.root_offset() + 6] = 1.0;
    }
    const auto status = [&](size_t row) { return outputs[row * layout.output_size() + layout.output.status_offset()]; };
    const auto scalar = layout.input.datastore_scalar_offset();
    constexpr double ok            = static_cast<double>(OutputLayout::OK);
    constexpr double worker_failed = static_cast<double>(OutputLayout::WORKER_FAILED);
    auto            &probe         = ProbeControl::instance();
    assert(probe.live == 0);

    for (size_t failure : {1, 2})
    {
        ControllersHost host(argv[1], rows, layout);
        rejects([&] { host.step(); }, "I/O blocks are not bound");
        rejects([&] { host.reset(); }, "I/O blocks are not bound");
        rejects([&] { host.set_blocks(IoInput(inputs).first(1), outputs); }, "Input block");
        rejects([&] { host.set_blocks(inputs, IoOutput(outputs).first(1)); }, "Output block");
        rejects([&] { host.set_blocks(inputs, IoOutput(inputs).first(outputs.size())); }, "must not overlap");
        host.initialize(inputs, outputs);
        assert(probe.live == 2);
        inputs[scalar + failure] = 1.0;
        host.step();
        assert(status(0) == worker_failed && status(1) == ok);
        assert(probe.live == 1);
        host.step();
        assert(status(0) == worker_failed && status(1) == ok);
        inputs[scalar + failure] = 0.0;
        host.reset();
        host.step();
        assert(status(0) == ok && status(1) == ok);
        assert(probe.live == 2);
    }
    assert(probe.live == 0);

    {
        ControllersHost host(argv[1], rows, layout);
        host.initialize(inputs, outputs);
        probe.fail_reset = true;
        host.reset();
        assert(status(0) == worker_failed && status(1) == worker_failed);
        assert(probe.live == 0);
        probe.fail_reset = false;
        host.reset();
        host.step();
        assert(status(0) == ok && status(1) == ok);
        assert(probe.live == 2);

        const auto previous    = outputs;
        auto       new_inputs  = inputs;
        auto       new_outputs = outputs;
        new_inputs[scalar]     = 3.0;
        host.set_blocks(new_inputs, new_outputs);
        host.reset();
        host.step();
        assert(outputs == previous);
        assert(new_outputs[layout.output.datastore_scalar_offset()] == 3.0);
        rejects([&] { host.set_blocks(inputs, IoOutput(outputs).first(1)); }, "Output block");
        new_inputs[scalar] = 4.0;
        host.step();
        assert(outputs == previous);
        assert(new_outputs[layout.output.datastore_scalar_offset()] == 4.0);
    }
    assert(probe.live == 0);

    for (const std::string callback : {"get_bool", "missing_callback"})
    {
        auto invalid                    = layout;
        invalid.output.datastore_scalar = {callback};
        ControllersHost host(argv[1], rows, invalid);
        rejects([&] { host.initialize(inputs, outputs); }, callback);
        host.step();
        assert(status(0) == worker_failed && status(1) == worker_failed);
    }
    assert(probe.live == 0);
}
