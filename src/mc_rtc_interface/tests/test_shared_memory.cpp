#include <boost/interprocess/shared_memory_object.hpp>
#include <cassert>
#include <cstdio>
#include <limits>
#include <string>
#include <unistd.h>
#include "hpp/ipc_socket.hpp"
#include "hpp/utils.hpp"

namespace
{
    IoLayout probe_layout()
    {
        IoLayout layout;
        layout.set_joint_order({"joint"});
        return layout;
    }

    // The row widths a Python-created block must have, so neither side has to guess them.
    int widths()
    {
        const auto layout = probe_layout();
        std::printf("%zu %zu\n", layout.input_size(), layout.output_size());
        return 0;
    }

    // Maps the blocks the parent created and writes each row's first input back, plus one.
    int view(char **argv)
    {
        const WorkerStartMessage message{
            probe_layout(),
            {argv[0], std::stoull(argv[1]), std::stoull(argv[2])},
            {argv[3], std::stoull(argv[4]), std::stoull(argv[5])},
        };
        std::span<const double> input;
        std::span<double>       output;
        const auto              mappings = utils::shared_memory::map_worker_io(message, input, output);
        const auto              in_width = message.layout.input_size();
        for (std::size_t row = 0; row < input.size() / in_width; ++row)
            output[row * message.layout.output_size()] = input[row * in_width] + 1.0;
        return 0;
    }
} // namespace

int main(int argc, char **argv)
{
    if (argc == 2 && std::string(argv[1]) == "--widths") return widths();
    if (argc == 8 && std::string(argv[1]) == "--view") return view(argv + 2);

    namespace bip                  = boost::interprocess;
    const auto                name = "/mc-worker-mapping-test-" + std::to_string(::getpid());
    bip::shared_memory_object object(bip::create_only, name.c_str(), bip::read_write);
    struct Remove
    {
            std::string name;
            ~Remove()
            {
                bip::shared_memory_object::remove(name.c_str());
            }
    } remove{name};
    object.truncate(4096);
    bip::mapped_region parent(object, bip::read_write);
    auto              *values = static_cast<double *>(parent.get_address());
    values[1]                 = 17.0;

    WorkerStartMessage start{};
    start.layout.set_joint_order({"joint"});
    start.input  = {name, sizeof(double), 2 * start.layout.input_size() * sizeof(double)};
    start.output = {name, 512, 2 * start.layout.output_size() * sizeof(double)};
    {
        std::span<const double> input;
        std::span<double>       output;
        auto                    mappings = utils::shared_memory::map_worker_io(start, input, output);
        assert(input.size() == 2 * start.layout.input_size());
        assert(output.size() == 2 * start.layout.output_size());
        assert(input[0] == 17.0);
        output[0] = 42.0;
        assert(values[512 / sizeof(double)] == 42.0);
        assert(mappings.input.get_mode() == bip::read_only);
        assert(mappings.output.get_mode() == bip::read_write);
    }
    bip::shared_memory_object still_exists(bip::open_only, name.c_str(), bip::read_only);

    auto reject = [](const WorkerStartMessage &message)
    {
        std::span<const double> input;
        std::span<double>       output;
        bool                    rejected = false;
        try
        {
            auto mappings = utils::shared_memory::map_worker_io(message, input, output);
        }
        catch (const std::exception &)
        {
            rejected = true;
        }
        assert(rejected && input.empty() && output.empty());
    };
    auto invalid         = start;
    invalid.input.offset = 1;
    reject(invalid);
    invalid              = start;
    invalid.input.offset = std::numeric_limits<std::size_t>::max() - 7;
    reject(invalid);
    invalid             = start;
    invalid.output.size = 0;
    reject(invalid);
    invalid = start;
    invalid.output.size *= 2;
    reject(invalid);
    invalid               = start;
    invalid.output.offset = start.input.offset;
    reject(invalid);
    invalid = start;
    invalid.output.file_name += "-missing";
    reject(invalid);
}
