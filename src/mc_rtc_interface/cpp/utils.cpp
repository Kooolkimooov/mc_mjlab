#include "hpp/utils.hpp"
#include <boost/interprocess/shared_memory_object.hpp>
#include <cerrno>
#include <stdexcept>
#include <sys/stat.h>
#include <system_error>
#include <utility>
#include "hpp/ipc_socket.hpp"

namespace utils::shared_memory
{
    namespace
    {
        struct stat
            validate(const boost::interprocess::shared_memory_object &object, const SharedMemoryDescription &region)
        {
            struct stat info{};
            if (::fstat(object.get_mapping_handle().handle, &info) == -1)
                throw std::system_error(errno, std::generic_category(), "fstat shared memory");
            if (region.size == 0 || region.offset % sizeof(double) || region.size % sizeof(double))
                throw std::invalid_argument("shared-memory region must contain aligned doubles");
            if (info.st_size < 0 || std::cmp_greater(region.offset, info.st_size) ||
                std::cmp_greater(region.size, info.st_size - region.offset))
                throw std::invalid_argument("shared-memory region exceeds backing object");
            return info;
        }
    } // namespace

    WorkerIoMappings
        map_worker_io(const WorkerStartMessage &message, std::span<const double> &input, std::span<double> &output)
    {
        namespace bip = boost::interprocess;
        if (!input.empty() || !output.empty()) throw std::invalid_argument("I/O spans must be empty before mapping");

        const auto               &in  = message.input;
        const auto               &out = message.output;
        bip::shared_memory_object input_object(bip::open_only, in.file_name.c_str(), bip::read_only);
        bip::shared_memory_object output_object(bip::open_only, out.file_name.c_str(), bip::read_write);
        const auto                input_info  = validate(input_object, in);
        const auto                output_info = validate(output_object, out);

        if (input_info.st_dev == output_info.st_dev && input_info.st_ino == output_info.st_ino &&
            in.offset < out.offset + out.size && out.offset < in.offset + in.size)
            throw std::invalid_argument("input and output shared-memory regions overlap");

        const auto input_width  = message.layout.input_size();
        const auto output_width = message.layout.output_size();
        const auto input_count  = in.size / sizeof(double);
        const auto output_count = out.size / sizeof(double);
        if (input_width == 0 || output_width == 0 || input_count % input_width || output_count % output_width ||
            input_count / input_width != output_count / output_width)
            throw std::invalid_argument("shared-memory regions must contain matching batches of complete I/O rows");

        WorkerIoMappings mappings{
            bip::mapped_region(input_object, bip::read_only, in.offset, in.size),
            bip::mapped_region(output_object, bip::read_write, out.offset, out.size),
        };
        input  = {static_cast<const double *>(mappings.input.get_address()), input_count};
        output = {static_cast<double *>(mappings.output.get_address()), output_count};
        return mappings;
    }
} // namespace utils::shared_memory
