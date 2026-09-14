#include <CLI/CLI.hpp>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include "hpp/controller_instance.hpp"
#include "hpp/controllers_host.hpp"
#include "hpp/ipc_socket.hpp"
#include "hpp/output_guard.hpp"
#include "hpp/utils.hpp"

int main(int argc, char **argv)
{
    CLI::App    app{"mc_rtc worker"};
    std::string endpoint;
    bool        logging;
    std::string mc_rtc_configuration_file;
    std::size_t num_controllers = 0;
    app.add_option("--endpoint", endpoint, "Nanomsg ipc:// endpoint")->required();
    app.add_option("--config", mc_rtc_configuration_file, "mc_rtc configuration file")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--num-controllers", num_controllers, "Number of controller instances")
        ->required()
        ->check(CLI::PositiveNumber);
    app.add_option("--logging", logging, "Allow worker's controller to log to terminal")->default_val(false);
    CLI11_PARSE(app, argc, argv);

    std::cout << std::format(
        "worker on {} running {} on {} controllers with logging {}\n",
        endpoint,
        mc_rtc_configuration_file,
        num_controllers,
        logging);

    try
    {
        OutputGuard guard(logging);

        IPCSocket                              socket(endpoint, IPCSocket::Mode::Connect);
        utils::shared_memory::WorkerIoMappings mappings;
        IoInput                                input;
        IoOutput                               output;
        std::unique_ptr<ControllersHost>       host;
        std::string                            failure;
        WorkerStartMessage                     worker_configuration;

        try
        {
            worker_configuration = socket.receive<WorkerStartMessage>(30000);
            mappings             = utils::shared_memory::map_worker_io(worker_configuration, input, output);
            host                 = std::make_unique<ControllersHost>(
                mc_rtc_configuration_file, num_controllers, worker_configuration.layout);
            host->set_blocks(input, output);
        }
        catch (const std::exception &error)
        {
            failure = error.what();
        }
        catch (...)
        {
            failure = "worker configuration failed";
        }

        socket.send(Reply{failure});

        while (true)
        {
            try
            {
                const auto command = socket.receive<Command>();

                if (command == Command::Stop)
                {
                    socket.send(Reply{});
                    return failure.empty() ? 0 : 1;
                }
                if (failure.empty())
                {
                    switch (command)
                    {
                        case Command::Initialize: host->initialize(input, output); break;
                        case Command::Reset: host->reset(); break;
                        case Command::Step: host->step(); break;
                        case Command::Stop: break;
                        default: throw std::invalid_argument("unknown worker command");
                    }
                }
            }
            catch (const std::exception &error)
            {
                failure = error.what();
            }
            catch (...)
            {
                failure = "worker operation failed";
            }
            socket.send(Reply{failure});
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "worker: " << error.what() << '\n';
        return 1;
    }
}
