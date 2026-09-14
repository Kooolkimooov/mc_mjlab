#include <cassert>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <spawn.h>
#include <stdexcept>
#include <sys/wait.h>
#include <system_error>
#include <unistd.h>
#include "hpp/ipc_socket.hpp"

extern char **environ;

namespace
{
    struct Endpoint
    {
            Endpoint()
            {
                char       pattern[] = "/tmp/mc-worker-ipc-XXXXXX";
                const auto result    = ::mkdtemp(pattern);
                assert(result);
                directory = result;
                address   = "ipc://" + (directory / "channel").string();
            }
            ~Endpoint()
            {
                std::filesystem::remove(directory / "channel");
                std::filesystem::remove(directory);
            }
            std::filesystem::path directory;
            std::string           address;
    };

    struct Child
    {
            Child(const char *executable, const std::vector<std::string> &arguments)
            {
                std::vector<char *> argv{const_cast<char *>(executable)};
                for (const auto &argument : arguments) argv.push_back(const_cast<char *>(argument.c_str()));
                argv.push_back(nullptr);
                assert(::posix_spawn(&pid, executable, nullptr, nullptr, argv.data(), environ) == 0);
            }
            ~Child()
            {
                if (pid > 0)
                {
                    ::kill(pid, SIGKILL);
                    int status;
                    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR)
                    {
                    }
                }
            }
            void wait(int expected)
            {
                int   status;
                pid_t result;
                do
                {
                    result = ::waitpid(pid, &status, 0);
                }
                while (result < 0 && errno == EINTR);
                assert(result == pid);
                pid = -1;
                assert(WIFEXITED(status) && WEXITSTATUS(status) == expected);
            }
            pid_t pid = -1;
    };

    int echo(const std::string &endpoint)
    {
        IPCSocket  channel(endpoint, IPCSocket::Mode::Connect);
        const auto configuration = channel.receive<WorkerStartMessage>(5000);
        assert(configuration.layout.input.joint_order.size() == 4000);
        assert(configuration.input.offset == 64);
        channel.send(Reply{});
        while (true)
        {
            const auto command = channel.receive<Command>(5000);
            if (command == Command::Stop)
            {
                channel.send(Reply{});
                return 0;
            }
            if (command == Command::Reset) continue;
            channel.send(Reply{command == Command::Step ? "probe failure" : ""});
        }
    }
} // namespace

int main(int argc, char **argv)
{
    if (argc == 3 && std::string(argv[1]) == "--echo") return echo(argv[2]);
    assert(argc == 3);

    IoLayout layout;
    for (int i = 0; i < 4000; ++i) layout.input.joint_order.push_back("joint_" + std::to_string(i));
    layout.output.joint_order       = layout.input.joint_order;
    layout.input.body_sensors       = {"FloatingBase"};
    layout.input.force_sensors      = {"LeftFoot"};
    layout.input.datastore_scalar   = {"set_scalar"};
    layout.input.datastore_vector3  = {"set_vector"};
    layout.output.datastore_scalar  = {"get_scalar"};
    layout.output.datastore_vector3 = {"get_vector"};
    const WorkerStartMessage configuration{layout, {"input", 64, 128}, {"output", 192, 256}};
    const auto               json = glz::write_json(configuration).value();
    assert(json.size() > 65536);
    WorkerStartMessage decoded;
    assert(!glz::read_json(decoded, json));
    assert(glz::write_json(decoded).value() == json);

    {
        Endpoint  endpoint;
        IPCSocket worker(endpoint.address, IPCSocket::Mode::Bind);
        Child     child(argv[0], {"--echo", endpoint.address});
        worker.send(configuration);
        assert(worker.receive<Reply>(5000).error.empty());
        worker.send(Command::Initialize);
        assert(worker.receive<Reply>(5000).error.empty());
        worker.send(Command::Step);
        assert(worker.receive<Reply>(5000).error == "probe failure");
        worker.send(Command::Stop);
        assert(worker.receive<Reply>(5000).error.empty());
        child.wait(0);
    }
    {
        Endpoint  endpoint;
        IPCSocket worker(endpoint.address, IPCSocket::Mode::Bind);
        Child     child(argv[0], {"--echo", endpoint.address});
        worker.send(configuration);
        assert(worker.receive<Reply>(5000).error.empty());
        worker.send(Command::Reset);
        bool timed_out = false;
        try
        {
            worker.receive<Reply>(20);
        }
        catch (const std::system_error &error)
        {
            timed_out = error.code().value() == ETIMEDOUT;
        }
        assert(timed_out);
        worker.send(Command::Stop);
        assert(worker.receive<Reply>(5000).error.empty());
        child.wait(0);
    }
    {
        Endpoint  endpoint;
        IPCSocket worker(endpoint.address, IPCSocket::Mode::Bind);
        Child     child(argv[1], {"--endpoint", endpoint.address, "--config", argv[2], "--num-controllers", "1"});
        worker.send(WorkerStartMessage{});
        const auto failure = worker.receive<Reply>(10000).error;
        assert(!failure.empty());
        for (const auto command : {Command::Initialize, Command::Reset, Command::Step})
        {
            worker.send(command);
            assert(worker.receive<Reply>(5000).error == failure);
        }
        worker.send(Command::Stop);
        assert(worker.receive<Reply>(5000).error.empty());
        child.wait(1);
    }

    {
        Endpoint  endpoint;
        IPCSocket parent(endpoint.address, IPCSocket::Mode::Bind);
        IPCSocket child(endpoint.address, IPCSocket::Mode::Connect);
        parent.send(configuration);
        assert(glz::write_json(child.receive<WorkerStartMessage>(1000)).value() == json);
        for (const auto command : {Command::Initialize, Command::Reset, Command::Step, Command::Stop})
        {
            parent.send(command);
            assert(child.receive<Command>(1000) == command);
        }

        parent.send(Command::Step);
        bool rejected = false;
        try
        {
            child.receive<WorkerStartMessage>(1000);
        }
        catch (const std::runtime_error &)
        {
            rejected = true;
        }
        assert(rejected);

        parent.send(std::map<std::string, int>{});
        rejected = false;
        try
        {
            child.receive<WorkerStartMessage>(1000);
        }
        catch (const std::runtime_error &)
        {
            rejected = true;
        }
        assert(rejected);
    }

    {
        Endpoint  endpoint;
        IPCSocket channel(endpoint.address, IPCSocket::Mode::Bind);
        bool      timed_out = false;
        try
        {
            channel.send(Command::Step, 20);
        }
        catch (const std::system_error &error)
        {
            timed_out = error.code().value() == ETIMEDOUT;
        }
        assert(timed_out);

        const std::string oversized(1024 * 1024, 'x');
        bool              rejected = false;
        try
        {
            channel.send(oversized);
        }
        catch (const std::length_error &)
        {
            rejected = true;
        }
        assert(rejected);
    }
}
