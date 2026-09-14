#include <cassert>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include "hpp/controllers_manager.hpp"

namespace
{
    using namespace std::chrono_literals;

    std::string worker_tag(const std::string &endpoint)
    {
        return std::filesystem::path(endpoint.substr(6)).filename();
    }

    int worker(const std::string &endpoint, const std::filesystem::path &directory)
    {
        const auto tag        = worker_tag(endpoint);
        const auto generation = std::stoul(tag.substr(tag.rfind('_') + 1));
        const auto first      = std::stoul(tag.substr(0, tag.find('_')));
        const auto mode       = directory.filename().string();
        IPCSocket  socket(endpoint, IPCSocket::Mode::Connect);
        socket.receive<WorkerStartMessage>();
        std::ofstream(directory / ("pid-" + tag)) << ::getpid();
        std::ofstream(directory / ("endpoint-" + tag)) << endpoint.substr(6);
        if ((mode == "startup_failure" && generation == 0) || (mode == "respawn_failure" && generation == 1))
        {
            socket.send(Reply{"probe startup failure"});
            std::this_thread::sleep_for(100ms);
            return 1;
        }
        socket.send(Reply{});
        while (true)
        {
            const auto command = socket.receive<Command>();
            if (command == Command::Stop)
            {
                socket.send(Reply{});
                std::this_thread::sleep_for(100ms);
                std::ofstream(directory / ("stopped-" + tag)) << "done";
                return 0;
            }
            // respawn_failure must fail its first step deterministically: replying
            // after the scenario's own timeout is a race the test used to lose.
            if (mode.starts_with("wedged") || (mode == "hang_once" && generation == 0) ||
                (mode == "hang_twice" && generation < 2) || (mode == "respawn_failure" && generation == 0) ||
                (mode == "partial_hang" && first == 0 && generation == 0))
                while (true) std::this_thread::sleep_for(1s);
            if (mode == "exit_once" && generation == 0) return 2;
            if (mode == "error_once" && generation == 0)
            {
                socket.send(Reply{"probe operation failure"});
                continue;
            }
            std::this_thread::sleep_for(50ms);
            socket.send(Reply{});
        }
    }

    pid_t read_pid(const std::filesystem::path &directory, const std::string &tag)
    {
        pid_t pid = -1;
        std::ifstream(directory / ("pid-" + tag)) >> pid;
        assert(pid > 0);
        return pid;
    }

    std::filesystem::path read_endpoint(const std::filesystem::path &directory, const std::string &tag)
    {
        std::string endpoint;
        std::ifstream(directory / ("endpoint-" + tag)) >> endpoint;
        assert(!endpoint.empty());
        return endpoint;
    }

    void check_reaped(const std::filesystem::path &directory, const std::string &tag)
    {
        const auto pid = read_pid(directory, tag);
        assert(::kill(pid, 0) == -1 && errno == ESRCH);
        assert(::waitpid(pid, nullptr, WNOHANG) == -1 && errno == ECHILD);
    }

    void check_closed(const std::filesystem::path &directory, const std::set<std::string> &graceful)
    {
        std::set<std::filesystem::path> parents;
        for (const auto &entry : std::filesystem::directory_iterator(directory))
        {
            const auto name = entry.path().filename().string();
            if (!name.starts_with("pid-")) continue;
            const auto tag = name.substr(4);
            check_reaped(directory, tag);
            parents.insert(read_endpoint(directory, tag).parent_path());
            assert(std::filesystem::exists(directory / ("stopped-" + tag)) == graceful.contains(tag));
        }
        assert(!parents.empty());
        for (const auto &parent : parents) assert(!std::filesystem::exists(parent));
    }

    template <typename Operation> void rejects_closed(Operation operation)
    {
        bool rejected = false;
        try
        {
            operation();
        }
        catch (const std::logic_error &)
        {
            rejected = true;
        }
        assert(rejected);
    }
} // namespace

int main(int argc, char **argv)
{
    if (argc == 7 && std::string(argv[1]) == "--endpoint") return worker(argv[2], argv[4]);
    assert(argc == 1);
    char       pattern[] = "/tmp/mc-manager-test-XXXXXX";
    const auto temporary = ::mkdtemp(pattern);
    assert(temporary);
    const std::filesystem::path root(temporary);
    const auto                  directory = [&](const char *name)
    {
        const auto path = root / name;
        std::filesystem::create_directory(path);
        return path;
    };

    const auto idle = directory("idle");
    {
        ControllersManager manager(idle.string(), 1, 1, {});
        manager.close();
        manager.close();
        check_closed(idle, {"0_0_0"});
        rejects_closed([&] { manager.dispatch(Command::Step); });
        rejects_closed([&] { manager.collect(); });
        rejects_closed([&] { manager.respawn({0}); });
    }

    const auto pending = directory("pending");
    {
        ControllersManager manager(pending.string(), 1, 1, {}, 2000);
        manager.dispatch(Command::Step);
        assert(manager.collect().empty());
        manager.dispatch(Command::Step);
        manager.close();
        check_closed(pending, {"0_0_0"});
    }

    const auto destructor = directory("destructor");
    {
        ControllersManager manager(destructor.string(), 1, 1, {});
        manager.dispatch(Command::Step);
    }
    check_closed(destructor, {"0_0_0"});

    const auto wedged = directory("wedged_collect");
    {
        ControllersManager manager(wedged.string(), 1, 1, {});
        manager.dispatch(Command::Step);
        assert(manager.collect() == std::vector<size_t>{0});
        check_reaped(wedged, "0_0_0");
        manager.respawn({0});
        manager.close();
        check_closed(wedged, {"0_0_1"});
    }
    const auto wedged_close = directory("wedged_close");
    {
        ControllersManager manager(wedged_close.string(), 1, 1, {});
        manager.dispatch(Command::Step);
        const auto start = std::chrono::steady_clock::now();
        manager.close();
        assert(std::chrono::steady_clock::now() - start < 5s);
        check_closed(wedged_close, {});
    }

    const auto hang_once = directory("hang_once");
    {
        ControllersManager manager(hang_once.string(), 3, 2, {}, 200);
        manager.dispatch(Command::Step);
        assert(manager.collect() == (std::vector<size_t>{0, 1, 2}));
        for (const auto *range : {"0_1", "2_2"})
        {
            check_reaped(hang_once, std::string(range) + "_0");
            assert(!std::filesystem::exists(hang_once / ("pid-" + std::string(range) + "_1")));
        }
        manager.respawn({0});
        assert(!std::filesystem::exists(hang_once / "pid-0_1_1"));
        manager.respawn({1, 2});
        for (const auto *range : {"0_1", "2_2"})
        {
            assert(
                read_endpoint(hang_once, std::string(range) + "_0") !=
                read_endpoint(hang_once, std::string(range) + "_1"));
        }
        manager.dispatch(Command::Step);
        assert(manager.collect().empty());
        manager.close();
        check_closed(hang_once, {"0_1_1", "2_2_1"});
    }

    const auto partial_hang = directory("partial_hang");
    {
        ControllersManager manager(partial_hang.string(), 3, 2, {}, 200);
        const auto         healthy_pid = read_pid(partial_hang, "2_2_0");
        manager.dispatch(Command::Step);
        assert(manager.collect() == (std::vector<size_t>{0, 1}));
        assert(read_pid(partial_hang, "2_2_0") == healthy_pid);
        manager.respawn({0});
        assert(!std::filesystem::exists(partial_hang / "pid-0_1_1"));
        manager.respawn({1});
        manager.dispatch(Command::Step);
        assert(manager.collect().empty());
        manager.close();
        check_closed(partial_hang, {"0_1_1", "2_2_0"});
    }

    const auto hang_twice = directory("hang_twice");
    {
        ControllersManager manager(hang_twice.string(), 1, 1, {}, 200);
        for (size_t generation = 0; generation < 2; ++generation)
        {
            manager.dispatch(Command::Step);
            assert(manager.collect() == std::vector<size_t>{0});
            check_reaped(hang_twice, "0_0_" + std::to_string(generation));
            manager.respawn({0});
        }
        manager.dispatch(Command::Step);
        assert(manager.collect().empty());
        manager.close();
        check_closed(hang_twice, {"0_0_2"});
    }

    for (const auto *name : {"exit_once", "error_once"})
    {
        const auto         casualty = directory(name);
        ControllersManager manager(casualty.string(), 1, 1, {}, 100);
        manager.dispatch(Command::Step);
        assert(manager.collect() == std::vector<size_t>{0});
        manager.respawn({0});
        manager.dispatch(Command::Step);
        assert(manager.collect().empty());
        manager.close();
        check_closed(casualty, {"0_0_1"});
    }

    const auto respawn_failure = directory("respawn_failure");
    {
        ControllersManager manager(respawn_failure.string(), 1, 1, {}, 50);
        manager.dispatch(Command::Step);
        assert(manager.collect() == std::vector<size_t>{0});
        bool rejected = false;
        try
        {
            manager.respawn({0});
        }
        catch (const std::runtime_error &error)
        {
            rejected = std::string(error.what()).find("respawn failed") != std::string::npos;
        }
        assert(rejected);
        rejects_closed([&] { manager.dispatch(Command::Step); });
        check_closed(respawn_failure, {});
    }

    const auto startup_failure = directory("startup_failure");
    {
        bool rejected = false;
        try
        {
            ControllersManager manager(startup_failure.string(), 1, 1, {});
        }
        catch (const std::runtime_error &error)
        {
            rejected = std::string(error.what()).find("startup failed") != std::string::npos;
        }
        assert(rejected);
        check_closed(startup_failure, {});
    }
    std::filesystem::remove_all(root);
}
