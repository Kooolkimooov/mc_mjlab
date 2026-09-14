#include "hpp/controllers_manager.hpp"
#include <algorithm>
#include <boost/asio/io_context.hpp>
#include <boost/process.hpp>
#include <chrono>
#ifndef WORKER_EXECUTABLE_PATH
    #include <dlfcn.h>
#endif
#include <filesystem>
#include <mc_rtc/log/Logger.h>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include "hpp/ipc_socket.hpp"

namespace
{
    std::filesystem::path worker_executable_path()
    {
#ifdef WORKER_EXECUTABLE_PATH
        return WORKER_EXECUTABLE_PATH;
#else
        Dl_info info{};
        if (::dladdr(reinterpret_cast<const void *>(&worker_executable_path), &info) == 0 || info.dli_fname == nullptr)
            throw std::runtime_error("failed to locate mc_rtc_interface module");
        return std::filesystem::absolute(info.dli_fname).parent_path() / "mc_rtc_interface_worker";
#endif
    }
} // namespace

ControllersManager::ControllersManager(
    std::string        mc_rtc_configuration_path,
    size_t             num_controllers,
    size_t             num_workers,
    WorkerStartMessage configuration,
    int                timeout_ms)
    : m_mc_rtc_configuration_path(std::move(mc_rtc_configuration_path)),
      m_num_controllers(num_controllers),
      m_num_workers(num_workers),
      m_configuration(std::move(configuration)),
      m_timeout_ms(timeout_ms)
{
    if (m_num_workers == 0 || m_num_workers > m_num_controllers)
    {
        throw std::runtime_error("num_workers must lie in [1, num_controllers]");
    }
    if (m_timeout_ms < 0) throw std::invalid_argument("timeout_ms must be nonnegative");

    m_workers.reserve(m_num_workers);

    m_ipc_parent = std::filesystem::temp_directory_path() /
                   ("mc_mjlab_" + std::to_string(boost::process::current_pid()) + "_" +
                    std::to_string(std::chrono::system_clock::now().time_since_epoch().count()));

    try
    {
        std::filesystem::create_directories(m_ipc_parent);
        std::filesystem::permissions(
            m_ipc_parent, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace);

        for (size_t index = 0, first_controller_index = 0; index < m_num_workers; ++index)
        {
            // Remainder spread over the leading workers, so shares differ by at most one row.
            const size_t num_controllers_per_worker =
                m_num_controllers / m_num_workers + (index < m_num_controllers % m_num_workers);

            m_workers.push_back(spawn_worker(first_controller_index, num_controllers_per_worker, 0));

            first_controller_index += num_controllers_per_worker;
        }

        for (auto &worker : m_workers) await_worker_start(worker);
    }
    catch (...)
    {
        close();
        throw;
    }
}

ControllersManager::~ControllersManager()
{
    close();
}

void ControllersManager::close() noexcept
{
    if (m_closed) return;
    m_closed = true;

    for (auto &worker : m_workers)
    {
        bool exited = false;
        try
        {
            if (worker.status == WorkerStatus::Completed || worker.status == WorkerStatus::Working)
            {
                // Drain the outstanding operation before expecting the Stop reply.
                if (worker.status == WorkerStatus::Working) worker.socket->receive<Reply>(1000);
                worker.socket->send(Command::Stop, 100);
                worker.socket->receive<Reply>(1000);

                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
                while (worker.child.running())
                {
                    if (std::chrono::steady_clock::now() >= deadline) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                exited = !worker.child.running();
            }
        }
        catch (...)
        {}
        if (!exited)
        {
            boost::system::error_code error;
            const bool                running = worker.child.running(error);
            if (!error && running) worker.child.terminate(error);
        }
        worker.socket.reset();
    }
    m_workers.clear();
    std::error_code error;
    std::filesystem::remove_all(m_ipc_parent, error);
}

void ControllersManager::dispatch(Command command)
{
    if (m_closed) throw std::logic_error("controller manager is closed");
    for (size_t index = 0; index < m_workers.size(); ++index)
    {
        auto &worker = m_workers[index];
        if (worker.status != WorkerStatus::Completed) continue;
        try
        {
            worker.socket->send(command);
            worker.status = WorkerStatus::Working;
            worker.error.clear();
        }
        catch (const std::exception &error)
        {
            worker.status = WorkerStatus::Failed;
            worker.error  = std::string("ipc send failed: ") + error.what();
        }
        catch (...)
        {
            worker.status = WorkerStatus::Failed;
            worker.error  = "ipc send failed";
        }
    }
}

std::vector<size_t> ControllersManager::collect()
{
    if (m_closed) throw std::logic_error("controller manager is closed");
    std::vector<size_t> failed_rows{};
    std::vector<size_t> failed_workers{};
    for (size_t index = 0; index < m_workers.size(); index++)
    {
        auto &worker = m_workers[index];

        if (worker.status == WorkerStatus::Failed || worker.status == WorkerStatus::TimedOut)
        {
            failed_workers.push_back(index);
            failed_rows.insert(failed_rows.end(), worker.row_ids.begin(), worker.row_ids.end());
            continue;
        }
        if (worker.status != WorkerStatus::Working) continue;

        Reply reply;

        try
        {
            // collect is supposed to be used at the start of loop
            // then dispatch and let it work until next loop
            reply = worker.socket->receive<Reply>(m_timeout_ms);
            if (!reply.error.empty())
            {
                worker.status = WorkerStatus::Failed;
                worker.error  = reply.error;
            }
        }
        catch (const std::system_error &error)
        {
            if (error.code().value() == ETIMEDOUT)
            {
                worker.status = WorkerStatus::TimedOut;
                worker.error  = "timed out";
            }
            else
            {
                worker.status = WorkerStatus::Failed;
                worker.error  = std::string("ipc receive failed: ") + error.what();
            }
        }
        catch (const std::exception &error)
        {
            worker.status = WorkerStatus::Failed;
            worker.error  = std::string("ipc receive failed: ") + error.what();
        }
        catch (...)
        {
            worker.status = WorkerStatus::Failed;
            worker.error  = "ipc receive failed";
        }
        if (worker.status == WorkerStatus::Failed || worker.status == WorkerStatus::TimedOut)
        {
            failed_workers.push_back(index);
            failed_rows.insert(failed_rows.end(), worker.row_ids.begin(), worker.row_ids.end());
        }
        else
        {
            worker.status = WorkerStatus::Completed;
        }
    }

    for (const auto index : failed_workers)
    {
        const auto &worker = m_workers[index];
        mc_rtc::log::error(
            "worker {} generation {} (rows {}..{}): {}",
            index,
            worker.generation,
            worker.row_ids.front(),
            worker.row_ids.back(),
            worker.error);
    }

    for (const auto index : failed_workers)
    {
        try
        {
            retire_worker(m_workers[index]);
        }
        catch (const std::exception &error)
        {
            const auto message = "mc_rtc worker " + std::to_string(index) + " retirement failed: " + error.what();
            close();
            throw std::runtime_error(message);
        }
    }

    return failed_rows;
}

void ControllersManager::respawn(const std::vector<size_t> &reset_row_ids)
{
    if (m_closed) throw std::logic_error("controller manager is closed");
    for (const auto row_id : reset_row_ids)
    {
        if (row_id >= m_num_controllers) throw std::out_of_range("controller row id is out of range");
        for (auto &worker : m_workers)
        {
            if (worker.status != WorkerStatus::Retired || row_id < worker.row_ids.front() ||
                row_id > worker.row_ids.back())
                continue;
            worker.reset_rows[row_id - worker.row_ids.front()] = true;
            break;
        }
    }

    std::vector<size_t> workers{};
    for (size_t index = 0; index < m_workers.size(); ++index)
    {
        const auto &worker = m_workers[index];
        if (worker.status == WorkerStatus::Retired &&
            std::all_of(worker.reset_rows.begin(), worker.reset_rows.end(), [](bool reset) { return reset; }))
            workers.push_back(index);
    }
    if (!workers.empty()) respawn_workers(workers);
}

ControllersManager::Worker
    ControllersManager::spawn_worker(size_t first_controller_index, size_t num_controllers, size_t generation)
{
    std::vector<size_t> row_ids(num_controllers);
    std::iota(row_ids.begin(), row_ids.end(), first_controller_index);

    const std::filesystem::path path =
        std::filesystem::path(m_ipc_parent) /
        (std::to_string(row_ids.front()) + "_" + std::to_string(row_ids.back()) + "_" + std::to_string(generation));

    const std::string endpoint = "ipc://" + path.string();

    auto socket = std::make_unique<IPCSocket>(endpoint, IPCSocket::Mode::Bind);

    // worker path defined by cmake.
    const auto worker_path   = worker_executable_path();
    auto       child_process = boost::process::process(
        m_io_context,
        worker_path.string(),
        {"--endpoint",
         endpoint,
         "--config",
         m_mc_rtc_configuration_path,
         "--num-controllers",
         std::to_string(num_controllers)});

    Worker worker{
        std::move(socket),
        WorkerStatus::Working,
        std::move(child_process),
        std::move(row_ids),
        std::vector<bool>(num_controllers, false),
        generation,
        {}};

    // The peer is still linking mc_rtc, far past IPCSocket's 1 s send default.
    worker.socket->send<WorkerStartMessage>(build_worker_configuration(first_controller_index, num_controllers), 30000);

    return worker;
}

void ControllersManager::await_worker_start(Worker &worker)
{
    Reply reply;
    try
    {
        reply = worker.socket->receive<Reply>(60000 * m_num_controllers / m_num_workers);
    }
    catch (const std::exception &error)
    {
        worker.status = WorkerStatus::Failed;
        worker.error  = std::string("mc_rtc worker startup failed: ") + error.what();
        throw std::runtime_error(worker.error);
    }
    if (!reply.error.empty())
    {
        worker.status = WorkerStatus::Failed;
        worker.error  = "mc_rtc worker startup failed: " + reply.error;
        throw std::runtime_error(worker.error);
    }
    worker.status = WorkerStatus::Completed;
    worker.error.clear();
}

void ControllersManager::retire_worker(Worker &worker)
{
    boost::system::error_code error;
    const bool                running = worker.child.running(error);
    if (error) throw std::runtime_error("failed to inspect worker process: " + error.message());
    if (running) worker.child.terminate(error);
    if (error) throw std::runtime_error("failed to terminate worker process: " + error.message());
    worker.socket.reset();
    worker.status = WorkerStatus::Retired;
    std::fill(worker.reset_rows.begin(), worker.reset_rows.end(), false);
}

void ControllersManager::respawn_workers(const std::vector<size_t> &worker_indices)
{
    size_t      active_index = worker_indices.front();
    const char *phase        = "launch";
    try
    {
        for (const auto index : worker_indices)
        {
            active_index          = index;
            auto      &worker     = m_workers[index];
            const auto first      = worker.row_ids.front();
            const auto count      = worker.row_ids.size();
            const auto generation = worker.generation + 1;
            worker                = spawn_worker(first, count, generation);
        }

        phase = "startup";
        for (const auto index : worker_indices)
        {
            active_index = index;
            auto &worker = m_workers[index];
            await_worker_start(worker);
            mc_rtc::log::warning(
                "worker {} generation {} (rows {}..{}) respawned",
                index,
                worker.generation,
                worker.row_ids.front(),
                worker.row_ids.back());
        }
    }
    catch (const std::exception &error)
    {
        const auto &worker  = m_workers[active_index];
        const auto  message = "mc_rtc worker " + std::to_string(active_index) + " generation " +
                              std::to_string(worker.generation) + " (rows " + std::to_string(worker.row_ids.front()) +
                              ".." + std::to_string(worker.row_ids.back()) + ") respawn failed during " + phase + ": " +
                              error.what();
        close();
        throw std::runtime_error(message);
    }
    catch (...)
    {
        const auto &worker  = m_workers[active_index];
        const auto  message = "mc_rtc worker " + std::to_string(active_index) + " generation " +
                              std::to_string(worker.generation) + " (rows " + std::to_string(worker.row_ids.front()) +
                              ".." + std::to_string(worker.row_ids.back()) + ") respawn failed during " + phase;
        close();
        throw std::runtime_error(message);
    }
}

WorkerStartMessage
    ControllersManager::build_worker_configuration(size_t first_controller_index, size_t num_controllers) const
{
    const auto slice = [&](SharedMemoryDescription block, size_t row_size)
    {
        block.offset += first_controller_index * row_size * sizeof(double);
        block.size = num_controllers * row_size * sizeof(double);
        return block;
    };

    WorkerStartMessage configuration = m_configuration;

    configuration.input  = slice(m_configuration.input, m_configuration.layout.input_size());
    configuration.output = slice(m_configuration.output, m_configuration.layout.output_size());

    return configuration;
}
