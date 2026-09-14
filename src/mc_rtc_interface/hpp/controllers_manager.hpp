#pragma once

#include <boost/asio/io_context.hpp>
#include <boost/process.hpp>
#include "hpp/ipc_socket.hpp"

class ControllersManager
{
    public:
        ControllersManager(
            std::string        mc_rtc_configuration_path,
            size_t             num_controllers,
            size_t             num_workers,
            WorkerStartMessage configuration,
            int                timeout_ms = 5);

        ControllersManager(const ControllersManager &)            = delete;
        ControllersManager &operator=(const ControllersManager &) = delete;

        ~ControllersManager();

        void                close() noexcept;
        void                dispatch(Command command);
        std::vector<size_t> collect();
        void                respawn(const std::vector<size_t> &reset_row_ids);

    private:
        struct Worker
        {
                std::unique_ptr<IPCSocket> socket;
                WorkerStatus               status;
                boost::process::process    child;
                std::vector<size_t>        row_ids;
                std::vector<bool>          reset_rows;
                size_t                     generation;
                std::string                error;
        };

        boost::asio::io_context m_io_context;
        std::vector<Worker>     m_workers;
        std::string             m_mc_rtc_configuration_path, m_ipc_parent;
        size_t                  m_num_controllers, m_num_workers;
        WorkerStartMessage      m_configuration;
        int                     m_timeout_ms;
        bool                    m_closed = false;

        WorkerStartMessage build_worker_configuration(size_t first_controller_index, size_t num_controllers) const;
        Worker             spawn_worker(size_t first_controller_index, size_t num_controllers, size_t generation);
        void               await_worker_start(Worker &worker);
        void               retire_worker(Worker &worker);
        void               respawn_workers(const std::vector<size_t> &worker_indices);
};
