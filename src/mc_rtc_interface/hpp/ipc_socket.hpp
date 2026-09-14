#pragma once

#include <glaze/json.hpp>
#include <stdexcept>
#include <string>
#include "hpp/io_layout.hpp"

enum class Command
{
    Initialize,
    Reset,
    Step,
    Stop
};
enum class WorkerStatus
{
    Working,
    Completed,
    TimedOut,
    Failed,
    Retired
};

struct Reply
{
        std::string error;
};

struct SharedMemoryDescription
{
        std::string file_name;
        std::size_t offset;
        std::size_t size;
};

struct WorkerStartMessage
{
        IoLayout                layout;
        SharedMemoryDescription input;
        SharedMemoryDescription output;
};

class IPCSocket
{
    public:
        enum class Mode
        {
            Bind,
            Connect
        };
        IPCSocket(const std::string &endpoint, Mode mode);

        IPCSocket(const IPCSocket &)            = delete;
        IPCSocket &operator=(const IPCSocket &) = delete;

        ~IPCSocket();

        template <typename T> void send(const T &message, int timeout_ms = 1000)
        {
            std::string payload;
            if (const auto error = glz::write_json(message, payload))
                throw std::runtime_error(glz::format_error(error, payload));
            send_payload(payload, timeout_ms);
        }

        template <typename T> T receive(int timeout_ms = -1)
        {
            const auto payload = receive_payload(timeout_ms);
            T          message{};
            if (const auto error = glz::read<glz::opts{.error_on_missing_keys = true}>(message, payload))
                throw std::runtime_error(glz::format_error(error, payload));
            return message;
        }

    private:
        int         m_socket = -1;
        void        send_payload(const std::string &payload, int timeout_ms);
        std::string receive_payload(int timeout_ms);
};
