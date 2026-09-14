#include "hpp/ipc_socket.hpp"
#include <cerrno>
#include <chrono>
#include <memory>
#include <nanomsg/nn.h>
#include <nanomsg/pair.h>
#include <stdexcept>
#include <system_error>

namespace
{
    constexpr int max_message_size = 1024 * 1024;

    void check(int result)
    {
        if (result < 0)
        {
            const int error = nn_errno();
            throw std::system_error(error, std::generic_category(), nn_strerror(error));
        }
    }

    template <typename Operation> int transfer(int socket, int option, int milliseconds, Operation operation)
    {
        if (milliseconds < -1) throw std::invalid_argument("timeout must be -1 or nonnegative");
        const auto start     = std::chrono::steady_clock::now();
        int        remaining = milliseconds;
        while (true)
        {
            check(nn_setsockopt(socket, NN_SOL_SOCKET, option, &remaining, sizeof(remaining)));
            const int result = operation();
            if (result >= 0) return result;
            if (nn_errno() != EINTR) check(result);
            if (milliseconds >= 0)
            {
                const auto elapsed =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
                        .count();
                if (elapsed >= milliseconds)
                    throw std::system_error(ETIMEDOUT, std::generic_category(), "IPC deadline expired");
                remaining = milliseconds - static_cast<int>(elapsed);
            }
        }
    }
} // namespace

IPCSocket::IPCSocket(const std::string &endpoint, Mode mode)
{
    if (!endpoint.starts_with("ipc://")) throw std::invalid_argument("channel requires an ipc:// endpoint");
    m_socket = nn_socket(AF_SP, NN_PAIR);
    check(m_socket);
    try
    {
        check(nn_setsockopt(m_socket, NN_SOL_SOCKET, NN_RCVMAXSIZE, &max_message_size, sizeof(max_message_size)));
        check(mode == Mode::Bind ? nn_bind(m_socket, endpoint.c_str()) : nn_connect(m_socket, endpoint.c_str()));
    }
    catch (...)
    {
        nn_close(m_socket);
        throw;
    }
}

IPCSocket::~IPCSocket()
{
    nn_close(m_socket);
}

void IPCSocket::send_payload(const std::string &payload, int timeout_ms)
{
    if (payload.size() > max_message_size) throw std::length_error("message exceeds 1 MiB");
    transfer(m_socket, NN_SNDTIMEO, timeout_ms, [&] { return nn_send(m_socket, payload.data(), payload.size(), 0); });
}

std::string IPCSocket::receive_payload(int timeout_ms)
{
    void     *buffer = nullptr;
    const int received =
        transfer(m_socket, NN_RCVTIMEO, timeout_ms, [&] { return nn_recv(m_socket, &buffer, NN_MSG, 0); });
    const std::unique_ptr<void, decltype(&nn_freemsg)> owner(buffer, &nn_freemsg);
    return std::string(static_cast<const char *>(buffer), received);
}
