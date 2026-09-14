#include "hpp/output_guard.hpp"
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

OutputGuard::OutputGuard(bool allow_logging)
{
    if (allow_logging) return;

    const int discard = ::open("/dev/null", O_WRONLY);
    if (discard < 0) return;

    m_stdout = ::dup(1);
    m_stderr = ::dup(2);
    if (m_stdout < 0 || m_stderr < 0)
    {
        if (m_stdout >= 0) ::close(m_stdout);
        if (m_stderr >= 0) ::close(m_stderr);
        m_stdout = m_stderr = -1;
        ::close(discard);
        return;
    }
    std::fflush(nullptr);
    ::dup2(discard, 1);
    ::dup2(discard, 2);
    ::close(discard);
}

OutputGuard::~OutputGuard()
{
    if (m_stdout < 0) return;

    std::fflush(nullptr);
    ::dup2(m_stdout, 1);
    ::dup2(m_stderr, 2);
    ::close(m_stdout);
    ::close(m_stderr);
}
