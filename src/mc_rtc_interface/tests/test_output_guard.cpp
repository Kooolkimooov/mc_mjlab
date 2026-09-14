#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include "hpp/output_guard.hpp"

int main()
{
    auto *capture = std::tmpfile();
    assert(capture);
    const int saved_out = ::dup(1), saved_err = ::dup(2);
    assert(::dup2(::fileno(capture), 1) >= 0);
    assert(::dup2(::fileno(capture), 2) >= 0);

    {
        OutputGuard visible(true);
        std::fputs("visible\n", stdout);
    }
    {
        OutputGuard outer(false);
        std::fputs("hidden stdout\n", stdout);
        std::fputs("hidden stderr\n", stderr);
        {
            OutputGuard inner(false);
            std::fputs("hidden nested\n", stdout);
        }
        {
            OutputGuard visible(true);
            std::fputs("still hidden\n", stdout);
        }
        std::fputs("hidden outer\n", stderr);
    }
    std::fputs("restored\n", stderr);
    try
    {
        OutputGuard guard(false);
        std::fputs("hidden exception\n", stdout);
        throw std::runtime_error("probe");
    }
    catch (const std::runtime_error &)
    {
        std::fputs("unwound\n", stdout);
    }
    std::fputs("finished\n", stdout);
    std::fflush(nullptr);
    ::dup2(saved_out, 1);
    ::dup2(saved_err, 2);
    ::close(saved_out);
    ::close(saved_err);
    std::rewind(capture);
    char       buffer[1024]{};
    const auto size = std::fread(buffer, 1, sizeof(buffer), capture);
    std::fclose(capture);
    assert(std::string(buffer, size) == "visible\nrestored\nunwound\nfinished\n");
}
