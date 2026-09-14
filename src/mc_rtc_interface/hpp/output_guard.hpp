#pragma once

class OutputGuard
{
    public:
        explicit OutputGuard(bool silent);
        ~OutputGuard();

        OutputGuard(const OutputGuard &)            = delete;
        OutputGuard &operator=(const OutputGuard &) = delete;

    private:
        int m_stdout = -1, m_stderr = -1;
};
