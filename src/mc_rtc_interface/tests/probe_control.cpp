#include "probe_control.hpp"

ProbeControl &ProbeControl::instance()
{
    static ProbeControl control;
    return control;
}

extern "C" int probe_live()
{
    return ProbeControl::instance().live;
}

extern "C" void probe_fail_reset(int fail)
{
    ProbeControl::instance().fail_reset = bool(fail);
}
