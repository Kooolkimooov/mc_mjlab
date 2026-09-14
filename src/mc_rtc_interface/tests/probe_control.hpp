#pragma once

struct ProbeControl
{
        static ProbeControl &instance();

        int  live            = 0;
        bool fail_reset      = false;
        bool decoy_datastore = false;
};
