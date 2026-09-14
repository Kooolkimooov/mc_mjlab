#pragma once

#include <mc_rtc/Configuration.h>
#include <memory>
#include <vector>
#include "hpp/controller_instance.hpp"
#include "hpp/io_layout.hpp"

using InstancePtrVector = std::vector<std::unique_ptr<ControllerInstance>>;

class ControllersHost
{
    public:
        ControllersHost(std::string configuration_path, size_t num_controllers, IoLayout layout);
        ~ControllersHost() = default;

        ControllersHost(const ControllersHost &)            = delete;
        ControllersHost &operator=(const ControllersHost &) = delete;

        void initialize(IoInput inputs, IoOutput outputs);
        void set_blocks(IoInput inputs, IoOutput outputs);
        void reset();
        void step();

        enum class Operation
        {
            Initialize,
            Reset,
            Step
        };

    private:
        GlobalConfiguration m_configuration;
        IoInput             m_inputs;
        IoOutput            m_outputs;
        const IoLayout      m_layout;
        InstancePtrVector   m_instances;

        bool m_blocks_valid;

        IoInput  input_row(size_t index) const;
        IoOutput output_row(size_t index);
        void     replace(size_t index);
        void     execute(Operation operation);
};
