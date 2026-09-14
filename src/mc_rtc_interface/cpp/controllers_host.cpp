#include "hpp/controllers_host.hpp"
#include <cstdint>
#include <exception>
#include <mc_rtc/logging.h>
#include <string>
#include "hpp/controller_instance.hpp"
#include "hpp/io_layout.hpp"


ControllersHost::ControllersHost(std::string configuration_path, size_t num_controllers, IoLayout layout)
    : m_configuration(configuration_path), m_layout(layout), m_blocks_valid(false)
{
    if (m_configuration.enable_gui_server) mc_rtc::log::error_and_throw("Controller host requires GUI server disabled");

    for (size_t i = 0; i < num_controllers; i++)
    {
        m_instances.emplace_back(std::make_unique<ControllerInstance>(m_configuration, m_layout));
    }
}

void ControllersHost::initialize(IoInput inputs, IoOutput outputs)
{
    set_blocks(inputs, outputs);
    execute(Operation::Initialize);
}

void ControllersHost::reset()
{
    execute(Operation::Reset);
}

void ControllersHost::step()
{
    execute(Operation::Step);
}

void ControllersHost::execute(Operation operation)
{
    if (!m_blocks_valid) mc_rtc::log::error_and_throw("I/O blocks are not bound; call initialize first");

    std::exception_ptr initialization_error;
    for (size_t index = 0; index < m_instances.size(); index++)
    {
        auto status           = OutputLayout::OK;
        bool replace_instance = false;
        try
        {
            auto &instance = *m_instances[index];
            switch (operation)
            {
                case Operation::Initialize:
                {
                    instance.initialize(input_row(index), output_row(index));
                    break;
                }
                case Operation::Reset:
                {
                    instance.reset(input_row(index), output_row(index));
                    break;
                }
                case Operation::Step:
                {
                    if (input_row(index)[m_layout.input.reset_offset()])
                    {
                        instance.reset(input_row(index), output_row(index));
                    }
                    status = instance.step(input_row(index), output_row(index));
                    break;
                }
            }
        }
        catch (...)
        {
            status = OutputLayout::WORKER_FAILED;
            if (operation == Operation::Initialize)
            {
                if (!initialization_error) initialization_error = std::current_exception();
            }
            else
                replace_instance = true;
        }
        // Destroy the exception before replacement can unload its controller library.
        if (replace_instance) replace(index);
        output_row(index)[m_layout.output.status_offset()] = status;
    }
    if (initialization_error) std::rethrow_exception(initialization_error);
}

IoInput ControllersHost::input_row(size_t index) const
{
    const auto width = m_layout.input_size();
    return m_inputs.subspan(index * width, width);
}
IoOutput ControllersHost::output_row(size_t index)
{
    const auto width = m_layout.output_size();
    return m_outputs.subspan(index * width, width);
}

namespace
{
    std::pair<std::uintptr_t, std::uintptr_t> block_range(std::span<const double> block)
    {
        const auto begin = reinterpret_cast<std::uintptr_t>(block.data());
        return {begin, begin + block.size_bytes()};
    }

    bool overlaps(std::pair<std::uintptr_t, std::uintptr_t> a, std::pair<std::uintptr_t, std::uintptr_t> b)
    {
        return a.first < a.second && b.first < b.second && a.first < b.second && b.first < a.second;
    }
} // namespace

void ControllersHost::set_blocks(IoInput inputs, IoOutput outputs)
{
    if (inputs.size() != m_layout.input_size() * m_instances.size())
        mc_rtc::log::error_and_throw("Input block is not of the appropirate size");
    if (outputs.size() != m_layout.output_size() * m_instances.size())
        mc_rtc::log::error_and_throw("Output block is not of the appropirate size");

    const auto input_range  = block_range(inputs);
    const auto output_range = block_range(outputs);
    if (overlaps(input_range, output_range)) mc_rtc::log::error_and_throw("I/O blocks must not overlap");

    m_inputs       = inputs;
    m_outputs      = outputs;
    m_blocks_valid = true;
}


void ControllersHost::replace(size_t index)
{
    m_instances[index] = std::make_unique<ControllerInstance>(m_configuration, m_layout);
}
