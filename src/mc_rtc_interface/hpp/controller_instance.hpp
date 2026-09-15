#pragma once

#include <mc_control/mc_global_controller.h>
#include <mc_rbdyn/RobotModule.h>
#include <mc_rtc/Configuration.h>
#include <memory>
#include <span>
#include "hpp/io_layout.hpp"

using IoInput               = std::span<const double>;
using IoOutput              = std::span<double>;
using GlobalConfiguration   = mc_control::MCGlobalController::GlobalConfiguration;
using MCGlobalControllerPtr = std::unique_ptr<mc_control::MCGlobalController>;

class ControllerInstance
{
    public:
        ControllerInstance(const GlobalConfiguration &configuration, const IoLayout &layout);
        ~ControllerInstance() = default;

        void                           initialize(IoInput input, IoOutput output);
        OutputLayout::ControllerStatus step(IoInput input, IoOutput output);
        void                           reset(IoInput input, IoOutput output);
        bool                           failed() const;
        bool                           initialized() const;

        std::vector<std::pair<std::string, std::string>> get_available_datastore_entries();

    private:
        MCGlobalControllerPtr      m_controller;
        const GlobalConfiguration &m_configuration;
        const IoLayout            &m_layout;
        bool                       m_initialized, m_failed;

        std::vector<double> m_q, m_qd, m_tau;
        std::vector<size_t> m_io_to_mbc;

        inline static constexpr std::size_t invalid_index = std::numeric_limits<std::size_t>::max();

        sva::PTransformd                        prepare_reset(IoInput input);
        void                                    finish_reset(IoInput input, const sva::PTransformd &pose);
        void                                    apply_input(IoInput input);
        void                                    apply_objects(IoInput input, bool seed_reference = false);
        std::map<std::string, sva::PTransformd> object_poses(IoInput input) const;
        void                                    apply_output(IoOutput output);
        void                                    io_to_encoders(IoInput io, std::vector<double> &encoders);
        void                                    mbc_to_io(IoOutput io, const std::vector<std::vector<double>> &mbc);
};
