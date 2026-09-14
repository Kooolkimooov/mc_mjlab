#pragma once

#include <Eigen/Core>
#include <functional>
#include <map>
#include <mc_control/MCController.h>
#include <string>
#include "hpp/io_layout.hpp"

namespace instance_datastore_plugin
{
    using ScalarGetter = std::function<double(const mc_control::MCController &)>;
    using VectorGetter = std::function<Eigen::Vector3d(const mc_control::MCController &)>;

    Eigen::Vector3d control_com(const mc_control::MCController &controller);
    Eigen::Vector3d control_com_vel(const mc_control::MCController &controller);
    Eigen::Vector3d planned_zmp(const mc_control::MCController &controller);
    double          support_foot(const mc_control::MCController &controller);

    const std::map<std::string, ScalarGetter> &scalar_getters();
    const std::map<std::string, VectorGetter> &vector_getters();

    void register_entries(mc_control::MCController &controller, const IoLayout &layout);
} // namespace instance_datastore_plugin
