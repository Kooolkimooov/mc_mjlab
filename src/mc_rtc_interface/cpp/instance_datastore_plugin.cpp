#include "hpp/instance_datastore_plugin.hpp"
#include <cmath>
#include <set>
#include <stdexcept>

namespace instance_datastore_plugin
{
    namespace
    {
        constexpr double gravity          = 9.81;
        constexpr double free_fall_margin = 1e-3;

        const std::string support_foot_name_callback = "ismpc_walking::support_foot_name";

        void register_name(mc_control::MCController &controller, const std::string &name)
        {
            const auto &prefix = IoLayout::plugin_prefix;
            if (name.rfind(prefix, 0) != 0) return;

            const auto function  = name.substr(prefix.size());
            auto      &datastore = controller.datastore();

            if (datastore.has(name))
                throw std::invalid_argument(
                    "controller already defines " + name + "; the plugin would redefine the same quantity");

            const auto scalar = scalar_getters().find(function);
            if (scalar != scalar_getters().end())
            {
                // The datastore holding this lambda is a member of the controller it captures.
                const auto getter = scalar->second;
                datastore.make_call(name, [&controller, getter]() -> double { return getter(controller); });
                return;
            }

            const auto vector = vector_getters().find(function);
            if (vector != vector_getters().end())
            {
                const auto getter = vector->second;
                datastore.make_call(name, [&controller, getter]() -> Eigen::Vector3d { return getter(controller); });
                return;
            }

            throw std::invalid_argument("no instance_datastore_plugin function named " + function);
        }
    } // namespace

    Eigen::Vector3d control_com(const mc_control::MCController &controller)
    {
        return controller.robot().com();
    }

    Eigen::Vector3d control_com_vel(const mc_control::MCController &controller)
    {
        return controller.robot().comVelocity();
    }

    Eigen::Vector3d planned_zmp(const mc_control::MCController &controller)
    {
        const auto com          = controller.robot().com();
        const auto acceleration = controller.robot().comAcceleration();
        const auto vertical     = acceleration.z() + gravity;

        Eigen::Vector3d zmp = Eigen::Vector3d::Zero();
        zmp.head<2>()       = com.head<2>();
        // Free fall has no control centroid; keep the historical CoM convention there.
        if (std::abs(vertical) >= free_fall_margin) zmp.head<2>() -= com.z() * acceleration.head<2>() / vertical;
        return zmp;
    }

    double support_foot(const mc_control::MCController &controller)
    {
        const auto &datastore = controller.datastore();
        if (!datastore.has(support_foot_name_callback))
            throw std::invalid_argument("support_foot requires " + support_foot_name_callback);

        const auto name = datastore.call<std::string>(support_foot_name_callback);
        if (name.rfind("Left", 0) == 0) return 1.0;
        if (name.rfind("Right", 0) == 0) return 0.0;
        throw std::invalid_argument("support foot " + name + " names neither side");
    }

    const std::map<std::string, ScalarGetter> &scalar_getters()
    {
        static const std::map<std::string, ScalarGetter> getters{{"support_foot", support_foot}};
        return getters;
    }

    const std::map<std::string, VectorGetter> &vector_getters()
    {
        static const std::map<std::string, VectorGetter> getters{
            {"control_com", control_com},
            {"control_com_vel", control_com_vel},
            {"planned_zmp", planned_zmp},
        };
        return getters;
    }

    void register_entries(mc_control::MCController &controller, const IoLayout &layout)
    {
        std::set<std::string> names;
        for (const auto *list :
             {&layout.input.datastore_scalar,
              &layout.input.datastore_vector3,
              &layout.output.datastore_scalar,
              &layout.output.datastore_vector3})
        {
            names.insert(list->begin(), list->end());
        }

        for (const auto &name : names) register_name(controller, name);
    }
} // namespace instance_datastore_plugin
