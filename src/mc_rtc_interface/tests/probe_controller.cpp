#include <chrono>
#include <mc_control/mc_controller.h>
#include <mc_rbdyn/RobotLoader.h>
#include <stdexcept>
#include <string>
#include <thread>
#include "probe_control.hpp"

struct ProbeController : mc_control::MCController
{
        ProbeController(mc_rbdyn::RobotModulePtr module, double dt, const mc_rtc::Configuration &config)
            : MCController(module, dt)
        {
            if (config("ProbeObject", false))
            {
                loadRobot(mc_rbdyn::RobotLoader::get_robot_module("object", MC_RTC_TEST_ASSET_DIR, "cart"), "obj");
                datastore().make_call(
                    "object_measured", [this]() -> Eigen::Vector3d { return realRobot("obj").posW().translation(); });
                datastore().make_call(
                    "object_reference", [this]() -> Eigen::Vector3d { return robot("obj").posW().translation(); });
                datastore().make_call(
                    "object_velocity", [this]() -> Eigen::Vector3d { return realRobot("obj").velW().linear(); });
                datastore().make_call(
                    "object_angular", [this]() -> Eigen::Vector3d { return realRobot("obj").velW().angular(); });
                datastore().make_call(
                    "object_axis", [this]() -> Eigen::Vector3d { return realRobot("obj").posW().rotation().row(0); });
                datastore().make_call(
                    "object_reset_reference", [this]() -> Eigen::Vector3d { return reset_object_reference; });
                datastore().make_call(
                    "object_reset_measured", [this]() -> Eigen::Vector3d { return reset_object_measured; });
            }
            auto &ds = datastore();
            ds.make_call("set_scalar", [this](double value) { scalar = value; });
            ds.make_call(
                "get_scalar",
                [this]()
                {
                    if (throw_output) throw std::runtime_error("probe output failure");
                    return scalar;
                });
            ds.make_call("set_vector", [this](const Eigen::Vector3d &value) { vector = value; });
            ds.make_call("get_vector", [this]() -> const Eigen::Vector3d & { return vector; });
            ds.make_call("set_vector_value", [this](Eigen::Vector3d value) { vector = value; });
            ds.make_call("get_vector_value", [this]() -> Eigen::Vector3d { return vector; });
            ds.make_call("get_bool", []() { return true; });
            ds.make_call("ismpc_walking::support_foot_name", [this]() -> std::string { return support_foot_name; });
            ds.make_call(
                "set_support_foot",
                [this](double value)
                {
                    support_foot_name = value > 0.0 ? "LeftFootCenter" : (value < 0.0 ? "Nowhere" : "RightFootCenter");
                });
            if (ProbeControl::instance().decoy_datastore)
                ds.make_call("mc_mjlab::control_com", []() -> Eigen::Vector3d { return Eigen::Vector3d::Zero(); });
            ds.make_call("set_throw", [this](double value) { throw_step = value != 0.0; });
            ds.make_call("set_throw_output", [this](double value) { throw_output = value != 0.0; });
            ds.make_call(
                "set_hang",
                [](double value)
                {
                    if (value != 0.0)
                        while (true) std::this_thread::sleep_for(std::chrono::seconds(1));
                });
            ds.make_call("reset_position", [this]() -> Eigen::Vector3d { return reset_pose.translation(); });
            ds.make_call("reset_axis", [this]() -> Eigen::Vector3d { return reset_pose.rotation().row(0); });
            ds.make_call("encoder_q", [this]() { return robot().encoderValues().front(); });
            const auto hip = robot().jointIndexByName("RCP");
            ds.make_call("default_q", [value = robot().mbc().q[hip][0]]() { return value; });
            ds.make_call("untouched_q", [this, hip]() { return robot().mbc().q[hip][0]; });
            for (const auto &sensor : robot().bodySensors())
            {
                const auto name = sensor.name();
                ds.make_call(
                    name + "/position",
                    [this, name]() -> Eigen::Vector3d { return robot().bodySensor(name).position(); });
                ds.make_call(
                    name + "/orientation",
                    [this, name]() -> Eigen::Vector3d { return robot().bodySensor(name).orientation().vec(); });
                ds.make_call(
                    name + "/velocity",
                    [this, name]() -> Eigen::Vector3d { return robot().bodySensor(name).linearVelocity(); });
                ds.make_call(
                    name + "/gyro",
                    [this, name]() -> Eigen::Vector3d { return robot().bodySensor(name).angularVelocity(); });
                ds.make_call(
                    name + "/accel",
                    [this, name]() -> Eigen::Vector3d { return robot().bodySensor(name).linearAcceleration(); });
            }
            for (const auto &sensor : robot().forceSensors())
            {
                const auto name = sensor.name();
                ds.make_call(
                    name + "/force",
                    [this, name]() -> Eigen::Vector3d { return robot().forceSensor(name).wrench().force(); });
                ds.make_call(
                    name + "/torque",
                    [this, name]() -> Eigen::Vector3d { return robot().forceSensor(name).wrench().couple(); });
            }
            ProbeControl::instance().live++;
        }

        ~ProbeController() override
        {
            ProbeControl::instance().live--;
        }

        void reset(const mc_control::ControllerResetData &) override
        {
            if (ProbeControl::instance().fail_reset) throw std::runtime_error("probe reset failure");
            reset_pose = realRobot().posW();
            if (robots().hasRobot("obj"))
            {
                reset_object_reference = robot("obj").posW().translation();
                reset_object_measured  = realRobot("obj").posW().translation();
            }
        }

        bool run() override
        {
            if (throw_step) throw std::runtime_error("probe step failure");
            if (scalar < 0.0) return false;
            const auto &order = robot().refJointOrder();
            for (std::size_t i = 0; i < order.size(); ++i)
            {
                if (!robot().hasJoint(order[i])) continue;
                const auto j = robot().jointIndexByName(order[i]);
                if (robot().mbc().alpha[j].size() != 1) continue;
                robot().mbc().alpha[j][0]       = robot().encoderVelocities()[i];
                robot().mbc().jointTorque[j][0] = robot().jointTorques()[i];
            }
            return true;
        }

        std::string      support_foot_name = "LeftFootCenter";
        double           scalar            = 0.0;
        bool             throw_step = false, throw_output = false;
        Eigen::Vector3d  vector                 = Eigen::Vector3d::Zero();
        sva::PTransformd reset_pose             = sva::PTransformd::Identity();
        Eigen::Vector3d  reset_object_reference = Eigen::Vector3d::Zero();
        Eigen::Vector3d  reset_object_measured  = Eigen::Vector3d::Zero();
};

CONTROLLER_CONSTRUCTOR("InstanceProbe", ProbeController)
