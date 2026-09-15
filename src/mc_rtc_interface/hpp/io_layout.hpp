#pragma once

#include <cstddef>
#include <string>
#include <vector>

/*
 * input col is:
 *  nj      q
 *  nj      qd
 *  nj      tau
 *  3       root position
 *  4       root orientation (quat)
 *  3       root linear velocity
 *  3 * nbs body sensor gyrometer
 *  3 * nbs body sensor accelerometer
 *  3 * nfs force sensor force
 *  3 * nfs force sensor torque
 *  13 * no  object position, xyzw quaternion and world linear/angular velocity
 *  nds     datastore scalars
 *  3 * ndv datastore vectors
 *  1       reset bool
 *
 * output col is:
 *  nj      q
 *  nj      qd
 *  nj      tau
 *  nds     datastore scalars
 *  3 * ndv datastore vectors
 *  1       status
 * */

struct InputLayout
{
        std::vector<std::string> joint_order;
        std::vector<std::string> body_sensors;
        std::vector<std::string> force_sensors;
        std::vector<std::string> objects;

        std::vector<std::string> datastore_scalar;
        std::vector<std::string> datastore_vector3;

        // Position, quaternion (xyzw), linear velocity.
        inline static constexpr std::size_t root_state_size   = 3 + 4 + 3;
        inline static constexpr std::size_t body_sensor_size  = 3 + 3;
        inline static constexpr std::size_t force_sensor_size = 3 + 3;
        inline static constexpr std::size_t object_state_size = 3 + 4 + 3 + 3;

        inline static const std::string floating_base_sensor = "FloatingBase";

        std::size_t q_offset() const
        {
            return 0;
        }

        std::size_t qd_offset() const
        {
            return joint_order.size();
        }

        std::size_t tau_offset() const
        {
            return 2 * joint_order.size();
        }

        std::size_t root_offset() const
        {
            return 3 * joint_order.size();
        }

        std::size_t body_sensors_offset() const
        {
            return root_offset() + root_state_size;
        }

        std::size_t force_sensors_offset() const
        {
            return body_sensors_offset() + body_sensor_size * body_sensors.size();
        }

        std::size_t objects_offset() const
        {
            return force_sensors_offset() + force_sensor_size * force_sensors.size();
        }

        std::size_t datastore_scalar_offset() const
        {
            return objects_offset() + object_state_size * objects.size();
        }

        std::size_t datastore_vector3_offset() const
        {
            return datastore_scalar_offset() + datastore_scalar.size();
        }

        std::size_t reset_offset() const
        {
            return datastore_vector3_offset() + 3 * datastore_vector3.size();
        }

        std::size_t size() const
        {
            return reset_offset() + 1;
        }
};


struct OutputLayout
{
        std::vector<std::string> joint_order;

        std::vector<std::string> datastore_scalar;
        std::vector<std::string> datastore_vector3;

        enum ControllerStatus
        {
            OK,
            QP_FAILED,
            WORKER_FAILED
        };

        std::size_t q_offset() const
        {
            return 0;
        }

        std::size_t qd_offset() const
        {
            return joint_order.size();
        }

        std::size_t tau_offset() const
        {
            return 2 * joint_order.size();
        }

        std::size_t datastore_scalar_offset() const
        {
            return 3 * joint_order.size();
        }

        std::size_t datastore_vector3_offset() const
        {
            return datastore_scalar_offset() + datastore_scalar.size();
        }

        std::size_t status_offset() const
        {
            return datastore_vector3_offset() + 3 * datastore_vector3.size();
        }

        std::size_t size() const
        {
            return status_offset() + 1;
        }
};

struct IoLayout
{
        InputLayout  input;
        OutputLayout output;

        // Names carrying this prefix are provided by instance_datastore_plugin, not the controller.
        inline static const std::string plugin_prefix = "mc_mjlab::";

        void set_joint_order(std::vector<std::string> joint_order)
        {
            input.joint_order  = joint_order;
            output.joint_order = joint_order;
        }

        std::size_t input_size() const
        {
            return input.size();
        }

        std::size_t output_size() const
        {
            return output.size();
        }
};
