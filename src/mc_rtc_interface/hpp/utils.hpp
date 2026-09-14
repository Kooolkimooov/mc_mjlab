#pragma once

#include <Eigen/Geometry>
#include <boost/interprocess/mapped_region.hpp>
#include <cstddef>
#include <functional>
#include <mc_rtc/DataStore.h>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

struct WorkerStartMessage;

namespace utils::geometry
{
    inline Eigen::Map<const Eigen::Vector3d> vector3(std::span<const double> row, std::size_t offset)
    {
        return Eigen::Map<const Eigen::Vector3d>(row.data() + offset);
    }

    inline Eigen::Quaterniond quaternion_xyzw(std::span<const double, 4> values)
    {
        return Eigen::Quaterniond(values[3], values[0], values[1], values[2]);
    }
} // namespace utils::geometry

namespace utils::datastore
{
    template <typename T> T read(const mc_rtc::DataStore &datastore, const std::string &name)
    {
        if (datastore.type(name) == mc_rtc::type_name<std::function<const T &()>>())
            return datastore.call<const T &>(name);
        return datastore.call<T>(name);
    }

    template <typename T> void write(const mc_rtc::DataStore &datastore, const std::string &name, const T &value)
    {
        if (datastore.type(name) == mc_rtc::type_name<std::function<void(const T &)>>())
            datastore.get<std::function<void(const T &)>>(name)(value);
        else
            datastore.get<std::function<void(T)>>(name)(value);
    }

    template <typename T>
    void validate(const mc_rtc::DataStore &datastore, const std::vector<std::string> &names, bool input)
    {
        const auto &value_type =
            input ? mc_rtc::type_name<std::function<void(T)>>() : mc_rtc::type_name<std::function<T()>>();
        const auto &reference_type = input ? mc_rtc::type_name<std::function<void(const T &)>>()
                                           : mc_rtc::type_name<std::function<const T &()>>();
        for (const auto &name : names)
        {
            const auto type = datastore.type(name);
            if (type != value_type && type != reference_type)
                throw std::invalid_argument("unsupported datastore callback: " + name + " (" + type + ")");
        }
    }
} // namespace utils::datastore

namespace utils::shared_memory
{
    struct WorkerIoMappings
    {
            boost::interprocess::mapped_region input;
            boost::interprocess::mapped_region output;
    };

    [[nodiscard]] WorkerIoMappings
        map_worker_io(const WorkerStartMessage &message, std::span<const double> &input, std::span<double> &output);
} // namespace utils::shared_memory
