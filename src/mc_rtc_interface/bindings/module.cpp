#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "hpp/controllers_manager.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(mc_rtc_interface, m)
{
    m.doc() = "mc_rtc controller interface";

    nb::class_<InputLayout>(m, "InputLayout")
        .def(nb::init<>())
        .def_rw("joint_order", &InputLayout::joint_order)
        .def_ro_static("floating_base_sensor", &InputLayout::floating_base_sensor)
        .def_rw("body_sensors", &InputLayout::body_sensors)
        .def_rw("force_sensors", &InputLayout::force_sensors)
        .def_rw("datastore_scalar", &InputLayout::datastore_scalar)
        .def_rw("datastore_vector3", &InputLayout::datastore_vector3)
        .def("q_offset", &InputLayout::q_offset)
        .def("qd_offset", &InputLayout::qd_offset)
        .def("tau_offset", &InputLayout::tau_offset)
        .def("root_offset", &InputLayout::root_offset)
        .def("body_sensors_offset", &InputLayout::body_sensors_offset)
        .def("force_sensors_offset", &InputLayout::force_sensors_offset)
        .def("datastore_scalar_offset", &InputLayout::datastore_scalar_offset)
        .def("datastore_vector3_offset", &InputLayout::datastore_vector3_offset)
        .def("reset_offset", &InputLayout::reset_offset)
        .def("size", &InputLayout::size);

    nb::class_<OutputLayout> output_layout(m, "OutputLayout");
    nb::enum_<OutputLayout::ControllerStatus>(output_layout, "Status", nb::is_arithmetic())
        .value("OK", OutputLayout::OK)
        .value("QP_FAILED", OutputLayout::QP_FAILED)
        .value("WORKER_FAILED", OutputLayout::WORKER_FAILED);

    output_layout.def(nb::init<>())
        .def_rw("joint_order", &OutputLayout::joint_order)
        .def_rw("datastore_scalar", &OutputLayout::datastore_scalar)
        .def_rw("datastore_vector3", &OutputLayout::datastore_vector3)
        .def("q_offset", &OutputLayout::q_offset)
        .def("qd_offset", &OutputLayout::qd_offset)
        .def("tau_offset", &OutputLayout::tau_offset)
        .def("status_offset", &OutputLayout::status_offset)
        .def("datastore_scalar_offset", &OutputLayout::datastore_scalar_offset)
        .def("datastore_vector3_offset", &OutputLayout::datastore_vector3_offset)
        .def("size", &OutputLayout::size);

    nb::class_<IoLayout>(m, "IoLayout")
        .def(nb::init<>())
        .def("set_joint_order", &IoLayout::set_joint_order, "joint_order"_a)
        .def_rw("input", &IoLayout::input)
        .def_rw("output", &IoLayout::output)
        .def_prop_ro("input_size", &IoLayout::input_size)
        .def_prop_ro("output_size", &IoLayout::output_size);


    nb::class_<SharedMemoryDescription>(m, "SharedMemoryDescription")
        .def(nb::init<>())
        .def(nb::init<std::string, size_t, size_t>(), "file_name"_a, "offset"_a, "size"_a)
        .def_rw("file_name", &SharedMemoryDescription::file_name)
        .def_rw("offset", &SharedMemoryDescription::offset)
        .def_rw("size", &SharedMemoryDescription::size);

    nb::class_<WorkerStartMessage>(m, "WorkerStartMessage")
        .def(nb::init<>())
        .def(nb::init<IoLayout, SharedMemoryDescription, SharedMemoryDescription>(), "layout"_a, "input"_a, "output"_a)
        .def_rw("layout", &WorkerStartMessage::layout)
        .def_rw("input", &WorkerStartMessage::input)
        .def_rw("output", &WorkerStartMessage::output);

    nb::enum_<Command>(m, "Command")
        .value("Initialize", Command::Initialize)
        .value("Reset", Command::Reset)
        .value("Step", Command::Step)
        .value("Stop", Command::Stop);

    nb::class_<ControllersManager> manager(m, "ControllersManager");
    manager
        .def(
            nb::init<std::string, size_t, size_t, WorkerStartMessage, int>(),
            "configuration_path"_a,
            "num_controllers"_a,
            "num_workers"_a,
            "configuration"_a,
            "timeout_ms"_a = 5,
            nb::call_guard<nb::gil_scoped_release>())
        .def("dispatch", &ControllersManager::dispatch, "command"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("collect", &ControllersManager::collect, nb::call_guard<nb::gil_scoped_release>())
        .def("respawn", &ControllersManager::respawn, "reset_row_ids"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("close", &ControllersManager::close, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "__enter__",
            [](ControllersManager &self) -> ControllersManager & { return self; },
            nb::rv_policy::reference)
        .def(
            "__exit__",
            [](ControllersManager &self, nb::handle, nb::handle, nb::handle) { self.close(); },
            "exc_type"_a.none(),
            "exc_value"_a.none(),
            "traceback"_a.none(),
            nb::call_guard<nb::gil_scoped_release>());
}
