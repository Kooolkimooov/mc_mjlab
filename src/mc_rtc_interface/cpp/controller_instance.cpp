#include "hpp/controller_instance.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <mc_control/MCController.h>
#include <stdexcept>
#include "hpp/instance_datastore_plugin.hpp"
#include "hpp/utils.hpp"

ControllerInstance::ControllerInstance(const GlobalConfiguration &configuration, const IoLayout &layout)
    : m_configuration(configuration), m_layout(layout), m_initialized(false), m_failed(false)
{}

void ControllerInstance::initialize(IoInput input, IoOutput output)
{
    if (m_initialized) throw std::logic_error("controller is already initialized");

    m_q.assign(m_layout.input.joint_order.size(), 0.0);
    m_qd.assign(m_q.size(), 0.0);
    m_tau.assign(m_q.size(), 0.0);

    m_controller = std::make_unique<mc_control::MCGlobalController>(m_configuration);

    std::vector<std::string> from = m_layout.input.joint_order;
    std::vector<std::string> to;

    for (const auto &joint : m_controller->robot().mb().joints())
    {
        to.push_back(joint.name());
    }

    m_io_to_mbc.clear();

    for (const auto &name : from)
    {
        const auto match = std::find(to.begin(), to.end(), name);
        m_io_to_mbc.push_back(match == to.end() ? invalid_index : static_cast<std::size_t>(match - to.begin()));
    }

    const auto name = m_controller->controller().robot().name();

    prepare_reset(input);
    m_controller->init({{name, m_q}}, m_reset_poses);

    finish_reset(input);
    apply_output(output);

    m_failed      = false;
    m_initialized = true;
}

void ControllerInstance::reset(IoInput input, IoOutput output)
{
    if (!m_initialized)
    {
        initialize(input, output);
        return;
    }

    const auto name = m_controller->controller().robot().name();

    prepare_reset(input);
    m_controller->reset({{name, m_q}}, m_reset_poses);

    finish_reset(input);
    apply_output(output);

    m_failed = false;
}

void ControllerInstance::prepare_reset(IoInput input)
{
    io_to_encoders(input.subspan(m_layout.input.q_offset()), m_q);

    const auto root = input.subspan(m_layout.input.root_offset(), InputLayout::root_state_size);

    m_reset_poses.clear();
    m_reset_poses.emplace(
        m_controller->controller().robot().name(),
        sva::PTransformd(
            utils::geometry::quaternion_xyzw(root.subspan<3, 4>()).inverse(), utils::geometry::vector3(root, 0)));

    for (std::size_t i = 0; i < m_layout.input.objects.size(); ++i)
    {
        const auto state = input.subspan(
            m_layout.input.objects_offset() + InputLayout::object_state_size * i, InputLayout::object_state_size);
        m_reset_poses.emplace(
            m_layout.input.objects[i],
            sva::PTransformd(
                utils::geometry::quaternion_xyzw(state.subspan<3, 4>()).inverse(), utils::geometry::vector3(state, 0)));
    }
}

void ControllerInstance::finish_reset(IoInput input)
{
    auto &controller = m_controller->controller();
    controller.realRobot().posW(m_reset_poses.at(controller.robot().name()));

    apply_input(input);

    // The reference robot is the ManipManager's to drive, so only a reset may move
    // it; apply_input places the measured one every step.
    for (const auto &object : m_layout.input.objects)
    {
        auto       &reference = controller.robot(object);
        const auto &measured  = controller.realRobot(object);
        reference.posW(measured.posW());
        reference.velW(measured.velW());
        reference.forwardKinematics();
        reference.forwardVelocity();
    }

    // Logger overwrite emits an add without a remove, shifting every later column.
    // The global init already registered this entry. docs/coupling.md#controller-logs
    controller.logger().removeLogEntry("perf_UpdateContacts");
    controller.reset({controller.robot().mbc().q});
    controller.resetObserverPipelines();

    instance_datastore_plugin::register_entries(controller, m_layout);

    utils::datastore::validate<double>(controller.datastore(), m_layout.input.datastore_scalar, true);
    utils::datastore::validate<Eigen::Vector3d>(controller.datastore(), m_layout.input.datastore_vector3, true);
    utils::datastore::validate<double>(controller.datastore(), m_layout.output.datastore_scalar, false);
    utils::datastore::validate<Eigen::Vector3d>(controller.datastore(), m_layout.output.datastore_vector3, false);

    m_controller->running = true;
}

OutputLayout::ControllerStatus ControllerInstance::step(IoInput input, IoOutput output)
{
    if (m_failed || !m_initialized)
    {
        return m_failed ? OutputLayout::QP_FAILED : OutputLayout::WORKER_FAILED;
    }

    apply_input(input);

    auto &datastore = m_controller->controller().datastore();

    for (std::size_t i = 0; i < m_layout.input.datastore_scalar.size(); ++i)
    {
        utils::datastore::write<double>(
            datastore, m_layout.input.datastore_scalar[i], input[m_layout.input.datastore_scalar_offset() + i]);
    }

    for (std::size_t i = 0; i < m_layout.input.datastore_vector3.size(); ++i)
    {
        utils::datastore::write<Eigen::Vector3d>(
            datastore,
            m_layout.input.datastore_vector3[i],
            utils::geometry::vector3(input, m_layout.input.datastore_vector3_offset() + 3 * i));
    }

    m_failed = !m_controller->run();

    if (!m_failed)
    {
        apply_output(output);
    }

    return m_failed ? OutputLayout::QP_FAILED : OutputLayout::OK;
}

void ControllerInstance::apply_input(IoInput input)
{
    io_to_encoders(input.subspan(m_layout.input.q_offset()), m_q);
    io_to_encoders(input.subspan(m_layout.input.qd_offset()), m_qd);
    io_to_encoders(input.subspan(m_layout.input.tau_offset()), m_tau);

    m_controller->setEncoderValues(m_q);
    m_controller->setEncoderVelocities(m_qd);
    m_controller->setJointTorques(m_tau);

    auto       &controller      = m_controller->controller();
    const auto  root            = input.subspan(m_layout.input.root_offset(), InputLayout::root_state_size);
    const auto  world_from_body = utils::geometry::quaternion_xyzw(root.subspan<3, 4>());
    const auto  body_from_world = world_from_body.inverse();
    const auto &base            = InputLayout::floating_base_sensor;

    m_controller->setSensorPositions({{base, utils::geometry::vector3(root, 0)}});
    m_controller->setSensorOrientations({{base, body_from_world}});
    m_controller->setSensorLinearVelocities({{base, utils::geometry::vector3(root, 7)}});

    std::map<std::string, Eigen::Vector3d> gyros, accelerometers;

    for (std::size_t i = 0; i < m_layout.input.body_sensors.size(); ++i)
    {
        const auto &name     = m_layout.input.body_sensors[i];
        const auto  offset   = m_layout.input.body_sensors_offset() + InputLayout::body_sensor_size * i;
        gyros[name]          = utils::geometry::vector3(input, offset);
        accelerometers[name] = utils::geometry::vector3(input, offset + 3);
    }

    m_controller->setSensorAngularVelocities(gyros);
    m_controller->setSensorLinearAccelerations(accelerometers);

    std::map<std::string, sva::ForceVecd> wrenches;

    for (std::size_t i = 0; i < m_layout.input.force_sensors.size(); ++i)
    {
        const auto offset = m_layout.input.force_sensors_offset() + InputLayout::force_sensor_size * i;
        // MuJoCo measures force on the site; mc_rtc expects the reaction on the robot.
        wrenches.emplace(
            m_layout.input.force_sensors[i],
            sva::ForceVecd(-utils::geometry::vector3(input, offset + 3), -utils::geometry::vector3(input, offset)));
    }

    m_controller->setWrenches(wrenches);

    for (std::size_t i = 0; i < m_layout.input.objects.size(); ++i)
    {
        const auto &object = m_layout.input.objects[i];
        const auto  state  = input.subspan(
            m_layout.input.objects_offset() + InputLayout::object_state_size * i, InputLayout::object_state_size);
        const auto object_from_world = utils::geometry::quaternion_xyzw(state.subspan<3, 4>()).inverse();
        const auto position          = utils::geometry::vector3(state, 0);
        const auto linear            = utils::geometry::vector3(state, 7);
        const auto angular           = utils::geometry::vector3(state, 10);
        const auto accel             = utils::geometry::vector3(state, 13);

        if (object == controller.robot().name() || !controller.robots().hasRobot(object) ||
            !controller.realRobots().hasRobot(object))
            throw std::invalid_argument("invalid controller object: " + object);

        // mc_mujoco feeds every floating-base robot these five and we did not; no
        // consumer is identified. docs/coupling.md#object-state-feedback
        m_controller->setSensorPositions(object, {{base, position}});
        m_controller->setSensorOrientations(object, {{base, object_from_world}});
        m_controller->setSensorLinearVelocities(object, {{base, linear}});
        m_controller->setSensorAngularVelocities(object, {{base, angular}});
        m_controller->setSensorLinearAccelerations(object, {{base, accel}});

        auto &measured = controller.realRobot(object);
        measured.posW({object_from_world, position});
        measured.velW({angular, linear});
        measured.forwardKinematics();
        measured.forwardVelocity();
    }
}

void ControllerInstance::apply_output(IoOutput output)
{
    const auto &mbc = m_controller->robot().mbc();

    const auto count = m_layout.output.joint_order.size();
    mbc_to_io(output.subspan(m_layout.output.q_offset(), count), mbc.q);
    mbc_to_io(output.subspan(m_layout.output.qd_offset(), count), mbc.alpha);
    mbc_to_io(output.subspan(m_layout.output.tau_offset(), count), mbc.jointTorque);

    const auto &datastore = m_controller->controller().datastore();

    for (std::size_t i = 0; i < m_layout.output.datastore_scalar.size(); ++i)
    {
        output[m_layout.output.datastore_scalar_offset() + i] =
            utils::datastore::read<double>(datastore, m_layout.output.datastore_scalar[i]);
    }

    for (std::size_t i = 0; i < m_layout.output.datastore_vector3.size(); ++i)
    {
        Eigen::Map<Eigen::Vector3d> target(output.data() + m_layout.output.datastore_vector3_offset() + 3 * i);
        target = utils::datastore::read<Eigen::Vector3d>(datastore, m_layout.output.datastore_vector3[i]);
    }
}

void ControllerInstance::io_to_encoders(IoInput io, std::vector<double> &encoders)
{
    std::copy_n(io.begin(), encoders.size(), encoders.begin());
}

void ControllerInstance::mbc_to_io(IoOutput io, const std::vector<std::vector<double>> &mbc)
{
    std::fill(io.begin(), io.end(), 0.0);

    for (std::size_t i = 0; i < m_io_to_mbc.size(); ++i)
    {
        const auto source = m_io_to_mbc[i];

        if (source != invalid_index && mbc[source].size() == 1)
        {
            io[i] = mbc[source][0];
        }
    }
}

std::vector<std::pair<std::string, std::string>> ControllerInstance::get_available_datastore_entries()
{
    std::vector<std::pair<std::string, std::string>> entries;

    const auto &datastore = m_controller->controller().datastore();

    for (const auto &key : datastore.keys())
    {
        entries.emplace_back(key, datastore.type(key));
    }

    return entries;
}

bool ControllerInstance::failed() const
{
    return m_failed;
}

bool ControllerInstance::initialized() const
{
    return m_initialized;
}
