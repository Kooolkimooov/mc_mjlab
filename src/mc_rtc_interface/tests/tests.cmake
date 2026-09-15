find_program(UV_EXECUTABLE uv REQUIRED)

add_executable(
  test_controllers_manager
  tests/test_controllers_manager.cpp
  cpp/controllers_manager.cpp
)

target_link_libraries(
  test_controllers_manager PRIVATE
  mc_rtc_interface_ipc Boost::process
)

target_compile_options(
  test_controllers_manager PRIVATE
  -UNDEBUG
)

# The test executable also acts as a worker with controlled delays and shutdown.
target_compile_definitions(
  test_controllers_manager PRIVATE
  WORKER_EXECUTABLE_PATH="$<TARGET_FILE:test_controllers_manager>"
)

add_test(
  NAME controllers_manager
  COMMAND test_controllers_manager
)

set_tests_properties(
  controllers_manager PROPERTIES
  TIMEOUT 30
)

add_test(
  NAME install_bundle
  COMMAND
    "${CMAKE_COMMAND}"
    "-DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
    "-DINSTALL_DIR=${MC_RTC_INTERFACE_INSTALL_DIR}"
    "-DMODULE_NAME=$<TARGET_FILE_NAME:mc_rtc_interface>"
    "-DWORKER_NAME=$<TARGET_FILE_NAME:mc_rtc_interface_worker>"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_install.cmake"
)

set_tests_properties(
  install_bundle PROPERTIES
  TIMEOUT 10
)

add_executable(
  test_shared_memory
  tests/test_shared_memory.cpp
)

target_link_libraries(
  test_shared_memory PRIVATE
  mc_rtc_interface_ipc
)

target_compile_options(
  test_shared_memory PRIVATE
  -UNDEBUG
)

add_test(
  NAME shared_memory
  COMMAND test_shared_memory
)

set_tests_properties(
  shared_memory PROPERTIES
  TIMEOUT 5
)

add_executable(
  test_output_guard
  tests/test_output_guard.cpp
  cpp/output_guard.cpp
)

target_include_directories(
  test_output_guard PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_options(
  test_output_guard PRIVATE
  -UNDEBUG
)

add_test(
  NAME output_guard
  COMMAND test_output_guard
)

set_tests_properties(
  output_guard PROPERTIES
  TIMEOUT 30
)

add_library(
  InstanceProbe SHARED
  tests/probe_controller.cpp
  tests/probe_control.cpp
)

target_link_libraries(
  InstanceProbe PRIVATE
  mc_rtc::mc_control
)

set_target_properties(
  InstanceProbe PROPERTIES
  PREFIX ""
)

file(GENERATE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/host_reset_outputs.yaml"
  CONTENT "MainRobot: HRP5P
Enabled: InstanceProbe
Timestep: 0.002
GUIServer: {Enable: false}
Log: false
ClearGlobalPluginPath: true
GlobalPluginPaths: []
Plugins: []
ControllerModulePaths: ['$<TARGET_FILE_DIR:InstanceProbe>']
"
)

foreach(test IN ITEMS host_failures host_io instance_datastore_plugin)
    add_executable(test_${test} tests/test_${test}.cpp)
    target_link_libraries(test_${test} PRIVATE mc_rtc_interface_core InstanceProbe)
    target_compile_options(test_${test} PRIVATE -UNDEBUG)
    add_test(
      NAME ${test}
      COMMAND test_${test} "${CMAKE_CURRENT_BINARY_DIR}/host_reset_outputs.yaml"
    )
    set_tests_properties(${test} PROPERTIES TIMEOUT 60)
endforeach()

add_executable(
  test_host_reset_outputs
  tests/test_host_reset_outputs.cpp
)

target_link_libraries(
  test_host_reset_outputs PRIVATE
  mc_rtc_interface_core
)

target_compile_options(
  test_host_reset_outputs PRIVATE
  -UNDEBUG
)

add_dependencies(
  test_host_reset_outputs
  InstanceProbe
)

add_test(
  NAME host_reset_outputs
  COMMAND test_host_reset_outputs
    "${CMAKE_CURRENT_BINARY_DIR}/host_reset_outputs.yaml"
)

set_tests_properties(
  host_reset_outputs PROPERTIES
  TIMEOUT 60
)

add_executable(
  test_worker_ipc
  tests/test_worker_ipc.cpp
)

target_link_libraries(
  test_worker_ipc PRIVATE
  mc_rtc_interface_ipc
)

target_compile_options(
  test_worker_ipc PRIVATE
  -UNDEBUG
)

add_dependencies(
  test_worker_ipc
  mc_rtc_interface_worker
  InstanceProbe
)

add_test(
  NAME worker_ipc
  COMMAND test_worker_ipc
    "$<TARGET_FILE:mc_rtc_interface_worker>"
    "${CMAKE_CURRENT_BINARY_DIR}/host_reset_outputs.yaml"
)

set_tests_properties(
  worker_ipc PROPERTIES
  TIMEOUT 30
)

add_test(
  NAME worker_cli_help
  COMMAND mc_rtc_interface_worker --help
)

set_tests_properties(
  worker_cli_help PROPERTIES
  PASS_REGULAR_EXPRESSION "--num-controllers"
  TIMEOUT 5
)

add_test(
  NAME worker_cli_missing
  COMMAND mc_rtc_interface_worker
)

set_tests_properties(
  worker_cli_missing PROPERTIES
  WILL_FAIL TRUE
  TIMEOUT 5
)

foreach(count IN ITEMS 0 -1 1x 1.5 18446744073709551616)
    add_test(
      NAME "worker_cli_count_${count}"
      COMMAND mc_rtc_interface_worker
        --endpoint ipc:///unused
        --config "${CMAKE_CURRENT_BINARY_DIR}/host_reset_outputs.yaml"
        --num-controllers "${count}"
    )

    set_tests_properties(
      "worker_cli_count_${count}" PROPERTIES
      WILL_FAIL TRUE
      TIMEOUT 5
    )
endforeach()

# The Python side lives in the repo's own tests/ directory: binding tests plus
# the deterministic action contracts, none of which the C++ tests above reach.
add_test(
  NAME python
  COMMAND "${UV_EXECUTABLE}" run --project "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    python -m pytest "${CMAKE_CURRENT_SOURCE_DIR}/../../tests" -q
)

set_tests_properties(
  python PROPERTIES
  ENVIRONMENT
    "MC_RTC_SHARED_MEMORY_TEST=$<TARGET_FILE:test_shared_memory>;MC_RTC_INSTANCE_PROBE_DIR=$<TARGET_FILE_DIR:InstanceProbe>"
  ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:${CMAKE_BINARY_DIR}"
  TIMEOUT 300
)

# Never attach ctest to the default target under scikit-build: the `python`
# test shells out to `uv run` on this same project, which deadlocks against
# the outer uv invocation driving the build. `--target check` still runs it.
if(DEFINED SKBUILD)
  set(MC_RTC_INTERFACE_CHECK_ALL "")
else()
  set(MC_RTC_INTERFACE_CHECK_ALL "ALL")
endif()

add_custom_target(
  check ${MC_RTC_INTERFACE_CHECK_ALL}
  COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# What the C++ pre-commit hook builds and runs: pytest has its own hook there,
# and running the `python` test from both doubles it onto every mixed commit.
add_custom_target(
  check-native
  COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -E "^python$"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

foreach(check_target check check-native)
  add_dependencies(
    ${check_target}
    test_controllers_manager
    test_shared_memory
    mc_rtc_interface
    InstanceProbe
    test_output_guard
    test_host_reset_outputs
    test_host_failures
    test_host_io
    test_worker_ipc
  )
endforeach()
