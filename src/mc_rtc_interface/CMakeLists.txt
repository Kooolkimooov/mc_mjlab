cmake_minimum_required(VERSION 3.26)

project(mc_rtc_interface VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(mc_rtc REQUIRED)

add_library(
  mc_rtc_interface_core STATIC
  cpp/controller_instance.cpp
  cpp/controllers_host.cpp
  cpp/instance_datastore_plugin.cpp
  cpp/output_guard.cpp
  hpp/io_layout.hpp
)

target_include_directories(
  mc_rtc_interface_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

set_target_properties(
  mc_rtc_interface_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(
  mc_rtc_interface_core PUBLIC
  mc_rtc::mc_control
)

include(FetchContent)

FetchContent_Declare(
  glaze
  GIT_REPOSITORY https://github.com/stephenberry/glaze.git
  GIT_TAG v7.0.2
  GIT_SUBMODULES ""
)

FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.6.1
  GIT_SUBMODULES ""
)

set(MC_RTC_INTERFACE_SKIP_INSTALL_RULES "${CMAKE_SKIP_INSTALL_RULES}")
set(CMAKE_SKIP_INSTALL_RULES ON)
FetchContent_MakeAvailable(glaze CLI11)
set_property(DIRECTORY "${glaze_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL ON)
set_property(DIRECTORY "${cli11_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL ON)
set(CMAKE_SKIP_INSTALL_RULES "${MC_RTC_INTERFACE_SKIP_INSTALL_RULES}")
unset(MC_RTC_INTERFACE_SKIP_INSTALL_RULES)

add_library(
  mc_rtc_interface_ipc STATIC
  cpp/ipc_socket.cpp
  cpp/utils.cpp
)

target_include_directories(
  mc_rtc_interface_ipc PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(
  mc_rtc_interface_ipc PUBLIC
  glaze::glaze nanomsg Boost::headers mc_rtc::mc_control
)

set_target_properties(
  mc_rtc_interface_ipc PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

add_executable(
  mc_rtc_interface_worker
  cpp/worker.cpp
)

target_link_libraries(
  mc_rtc_interface_worker PRIVATE
  mc_rtc_interface_core
  mc_rtc_interface_ipc
  CLI11::CLI11
)

find_package(Boost REQUIRED COMPONENTS process)

add_library(
  mc_rtc_interface_manager STATIC
  cpp/controllers_manager.cpp
)

target_include_directories(
  mc_rtc_interface_manager PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(
  mc_rtc_interface_manager PUBLIC
  mc_rtc_interface_ipc
  Boost::process
  ${CMAKE_DL_LIBS}
)

set_target_properties(
  mc_rtc_interface_manager PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# The manager launches the worker executable at runtime.
add_dependencies(
  mc_rtc_interface_manager
  mc_rtc_interface_worker
)

set(
  Python_VENV_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/bin/python"
)

if(NOT DEFINED Python_EXECUTABLE AND EXISTS "${Python_VENV_PATH}")
    set(
      Python_EXECUTABLE
      "${Python_VENV_PATH}"
    )
endif()

find_package(
  Python REQUIRED COMPONENTS
  Interpreter
  Development.Module
)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

find_package(nanobind CONFIG REQUIRED)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND NOT DEFINED SKBUILD)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix)"
    OUTPUT_VARIABLE Python_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  set(
    CMAKE_INSTALL_PREFIX "${Python_PREFIX}"
    CACHE PATH "Install path prefix" FORCE
  )
endif()

execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c
    "import sysconfig; print(sysconfig.get_path('platlib', vars={'base': '', 'platbase': ''}).lstrip('/'))"
  OUTPUT_VARIABLE Python_RELATIVE_SITEARCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

nanobind_add_module(
  mc_rtc_interface
  bindings/module.cpp
)

target_link_libraries(
  mc_rtc_interface PRIVATE
  mc_rtc_interface_manager
)

if(DEFINED SKBUILD)
  set(MC_RTC_INTERFACE_INSTALL_DIR ".")
else()
  set(MC_RTC_INTERFACE_INSTALL_DIR "${Python_RELATIVE_SITEARCH}")
endif()

install(
  TARGETS mc_rtc_interface mc_rtc_interface_worker
  LIBRARY DESTINATION "${MC_RTC_INTERFACE_INSTALL_DIR}"
  RUNTIME DESTINATION "${MC_RTC_INTERFACE_INSTALL_DIR}"
)

message(
  STATUS
  "mc_rtc_interface installs to ${CMAKE_INSTALL_PREFIX}/${MC_RTC_INTERFACE_INSTALL_DIR}"
)

add_custom_target(
  uninstall
  COMMAND
    "${CMAKE_COMMAND}" -E rm -f
    "${CMAKE_INSTALL_PREFIX}/${MC_RTC_INTERFACE_INSTALL_DIR}/$<TARGET_FILE_NAME:mc_rtc_interface>"
    "${CMAKE_INSTALL_PREFIX}/${MC_RTC_INTERFACE_INSTALL_DIR}/$<TARGET_FILE_NAME:mc_rtc_interface_worker>"
  COMMENT "Removing installed mc_rtc_interface module and worker"
  VERBATIM
)

option(
  MC_RTC_INTERFACE_TESTS
  "Run the tests as part of the build"
  ON
)

if(MC_RTC_INTERFACE_TESTS)
    enable_testing()
    include(tests/tests.cmake)
endif()
