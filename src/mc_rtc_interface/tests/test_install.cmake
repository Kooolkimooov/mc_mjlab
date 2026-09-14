set(PREFIX "${BUILD_DIR}/test-install")
file(REMOVE_RECURSE "${PREFIX}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${PREFIX}"
  RESULT_VARIABLE INSTALL_RESULT
  OUTPUT_VARIABLE INSTALL_OUTPUT
  ERROR_VARIABLE INSTALL_ERROR
)
if(NOT INSTALL_RESULT EQUAL 0)
  message(FATAL_ERROR "install failed:\n${INSTALL_OUTPUT}${INSTALL_ERROR}")
endif()

set(BUNDLE_DIR "${PREFIX}/${INSTALL_DIR}")
if(NOT EXISTS "${BUNDLE_DIR}/${MODULE_NAME}")
  message(FATAL_ERROR "installed module is missing")
endif()
if(NOT EXISTS "${BUNDLE_DIR}/${WORKER_NAME}")
  message(FATAL_ERROR "installed worker is missing")
endif()

file(GLOB_RECURSE INSTALLED_FILES LIST_DIRECTORIES false "${PREFIX}/*")
list(LENGTH INSTALLED_FILES INSTALLED_COUNT)
if(NOT INSTALLED_COUNT EQUAL 2)
  message(FATAL_ERROR "install contains unexpected files: ${INSTALLED_FILES}")
endif()

execute_process(
  COMMAND "${BUNDLE_DIR}/${WORKER_NAME}" --help
  RESULT_VARIABLE WORKER_RESULT
  OUTPUT_QUIET
  ERROR_QUIET
)
if(NOT WORKER_RESULT EQUAL 0)
  message(FATAL_ERROR "installed worker is not executable")
endif()

file(REMOVE_RECURSE "${PREFIX}")
