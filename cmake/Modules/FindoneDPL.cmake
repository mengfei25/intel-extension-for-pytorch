# - Try to find oneDPL
#

IF (NOT ONEDPL_FOUND)
SET(ONEDPL_FOUND OFF)

SET(ONEDPL_LIBRARIES)
SET(ONEDPL_INCLUDE_DIR)

SET(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/third_party")
SET(ONEDPL_DIR "oneDPL")
SET(ONEDPL_ROOT "${THIRD_PARTY_DIR}/${ONEDPL_DIR}")

FIND_PATH(ONEDPL_INCLUDE_DIR oneapi/dpl/algorithm PATHS ${ONEDPL_ROOT} PATH_SUFFIXES include)
IF (NOT ONEDPL_INCLUDE_DIR)
  EXECUTE_PROCESS(
    COMMAND git${CMAKE_EXECUTABLE_SUFFIX} submodule update --init ${ONEDPL_DIR}
    WORKING_DIRECTORY ${THIRD_PARTY_DIR})
  FIND_PATH(ONEDPL_INCLUDE_DIR oneapi/dpl/algorithm PATHS ${ONEDPL_ROOT} PATH_SUFFIXES include)
ENDIF(NOT ONEDPL_INCLUDE_DIR)

IF (NOT ONEDPL_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "oneDPL headers not found!")
ENDIF(NOT ONEDPL_INCLUDE_DIR)

add_library(oneDPL INTERFACE)
IF(NOT TARGET oneDPL)
  MESSAGE(FATAL_ERROR "Failed to include oneDPL target")
ENDIF(NOT TARGET oneDPL)

TARGET_INCLUDE_DIRECTORIES(oneDPL INTERFACE ${ONEDPL_INCLUDE_DIR})
SET(ONEDPL_INCLUDE_DIR ${ONEDPL_INCLUDE_DIR} PARENT_SCOPE)
SET(ONEDPL_FOUND ON)

ENDIF(NOT ONEDPL_FOUND)
