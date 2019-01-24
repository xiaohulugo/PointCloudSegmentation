# set project's name
PROJECT( PointCloudSegmentation )
macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD 11)
  endif ()
endmacro(use_cxx11)
###############################################################################
# CMake settings
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
use_cxx11()

# OpenCV
FIND_PACKAGE(OpenCV REQUIRED)

FILE(GLOB_RECURSE HDRS_FILES "src/*.h" "src/*.hpp")
FILE(GLOB_RECURSE SRCS_FILES "src/*.c" "src/*.cpp")

ADD_EXECUTABLE(${PROJECT_NAME} ${SRCS_FILES} ${HDRS_FILES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS})