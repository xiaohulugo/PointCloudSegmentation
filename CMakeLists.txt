# set project's name
PROJECT( PointCloudSegmentation )

###############################################################################
# CMake settings
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)

# OpenCV
FIND_PACKAGE(OpenCV REQUIRED)

FILE(GLOB_RECURSE HDRS_FILES "src/*.h" "src/*.hpp")
FILE(GLOB_RECURSE SRCS_FILES "src/*.c" "src/*.cpp")

ADD_EXECUTABLE(${PROJECT_NAME} ${SRCS_FILES} ${HDRS_FILES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS})