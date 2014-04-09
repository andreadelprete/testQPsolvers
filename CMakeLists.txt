#
# Copyright 2014 CNRS
# Author: Andrea Del Prete
#

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/cpack.cmake)
INCLUDE(cmake/eigen.cmake)

SET(PROJECT_NAME testQPsolvers)
SET(PROJECT_DESCRIPTION "Tests of different QP solvers")
SET(PROJECT_URL "http://github.com/andreadelprete/testQPsolvers")

SET(CXX_DISABLE_WERROR True)
SET(CMAKE_VERBOSE_MAKEFILE true)

project(${PROJECT_NAME})

#SETUP_PROJECT()
SEARCH_FOR_EIGEN()

file(GLOB ${PROJECT_NAME}_H    "include/testQPsolvers/*.h")
file(GLOB ${PROJECT_NAME}_SRC  "src/*.cpp")

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include
                    /home/adelpret/devel/yarp_gazebo/src/qpOASES-3.0beta/include)

link_directories(/home/adelpret/devel/yarp_gazebo/src/qpOASES-3.0beta/bin)

add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_H} ${${PROJECT_NAME}_SRC})

target_link_libraries(${PROJECT_NAME} libqpOASES.so)

PKG_CONFIG_APPEND_CFLAGS(${_Eigen_CFLAGS})
#SETUP_PROJECT_FINALIZE()
#SETUP_PROJECT_CPACK()

