cmake_minimum_required(VERSION 2.8) 
set(targetname Demo_LPM) 
project(${targetname}) 

# #####
if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3")
endif (CMAKE_COMPILER_IS_GNUCXX)
# #####

find_package(OpenCV 2.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE Header ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
file(GLOB Source ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/data)

add_executable(demo_lpm demo_lpm.cpp ${Source} ${Header})

target_link_libraries(demo_lpm ${OpenCV_LIBS} )

