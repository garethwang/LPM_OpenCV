cmake_minimum_required(VERSION 2.8) 
set(targetname Demo_LPM) 
project(${targetname}) 

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(demo_lpm demo_lpm.cpp ${SOURCE_FILES} ${HEADER_FILES})

target_link_libraries(demo_lpm ${OpenCV_LIBS})

