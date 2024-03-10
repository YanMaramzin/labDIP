cmake_minimum_required(VERSION 3.28)
project(labDIP)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(src)

add_executable(${PROJECT_NAME} main.cpp
               src/imageutils.h
               src/imageutils.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})