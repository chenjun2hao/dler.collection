project(rgb)
cmake_minimum_required()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(rgb rgb.cpp)
target_link_libraries(rgb ${OpenCV_LIBS})