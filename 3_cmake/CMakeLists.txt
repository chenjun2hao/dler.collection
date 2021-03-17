cmake_minimum_required(VERSION 3.2)
project(test)

option(ONTX2 "build project on tx2" OFF)
IF(ONTX2)
    add_definitions(-Dtx2)
ENDIF(ONTX2)

set(source net.cpp)
add_executable(test ${source})