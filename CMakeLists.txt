cmake_minimum_required(VERSION 3.16)
project(RingBuffer VERSION 1.0 DESCRIPTION "A Lock-free ring buffer" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

if (MSVC)
    set(CMAKE_CXX_FLAGS "/DWIN32_LEAN_AND_MEAN /DNOGDI /DNOMINMAX /EHsc")
endif (MSVC)

add_executable(RingBuffer main.cpp pixel_dist_poc.cpp RingBuffer.h Atomic.h)

if (UNIX)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(RingBuffer PRIVATE Threads::Threads atomic)

    #set(CMAKE_EXE_LINKER_FLAGS "-latomic")
endif (UNIX)

