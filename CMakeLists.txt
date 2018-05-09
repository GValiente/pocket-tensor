cmake_minimum_required(VERSION 3.4)
project(pocket-tensor-project)

# Enable C++11:
set(CMAKE_CXX_STANDARD 11)

# Detect Clang:
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_COMPILER_IS_CLANGXX 1)
endif()

# Set compiler warnings
# (https://github.com/lefticus/cppbestpractices/blob/master/02-Use_the_Tools_Available.md#compilers)
# (https://stackoverflow.com/questions/3375697/useful-gcc-flags-for-c)
# (https://stackoverflow.com/questions/5088460/flags-to-enable-thorough-and-verbose-g-warnings):
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANGXX)
    add_definitions("-Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wcast-align -Wcast-qual -Wunused
	-Woverloaded-virtual -Wpedantic -Wdouble-promotion -Wformat=2 -Wfloat-equal -Wpointer-arith
	-Wwrite-strings -Wswitch-enum -Wunreachable-code -Winit-self -Wuninitialized
	-Wno-unused-result -Wctor-dtor-privacy -Wdisabled-optimization -Wmissing-declarations
        -Wredundant-decls -Wno-unused")
endif()

# Add fast math support
# (https://stackoverflow.com/questions/7420665/what-does-gccs-ffast-math-actually-do):
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANGXX)
    add_definitions("-ffast-math")
elseif(MSVC)
    add_definitions("/fp:fast")
endif()

# Add native CPU features support:
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANGXX)
    add_definitions("-march=native")

    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
        add_definitions("-mfpu=neon")
    endif()
elseif(MSVC)
    add_definitions("/arch:AVX")
endif()

# Define libsimdpp library path:
set(PT_LIBSIMDPP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/libsimdpp" CACHE STRING "libsimdpp library path")

# Add library subdirectory:
add_subdirectory(lib)

# Define build all artefacts option:
option(PT_BUILD_ALL "Build all PocketTensor artefacts" OFF)

# Add tests subdirectory:
option(PT_BUILD_TESTS "Build PocketTensor tests" OFF)
if(PT_BUILD_TESTS OR PT_BUILD_ALL)
    add_subdirectory(tests)
endif()
