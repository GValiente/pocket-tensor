cmake_minimum_required(VERSION 3.4)
project(catch-mini)

# Enable C++11:
set(CMAKE_CXX_STANDARD 11)

# Define sources:
set(SOURCES
    src/catch.cpp
)

# Add a library with the above sources:
add_library(${PROJECT_NAME} ${SOURCES})

# Define include directories:
target_include_directories(${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
