cmake_minimum_required(VERSION 3.4)
project(pocket-tensor-benchmark)

# Define sources:
set(SOURCES
    src/main.cpp
    src/kf_bench.cpp
    src/fd_bench.cpp
    src/pt_bench.cpp
)

# Define data folder:
add_definitions(-DPT_BENCHMARK_MODELS_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}/models")

# Add a executable with the above sources:
add_executable(${PROJECT_NAME} ${SOURCES})

# Define include directories:
target_include_directories(${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

# Link static libraries:
target_link_libraries(${PROJECT_NAME} pocket-tensor)

# Include kerasify library:
add_subdirectory(3rd_party/kerasify)
target_link_libraries(${PROJECT_NAME} kerasify)

# Include frugally-deep library:
add_subdirectory(3rd_party/frugally-deep)
target_link_libraries(${PROJECT_NAME} frugally-deep)
