cmake_minimum_required(VERSION 3.4)
project(kerasify)

# Define sources:
set(SOURCES
    src/layer.cpp
    src/layers/activation.cpp
    src/layers/conv1d.cpp
    src/layers/conv2d.cpp
    src/layers/dense.cpp
    src/layers/elu.cpp
    src/layers/embedding.cpp
    src/layers/flatten.cpp
    src/layers/lstm.cpp
    src/layers/locally1d.cpp
    src/layers/locally2d.cpp
    src/layers/maxpooling2d.cpp
    src/layers/normalization.cpp
    src/model.cpp
    src/tensor.cpp
    src/utility.cpp
)

# Add a library with the above sources:
add_library(${PROJECT_NAME} ${SOURCES})

# Define include directories:
target_include_directories(${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
