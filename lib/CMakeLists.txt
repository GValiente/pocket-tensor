cmake_minimum_required(VERSION 3.4)
project(pocket-tensor)

# Define sources:
set(SOURCES
    src/pt_tensor.cpp
    src/pt_dispatcher.cpp
    src/pt_layer.cpp
    src/pt_dense_layer.cpp
    src/pt_conv_1d_layer.cpp
    src/pt_conv_2d_layer.cpp
    src/pt_locally_connected_1d_layer.cpp
    src/pt_elu_layer.cpp
    src/pt_activation_layer.cpp
    src/pt_max_pooling_2d_layer.cpp
    src/pt_lstm_layer.cpp
    src/pt_embedding_layer.cpp
    src/pt_batch_normalization_layer.cpp
    src/pt_leaky_relu_layer.cpp
    src/pt_model.cpp
)

# Add a library with the above sources:
add_library(${PROJECT_NAME} ${SOURCES})

# Define include directories:
target_include_directories(${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

# Include libsimdpp library:
target_include_directories(${PROJECT_NAME}
    PUBLIC ${PT_LIBSIMDPP_PATH}
)

# Include pthread (GCC only):
if(CMAKE_COMPILER_IS_GNUCC)
    target_link_libraries(${PROJECT_NAME} pthread)
endif()
