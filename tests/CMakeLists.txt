cmake_minimum_required(VERSION 3.4)
project(pocket-tensor-tests)

# Define sources:
set(SOURCES
    src/test_util.cpp
    src/conv1d_2_test.cpp
    src/conv1d_3_test.cpp
    src/conv1d_3x3_test.cpp
    src/conv_2x2_test.cpp
    src/conv_3x3_test.cpp
    src/conv_3x3x3_test.cpp
    src/locally_connected_1d_2_test.cpp
    src/locally_connected_1d_3_test.cpp
    src/locally_connected_1d_3x3_test.cpp
    src/conv_hard_sigmoid_2x2_test.cpp
    src/conv_sigmoid_2x2_test.cpp
    src/conv_softplus_2x2_test.cpp
    src/dense_10x10_test.cpp
    src/dense_10x10x10_test.cpp
    src/dense_10x1_test.cpp
    src/dense_1x1_test.cpp
    src/dense_2x2_test.cpp
    src/dense_elu_10_test.cpp
    src/dense_relu_10_test.cpp
    src/dense_softmax_10_test.cpp
    src/dense_softsign_10_test.cpp
    src/dense_tanh_10_test.cpp
    src/dense_selu_10_test.cpp
    src/elu_10_test.cpp
    src/leaky_relu_10_test.cpp
    src/maxpool2d_1x1_test.cpp
    src/maxpool2d_2x2_test.cpp
    src/maxpool2d_3x2x2_test.cpp
    src/maxpool2d_3x3x3_test.cpp
    src/maxpool2d_8x2x2_test.cpp
    src/maxpool2d_8x3x3_test.cpp
    src/global_maxpool2d_1_test.cpp
    src/global_maxpool2d_3_test.cpp
    src/global_maxpool2d_8_test.cpp
    src/relu_10_test.cpp
    src/embedding_64_test.cpp
    src/lstm_simple_7x20_test.cpp
    src/lstm_simple_stacked_16x9_test.cpp
    src/lstm_stacked_64x83_test.cpp
    src/input_test.cpp
    src/repeat_vector_test.cpp
)

# Define data folder:
add_definitions(-DPT_TEST_MODELS_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}/models")

# Add a executable with the above sources:
add_executable(${PROJECT_NAME} ${SOURCES})

# Define include directories:
target_include_directories(${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

# Link static libraries:
target_link_libraries(${PROJECT_NAME} pocket-tensor)

# Include catch-mini library:
add_subdirectory(3rd_party/catch-mini)
target_link_libraries(${PROJECT_NAME} catch-mini)
