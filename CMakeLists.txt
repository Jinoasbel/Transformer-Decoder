cmake_minimum_required (VERSION 3.20 )
project (TRANSFORMER_DECODER VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "D:/Library/libtorch")
set(DEPENDENCY_PATH "D:/Library/libtorch/lib")

set(LIB_NAME "TRANSFORMER_DECODER")

find_package(
    torch
    REQUIRED
)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(
    ${LIB_NAME}
    STATIC
    src/transformer.cpp
    src/embedding.cpp
    src/positionalEncoder.cpp
    src/decoder.cpp
    src/feedForwardNN.cpp
    src/attention.cpp
    src/heads.cpp
)


target_include_directories(
    ${LIB_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)


target_link_libraries(
    ${LIB_NAME}
    PUBLIC 
    ${TORCH_LIBRARIES}
)

add_custom_command(
    TARGET ${LIB_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E
    copy_directory_if_different
    ${DEPENDENCY_PATH}
    "$<TARGET_FILE_DIR:${LIB_NAME}>"
)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${LIB_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${LIB_NAME}>)
endif (MSVC)