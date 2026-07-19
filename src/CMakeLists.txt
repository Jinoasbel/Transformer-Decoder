cmake_minimum_required (VERSION 3.20 )
project (Transformer VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "D:/Library/libtorch")
set(DEPENDENCY_PATH "D:/Library/libtorch/lib")

find_package(
    torch
    REQUIRED
)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(
    ${CMAKE_PROJECT_NAME}
    test.cpp
    transformer.cpp
    attention.cpp
    embedding.cpp
    decoder.cpp
    feedForwardNN.cpp
    positionalEncoder.cpp
)

target_include_directories(
    ${CMAKE_PROJECT_NAME}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(
    ${CMAKE_PROJECT_NAME}
    PRIVATE 
    ${TORCH_LIBRARIES}
)

add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E
    copy_directory
    ${DEPENDENCY_PATH}
    "$<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>"
)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${CMAKE_PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>)
endif (MSVC)