cmake_minimum_required(VERSION 3.16)
project(voxoctis_core LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Put all outputs in core/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Your first runnable program
add_executable(hello_core
  src/main.cpp
)

# If you add headers under core/include:
target_include_directories(hello_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
