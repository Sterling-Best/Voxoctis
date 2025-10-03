cmake_minimum_required(VERSION 3.16)
project(voxoctis_core LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Detect x86/x64
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _cpu)
if (_cpu MATCHES "x86_64|amd64|x86|i[3-6]86")
  set(VOX_ON_X86 TRUE)
else()
  set(VOX_ON_X86 FALSE)
endif()

# CPU feature flags (BMI2/AVX2) – only on x86/x64
if (VOX_ON_X86)
  if (MSVC)
    add_compile_options(/arch:AVX2)      # enables AVX2 + BMI2 intrinsics on MSVC
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-mbmi2)          # BMI2
    # Optional: match MSVC by also turning on AVX2
    add_compile_options(-mavx2)
  endif()
endif()

# Paths
set(VOX_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(VOX_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Put outputs in core/bin (same folder for all configs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
foreach(cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${cfg} ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${cfg} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endforeach()

# Library
add_library(vox_depthlocationcode STATIC
  ${VOX_SRC_DIR}/depth_location_code.cpp
)
target_include_directories(vox_depthlocationcode PUBLIC ${VOX_INC_DIR})

# (Optional) a little warning hygiene
if (MSVC)
  target_compile_options(vox_depthlocationcode PRIVATE /W4 /permissive-)
else()
  target_compile_options(vox_depthlocationcode PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Executable
add_executable(voxoctis
  ${VOX_SRC_DIR}/main.cpp
)
target_include_directories(voxoctis PRIVATE ${VOX_INC_DIR})
target_link_libraries(voxoctis PRIVATE vox_depthlocationcode)