# Native Data2G: C++20 core, Qt front end to follow.
#
# See docs/native-port-plan.md. The Python package stays the normative
# definition of the on-air format; everything here is checked against it.
#
#   tools/build_native.sh --test     # configure, build, ctest, pytest --native
#
# Options:
#   DATA2G_BUILD_PYMODULE  pybind11 module, so pytest can drive the C++ core
#   DATA2G_SANITIZE        ASan + UBSan

cmake_minimum_required(VERSION 3.24)
project(data2g_native VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(DATA2G_BUILD_PYMODULE "Build the pybind11 module for pytest parity runs" ON)
option(DATA2G_SANITIZE "Build with AddressSanitizer and UBSan" OFF)

# Parity is checked to tight tolerances; fast-math would let the optimizer
# reassociate exactly the sums being compared.
if(MSVC)
  add_compile_options(/fp:precise /W4)
else()
  add_compile_options(-fno-fast-math -Wall -Wextra)
endif()

if(DATA2G_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

# --- core: no Qt, no files to find at run time (tools/check_layering.py) ----
add_library(data2g_core STATIC
  core/codes/codes.cpp
  core/generated/format.cpp
)
target_include_directories(data2g_core PUBLIC core)
set_target_properties(data2g_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

enable_testing()
add_subdirectory(tests)

if(DATA2G_BUILD_PYMODULE)
  add_subdirectory(bindings)
endif()
