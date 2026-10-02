# The pybind11 module that lets pytest drive the C++ core. It must be built
# for the interpreter that runs pytest (tools/build_native.sh passes the
# venv's), and lands in build/python, which tests/conftest.py puts on
# sys.path: a test fixture, never installed, so a stale copy can't shadow it.
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(data2g_native MODULE module.cpp)
target_link_libraries(data2g_native PRIVATE data2g_core)
set_target_properties(data2g_native PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)
message(STATUS "pybind11 module will be built for ${Python3_EXECUTABLE}")
