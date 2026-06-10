cmake_minimum_required(VERSION 3.28)
project(citadel_alpha CXX)

# ── Language standard ──────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build type ─────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── Compiler flags ─────────────────────────────────────────────────────────
add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -march=native -funroll-loops -ffast-math)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-g -fsanitize=address,undefined)
  add_link_options(-fsanitize=address,undefined)
endif()

# ── Python ─────────────────────────────────────────────────────────────────
find_package(Python 3.13 COMPONENTS Interpreter Development.Module REQUIRED)

# ── nanobind: pip-installed (Docker/offline) or FetchContent (CI/local) ────
include(FetchContent)

find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
  message(STATUS "nanobind not found via find_package — fetching via FetchContent")
  FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG        v2.4.0
  )
  FetchContent_MakeAvailable(nanobind)
endif()

# ── Google Test (skipped when BUILD_TESTING=OFF, e.g. in Docker) ──────────
option(BUILD_TESTING "Build C++ unit tests" ON)
if(BUILD_TESTING)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
  )
  FetchContent_MakeAvailable(googletest)
endif()

# ── Core library ───────────────────────────────────────────────────────────
add_library(alpha_engine_lib STATIC src/cpp/alpha_engine.cpp)
target_include_directories(alpha_engine_lib PUBLIC src/cpp)
target_compile_features(alpha_engine_lib PUBLIC cxx_std_26)
set_target_properties(alpha_engine_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ── nanobind extension module ──────────────────────────────────────────────
nanobind_add_module(_citadel_alpha_cpp STABLE_ABI src/cpp/bindings.cpp)
target_link_libraries(_citadel_alpha_cpp PRIVATE alpha_engine_lib)
target_include_directories(_citadel_alpha_cpp PRIVATE src/cpp)

install(TARGETS _citadel_alpha_cpp
  LIBRARY DESTINATION src/python/citadel_alpha
)

# ── C++ unit tests ─────────────────────────────────────────────────────────
if(BUILD_TESTING)
  enable_testing()
  add_executable(test_cpp_lib tests/cpp/test_cpp_lib.cpp)
  target_link_libraries(test_cpp_lib PRIVATE alpha_engine_lib GTest::gtest_main)
  target_include_directories(test_cpp_lib PRIVATE src/cpp)
  include(GoogleTest)
  gtest_discover_tests(test_cpp_lib)
endif()
