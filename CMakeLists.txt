cmake_minimum_required(VERSION 3.25)

project(InplaceVector VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
enable_testing()

# Declare GoogleTest as a dependency
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)

# Make GoogleTest available to your project
FetchContent_MakeAvailable(googletest)

add_subdirectory(src)
