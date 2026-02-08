add_subdirectory(test)

add_library(InplaceVector INTERFACE inplace_vector.hpp)
target_include_directories(InplaceVector INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})