add_subdirectory(test)

add_library(
    InplaceVector INTERFACE
    inplace_vector.hpp
    detail/attic.hpp
    detail/container_compatible_range.hpp
    detail/inplace_vector_forward.hpp
    detail/iterator.hpp
    detail/storage.hpp
)
target_include_directories(
    InplaceVector INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)