include(GoogleTest)

add_executable(
    InplaceVectorTest
    inplace_vector_test.cpp
)
target_compile_definitions(
    InplaceVectorTest PUBLIC CHECKED_ITERATORS=1
)
target_link_libraries(
    InplaceVectorTest
    InplaceVector
    GTest::gmock
    GTest::gtest_main 
)
add_test(
    NAME InplaceVectorTest
    COMMAND $<TARGET_FILE:InplaceVectorTest>
)

gtest_discover_tests(InplaceVectorTest)