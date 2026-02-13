include(GoogleTest)

set(CMAKE_BUILD_TYPE "Debug")

add_executable(
    InplaceVectorTest
    inplace_vector_test.cpp
)
target_compile_definitions(
    InplaceVectorTest PRIVATE
    CHECKED_ITERATORS=1
)
target_compile_options(
    InplaceVectorTest PRIVATE
    -fprofile-instr-generate 
    -fcoverage-mapping
)
target_link_libraries(
    InplaceVectorTest
    InplaceVector
    GTest::gmock
    GTest::gtest_main 
)
target_link_options(
    InplaceVectorTest PRIVATE 
    -fprofile-instr-generate 
    -fcoverage-mapping
)
add_test(
    NAME InplaceVectorTest
    COMMAND $<TARGET_FILE:InplaceVectorTest>
)

gtest_discover_tests(InplaceVectorTest)

find_program(LLVM_PROFDATA llvm-profdata REQUIRED)
find_program(LLVM_COV llvm-cov REQUIRED)
find_program(BROWSER NAMES x-www-browser REQUIRED)

add_custom_target(
  coverage
  COMMAND ${LLVM_PROFDATA} merge -o ${CMAKE_BINARY_DIR}/default.profdata $<TARGET_FILE_DIR:InplaceVectorTest>/default.profraw
  COMMAND ${LLVM_COV} show -Xdemangler=c++filt -instr-profile=default.profdata $<TARGET_FILE:InplaceVectorTest> -format=html -output-dir=coverage_report
  COMMAND echo ${CMAKE_BINARY_DIR}/coverage-report/index.html
  COMMAND ${BROWSER} coverage_report/index.html
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  VERBATIM USES_TERMINAL
)
add_dependencies(coverage InplaceVectorTest)
