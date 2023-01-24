# For use in subdirs.
set(SPUTNIK_SRCS)

set(SPUTNIK_SPMM_TEST_SRCS)
set(SPUTNIK_SPMM_BENCHMARK_SRCS)
add_subdirectory(spmm)

set(SPUTNIK_SDDMM_TEST_SRCS)
set(SPUTNIK_SDDMM_BENCHMARK_SRCS)
add_subdirectory(sddmm)

set(SPUTNIK_DEPTHWISE_TEST_SRCS)
set(SPUTNIK_DEPTHWISE_BENCHMARK_SRCS)
add_subdirectory(depthwise)

add_subdirectory(bias_relu)
add_subdirectory(softmax)
add_subdirectory(utils)
add_subdirectory(block/transpose)
add_subdirectory(block/bitmask)
add_subdirectory(block/row_indices)

set(SPUTNIK_BDSD_TEST_SRCS)
set(SPUTNIK_BDSD_BENCHMARK_SRCS)
add_subdirectory(block/dsd)

set(SPUTNIK_BDDS_TEST_SRCS)
set(SPUTNIK_BDDS_BENCHMARK_SRCS)
add_subdirectory(block/dds)

set(SPUTNIK_BSDD_TEST_SRCS)
set(SPUTNIK_BSDD_BENCHMARK_SRCS)
add_subdirectory(block/sdd)

set(SPUTNIK_BSSD_TEST_SRCS)
set(SPUTNIK_BSSD_BENCHMARK_SRCS)
add_subdirectory(block/ssd)

set(SPUTNIK_BSDS_TEST_SRCS)
set(SPUTNIK_BSDS_BENCHMARK_SRCS)
add_subdirectory(block/sds)

set(SPUTNIK_BDSS_TEST_SRCS)
set(SPUTNIK_BDSS_BENCHMARK_SRCS)
add_subdirectory(block/dss)

# TODO(tgale): Clean up build.
add_subdirectory(block/cutlass)

set(DIR_SRCS)

##
### Find all sources in this directory.
##

# Lib srcs.
file(GLOB TMP *.h)
list(APPEND DIR_SRCS ${TMP})
file(GLOB TMP *.cc)
list(APPEND DIR_SRCS ${TMP})

##
### Filter files that we don't want in the main library.
##

set(FILTER_SRCS)

# Don't want test related code.
file(GLOB TMP test_utils*)
list(APPEND FILTER_SRCS ${TMP})

# Don't need matrix utilities.
file(GLOB TMP matrix_utils*)
list(APPEND FILTER_SRCS ${TMP})

foreach(FILE ${FILTER_SRCS})
  list(REMOVE_ITEM DIR_SRCS ${FILE})
endforeach(FILE)

# Add the sources to the build.
list(APPEND SPUTNIK_SRCS ${DIR_SRCS})

# Set .cu.cc files to be compiled as CUDA.
set(SPUTNIK_CUDA_SRCS ${SPUTNIK_SRCS})
list(FILTER SPUTNIK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
set_source_files_properties(${SPUTNIK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)

# Create libsputnik.
add_library(sputnik STATIC ${SPUTNIK_SRCS})
target_link_libraries(sputnik ${SPUTNIK_LIBS})

# Library installation.
install(TARGETS sputnik LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

set(INSTALL_BASE "include/sputnik")
install(FILES "cuda_utils.h" DESTINATION ${INSTALL_BASE})
install(FILES "test_utils.h" DESTINATION ${INSTALL_BASE})
install(FILES "sputnik.h" DESTINATION ${INSTALL_BASE})
install(FILES "bias_relu/bias_relu.h" DESTINATION "${INSTALL_BASE}/bias_relu")
install(FILES "depthwise/cuda_depthwise.h" DESTINATION "${INSTALL_BASE}/depthwise")
install(FILES "spmm/cuda_spmm.h" DESTINATION "${INSTALL_BASE}/spmm")
install(FILES "sddmm/cuda_sddmm.h" DESTINATION "${INSTALL_BASE}/sddmm")
install(FILES "softmax/softmax.h" DESTINATION "${INSTALL_BASE}/softmax")
install(FILES "softmax/sparse_softmax.h" DESTINATION "${INSTALL_BASE}/softmax")
install(FILES "utils/index_format.h" DESTINATION "${INSTALL_BASE}/utils")
install(FILES "block/arguments.h" DESTINATION "${INSTALL_BASE}/block")
install(FILES "block/dsd/dsd.h" DESTINATION "${INSTALL_BASE}/block/dsd")
install(FILES "block/dds/dds.h" DESTINATION "${INSTALL_BASE}/block/dds")
install(FILES "block/sdd/sdd.h" DESTINATION "${INSTALL_BASE}/block/sdd")
install(FILES "block/ssd/ssd.h" DESTINATION "${INSTALL_BASE}/block/ssd")
install(FILES "block/sds/sds.h" DESTINATION "${INSTALL_BASE}/block/sds")
install(FILES "block/dss/dss.h" DESTINATION "${INSTALL_BASE}/block/dss")
install(FILES "block/row_indices/row_indices.h" DESTINATION "${INSTALL_BASE}/block/row_indices")

# Optionally build the test suite.
if (BUILD_TEST)
  # Test sources for all targets.
  set(SPUTNIK_TEST_SRCS)
  file(GLOB TMP matrix_utils*)
  list(APPEND SPUTNIK_TEST_SRCS ${TMP})
  file(GLOB TMP block/matrix_utils*)
  list(APPEND SPUTNIK_TEST_SRCS ${TMP})
  file(GLOB TMP test_utils*)
  list(APPEND SPUTNIK_TEST_SRCS ${TMP})

  # SpMM test build.
  list(APPEND SPUTNIK_SPMM_TEST_SRCS ${SPUTNIK_TEST_SRCS})

  set(SPUTNIK_SPMM_TEST_CUDA_SRCS ${SPUTNIK_SPMM_TEST_SRCS})
  list(FILTER SPUTNIK_SPMM_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_SPMM_TEST_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
  add_executable(spmm_test ${SPUTNIK_SPMM_TEST_SRCS})
  target_link_libraries(spmm_test sputnik ${SPUTNIK_TEST_LIBS})

  # SDDMM test build.
  list(APPEND SPUTNIK_SDDMM_TEST_SRCS ${SPUTNIK_TEST_SRCS})

  set(SPUTNIK_SDDMM_TEST_CUDA_SRCS ${SPUTNIK_SDDMM_TEST_SRCS})
  list(FILTER SPUTNIK_SDDMM_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_SDDMM_TEST_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
  add_executable(sddmm_test ${SPUTNIK_SDDMM_TEST_SRCS})
  target_link_libraries(sddmm_test sputnik ${SPUTNIK_TEST_LIBS})

  # Depthwise test build.
  list(APPEND SPUTNIK_DEPTHWISE_TEST_SRCS ${SPUTNIK_TEST_SRCS})

  set(SPUTNIK_DEPTHWISE_TEST_CUDA_SRCS ${SPUTNIK_DEPTHWISE_TEST_SRCS})
  list(FILTER SPUTNIK_DEPTHWISE_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(
    ${SPUTNIK_DEPTHWISE_TEST_CUDA_SRCS}
    PROPERTIES LANGUAGE CUDA)
  add_executable(depthwise_test ${SPUTNIK_DEPTHWISE_TEST_SRCS})
  target_link_libraries(depthwise_test sputnik ${SPUTNIK_TEST_LIBS})

  # BDSD test build.
  list(APPEND SPUTNIK_BDSD_TEST_SRCS ${SPUTNIK_TEST_SRCS})

  set(SPUTNIK_BDSD_TEST_CUDA_SRCS ${SPUTNIK_BDSD_TEST_SRCS})
  list(FILTER SPUTNIK_BDSD_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_BDSD_TEST_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
  add_executable(dsd_test ${SPUTNIK_BDSD_TEST_SRCS})
  target_link_libraries(dsd_test sputnik ${SPUTNIK_TEST_LIBS})

  # BDDS test build.
  list(APPEND SPUTNIK_BDDS_TEST_SRCS ${SPUTNIK_TEST_SRCS})

  set(SPUTNIK_BDDS_TEST_CUDA_SRCS ${SPUTNIK_BDDS_TEST_SRCS})
  list(FILTER SPUTNIK_BDDS_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_BDDS_TEST_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
  add_executable(dds_test ${SPUTNIK_BDDS_TEST_SRCS})
  target_link_libraries(dds_test sputnik ${SPUTNIK_TEST_LIBS})

  # BSDD test build.
  list(APPEND SPUTNIK_BSDD_TEST_SRCS ${SPUTNIK_TEST_SRCS})

  set(SPUTNIK_BSDD_TEST_CUDA_SRCS ${SPUTNIK_BSDD_TEST_SRCS})
  list(FILTER SPUTNIK_BSDD_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_BSDD_TEST_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
  add_executable(sdd_test ${SPUTNIK_BSDD_TEST_SRCS})
  target_link_libraries(sdd_test sputnik ${SPUTNIK_TEST_LIBS})

  # BSSD test build.
  list(APPEND SPUTNIK_BSSD_TEST_SRCS ${SPUTNIK_TEST_SRCS})

  set(SPUTNIK_BSSD_TEST_CUDA_SRCS ${SPUTNIK_BSSD_TEST_SRCS})
  list(FILTER SPUTNIK_BSSD_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_BSSD_TEST_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
  add_executable(ssd_test ${SPUTNIK_BSSD_TEST_SRCS})
  target_link_libraries(ssd_test sputnik ${SPUTNIK_TEST_LIBS})

  # BSDS test build.
  list(APPEND SPUTNIK_BSDS_TEST_SRCS ${SPUTNIK_TEST_SRCS})

  set(SPUTNIK_BSDS_TEST_CUDA_SRCS ${SPUTNIK_BSDS_TEST_SRCS})
  list(FILTER SPUTNIK_BSDS_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_BSDS_TEST_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
  add_executable(sds_test ${SPUTNIK_BSDS_TEST_SRCS})
  target_link_libraries(sds_test sputnik ${SPUTNIK_TEST_LIBS})

  # BDSS test build.
  list(APPEND SPUTNIK_BDSS_TEST_SRCS ${SPUTNIK_TEST_SRCS})

  set(SPUTNIK_BDSS_TEST_CUDA_SRCS ${SPUTNIK_BDSS_TEST_SRCS})
  list(FILTER SPUTNIK_BDSS_TEST_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_BDSS_TEST_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
  add_executable(dss_test ${SPUTNIK_BDSS_TEST_SRCS})
  target_link_libraries(dss_test sputnik ${SPUTNIK_TEST_LIBS})
endif()

# Optionally build the benchmark suite.
if (BUILD_BENCHMARK)
  # Benchmark sources for all targets.
  set(SPUTNIK_BENCHMARK_SRCS)
  file(GLOB TMP matrix_utils*)
  list(APPEND SPUTNIK_BENCHMARK_SRCS ${TMP})
  file(GLOB TMP block/matrix_utils*)
  list(APPEND SPUTNIK_BENCHMARK_SRCS ${TMP})
  file(GLOB TMP test_utils*)
  list(APPEND SPUTNIK_BENCHMARK_SRCS ${TMP})

  # SpMM benchmark build.
  list(APPEND SPUTNIK_SPMM_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})

  set(SPUTNIK_SPMM_BENCHMARK_CUDA_SRCS ${SPUTNIK_SPMM_BENCHMARK_SRCS})
  list(FILTER SPUTNIK_SPMM_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_SPMM_BENCHMARK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
  add_executable(spmm_benchmark ${SPUTNIK_SPMM_BENCHMARK_SRCS})
  target_link_libraries(spmm_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})

  # SDDMM benchmark build.
  list(APPEND SPUTNIK_SDDMM_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})

  set(SPUTNIK_SDDMM_BENCHMARK_CUDA_SRCS ${SPUTNIK_SDDMM_BENCHMARK_SRCS})
  list(FILTER SPUTNIK_SDDMM_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_SDDMM_BENCHMARK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
  add_executable(sddmm_benchmark ${SPUTNIK_SDDMM_BENCHMARK_SRCS})
  target_link_libraries(sddmm_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})

  # Depthwise benchmark build.
  list(APPEND SPUTNIK_DEPTHWISE_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})

  set(SPUTNIK_DEPTHWISE_BENCHMARK_CUDA_SRCS ${SPUTNIK_DEPTHWISE_BENCHMARK_SRCS})
  list(FILTER SPUTNIK_DEPTHWISE_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(
    ${SPUTNIK_DEPTHWISE_BENCHMARK_CUDA_SRCS}
    PROPERTIES LANGUAGE CUDA)
  add_executable(depthwise_benchmark ${SPUTNIK_DEPTHWISE_BENCHMARK_SRCS})
  target_link_libraries(depthwise_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})

  # BDSD benchmark build.
  list(APPEND SPUTNIK_BDSD_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})

  set(SPUTNIK_BDSD_BENCHMARK_CUDA_SRCS ${SPUTNIK_BDSD_BENCHMARK_SRCS})
  list(FILTER SPUTNIK_BDSD_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_BDSD_BENCHMARK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
  add_executable(dsd_benchmark ${SPUTNIK_BDSD_BENCHMARK_SRCS})
  target_link_libraries(dsd_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})

  # BDDS benchmark build.
  list(APPEND SPUTNIK_BDDS_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})

  set(SPUTNIK_BDDS_BENCHMARK_CUDA_SRCS ${SPUTNIK_BDDS_BENCHMARK_SRCS})
  list(FILTER SPUTNIK_BDDS_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_BDDS_BENCHMARK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
  add_executable(dds_benchmark ${SPUTNIK_BDDS_BENCHMARK_SRCS})
  target_link_libraries(dds_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})

  # BSDD benchmark build.
  list(APPEND SPUTNIK_BSDD_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})

  set(SPUTNIK_BSDD_BENCHMARK_CUDA_SRCS ${SPUTNIK_BSDD_BENCHMARK_SRCS})
  list(FILTER SPUTNIK_BSDD_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_BSDD_BENCHMARK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
  add_executable(sdd_benchmark ${SPUTNIK_BSDD_BENCHMARK_SRCS})
  target_link_libraries(sdd_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})

  # BSSD benchmark build.
  list(APPEND SPUTNIK_BSSD_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})

  set(SPUTNIK_BSSD_BENCHMARK_CUDA_SRCS ${SPUTNIK_BSSD_BENCHMARK_SRCS})
  list(FILTER SPUTNIK_BSSD_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_BSSD_BENCHMARK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
  add_executable(ssd_benchmark ${SPUTNIK_BSSD_BENCHMARK_SRCS})
  target_link_libraries(ssd_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})

  # BSDS benchmark build.
  list(APPEND SPUTNIK_BSDS_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})

  set(SPUTNIK_BSDS_BENCHMARK_CUDA_SRCS ${SPUTNIK_BSDS_BENCHMARK_SRCS})
  list(FILTER SPUTNIK_BSDS_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_BSDS_BENCHMARK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
  add_executable(sds_benchmark ${SPUTNIK_BSDS_BENCHMARK_SRCS})
  target_link_libraries(sds_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})

  # BDSS benchmark build.
  list(APPEND SPUTNIK_BDSS_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})

  set(SPUTNIK_BDSS_BENCHMARK_CUDA_SRCS ${SPUTNIK_BDSS_BENCHMARK_SRCS})
  list(FILTER SPUTNIK_BDSS_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPUTNIK_BDSS_BENCHMARK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
  add_executable(dss_benchmark ${SPUTNIK_BDSS_BENCHMARK_SRCS})
  target_link_libraries(dss_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})
endif()
