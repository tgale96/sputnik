# For use in subdirs.
set(SPUTNIK_SRCS)

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
file(GLOB TMP *.cu)
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

# Create libsputnik.
add_library(sputnik SHARED ${SPUTNIK_SRCS})
target_link_libraries(sputnik ${SPUTNIK_LIBS})

# Library installation.
install(TARGETS sputnik LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

set(INSTALL_BASE "include/sputnik")
install(FILES "cuda_utils.h" DESTINATION ${INSTALL_BASE})
install(FILES "test_utils.h" DESTINATION ${INSTALL_BASE})
install(FILES "sputnik.h" DESTINATION ${INSTALL_BASE})
install(FILES "logging.h" DESTINATION ${INSTALL_BASE})
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

  # BDSD test build.
  list(APPEND SPUTNIK_BDSD_TEST_SRCS ${SPUTNIK_TEST_SRCS})

  add_executable(dsd_test ${SPUTNIK_BDSD_TEST_SRCS})
  target_link_libraries(dsd_test sputnik ${SPUTNIK_TEST_LIBS})

  # BDDS test build.
  list(APPEND SPUTNIK_BDDS_TEST_SRCS ${SPUTNIK_TEST_SRCS})

  add_executable(dds_test ${SPUTNIK_BDDS_TEST_SRCS})
  target_link_libraries(dds_test sputnik ${SPUTNIK_TEST_LIBS})

  # BSDD test build.
  list(APPEND SPUTNIK_BSDD_TEST_SRCS ${SPUTNIK_TEST_SRCS})

  add_executable(sdd_test ${SPUTNIK_BSDD_TEST_SRCS})
  target_link_libraries(sdd_test sputnik ${SPUTNIK_TEST_LIBS})

  # BSSD test build.
  list(APPEND SPUTNIK_BSSD_TEST_SRCS ${SPUTNIK_TEST_SRCS})

  add_executable(ssd_test ${SPUTNIK_BSSD_TEST_SRCS})
  target_link_libraries(ssd_test sputnik ${SPUTNIK_TEST_LIBS})

  # BSDS test build.
  list(APPEND SPUTNIK_BSDS_TEST_SRCS ${SPUTNIK_TEST_SRCS})

  add_executable(sds_test ${SPUTNIK_BSDS_TEST_SRCS})
  target_link_libraries(sds_test sputnik ${SPUTNIK_TEST_LIBS})

  # BDSS test build.
  list(APPEND SPUTNIK_BDSS_TEST_SRCS ${SPUTNIK_TEST_SRCS})

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

  # BDSD benchmark build.
  list(APPEND SPUTNIK_BDSD_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})

  add_executable(dsd_benchmark ${SPUTNIK_BDSD_BENCHMARK_SRCS})
  target_link_libraries(dsd_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})

  # BDDS benchmark build.
  list(APPEND SPUTNIK_BDDS_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})

  add_executable(dds_benchmark ${SPUTNIK_BDDS_BENCHMARK_SRCS})
  target_link_libraries(dds_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})

  # BSDD benchmark build.
  list(APPEND SPUTNIK_BSDD_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})

  add_executable(sdd_benchmark ${SPUTNIK_BSDD_BENCHMARK_SRCS})
  target_link_libraries(sdd_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})

  # BSSD benchmark build.
  list(APPEND SPUTNIK_BSSD_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})

  add_executable(ssd_benchmark ${SPUTNIK_BSSD_BENCHMARK_SRCS})
  target_link_libraries(ssd_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})

  # BSDS benchmark build.
  list(APPEND SPUTNIK_BSDS_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})

  add_executable(sds_benchmark ${SPUTNIK_BSDS_BENCHMARK_SRCS})
  target_link_libraries(sds_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})

  # BDSS benchmark build.
  list(APPEND SPUTNIK_BDSS_BENCHMARK_SRCS ${SPUTNIK_BENCHMARK_SRCS})

  add_executable(dss_benchmark ${SPUTNIK_BDSS_BENCHMARK_SRCS})
  target_link_libraries(dss_benchmark sputnik ${SPUTNIK_BENCHMARK_LIBS})
endif()
