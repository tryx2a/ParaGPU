cmake_minimum_required(VERSION 2.8)
project(mc-pricer CXX)

# Prefix
if (NOT PREFIX)
    set(PREFIX ${CMAKE_CURRENT_BINARY_DIR})
endif (NOT PREFIX)
set(CMAKE_INSTALL_PREFIX ${PREFIX} CACHE STRING "Installation prefix." FORCE)

if (CXX)
    set(CMAKE_CXX_COMPILER ${CXX})
endif (CXX)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# Release or Debug
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif ()


# Detect Boost
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# Detect PNL
find_package(Pnl REQUIRED)
set(LIBS ${LIBS} ${PNL_LIBRARIES})
include_directories(${PNL_INCLUDE_DIRS})

# Add math lib
set(LIBS ${LIBS} m)

# Testing the parser
add_executable(parser-test src/parser.cpp src/test_parser.cpp)
target_link_libraries(parser-test ${LIBS})
install(TARGETS parser-test DESTINATION bin)

# Testing BS
 add_executable(bs-test src/test_bs.cpp src/bs.cpp src/parser.cpp src/utils.cpp src/option.cpp src/mc.cpp src/optionAsian.cpp src/optionBarrier.cpp src/optionBarrierLow.cpp src/optionBarrierUp.cpp src/optionBasket.cpp src/optionPerformance.cpp)
 target_link_libraries(bs-test ${LIBS})
 install(TARGETS bs-test DESTINATION bin)
 

add_executable(simul-market-test src/test_simul_market.cpp src/bs.cpp src/parser.cpp src/utils.cpp src/option.cpp src/mc.cpp src/optionAsian.cpp src/optionBarrier.cpp src/optionBarrierLow.cpp src/optionBarrierUp.cpp src/optionBasket.cpp src/optionPerformance.cpp)
target_link_libraries(simul-market-test ${LIBS})
install(TARGETS simul-market-test DESTINATION bin)

# Add google test
if(EXISTS /usr/src/gtest)
    add_subdirectory(/usr/src/gtest ./gtest)
else(EXISTS /usr/src/gtest)
  if(EXISTS /usr/bin/gtest-config)
    execute_process(COMMAND gtest-config --cppflags --cxxflags OUTPUT_VARIABLE CENTOS_GTEST_CFLAGS)
    execute_process(COMMAND gtest-config --ldflags --libs OUTPUT_VARIABLE CENTOS_GTEST_LDFLAGS)
    string(STRIP "${CENTOS_GTEST_CFLAGS}" CENTOS_GTEST_CFLAGS)
    string(STRIP "${CENTOS_GTEST_LDFLAGS}" CENTOS_GTEST_LDFLAGS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CENTOS_GTEST_CFLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CENTOS_GTEST_LDFLAGS} -lgtest_main")
  else(EXISTS /usr/bin/gtest-config)
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})
  endif(EXISTS /usr/bin/gtest-config)
endif(EXISTS /usr/src/gtest)

# Link test executable against gtest & gtest_main

add_executable(runUnitTests 
    src/allTests.cpp
    src/utils.cpp
    src/bs.cpp
    src/optionBarrierLow.cpp 
    src/optionBarrierUp.cpp 
    src/optionBarrier.cpp
    src/optionAsian.cpp 
    test/optionAsian-test.cpp
    src/optionBasket.cpp 
    src/optionPerformance.cpp
    test/optionBasket-test.cpp 
    test/optionBarrier-test.cpp 
    test/optionPerformance-test.cpp
    src/option.cpp 
    test/option-test.cpp
    src/mc.cpp
    src/parser.cpp
    src/bs.cpp
    )

target_link_libraries(runUnitTests ${LIBS} gtest gtest_main)
add_test(RunUnitTests runUnitTests)


add_test(
    NAME runUnitTests
    COMMAND runUnitTests
)

# Main for tests 

# Price option
add_executable(price-option test/integration/price_option.cpp src/bs.cpp src/parser.cpp src/utils.cpp src/option.cpp src/mc.cpp src/optionAsian.cpp src/optionBarrier.cpp src/optionBarrierLow.cpp src/optionBarrierUp.cpp src/optionBasket.cpp src/optionPerformance.cpp)
target_link_libraries(price-option ${LIBS})
install(TARGETS price-option DESTINATION bin)

add_executable(compute-delta test/integration/compute_delta.cpp src/bs.cpp src/parser.cpp src/utils.cpp src/option.cpp src/mc.cpp src/optionAsian.cpp src/optionBarrier.cpp src/optionBarrierLow.cpp src/optionBarrierUp.cpp src/optionBasket.cpp src/optionPerformance.cpp)
target_link_libraries(compute-delta ${LIBS})
install(TARGETS compute-delta DESTINATION bin)

add_executable(profit-and-loss test/integration/profit_and_loss.cpp src/bs.cpp src/parser.cpp src/utils.cpp src/option.cpp src/mc.cpp src/optionAsian.cpp src/optionBarrier.cpp src/optionBarrierLow.cpp src/optionBarrierUp.cpp src/optionBasket.cpp src/optionPerformance.cpp)
target_link_libraries(profit-and-loss ${LIBS})
install(TARGETS profit-and-loss DESTINATION bin)

configure_file(src/pricer.py pricer COPYONLY)