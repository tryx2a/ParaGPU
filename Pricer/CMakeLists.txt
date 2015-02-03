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
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch sm_12)


# Release or Debug
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif ()

# Detect Cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIR})

# Detect Boost
set(Boost_NO_BOOST_CMAKE on)
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# Detect PNL
find_package(Pnl REQUIRED)
set(LIBS ${LIBS} ${PNL_LIBRARIES})
include_directories(${PNL_INCLUDE_DIRS})

# Add math lib
set(LIBS ${LIBS} m)


# Price option
#add_executable(price-option test/integration/price_option.cpp src/bs.cpp src/parser.cpp src/utils.cpp src/option.cpp src/mc.cpp src/optionAsian.cpp src/optionBarrier.cpp src/optionBarrierLow.cpp src/optionBarrierUp.cpp src/optionBasket.cpp src/optionPerformance.cpp)
#target_link_libraries(price-option ${LIBS})
#install(TARGETS price-option DESTINATION bin)

#add_executable(compute-delta test/integration/compute_delta.cpp src/bs.cpp src/parser.cpp src/utils.cpp src/option.cpp src/mc.cpp src/optionAsian.cpp src/optionBarrier.cpp src/optionBarrierLow.cpp src/optionBarrierUp.cpp src/optionBasket.cpp src/optionPerformance.cpp)
#target_link_libraries(compute-delta ${LIBS})
#install(TARGETS compute-delta DESTINATION bin)

#add_executable(profit-and-loss test/integration/profit_and_loss.cpp src/bs.cpp src/parser.cpp src/utils.cpp src/option.cpp src/mc.cpp src/optionAsian.cpp src/optionBarrier.cpp src/optionBarrierLow.cpp src/optionBarrierUp.cpp src/optionBasket.cpp src/optionPerformance.cpp)
#target_link_libraries(profit-and-loss ${LIBS})
#install(TARGETS profit-and-loss DESTINATION bin)


cuda_add_executable(cuda_info src/cuda_info.cu)
target_link_libraries(cuda_info ${LIBS})
install(TARGETS cuda_info DESTINATION bin)

#cuda_add_executable(matrix_simple src/matrix-simple.cu)
#target_link_libraries(matrix_simple ${LIBS})
#install(TARGETS matrix_simple DESTINATION bin)

cuda_add_executable(mainCuda src/mainCuda.cu src/cudaLib.cu src/bs.cpp src/parser.cpp src/utils.cpp src/option.cpp src/mc.cpp src/optionAsian.cpp src/optionBarrier.cpp src/optionBarrierLow.cpp src/optionBarrierUp.cpp src/optionBasket.cpp src/optionPerformance.cpp)
target_link_libraries(mainCuda ${LIBS})
install(TARGETS mainCuda DESTINATION bin)

cuda_add_executable(test_rand src/rand.cu)
target_link_libraries(test_rand ${LIBS})
install(TARGETS test_rand DESTINATION bin)

cuda_add_executable(test_rand_2 src/rand2.cu)
target_link_libraries(test_rand_2 ${LIBS})
install(TARGETS test_rand_2 DESTINATION bin)
