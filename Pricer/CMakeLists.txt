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


cuda_add_executable(cuda_info src/Cuda/cuda_info.cu)
target_link_libraries(cuda_info ${LIBS})
install(TARGETS cuda_info DESTINATION bin)

cuda_add_executable(mainCuda src/Cuda/mainCuda.cu src/Cuda/cudaLib.cu src/Cuda/pricer_kernel.cuh src/Model/bs.cpp src/parser.cpp src/Options/utils.cpp src/Options/option.cpp src/Method/mc.cpp src/Options/optionAsian.cpp src/Options/optionBarrier.cpp src/Options/optionBarrierLow.cpp src/Options/optionBarrierUp.cpp src/Options/optionBasket.cpp src/Options/optionPerformance.cpp)
target_link_libraries(mainCuda ${LIBS})

