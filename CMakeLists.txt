cmake_minimum_required(VERSION 3.13)

project(iPic3D)

set(CMAKE_CXX_STANDARD 17)

cmake_policy(SET CMP0042 NEW)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Macro for debugging output
macro(DP var)
    message(NOTICE "${var} = '${${var}}'")
endmacro()

############### SITE CONFIGURATION ###############

function(get_sites OUTVAR)
    file(GLOB site_incs RELATIVE "${CMAKE_SOURCE_DIR}/cmake/sites" "${CMAKE_SOURCE_DIR}/cmake/sites/*.cmake")
    set(sites "")
    foreach(inc "${site_incs}")
        string(REPLACE ".cmake" "" site "${inc}")
        list(APPEND sites "${site}")
    endforeach()
    list(FIND sites "default" def_ind)
    list(REMOVE_ITEM sites "default")
    list(PREPEND sites "default")
    set(${OUTVAR} "${sites}" PARENT_SCOPE)
endfunction()
get_sites(KNOWN_SITES)

set(SITE "default" CACHE STRING "A predefined site configuration to use")
set_property(CACHE SITE PROPERTY STRINGS "${KNOWN_SITES}")
include("${CMAKE_SOURCE_DIR}/cmake/sites/${SITE}.cmake")

############### BUILD OPTIONS ###############

option(BUILD_SHARED_LIBS "Use shared libraries if ON, static if OFF" ON)
option(USE_CATALYST "Use Catalyst adaptor" OFF)
option(USE_BATSRUS "Use BATSRUS flag" OFF)
option(USE_HDF5 "Use HDF5 library" ON)
option(BENCH_MARK "Print tasks time" OFF)
option(USE_OPENMP "Use OpenMP in support loops" ON)
option(PROFILING "Measure runtimes of substages" OFF)
option(PROFILE_MOMENTS "Profile Moment Gatherer" OFF)
option(PROFILE_FIELDS "Profile Field Solver" OFF)
option(USE_PETSC "Use PETSc library for field solver" OFF)

include(GNUInstallDirs)
include(CheckCSourceCompiles)

############### PARAVIEW/CATALYST ###############

if(USE_CATALYST)
    find_package(ParaView 5.7 REQUIRED)
    if (NOT TARGET ParaView::PythonCatalyst)
        message(FATAL_ERROR "ParaView must be built with Catalyst and Python support.")
    endif()
    add_library(iPICAdaptor ${CMAKE_SOURCE_DIR}/catalyst/Adaptor.cxx)
    if (BUILD_SHARED_LIBS)
        install(TARGETS iPICAdaptor)
    endif()
    target_link_libraries(iPICAdaptor PRIVATE ParaView::PythonCatalyst VTK::CommonDataModel Python3::Python)
    target_include_directories(iPICAdaptor INTERFACE ${CMAKE_SOURCE_DIR}/catalyst)
    target_compile_definitions(iPICAdaptor INTERFACE USE_CATALYST)
endif()

############### MPI and OpenMP ###############

# Find MPI early so HDF5 parallel checks can link against MPI::MPI_C
find_package(MPI REQUIRED COMPONENTS C CXX)

if(USE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

############### HDF5 ###############

if (USE_HDF5)
    if(BUILD_SHARED_LIBS)
        set(HDF5_USE_STATIC_LIBRARIES OFF)
    else()
        set(HDF5_USE_STATIC_LIBRARIES ON)
    endif()

    # Conda-forge h5pcc hardcodes build-time compiler paths that don't exist;
    # override with the actual C compiler so the wrapper test succeeds
    if(DEFINED ENV{CONDA_PREFIX})
        set(ENV{HDF5_CC} "${CMAKE_C_COMPILER}")
        set(ENV{HDF5_CLINKER} "${CMAKE_C_COMPILER}")
    endif()

    find_package(HDF5 REQUIRED COMPONENTS C HL)

    # Parallel-HDF5 detection (Better than relying on CMake's HDF5 discovery)
    # Try compiling+linking a tiny program that uses the MPIO file driver API
    set(_CMAKE_REQUIRED_INCLUDES_SAVE "${CMAKE_REQUIRED_INCLUDES}")
    set(_CMAKE_REQUIRED_LIBRARIES_SAVE "${CMAKE_REQUIRED_LIBRARIES}")

    set(CMAKE_REQUIRED_INCLUDES "${HDF5_INCLUDE_DIRS}")
    set(CMAKE_REQUIRED_LIBRARIES "${HDF5_LIBRARIES};MPI::MPI_C")

    check_c_source_compiles("
        #include <mpi.h>
        #include <hdf5.h>
        int main(void) {
            hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
            if (fapl < 0) return 1;
            herr_t status = H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
            (void)status;
            H5Pclose(fapl);
            return 0;
        }
    " HDF5_HAS_PARALLEL_MPIO)

    set(CMAKE_REQUIRED_INCLUDES "${_CMAKE_REQUIRED_INCLUDES_SAVE}")
    set(CMAKE_REQUIRED_LIBRARIES "${_CMAKE_REQUIRED_LIBRARIES_SAVE}")

    if (HDF5_HAS_PARALLEL_MPIO)
        add_compile_definitions(PHDF5 USING_PARALLEL_HDF5)
        message(STATUS "HDF5 is parallel: enabling PHDF5/USING_PARALLEL_HDF5")
    else()
        message(STATUS "HDF5 is NOT parallel (no H5Pset_fapl_mpio): PHDF5 disabled")
    endif()
endif()

############### PETSc ###############

if(USE_PETSC)
    find_package(PkgConfig QUIET)
    if(NOT PkgConfig_FOUND)
        message(FATAL_ERROR
            "USE_PETSC=ON requires pkg-config to locate PETSc, but pkg-config was not found.\n"
            "  Install it (e.g. 'brew install pkg-config' or 'apt install pkg-config'),\n"
            "  then re-run CMake.")
    endif()
    pkg_check_modules(PETSC QUIET IMPORTED_TARGET PETSc)
    if(NOT PETSC_FOUND)
        message(FATAL_ERROR
            "USE_PETSC=ON but PETSc was not found by pkg-config.\n"
            "  Install PETSc (e.g. 'brew install petsc') or set PKG_CONFIG_PATH to include\n"
            "  the directory containing PETSc.pc (typically <petsc-prefix>/lib/pkgconfig).")
    endif()
    message(STATUS "PETSc found: ${PETSC_PREFIX}")
endif()

############### Source Files ###############

file(GLOB
    src_files
    ConfigFile/src/*.cpp
    PSKOutput3D/*.cpp
    bc/*.cpp
    communication/*.cpp
    fields/*.cpp
    grids/*.cpp
    inputoutput/*.cpp
    mathlib/*.cpp
    mpidata/*.cpp
    particles/*.cpp
    performances/*.cpp
    processtopology/*.cpp
    solvers/*.cpp
    utility/*.cpp
    main/*.cpp
)

add_library(iPIC3Dlib ${src_files})
target_include_directories(iPIC3Dlib PUBLIC include)

############### Link Libraries ###############

target_link_libraries(iPIC3Dlib PUBLIC MPI::MPI_CXX)

if(USE_OPENMP)
    target_link_libraries(iPIC3Dlib PUBLIC OpenMP::OpenMP_CXX)
endif()

if(USE_HDF5)
    target_include_directories(iPIC3Dlib PRIVATE ${HDF5_INCLUDE_DIRS})

    # Ensure the linker can find HDF5 libs installed in non-standard prefixes
    if (DEFINED HDF5_LIBRARY_DIRS)
        target_link_directories(iPIC3Dlib PRIVATE ${HDF5_LIBRARY_DIRS})
    endif()

    target_link_libraries(iPIC3Dlib PRIVATE ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
endif()

if(USE_CATALYST)
    target_link_libraries(iPIC3Dlib PUBLIC iPICAdaptor PRIVATE)
endif()

if(USE_PETSC)
    target_link_libraries(iPIC3Dlib PUBLIC PkgConfig::PETSC)
    target_compile_definitions(iPIC3Dlib PUBLIC USE_PETSC)
endif()

############### Compiler Definitions ###############

if(USE_BATSRUS)
    target_compile_definitions(iPIC3Dlib PUBLIC BATSRUS)
    message("WARNING: BATSRUS flag is active.")
else()
    message("INFO: BATSRUS is not active.")
endif()

if(NOT USE_HDF5)
    target_compile_definitions(iPIC3Dlib PUBLIC NO_HDF5)
endif()

if(PROFILING)
    add_definitions(-D__PROFILING__)
endif()

if(PROFILE_MOMENTS)
    add_definitions(-D__PROFILE_MOMENTS__)
endif()

if(PROFILE_FIELDS)
    add_definitions(-D__PROFILE_FIELDS__)
endif()

if(BENCH_MARK)
    target_compile_definitions(iPIC3Dlib PUBLIC LOG_TASKS_TOTAL_TIME)
endif()

############### Executable ###############

add_executable(iPIC3D iPIC3D.cpp)
target_link_libraries(iPIC3D iPIC3Dlib)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set_target_properties(iPIC3D PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} DEBUG_POSTFIX "_d")

############### RPATH Stuff ###############

function(add_install_libdir_to_rpath tgt)
    if(BUILD_SHARED_LIBS)
        get_target_property(install_rpaths ${tgt} INSTALL_RPATH)
        if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
            list(APPEND install_rpaths "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
            list(APPEND install_rpaths "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
        endif()
        list(REMOVE_DUPLICATES install_rpaths)
        set_target_properties(${tgt} PROPERTIES INSTALL_RPATH "${install_rpaths}")
    endif()
endfunction()
add_install_libdir_to_rpath(iPIC3D)

############### Install ###############

install(TARGETS iPIC3D)
add_subdirectory(inputfiles)

############### Debug Info ###############

message("System Information:")
message("HOSTNAME               = ${myhostname}")
message("CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
message("CMAKE_CXX_COMPILER     = ${CMAKE_CXX_COMPILER}")
message("CMAKE_CXX_FLAGS        = ${CMAKE_CXX_FLAGS}")
message("HDF5_INCLUDE_DIRS      = ${HDF5_INCLUDE_DIRS}")
message("HDF5_LIBRARIES         = ${HDF5_LIBRARIES}")
message("HDF5_HL_LIBRARIES      = ${HDF5_HL_LIBRARIES}")
message("MPI_LIBRARIES          = ${MPI_LIBRARIES}")
message("USE_PETSC              = ${USE_PETSC}")
