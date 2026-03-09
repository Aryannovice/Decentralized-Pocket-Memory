cmake_minimum_required(VERSION 3.16)
project(pocket_memory_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_FAISS "Enable FAISS backend" OFF)

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

if(USE_FAISS)
    find_package(faiss REQUIRED)
    add_compile_definitions(USE_FAISS=1)
endif()

# Define the pybind11 module target
pybind11_add_module(pocket_memory_cpp
    src/bindings.cpp
    src/vector_index.cpp
)

# Add include directories
target_include_directories(pocket_memory_cpp PRIVATE src)

# Link OpenMP after the target is created
target_link_libraries(pocket_memory_cpp PRIVATE OpenMP::OpenMP_CXX)

# Link FAISS if enabled
if(USE_FAISS)
    if(TARGET faiss)
        target_link_libraries(pocket_memory_cpp PRIVATE faiss)
    elseif(TARGET faiss::faiss)
        target_link_libraries(pocket_memory_cpp PRIVATE faiss::faiss)
    else()
        message(FATAL_ERROR "FAISS target not found after find_package(faiss)")
    endif()
endif()