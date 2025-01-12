# set example name according to its directory name (without the leading digits followed by an underscore)
get_filename_component(DIR_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
string(REGEX REPLACE "^[0-9]+_" "" TRIMMED_DIR_NAME "${DIR_NAME}")

# add prefix "tile_example_" to the processed directory name
set(EXAMPLE_NAME "tile_example_${TRIMMED_DIR_NAME}")

set(CONDA_PREFIX "/opt/conda")
set(TORCH_CONFIG_CMAKE "TorchConfig.cmake")

function(find_file_recursively OUTPUT_VAR FILENAME START_DIR)
    # Recursively search for the file
    file(GLOB_RECURSE FOUND_FILES "${START_DIR}/*")

    # Loop through all found files to locate the target file
    foreach(FILE_PATH ${FOUND_FILES})
        if(FILE_PATH MATCHES "${FILENAME}$") # Match the file name
            set(${OUTPUT_VAR} "${FILE_PATH}" PARENT_SCOPE) # Return the full path
            return()
        endif()
    endforeach()

    # If the file is not found, set the output variable to an empty string
    set(${OUTPUT_VAR} "" PARENT_SCOPE)
endfunction()

# Find TorchConfig.cmake recursively
find_file_recursively(FOUND_TORCH_CONFIG_CMAKE "${TORCH_CONFIG_CMAKE}" "${CONDA_PREFIX}")

if(FOUND_TORCH_CONFIG_CMAKE)
    message(STATUS "File found: ${FOUND_TORCH_CONFIG_CMAKE}")

    # Extract the directory of TorchConfig.cmake
    get_filename_component(FILE_DIRECTORY "${FOUND_TORCH_CONFIG_CMAKE}" DIRECTORY)

    # Add the directory to CMAKE_PREFIX_PATH for find_package
    list(APPEND CMAKE_PREFIX_PATH "${FILE_DIRECTORY}")
else()
    message(FATAL_ERROR "File not found: ${TORCH_CONFIG_CMAKE} in ${CONDA_PREFIX}")
endif()

# Use find_package() to locate Torch
find_package(Torch REQUIRED)

add_executable(${EXAMPLE_NAME} EXCLUDE_FROM_ALL main.cpp itfs/paged_attention.cpp)
target_include_directories(${EXAMPLE_NAME} 
  AFTER PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${EXAMPLE_NAME}
  SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(${EXAMPLE_NAME} PRIVATE USE_ROCM)
target_compile_options(${EXAMPLE_NAME} PRIVATE
  ${TORCH_CXX_FLAGS}
  -Wno-undefined-reinterpret-cast
  -Wno-unused-variable
  -Wno-unused-parameter
  -Wno-old-style-cast
  -Wno-deprecated-copy
  -Wno-shadow
)