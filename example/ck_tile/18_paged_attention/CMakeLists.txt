# set example name according to its directory name (without the leading digits followed by an underscore)
get_filename_component(DIR_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
string(REGEX REPLACE "^[0-9]+_" "" TRIMMED_DIR_NAME "${DIR_NAME}")

# add prefix "tile_example_" to the processed directory name
set(EXAMPLE_NAME "tile_example_${TRIMMED_DIR_NAME}")

add_executable(${EXAMPLE_NAME} EXCLUDE_FROM_ALL paged_attention.cpp attention_launcher.cpp)
target_include_directories(${EXAMPLE_NAME} AFTER PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(${EXAMPLE_NAME} PRIVATE USE_ROCM)
target_compile_options(${EXAMPLE_NAME} PRIVATE
  -Wno-undefined-reinterpret-cast
  -Wno-unused-variable
  -Wno-unused-parameter
  -Wno-old-style-cast
  -Wno-deprecated-copy
  -Wno-shadow
)