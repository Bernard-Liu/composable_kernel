include_directories(AFTER
  ${CMAKE_CURRENT_LIST_DIR}
)

# add all example subdir
file(GLOB dir_list LIST_DIRECTORIES true *)
FOREACH(subdir ${dir_list})
    if(IS_DIRECTORY "${subdir}" AND EXISTS "${subdir}/CMakeLists.txt")
        add_subdirectory(${subdir})
    ENDIF()
ENDFOREACH()