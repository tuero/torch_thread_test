# Source files
set(COMMON_SOURCES
    model.cpp
    search.cpp
    rnd/util.cpp 
    rnd/stonesngems_base.cpp
)

# # Main entry point
# add_executable(main_works main_works.cpp ${COMMON_SOURCES})
# target_compile_options(main_works PRIVATE 
#     -Wall -Wextra 
#     $<$<CONFIG:RELEASE>:-O3> $<$<CONFIG:RELEASE>:-DNDEBUG>
#     $<$<CONFIG:DEBUG>:-g> $<$<CONFIG:DEBUG>:-O0> $<$<CONFIG:DEBUG>:-DDEBUG> $<$<CONFIG:DEBUG>:-pg>
# )
# target_include_directories(main_works PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(main_works ${TORCH_LIBRARIES})

# # Main entry point
# add_executable(main_crashes main_crashes.cpp ${COMMON_SOURCES})
# target_compile_options(main_crashes PRIVATE 
#     -Wall -Wextra 
#     $<$<CONFIG:RELEASE>:-O3> $<$<CONFIG:RELEASE>:-DNDEBUG>
#     $<$<CONFIG:DEBUG>:-g> $<$<CONFIG:DEBUG>:-O0> $<$<CONFIG:DEBUG>:-DDEBUG> $<$<CONFIG:DEBUG>:-pg>
# )
# target_include_directories(main_crashes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(main_crashes ${TORCH_LIBRARIES})



# Main entry point
add_executable(main main.cpp ${COMMON_SOURCES})
target_compile_options(main PRIVATE 
    -Wall -Wextra 
    $<$<CONFIG:RELEASE>:-O3> $<$<CONFIG:RELEASE>:-DNDEBUG>
    $<$<CONFIG:DEBUG>:-g> $<$<CONFIG:DEBUG>:-O0> $<$<CONFIG:DEBUG>:-DDEBUG> $<$<CONFIG:DEBUG>:-pg>
)
target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(main ${TORCH_LIBRARIES})