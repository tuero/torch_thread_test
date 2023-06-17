cmake_minimum_required (VERSION 3.12)
project(torch_thread_safe)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# set(ABSL_PROPAGATE_CXX_STD ON)
# add_subdirectory(external/abseil-cpp)
# link_libraries( 
#     absl::base
#     absl::synchronization
#     absl::optional
#     absl::time
# )

add_subdirectory(src)