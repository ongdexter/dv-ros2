cmake_minimum_required(VERSION 3.8)
project(torch_converter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- PyTorch C++ API configuration ---
execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)


list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(dv_ros2_msgs REQUIRED)
find_package(dv_ros2_messaging REQUIRED)
find_package(dv-processing REQUIRED)
find_package(Torch REQUIRED)

set(dependencies "std_msgs" "rclcpp" "geometry_msgs" "sensor_msgs" "dv_ros2_msgs" "dv_ros2_messaging" "rclcpp_components")


# --- PyTorch C++ API configuration ---
# set(Torch_DIR /usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch)
# set(ATen_DIR /usr/local/lib/python3.10/dist-packages/torch/share/cmake/ATen)
# set(Tensorpipe_DIR /usr/local/lib/python3.10/dist-packages/torch/share/cmake/Tensorpipe)
# find_package(Torch REQUIRED)
# include_directories(${TORCH_INCLUDE_DIRS})
# link_directories(/usr/local/lib/python3.10/dist-packages/torch/lib)

# support eigen and openmp
# find_package(Eigen3 REQU

# --- CUDA configuration ---
set(CUDA 12.6)
set(CUDA_HOME /usr/local/cuda-${CUDA})
set(CUDA_PATH /usr/local/cuda-${CUDA})
set(NVCC /usr/local/cuda-$CUDA/bin/nvcc)

set(CMAKE_CXX_STANDARD 17)

include_directories(include())

# Add CUDA include directories manually
include_directories(${CUDA_HOME}/include)
include_directories(${CUDA_HOME}/targets/aarch64-linux/include)

# Include project-specific include directory
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/")

### Optional for .cu files I guess?
# enable_language(CUDA)
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(engine_loader CUDA::cudart nvinfer)

# --- TensorRT paths ---
# Adjust this path if TensorRT is not under /usr/lib or /usr/local/lib
# link_directories(
#   /usr/lib
#   /usr/local/lib
#   ${CUDA_HOME}/lib64
# )

# Optional: print paths for debugging
message(STATUS "Using CUDA from ${CUDA_HOME}")

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

file(GLOB_RECURSE SOURCES src/*.cpp)


### ENGINE LOAD CHECK EXECUTABLE ###
# add_executable(${PROJECT_NAME}_engine_load_check
#   src/trt_engine_main.cpp
#   src/trt_engine.cpp
# )

# install(TARGETS
#   ${PROJECT_NAME}_engine_load_check
#   DESTINATION bin
# )

# # --- Link libraries ---
# target_link_libraries(${PROJECT_NAME}_engine_load_check
#     nvinfer       # TensorRT runtime
#     cudart        # CUDA runtime
# )
######################################

### INFERENCE NODE COMPONENT ###
# add_library(${PROJECT_NAME}_component SHARED
#   src/inference_node.cpp
#   src/trt_engine.cpp
# )
# ament_target_dependencies(${PROJECT_NAME}_component rclcpp rclcpp_components std_msgs sensor_msgs dv_ros2_messaging dv_ros2_msgs geometry_msgs)
# rclcpp_components_register_nodes(${PROJECT_NAME}_component "ros2_cpp_evfn_infer::InferenceNode")
# target_link_libraries(${PROJECT_NAME}_component
#   ${catkin_LIBRARIES}
#   dv::processing
#   nvinfer
#   cudart
# )

# install(TARGETS
#   ${PROJECT_NAME}_component
#   ARCHIVE DESTINATION lib
# )
################################

### TORCH CONVERTER NODE COMPONENT ###

add_library(torch_converter_component SHARED
  src/torch_converter.cpp
)
ament_target_dependencies(torch_converter_component rclcpp rclcpp_components std_msgs nav_msgs sensor_msgs dv_ros2_messaging dv_ros2_msgs geometry_msgs)
rclcpp_components_register_nodes(torch_converter_component "torch_converter::TorchConverter")

target_link_libraries(torch_converter_component
  ${catkin_LIBRARIES}
  dv::processing
  "${TORCH_LIBRARIES}"
)
set_property(TARGET torch_converter_component PROPERTY CXX_STANDARD 17)

install(TARGETS
  torch_converter_component
  ARCHIVE DESTINATION lib
)

#################################

# TODO: check if config directory is required here
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
