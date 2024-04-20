cmake_minimum_required(VERSION 3.14)
project(ml_controller)

find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(Eigen3 REQUIRED)
find_package(tensorrt_common)
find_package(cuda_utils)
if(NOT ${tensorrt_common_FOUND})
  message(WARNING "The tensorrt_common package is not found. Please check its dependencies.")
  return()
endif()
find_package(OpenMP)
if(OpenMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


include_directories(
  SYSTEM
    ${EIGEN3_INCLUDE_DIRS}
)

# ml_controller_core
ament_auto_add_library(ml_controller_core SHARED
  src/ml_controller_core/planning_utils.cpp
  src/ml_controller_core/ml_controller.cpp
  src/ml_controller_core/interpolate.cpp
)

# ml_controller
ament_auto_add_library(ml_controller_node SHARED
  src/ml_controller/ml_controller_node.cpp
  src/ml_controller/ml_controller_viz.cpp
)

target_link_libraries(ml_controller_node
  ml_controller_core
)

rclcpp_components_register_node(ml_controller_node
  PLUGIN "ml_controller::MlControllerNode"
  EXECUTABLE ml_controller_node_exe
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(INSTALL_TO_SHARE
  launch
  config
)
