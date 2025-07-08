cmake_minimum_required(VERSION 3.0.2)
project(pclprocess)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(pclprocess)
# Mengatur standar C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## ----------------------------------------------------------
## 1. Menemukan Dependensi
## ----------------------------------------------------------
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  sensor_msgs
  message_filters
  nav_core
  costmap_2d
  pluginlib
  tf2_ros
)

# Menemukan PCL (Point Cloud Library)
find_package(PCL REQUIRED)

# Menemukan OpenCV
find_package(OpenCV REQUIRED)

# Menemukan LibTorch menggunakan metode execute_process yang sudah bekerja untuk Anda
execute_process(
  COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TorchPath
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH "${TorchPath};${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)

## ----------------------------------------------------------
## 2. Mendefinisikan Paket
## ----------------------------------------------------------
catkin_package(
  INCLUDE_DIRS include ${PCL_INCLUDE_DIRS}
  LIBRARIES drl_local_planner_plugin
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    cv_bridge
    sensor_msgs
    message_filters
    nav_core
    costmap_2d
    pluginlib
    tf2_ros
  DEPENDS PCL OpenCV Torch
)

## ----------------------------------------------------------
## 3. Menambahkan Include Directories
## ----------------------------------------------------------
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Torch_INCLUDE_DIRS}
)

## ----------------------------------------------------------
## 4. Membangun Target
## ----------------------------------------------------------

# Plugin sebagai library
add_library(drl_local_planner_plugin src/drl_local_planner_plugin.cpp)

# Menautkan library plugin dengan dependensi yang diperlukan
target_link_libraries(drl_local_planner_plugin
  ${catkin_LIBRARIES}
  ${TORCH_LIBRARIES}
)

# Executable node lainnya
add_executable(pcl_slice_node src/pcl_slice.cpp)
target_link_libraries(pcl_slice_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(pcl_slice_5_node src/pcl_slice_5.cpp)
target_link_libraries(pcl_slice_5_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})

## ----------------------------------------------------------
## 5. Menginstal Target
## ----------------------------------------------------------
install(TARGETS
  drl_local_planner_plugin
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# ... (sisa file install Anda) ...
install(PROGRAMS
  src/rovid_env.py
  src/train_rovid.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
install(DIRECTORY config/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config)
install(DIRECTORY worlds/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/worlds)
