cmake_minimum_required(VERSION 3.0.2)
project(pclprocess)

## ----------------------------------------------------------
## 1. Temukan dan deklarasikan dependency
## ----------------------------------------------------------
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  sensor_msgs
  message_filters
  tf2_ros         # ✅ Diperlukan untuk StaticTransformBroadcaster
)

find_package(OpenCV REQUIRED)

## ----------------------------------------------------------
## 2. Definisikan package ini sebagai catkin package
## ----------------------------------------------------------
catkin_package(
  LIBRARIES pclprocess
  CATKIN_DEPENDS roscpp rospy std_msgs cv_bridge sensor_msgs message_filters tf2_ros
  DEPENDS OpenCV
)

## ----------------------------------------------------------
## 3. Include direktori
## ----------------------------------------------------------
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}   # ✅ Tambahan opsional tapi baik
)

## ----------------------------------------------------------
## 4. Tambah executable dan target
## ----------------------------------------------------------
add_executable(pcl_slice_node src/pcl_slice.cpp)

add_dependencies(pcl_slice_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(pcl_slice_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)


add_executable(pcl_slice_5_node src/pcl_slice_5.cpp)

add_dependencies(pcl_slice_5_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(pcl_slice_5_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

## ----------------------------------------------------------
## 5. Install Python scripts (jika ada)
## ----------------------------------------------------------
install(PROGRAMS
  src/rovid_env.py
  src/train_rovid.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## ----------------------------------------------------------
## 6. Install launch file
## ----------------------------------------------------------
install(DIRECTORY
  launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

## ----------------------------------------------------------
## 7. Install konfigurasi (YAML) file
## ----------------------------------------------------------
install(DIRECTORY
  config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

## ----------------------------------------------------------
## 8. Install world file untuk Gazebo (jika ada)
## ----------------------------------------------------------
install(DIRECTORY
  worlds/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/worlds
)

# Jika punya model Gazebo custom:
# install(DIRECTORY
#   models/
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
# )
