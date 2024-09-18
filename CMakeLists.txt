cmake_minimum_required(VERSION 3.5)
project(forest)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Open3D REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)


# Add the executable
add_executable(cybathlon src/cybathlon.cc)
add_executable(temp src/temp.cc)

ament_target_dependencies(cybathlon rclcpp Eigen3 OpenCV Open3D cv_bridge nav_msgs)
ament_target_dependencies(temp rclcpp Eigen3 OpenCV Open3D cv_bridge nav_msgs)
# # Link libraries
# target_include_directories(cybathlon PRIVATE
#   ${EIGEN3_INCLUDE_DIR}
#   ${OpenCV_INCLUDE_DIRS}
#   ${Open3D_INCLUDE_DIRS}
# )

# target_link_libraries(cybathlon
#   ${OpenCV_LIBRARIES}
#   ${Open3D_LIBRARIES}
# )

install(TARGETS cybathlon temp
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
