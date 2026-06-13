cmake_minimum_required(VERSION 3.8)
project(phantom_echo_reveal)

find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY
  launch worlds params urdf
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
