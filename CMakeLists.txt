# cmake_minimum_required(VERSION 3.24.0)
# project(yunet_demo)

# # 设置 C++ 标准
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # 查找 OpenCV
# find_package(OpenCV REQUIRED)

# # 添加可执行文件
# add_executable(demo main.cpp)

# # 链接 OpenCV 库
# target_link_libraries(demo ${OpenCV_LIBS}) 

cmake_minimum_required(VERSION 3.24.0)
project(opencv_zoo_face_detection_yunet)

set(OPENCV_VERSION "4.10.0")
set(OPENCV_INSTALLATION_PATH "E:/OpenCV/opencv")

# Find OpenCV
find_package(OpenCV ${OPENCV_VERSION} REQUIRED HINTS ${OPENCV_INSTALLATION_PATH})

add_executable(demo demo.cpp)
target_link_libraries(demo ${OpenCV_LIBS})