cmake_minimum_required(VERSION 3.20)
project(theta_streamer C)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)

include_directories(
	${GST_INCLUDE_DIRS}
	${GST_APP_INCLUDE_DIRS}
)

link_directories(${GST_LIBRARY_DIRS})

add_executable(theta_streamer src/main.c)

target_link_libraries(theta_streamer ${GST_LIBRARIES} ${GST_APP_LIBRARIES})

target_compile_options(theta_streamer PRIVATE ${GST_CFLAGS_OTHER})
