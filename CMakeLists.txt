cmake_minimum_required(VERSION 3.17)
project(opencv_svm_testing)

set(CMAKE_CXX_STANDARD 20)

add_executable(opencv_svm_testing
#        ocv_svm_hog_detector.cpp
#        ocv_svm_hog_train.cpp
        train_HOG.cpp
)

target_link_libraries(opencv_svm_testing seasocks "${ZLIB_LIBRARIES}")
target_link_libraries(opencv_svm_testing pthread z)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_svm_testing ${OpenCV_LIBS})

find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(DLIB "dlib-1")
    if (DLIB_FOUND)
        target_link_libraries(opencv_svm_testing ${DLIB_LIBRARIES})
        add_definitions(${DLIB_CFLAGS} ${DLIB_CFLAGS_OTHER})
    endif()
endif()

find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(GLIB "glib-2.0")
    if (GLIB_FOUND)
        target_link_libraries(opencv_svm_testing ${GLIB_LIBRARIES})
        add_definitions(${GLIB_CFLAGS} ${GLIB_CFLAGS_OTHER})
    endif()
endif()

find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(GST "gstreamer-1.0")
    if (GST_FOUND)
        target_link_libraries(opencv_svm_testing ${GST_LIBRARIES})
        add_definitions(${GST_CFLAGS} ${GST_CFLAGS_OTHER})
    endif()
endif()

find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(SPD "gstreamer-sdp-1.0")
    if (SPD_FOUND)
        target_link_libraries(opencv_svm_testing ${SPD_LIBRARIES})
        add_definitions(${SPD_CFLAGS} ${SPD_CFLAGS_OTHER})
    endif()
endif()

find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(WEBRTC "gstreamer-webrtc-1.0")
    if (WEBRTC_FOUND)
        target_link_libraries(opencv_svm_testing ${WEBRTC_LIBRARIES})
        add_definitions(${WEBRTC_CFLAGS} ${WEBRTC_CFLAGS_OTHER})
    endif()
endif()

find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(JSON "json-glib-1.0")
    if (JSON_FOUND)
        target_link_libraries(opencv_svm_testing ${JSON_LIBRARIES})
        add_definitions(${JSON_CFLAGS} ${JSON_CFLAGS_OTHER})
    endif()
endif()

find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(SOUP "libsoup-2.4")
    if (SOUP_FOUND)
        target_link_libraries(opencv_svm_testing ${SOUP_LIBRARIES})
        add_definitions(${SOUP_CFLAGS} ${SOUP_CFLAGS_OTHER})
    endif()
endif()

add_custom_command(TARGET opencv_svm_testing POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/src $<TARGET_FILE_DIR:opencv_svm_testing>/src)