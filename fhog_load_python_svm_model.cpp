//
// Created by biba_bo on 2020-09-18.
//

#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/opencv.h>

#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>


using namespace std;
using namespace dlib;

int main() {
    //dlib::array<array2d<unsigned char>> images_train;
    typedef scan_fhog_pyramid<pyramid_down<4>> image_scanner_type;
    object_detector<image_scanner_type> detector;
    //deserialize("src/Hand_Detector_v9_c10_t4.svm") >> detector;
    deserialize("src/Hand_Detector_v6_c20.svm");

    std::vector<object_detector<image_scanner_type>> my_detectors_lst;
    my_detectors_lst.push_back(detector);
    //std::vector<rectangle> dets_lst = evaluate_detectors(my_detectors_lst, images_train[0]);

    cv::VideoCapture cap(0);
    //cap.open("/dev/video0");
    std::string named_window_name = "testing_window";
    namedWindow(named_window_name, cv::WINDOW_NORMAL);

    cv::Mat current_frame, resized_frame;
    int new_width = 320;
    int new_height = 240;
    int delay = 0;
    dlib::array2d<unsigned char> dlib_format_frame;
    while(true) {
        if (cap.isOpened()) {
            cap >> current_frame;
            delay = 1;
        } else
            break;
        //trying to recognize:
        cv::resize(current_frame, resized_frame, cv::Size(new_width, new_height));
        cv::cvtColor(resized_frame, resized_frame, cv::COLOR_BGR2GRAY);

        std::cout << "frame " << current_frame.size << "\n";

        //cv::Ptr<IplImage> iplimg = resized_frame.data;
        //dlib_format_frame.set_size(resized_frame.rows, resized_frame.cols);
        dlib::assign_image(dlib_format_frame, dlib::cv_image<unsigned char>(resized_frame));
        //std::vector<rectangle> dets_lst = detector(dlib_format_frame);
        std::vector<rectangle> dets_lst = evaluate_detectors(my_detectors_lst, dlib_format_frame);
        for (int inx = 0; inx < dets_lst.size(); inx++) {
            cv::rectangle(current_frame, cv::Point(dets_lst[inx].left() * 4, dets_lst[inx].top() * 4),
                          cv::Point(dets_lst[inx].right() * 4, dets_lst[inx].bottom() * 4), cv::Scalar(0, 255,0));
        }
        std::cout << "detection count: " << dets_lst.size() << "\n";
        cv::imshow(named_window_name, current_frame);
        if (cv::waitKey(delay) == 27)
            break;
    }
}