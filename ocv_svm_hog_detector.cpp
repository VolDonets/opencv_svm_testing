//
// Created by biba_bo on 2020-09-16.
//

#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

void test_trained_detector(std::string obj_det_filename, std::string videofilename) {
    std::cout << "Testing trained detector..." << "\n";
    cv::HOGDescriptor hog;
    hog.load(obj_det_filename);

    int delay = 0;
    cv::VideoCapture cap;
    if (videofilename != "") {
        if (videofilename.size() == 1 && isdigit(videofilename[0]))
            cap.open(videofilename[0] - '0');
        else
            cap.open(videofilename);
    }
    obj_det_filename = "testing " + obj_det_filename;
    namedWindow(obj_det_filename, cv::WINDOW_NORMAL);
    for (size_t i = 0;; i++) {
        cv::Mat img;
        if (cap.isOpened()) {
            cap >> img;
            delay = 1;
        }
        if (img.empty()) {
            return;
        }
        std::vector<cv::Rect> detections;
        std::vector<double> foundWeights;
        hog.detectMultiScale(img, detections, foundWeights);
        for (size_t j = 0; j < detections.size(); j++) {
            cv::Scalar color = cv::Scalar(0, foundWeights[j] * foundWeights[j] * 200, 0);
            rectangle(img, detections[j], color, img.cols / 400 + 1);
        }
        std::cout << "img params: " << img.cols << " * " << img.rows << "\n";
        cv::imshow(obj_det_filename, img);
        if (cv::waitKey(delay) == 27) {
            return;
        }
    }
}

int main() {
    std::string path_to_saved_detector = "src/my_detector_2.xml";
    test_trained_detector(path_to_saved_detector, "/dev/video0");
}
