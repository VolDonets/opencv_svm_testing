//
// Created by biba_bo on 2020-09-15.
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

void calculateFeaturesFromInput(cv::Mat imageData, std::vector<float>& featuresVector, cv::HOGDescriptor& hog) {
    hog.compute(imageData, featuresVector, cv::Size(8,8), cv::Size(0,0));
}

int main() {
    std::ifstream inputFileStream("src/boxes_h.txt");
    if (!inputFileStream.is_open()) {
        std::cerr << "File does NOT exist!!!\n";
        return 1;
    }

    std::string current_line, mat_name;
    cv::Mat cur_img;
    std::vector<cv::Mat> train_rois;
    while(!inputFileStream.eof()) {
        inputFileStream >> current_line;
        std::stringstream ss(current_line);
        int img_num, x1, y1, x2, y2;
        ss >> img_num;
        ss.ignore();
        ss >> x1;
        ss.ignore();
        ss >> y1;
        ss.ignore();
        ss >> x2;
        ss.ignore();
        ss >> y2;

        mat_name = "src/no_ok_train_images_h/" + std::to_string(img_num) + ".png";
        cur_img = cv::imread(mat_name);
        train_rois.push_back(cur_img(cv::Rect(x1, y1, x2-x1, y2-y1)).clone());
        cur_img.release();

    }

    for (int i = 0; i < train_rois.size(); i++) {
        mat_name = "src/no_okROIs/" + std::to_string(i) + ".png";
        cv::imwrite(mat_name, train_rois[i]);
    }

    exit(0);

    cv::HOGDescriptor hog(
            cv::Size(190,190), //winSize,
            cv::Size(10,10), //blocksize,
            cv::Size(5,5), //blockStride (50% of blockSize),
            cv::Size(80,80), //cellSize,
            9, //nbins,
            1, //derivApper,
            -1, //winSigma,
            cv::HOGDescriptor::L2Hys, //histogramNormType,
            0.2, //L2HysThresh,
            true, //gammal correction,
            cv::HOGDescriptor::DEFAULT_NLEVELS, //nleverls=64
            true //use signed gradients
            );

    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setC(12.5);
    svm->setGamma(0.50625);

    std::vector<int> train_labels;
    //for (int i = 0; i < train_rois.size(); i++)
        train_labels.push_back(1);
    cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(train_rois[0], cv::ml::ROW_SAMPLE, train_labels);
    svm->train(td);
    svm->save("digit_svm_model.yml");
}