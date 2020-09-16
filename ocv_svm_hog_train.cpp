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

std::vector<float> get_svm_detector( const cv::Ptr<cv::ml::SVM>& svm );
void sample_neg(const std::vector<cv::Mat> &full_neg_lst, std::vector<cv::Mat> &neg_lst, const cv::Size &size);
void computeHOGs(const cv::Size wsize, const std::vector<cv::Mat> & img_lst, std::vector<cv::Mat> &gradient_lst,  bool use_flip);
void convert_to_ml(const std::vector<cv::Mat> &train_samples, cv::Mat &trainData);
void load_images( const std::string &images_path, const std::string &image_resolution, const std::string boxes_path, std::vector<cv::Mat> & img_lst);


int main() {
    int roi_width = 190;
    int roi_height = 190;
    int box_width = 128;
    int box_height = 128;
    bool flip_samples = false;
    bool train_twice = true;

    std::string ok_boxes_path = "src/boxes_h.txt";
    std::string no_ok_boxes_path = "src/boxes_h.txt";

    std::string ok_images_path = "src/train_images_h/";
    std::string no_ok_images_path = "src/no_ok_train_images_h/";
    std::string image_resolution = ".png";

    std::string path_to_saved_detector = "src/my_detector_2.xml";

    std::vector<cv::Mat> ok_train_rois, no_ok_train_rois;
    std::vector<int> labels;
    std::vector<cv::Mat> gradient_lst;
    cv::Mat train_data;
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();

    load_images(ok_images_path, image_resolution, ok_boxes_path, ok_train_rois);
    load_images(no_ok_images_path, image_resolution, no_ok_boxes_path, no_ok_train_rois);

    std::cout << "Positive ROIs loaded: " << ok_train_rois.size() << "\n";
    std::cout << "Negative ROIs loaded: " << no_ok_train_rois.size() << "\n";

    computeHOGs(cv::Size(box_width, box_height), ok_train_rois, gradient_lst, flip_samples);
    computeHOGs(cv::Size(box_width, box_height), no_ok_train_rois, gradient_lst, flip_samples);

    labels.insert(labels.end(), ok_train_rois.size(), 1);
    labels.insert(labels.end(), no_ok_train_rois.size(), -1);

    convert_to_ml(gradient_lst, train_data);

    std::cout << "Training SVM ...";

    svm->setCoef0( 0.0 );
    svm->setDegree( 3 );
    svm->setTermCriteria( cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-3 ) );
    svm->setGamma( 0 );
    svm->setKernel( cv::ml::SVM::LINEAR);
    svm->setNu( 0.5 );
    svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?
    svm->setC( 0.01 ); // From paper, soft classifier
    svm->setType( cv::ml::SVM::EPS_SVR ); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
    svm->train( train_data, cv::ml::ROW_SAMPLE, labels );

    if (train_twice) {
        std::cout << "Testing trained detector on negative images. This might take a few minutes...";
        cv::HOGDescriptor my_hog;
        my_hog.winSize = cv::Size(box_width, box_height);
        // Set the trained svm to my_hog
        my_hog.setSVMDetector( get_svm_detector( svm ) );
        vector< Rect > detections;
        vector< double > foundWeights;
        for ( size_t i = 0; i < full_neg_lst.size(); i++ )
        {
            if ( full_neg_lst[i].cols >= pos_image_size.width && full_neg_lst[i].rows >= pos_image_size.height )
                my_hog.detectMultiScale( full_neg_lst[i], detections, foundWeights );
            else
                detections.clear();
            for ( size_t j = 0; j < detections.size(); j++ )
            {
                Mat detection = full_neg_lst[i]( detections[j] ).clone();
                resize( detection, detection, pos_image_size, 0, 0, INTER_LINEAR_EXACT);
                neg_lst.push_back( detection );
            }
            if ( visualization )
            {
                for ( size_t j = 0; j < detections.size(); j++ )
                {
                    rectangle( full_neg_lst[i], detections[j], Scalar( 0, 255, 0 ), 2 );
                }
                imshow( "testing trained detector on negative images", full_neg_lst[i] );
                waitKey( 5 );
            }
    }

    std::cout << " DONE\n";

    cv::HOGDescriptor hog;
    hog.winSize = cv::Size(box_width, box_height);
    hog.setSVMDetector(get_svm_detector(svm));
    hog.save(path_to_saved_detector);
    return 0;
}

void load_images( const std::string &images_path, const std::string &image_resolution, const std::string boxes_path, std::vector<cv::Mat> & img_lst) {
    std::string current_line, mat_name;
    cv::Mat cur_img;


    std::ifstream boxes_file_stream(boxes_path);
    if (!boxes_file_stream.is_open()) {
        std::cerr << "File \"" + boxes_path + "\" does NOT exist!!!\n";
        return;
    }

    while(!boxes_file_stream.eof()) {
        boxes_file_stream >> current_line;
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

        mat_name = images_path + std::to_string(img_num) + image_resolution;
        cur_img = cv::imread(mat_name);
        img_lst.push_back(cur_img(cv::Rect(x1, y1, x2-x1, y2-y1)).clone());
        cur_img.release();
    }

    boxes_file_stream.close();
}

void sample_neg( const std::vector<cv::Mat> & full_neg_lst, std::vector<cv::Mat> & neg_lst, const cv::Size & size ) {
    cv::Rect box;
    box.width = size.width;
    box.height = size.height;
    srand( time( NULL ) );
    for ( size_t i = 0; i < full_neg_lst.size(); i++ )
        if ( full_neg_lst[i].cols > box.width && full_neg_lst[i].rows > box.height ) {
            box.x = rand() % ( full_neg_lst[i].cols - box.width );
            box.y = rand() % ( full_neg_lst[i].rows - box.height );
            cv::Mat roi = full_neg_lst[i]( box );
            neg_lst.push_back( roi.clone() );
        }
}

void computeHOGs(const cv::Size wsize, const std::vector<cv::Mat> &img_lst, std::vector<cv::Mat> &gradient_lst, bool use_flip ) {
    cv::HOGDescriptor hog;
    hog.winSize = wsize;
    cv::Mat gray;
    std::vector<float> descriptors;
        for( size_t i = 0 ; i < img_lst.size(); i++ ) {
        if ( img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height ) {
            cv::Rect r = cv::Rect(( img_lst[i].cols - wsize.width ) / 2,
                          ( img_lst[i].rows - wsize.height ) / 2,
                          wsize.width,
                          wsize.height);
            cv::cvtColor( img_lst[i](r), gray, cv::COLOR_BGR2GRAY );
            hog.compute( gray, descriptors, cv::Size( 8, 8 ), cv::Size( 0, 0 ) );
            gradient_lst.push_back( cv::Mat( descriptors ).clone() );
            if ( use_flip ) {
                flip( gray, gray, 1 );
                hog.compute( gray, descriptors, cv::Size( 8, 8 ), cv::Size( 0, 0 ) );
                gradient_lst.push_back( cv::Mat( descriptors ).clone() );
            }
        }
    }
}

void convert_to_ml( const std::vector<cv::Mat> &train_samples, cv::Mat &trainData ) {
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows);
    cv::Mat tmp( 1, cols, CV_32FC1); //< used for transposition if needed
    trainData = cv::Mat( rows, cols, CV_32FC1);
    for( size_t i = 0 ; i < train_samples.size(); ++i ){
        CV_Assert( train_samples[i].cols == 1 || train_samples[i].rows == 1);
        if(train_samples[i].cols == 1) {
            transpose( train_samples[i], tmp);
            tmp.copyTo( trainData.row( (int)i ));
        }
        else if(train_samples[i].rows == 1) {
            train_samples[i].copyTo( trainData.row( (int)i ) );
        }
    }
}

std::vector<float> get_svm_detector(const cv::Ptr<cv::ml::SVM>& svm) {
    // get the support vectors
    cv::Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    cv::Mat alpha, svidx;
    double rho = svm->getDecisionFunction( 0, alpha, svidx );
    CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
               (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    CV_Assert( sv.type() == CV_32F );
    std::vector< float > hog_detector( sv.cols + 1 );
    memcpy( &hog_detector[0], sv.ptr(), sv.cols*sizeof( hog_detector[0] ) );
    hog_detector[sv.cols] = (float)-rho;
    return hog_detector;
}