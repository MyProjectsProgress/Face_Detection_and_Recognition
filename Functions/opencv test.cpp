//#include <vector>
//#include <cmath>
//#include<iostream>
//#include <stdio.h>
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
//using namespace std;
//using namespace cv;
////
////void BGRtoLuv(cv::Mat& bgr_image, cv::Mat& luv_image) {
////    const float epsilon = 0.008856;
////    const float kappa = 903.3;
////    const cv::Vec3f whitePoint(0.9505, 1.0000, 1.0890);
////
////    // resize the input image
////    cv::resize(bgr_image, bgr_image, cv::Size(500, 500));
////
////    // create a new image to store the Luv values
////    luv_image.create(bgr_image.size(), CV_32FC3);
////
////    for (int i = 0; i < bgr_image.rows; i++) {
////        for (int j = 0; j < bgr_image.cols; j++) {
////            cv::Vec3f bgr = bgr_image.at<cv::Vec3f>(i, j);
////            float b = bgr[0];
////            float g = bgr[1];
////            float r = bgr[2];
////
////            // convert BGR to XYZ
////            float x = 0.4124 * r + 0.3576 * g + 0.1805 * b;
////            float y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
////            float z = 0.0193 * r + 0.1192 * g + 0.9505 * b;
////
////            // normalize XYZ values
////            cv::Vec3f xyz(x, y, z);
////            cv::divide(xyz, whitePoint, xyz);
////
////            // calculate L*, u*, and v* values
////            float L, u, v;
////            if (xyz[1] > epsilon) {
////                L = 116.0 * pow(xyz[1], 1.0 / 3.0) - 16.0;
////            }
////            else {
////                L = kappa * xyz[1];
////            }
////
////            float uPrime = 4.0 * xyz[0] / (xyz[0] + 15.0 * xyz[1] + 3.0 * xyz[2]);
////            float vPrime = 9.0 * xyz[1] / (xyz[0] + 15.0 * xyz[1] + 3.0 * xyz[2]);
////            u = 13.0 * L * (uPrime - whitePoint[0]);
////            v = 13.0 * L * (vPrime - whitePoint[1]);
////
////            // store Luv values in the new image
////            cv::Vec3f luv(L, u, v);
////            luv_image.at<cv::Vec3f>(i, j) = luv;
////        }
////    }
////}
//
////#include <iostream>
////#include <vector>
////#include <opencv2/opencv.hpp>
////
////using namespace cv;
////using namespace std;
////
////void kmeans_euclidean(const Mat& data, int num_clusters, Mat& labels, Mat& centers, int max_iterations = 100)
////{
////    // Initialize random centroids
////    Mat centers_old, centers_new;
////    data.rowRange(0, num_clusters).copyTo(centers_new);
////    centers_new.copyTo(centers_old);
////
////    // Loop until convergence or maximum iterations reached
////    int iter = 0;
////    while (iter < max_iterations)
////    {
////        // Assign each data point to the nearest centroid
////        for (int i = 0; i < data.rows; i++)
////        {
////            double min_dist = numeric_limits<double>::max();
////            int label = 0;
////
////            for (int j = 0; j < num_clusters; j++)
////            {
////                double dist = norm(data.row(i) - centers_new.row(j));
////                if (dist < min_dist)
////                {
////                    min_dist = dist;
////                    label = j;
////                }
////            }
////
////            labels.at<int>(i) = label;
////        }
////
////        // Update the centroid of each cluster
////        centers_new.setTo(0);
////        vector<int> count(num_clusters, 0);
////
////        for (int i = 0; i < data.rows; i++)
////        {
////            int label = labels.at<int>(i);
////            centers_new.row(label) += data.row(i);
////            count[label]++;
////        }
////
////        for (int j = 0; j < num_clusters; j++)
////        {
////            if (count[j] != 0)
////            {
////                centers_new.row(j) /= count[j];
////            }
////        }
////
////        // Check for convergence
////        double epsilon = 0.001;
////        double diff = norm(centers_new - centers_old);
////        if (diff < epsilon)
////        {
////            break;
////        }
////
////        // Update old centroids and increment iteration counter
////        centers_new.copyTo(centers_old);
////        iter++;
////    }
////
////    // Set final cluster centers
////    centers_new.copyTo(centers);
////}
////
////
////int main() {
////    // Reading the image
////    Mat image = imread("D:/Screenshotfrom20200703024321.png", IMREAD_COLOR);
////
////    // Convert the image to LUV color space
////    Mat luv_image;
////    cvtColor(image, luv_image, COLOR_BGR2Luv);
////
////    // Convert the image to a 2D matrix for k-means clustering
////    Mat data;
////    luv_image.convertTo(data, CV_32F);
////    data = data.reshape(1, data.rows * data.cols);
////
////    // Perform k-means clustering
////    int num_clusters = 10;
////    Mat labels = Mat::zeros(data.rows, 1, CV_32SC1);
////    Mat centers;
////    //kmeans(data, num_clusters, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
////    kmeans_euclidean(data, num_clusters, labels, centers, 5);
////    // Reshape the labels and centers to match the LUV image size
////    labels = labels.reshape(1, luv_image.rows);
////    centers = centers.reshape(1, num_clusters);
////
////    // Generate the segmented image
////    Mat segmented_image(luv_image.size(), luv_image.type());
////    for (int i = 0; i < luv_image.rows; i++)
////    {
////        for (int j = 0; j < luv_image.cols; j++)
////        {
////            int label = labels.at<int>(i, j);
////            segmented_image.at<Vec3b>(i, j) = centers.at<Vec3f>(label, 0);
////        }
////    }
////
////    // Convert the segmented image back to BGR color space
////    Mat bgr_segmented_image;
////    cvtColor(segmented_image, bgr_segmented_image, COLOR_Luv2BGR);
////
////    // Display the original and segmented images
////    namedWindow("Original Image", WINDOW_NORMAL);
////    imshow("Original Image", image);
////    namedWindow("Segmented Image", WINDOW_NORMAL);
////    imshow("Segmented Image", bgr_segmented_image);
////    waitKey(0);
////
////    return 0;
////}
//
//
//#include <iostream>
//#include <vector>
//#include <opencv2/opencv.hpp>
//
//using namespace std;
//using namespace cv;
//
////void kmeans_euclidean(const Mat& X, int K, Mat& idx, Mat& centroids, int max_iters) {
////    int m = X.rows;
////    int n = X.cols;
////
////    // Randomly initialize K centroids
////    RNG rng;
////    centroids = Mat::zeros(K, n, CV_32F);
////    for (int i = 0; i < K; i++) {
////        int row = rng.uniform(0, m);
////        X.row(row).copyTo(centroids.row(i));
////    }
////
////    Mat previous_centroids;
////    for (int iter = 0; iter < max_iters; iter++) {
////        // Assign each data point to the closest centroid
////        idx = Mat::zeros(m, 1, CV_32S);
////        for (int i = 0; i < m; i++) {
////            float min_dist = numeric_limits<float>::max();
////            int closest_centroid = 0;
////            for (int j = 0; j < K; j++) {
////                float dist = norm(X.row(i), centroids.row(j));
////                if (dist < min_dist) {
////                    min_dist = dist;
////                    closest_centroid = j;
////                }
////            }
////            idx.at<int>(i, 0) = closest_centroid;
////        }
////
////        // Update centroids
////        previous_centroids = centroids.clone();
////        for (int i = 0; i < K; i++) {
////            Mat points;
////            for (int j = 0; j < m; j++) {
////                if (idx.at<int>(j, 0) == i) {
////                    points.push_back(X.row(j));
////                }
////            }
////            if (!points.empty()) {
////                reduce(points, centroids.row(i), 0, REDUCE_AVG);
////            }
////        }
////
////        // Check for convergence
////        double delta = norm(centroids, previous_centroids);
////        if (delta < 1e-5) {
////            break;
////        }
////    }
////}
//void kmeans_euclidean(const Mat& X, int K, Mat& idx, Mat& centroids, int max_iters) {
//    int m = X.rows;
//    int n = X.cols;
//
//    // Randomly initialize K centroids
//    RNG rng;
//    centroids = Mat::zeros(K, n, CV_32F);
//    for (int i = 0; i < K; i++) {
//        int row = rng.uniform(0, m);
//        X.row(row).copyTo(centroids.row(i));
//    }
//
//    Mat previous_centroids;
//    for (int iter = 0; iter < max_iters; iter++) {
//        // Assign each data point to the closest centroid
//        idx = Mat::zeros(m, 1, CV_32S);
//        for (int i = 0; i < m; i++) {
//            float min_dist = numeric_limits<float>::max();
//            int closest_centroid = 0;
//            for (int j = 0; j < K; j++) {
//                float dist = norm(X.row(i), centroids.row(j));
//                if (dist < min_dist) {
//                    min_dist = dist;
//                    closest_centroid = j;
//                }
//            }
//            idx.at<int>(i, 0) = closest_centroid;
//        }
//
//        // Update centroids
//        previous_centroids = centroids.clone();
//        for (int i = 0; i < K; i++) {
//            Mat points;
//            for (int j = 0; j < m; j++) {
//                if (idx.at<int>(j, 0) == i) {
//                    points.push_back(X.row(j));
//                }
//            }
//            if (!points.empty()) {
//                Mat1f sum;
//                reduce(points, sum, 0, REDUCE_SUM);
//                centroids.row(i) = sum / static_cast<float>(points.rows);
//            }
//        }
//
//        // Check for convergence
//        double delta = norm(centroids, previous_centroids);
//        if (delta < 1e-5) {
//            break;
//        }
//    }
//}
//
//
//int main() {
//    // Reading the image
//    Mat image = imread("D:/16476928522307.jpg", IMREAD_COLOR);
//    resize(image, image, Size(512, 512));
//
//    // Convert the image to LUV color space
//    Mat luv_image;
//    cvtColor(image, luv_image, COLOR_BGR2Luv);
//    // Convert the image to a 2D matrix for k-means clustering
//    Mat data;
//    luv_image.convertTo(data, CV_32F);
//    data = data.reshape(1, data.rows * data.cols);
//
//    // Perform k-means clustering
//    int num_clusters = 2;
//    Mat labels, centers;
//    kmeans_euclidean(data, num_clusters, labels, centers, 5);
//
//    // Reshape the labels and centers to match the LUV image size
//    labels = labels.reshape(1, luv_image.rows);
//    centers = centers.reshape(1, num_clusters);
//
//    // Generate the segmented image
//    Mat segmented_image(luv_image.size(), luv_image.type());
//    for (int i = 0; i < luv_image.rows; i++) {
//        for (int j = 0; j < luv_image.cols; j++) {
//            int label = labels.at<int>(i, j);
//            segmented_image.at<Vec3b>(i, j) = centers.at<Vec3f>(label, 0);
//        }
//    }
//    // Convert the segmented image back to BGR color space
//    Mat bgr_image;
//    cvtColor(segmented_image, bgr_image, COLOR_Luv2BGR);
//    // Display the original and segmented images
//    imshow("Original Image", image);
//    imshow("Segmented Image", bgr_image);
//    waitKey(0);
//
//    return 0;
//}
//

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "ReadFile.h"
#include "GetFrame.h"
#include "MyPCA.h"
#include "FaceRecognizer.h"
#include "WriteTrainData.h"
#include "FaceDetector.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    string trainListFilePath = "C:/Users/ahmad/source/repos/opencv test/list/train_list.txt";
    vector<string> trainFacesPath;
    vector<string> trainFacesID;
    vector<string> loadedFacesID;
    //read training list and ID from txt file
    readList(trainListFilePath, trainFacesPath, trainFacesID);
    //read training data(faces, eigenvector, average face) from txt file
    Mat avgVec, eigenVec, facesInEigen;
    facesInEigen = readFaces(int(trainFacesID.size()), loadedFacesID);
    avgVec = readMean();
    eigenVec = readEigen(int(trainFacesID.size()));

    Mat frame, processed, testImg;

    cout << "++++++Welcome to Face Recognisation System++++++" << endl;
    cout << "Prepare Faces(0) or Training(1) or Recognise(2), Input your number:  ";
    int choice;
    cin >> choice;

    if (choice == 0) {
        cout << "Prepare Face Start......" << endl;
        //Initialize capture
        GetFrame getFrame(1);
        int facesCount = 0;
        while (getFrame.getNextFrame(frame)) {
            //TO DO FACE DETECTION
            FaceDetector faceDetector;
            faceDetector.findFacesInImage(frame, processed);
            resize(processed, processed, Size(480, 480));
            imshow("Face Recognisation", processed);

            if (faceDetector.goodFace()) {
                testImg = faceDetector.getFaceToTest();
            }
            int key = waitKey(30);
            if (key != -1) {
                if ((key & 255) == 27) {
                    break;
                }
                else {
                    facesCount++;
                    string tempPath = "C:/Users/ahmad/source/repos/opencv test/faces/temp/s";
                    tempPath += to_string(facesCount);
                    tempPath += ".bmp";
                    imwrite(tempPath, testImg);
                    cout << facesCount << " Face Finished." << endl;
                }
            }
        }
        //after prepare one group faces, copy folder to "faces" folder
        cout << "Prepare Face Finished." << endl;
        //Prepare finish.
    }
    else if (choice == 1) {
        cout << "Traning Start......" << endl;
        //do PCA analysis for training faces
        MyPCA myPCA = MyPCA(trainFacesPath);
        //Write trainning data to file
        WriteTrainData wtd = WriteTrainData(myPCA, trainFacesID);
        //training finsih.
        cout << "Training finsih." << endl;
    }
    else if (choice == 2) {
        cout << "Recognise Start......" << endl;
                Mat testImg = imread("C:/Users/ahmad/source/repos/opencv test/New_test/2-10.bmp",0);
                cout << testImg.rows << "X" << testImg.cols;
                //final step: recognize new face from training faces
                FaceRecognizer faceRecognizer = FaceRecognizer(testImg, avgVec, eigenVec, facesInEigen, loadedFacesID);
                // Show Result
                string faceID = faceRecognizer.getClosetFaceID();
                cout << faceID;  
    }
    else {
        cout << "Input wrong choice......" << endl;
    }

    cout << "Program End." << endl;
    return 0;
}
