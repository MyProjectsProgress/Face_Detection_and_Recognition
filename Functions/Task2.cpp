#include <vector>
#include <cmath>
#include<iostream>
#include <stdio.h>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace std;
using namespace cv;


//#include <vector>
//#include <cmath>
//#include<iostream>
//#include <stdio.h>
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
//using namespace std;
//using namespace cv;
//
//
////// Transfering RGB to L*U*V* color Space
////void RGBtoLuv(cv::Mat& image) {
////    const float epsilon = 0.008856;
////    const float kappa = 903.3;
////    const cv::Vec3f whitePoint(0.9505, 1.0000, 1.0890);
////
////    // resize the input image
////    cv::resize(image, image, cv::Size(500, 500));
////
////    // create a new image to store the Luv values
////    cv::Mat luvImage(image.rows, image.cols, CV_32FC3);
////
////    for (int i = 0; i < image.rows; i++) {
////        for (int j = 0; j < image.cols; j++) {
////            cv::Vec3b rgb = image.at<cv::Vec3b>(i, j);
////            float r = static_cast<float>(rgb[2]) / 255.0;
////            float g = static_cast<float>(rgb[1]) / 255.0;
////            float b = static_cast<float>(rgb[0]) / 255.0;
////
////            // convert RGB to XYZ
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
////            luvImage.at<cv::Vec3f>(i, j) = luv;
////        }
////    }
////}
////
////
////void kMeansSegmentation(cv::Mat& luvImage, int k) {
////
////    // Reshape the Luv image to a 2D array of pixels
////    cv::Mat pixels = luvImage.reshape(1, luvImage.rows * luvImage.cols);
////
////    // Convert the pixel values to 32-bit float values
////    pixels.convertTo(pixels, CV_32F);
////
////    // Run k-means clustering on the pixels
////    cv::Mat labels, centers;
////    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0);
////    cv::kmeans(pixels, k, labels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);
////
////    // Replace the pixel values with the cluster centers
////    for (int i = 0; i < pixels.rows; i++) {
////        int label = labels.at<int>(i);
////        pixels.at<float>(i, 0) = centers.at<float>(label, 0);
////        pixels.at<float>(i, 1) = centers.at<float>(label, 1);
////        pixels.at<float>(i, 2) = centers.at<float>(label, 2);
////    }
////
////    // Reshape the pixel array to the original image shape
////    cv::Mat segmentedLuvImage = pixels.reshape(3, luvImage.rows);
////
////    // Convert the segmented Luv image back to BGR color space
////    cv::Mat segmentedBgrImage;
////    cv::cvtColor(segmentedLuvImage, segmentedBgrImage, cv::COLOR_Luv2BGR);
////
////    // Display the segmented image
////    cv::imshow("Segmented Image", segmentedBgrImage);
////    cv::waitKey(0);
////}
////
////int main() {
////    cv::Mat originalImg = cv::imread("D:/Screenshotfrom20200703024321.png");
////    cv::cvtColor(originalImg, originalImg, cv::COLOR_BGR2RGB);
////
////    // Resize the image to 500x500
////    resize(originalImg, originalImg, cv::Size(500, 500));
////
////    // Convert the image to Luv color space
////    //RGBtoLuv(originalImg);
////    cv::cvtColor(originalImg, originalImg, cv::COLOR_RGB2Luv);
////
////    // Segment the image using k-means clustering with k = 5
////    kMeansSegmentation(originalImg, 3);
////
////    return 0;
////}
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
//void BGRtoLuv(cv::Mat& bgr_image, cv::Mat& luv_image) {
//    const float epsilon = 0.008856;
//    const float kappa = 903.3;
//    const cv::Vec3f whitePoint(0.9505, 1.0000, 1.0890);
//
//    // resize the input image
//    cv::resize(bgr_image, bgr_image, cv::Size(500, 500));
//
//    // create a new image to store the Luv values
//    luv_image.create(bgr_image.size(), CV_32FC3);
//
//    for (int i = 0; i < bgr_image.rows; i++) {
//        for (int j = 0; j < bgr_image.cols; j++) {
//            cv::Vec3f bgr = bgr_image.at<cv::Vec3f>(i, j);
//            float b = bgr[0];
//            float g = bgr[1];
//            float r = bgr[2];
//
//            // convert BGR to XYZ
//            float x = 0.4124 * r + 0.3576 * g + 0.1805 * b;
//            float y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
//            float z = 0.0193 * r + 0.1192 * g + 0.9505 * b;
//
//            // normalize XYZ values
//            cv::Vec3f xyz(x, y, z);
//            cv::divide(xyz, whitePoint, xyz);
//
//            // calculate L*, u*, and v* values
//            float L, u, v;
//            if (xyz[1] > epsilon) {
//                L = 116.0 * pow(xyz[1], 1.0 / 3.0) - 16.0;
//            }
//            else {
//                L = kappa * xyz[1];
//            }
//
//            float uPrime = 4.0 * xyz[0] / (xyz[0] + 15.0 * xyz[1] + 3.0 * xyz[2]);
//            float vPrime = 9.0 * xyz[1] / (xyz[0] + 15.0 * xyz[1] + 3.0 * xyz[2]);
//            u = 13.0 * L * (uPrime - whitePoint[0]);
//            v = 13.0 * L * (vPrime - whitePoint[1]);
//
//            // store Luv values in the new image
//            cv::Vec3f luv(L, u, v);
//            luv_image.at<cv::Vec3f>(i, j) = luv;
//        }
//    }
//}
//
//#include <iostream>
//#include <vector>
//#include <opencv2/opencv.hpp>
//
//using namespace cv;
//using namespace std;
//
//void kmeans(const Mat& data, int num_clusters, Mat& labels, Mat& centers, int max_iterations = 100)
//{
//    // Initialize random centroids
//    Mat centers_old, centers_new;
//    data.rowRange(0, num_clusters).copyTo(centers_new);
//    centers_new.copyTo(centers_old);
//
//    // Loop until convergence or maximum iterations reached
//    int iter = 0;
//    while (iter < max_iterations)
//    {
//        // Assign each data point to the nearest centroid
//        for (int i = 0; i < data.rows; i++)
//        {
//            double min_dist = numeric_limits<double>::max();
//            int label = 0;
//
//            for (int j = 0; j < num_clusters; j++)
//            {
//                double dist = norm(data.row(i) - centers_new.row(j));
//                if (dist < min_dist)
//                {
//                    min_dist = dist;
//                    label = j;
//                }
//            }
//
//            labels.at<int>(i) = label;
//        }
//
//        // Update the centroid of each cluster
//        centers_new.setTo(0);
//        vector<int> count(num_clusters, 0);
//
//        for (int i = 0; i < data.rows; i++)
//        {
//            int label = labels.at<int>(i);
//            centers_new.row(label) += data.row(i);
//            count[label]++;
//        }
//
//        for (int j = 0; j < num_clusters; j++)
//        {
//            if (count[j] != 0)
//            {
//                centers_new.row(j) /= count[j];
//            }
//        }
//
//        // Check for convergence
//        double epsilon = 0.001;
//        double diff = norm(centers_new - centers_old);
//        if (diff < epsilon)
//        {
//            break;
//        }
//
//        // Update old centroids and increment iteration counter
//        centers_new.copyTo(centers_old);
//        iter++;
//    }
//
//    // Set final cluster centers
//    centers_new.copyTo(centers);
//}
//
//
//int main() {
//    // Reading the image
//    Mat image = imread("D:/Screenshotfrom20200703024321.png", IMREAD_COLOR);
//
//    // Convert the image to LUV color space
//    Mat luv_image;
//    cvtColor(image, luv_image, COLOR_BGR2Luv);
//
//    // Convert the image to a 2D matrix for k-means clustering
//    Mat data;
//    luv_image.convertTo(data, CV_32F);
//    data = data.reshape(1, data.rows * data.cols);
//
//    // Perform k-means clustering
//    int num_clusters = 10;
//    Mat labels, centers;
//    kmeans(data, num_clusters, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
//
//    // Reshape the labels and centers to match the LUV image size
//    labels = labels.reshape(1, luv_image.rows);
//    centers = centers.reshape(1, num_clusters);
//
//    // Generate the segmented image
//    Mat segmented_image(luv_image.size(), luv_image.type());
//    for (int i = 0; i < luv_image.rows; i++)
//    {
//        for (int j = 0; j < luv_image.cols; j++)
//        {
//            int label = labels.at<int>(i, j);
//            segmented_image.at<Vec3b>(i, j) = centers.at<Vec3f>(label, 0);
//        }
//    }
//
//    // Convert the segmented image back to BGR color space
//    Mat bgr_segmented_image;
//    cvtColor(segmented_image, bgr_segmented_image, COLOR_Luv2BGR);
//
//    // Display the original and segmented images
//    namedWindow("Original Image", WINDOW_NORMAL);
//    imshow("Original Image", image);
//    namedWindow("Segmented Image", WINDOW_NORMAL);
//    imshow("Segmented Image", bgr_segmented_image);
//    waitKey(0);
//
//    return 0;
//}
//
//
//
//
//
//
////int main() {
////    cv::Mat originalImg = cv::imread("D:/16476928522307.jpg");
////
////     Resize the image to 500x500
////    resize(originalImg, originalImg, cv::Size(500, 500));
////
////     Convert the image to Luv color space
////    RGBtoLuv(originalImg);
////
////     Segment the image using k-means clustering with k = 5
////    kMeansSegmentation(originalImg, 5);
////
////    return 0;
////}
//
////int SSDMethod(cv::Mat originalImg, cv::Mat compImg) {
////	int result=0;
////	for (int i = 0; i < originalImg.rows - 1; i++)
////	{
////		for (int j = 0; j < originalImg.cols - 1; j++)
////		{
////			int SSD = (originalImg.at<uchar>(i, j) - compImg.at<uchar>(i, j)) * (originalImg.at<uchar>(i, j) - compImg.at<uchar>(i, j));
////			if (SSD <= 0) {
////				SSD *= -1;
////			}
////			result += SSD;
////		}
////	}
////	return result;
////}
////
////cv::Mat getSimlarImage(cv::Mat originalImg ,std::vector<cv::Mat> images) {
////	vector<int> imageScores;
////	for (int i = 0; i < images.size(); i++) {
////		imageScores.push_back(SSDMethod(originalImg ,images[i]));
////	}
////	auto minIterator = std::min_element(imageScores.begin(), imageScores.end());
////	int minIndex = std::distance(imageScores.begin(), minIterator);
////	return images[minIndex];
////}
////
////int main() {
////	cv::Mat originalImg = cv::imread("D:/one_less_traveled_513313.jpg", cv::IMREAD_GRAYSCALE);
////	cv::Mat secImg = cv::imread("D:/16476928522307.jpg", cv::IMREAD_GRAYSCALE);
////	cv::Mat thirdImg = cv::imread("D:/pexels-kaique-rocha-65438.jpg", cv::IMREAD_GRAYSCALE);
////	resize(originalImg, originalImg, cv::Size(500, 500));
////	resize(secImg, secImg, cv::Size(500, 500));
////	resize(thirdImg, thirdImg, cv::Size(500, 500));
////	vector<cv::Mat> images;
////	images.push_back(secImg);
////	images.push_back(thirdImg);
////	cv::Mat finalResult = getSimlarImage(originalImg, images);
////	cv::imshow("Output", finalResult);
////	cv::waitKey(0);
////	return(0);
////}
////
////
////
//
//
//
//
//
//
//
//
////vector<int> calculateChainCodeWithSlope(int x[], int y[], int n) {
////    vector<int> chainCode;
////    for (int i = 0; i < n-1; i++) {
////        int dx = x[i + 1]-x[i];
////        int dy = y[i+1]-y[i];
////
////        double slope = (double)dy / (double)dx;
////        double angle = atan(slope);
////        angle = abs(angle * 180 / 3.14);
////        // Calculate the quadrant
////        int quadrant;
////        if (dx > 0 && dy >= 0) {
////            quadrant = 1;
////        }
////        else if (dx <= 0 && dy > 0) {
////            quadrant = 2;
////            angle = 180 - angle;
////        }
////        else if (dx < 0 && dy <= 0) {
////            quadrant = 3;
////            angle = 180 + angle;
////
////        }
////        else if (dx >= 0 && dy < 0) {
////            quadrant = 4;
////            angle = 360 - angle;
////        }
////        cout << angle << endl;
////        // Calculate the chaincode for the quadrant
////        int code;
////        if (angle >= 337.5 || angle < 22.5) {
////            code = 0;
////        }
////        else if (angle >= 22.5 && angle < 67.5) {
////            code = 1;
////        }
////        else if (angle >= 67.5 && angle < 112.5) {
////            code = 2;
////        }
////        else if (angle >= 112.5 && angle < 157.5) {
////            code = 3;
////        }
////        else if (angle >= 157.5 && angle < 202.5) {
////            code = 4;
////        }
////        else if (angle >= 202.5 && angle < 247.5) {
////            code = 5;
////        }
////        else if (angle >= 247.5 && angle < 292.5) {
////            code = 6;
////        }
////        else if (angle >= 292.5 && angle < 337.5) {
////            code = 7;
////        }
////        // Add the chaincode to the sequence
////        chainCode.push_back(code);
////    }
////
////    return chainCode;
////}
//
////#include <iostream>
////#include <vector>
////
////using namespace std;
////
////vector<int> calculateChainCodeWithSlope(int x[], int y[], int n);
//
////int main() {
////    // Define the contour as an array of x-coordinates and y-coordinates
////    int x[] = { -7, -4};
////    int y[] = { 9, 3};
////    int n = 2;
////
////    // Calculate the chaincode using the slope method
////    vector<int> chainCode = calculateChainCodeWithSlope(x, y, n);
////
////    // Print the chaincode
////    cout << "Chaincode sequence: ";
////    for (int i = 0; i < chainCode.size(); i++) {
////        cout << chainCode[i] << " ";
////    }
////    cout << endl;
////
////    return 0;
////}
//
//
//
////#include <iostream>
////#include <opencv2/opencv.hpp>
////
////std::vector<int> get_4_chain_code(const std::vector<cv::Point>& contour) {
////    std::vector<int> code;
////    int dx, dy;
////    for (int i = 1; i < contour.size(); i++) {
////        dx = contour[i].x - contour[i - 1].x;
////        dy = contour[i].y - contour[i - 1].y;
////        if (dx == 0 && dy < 0) {
////            code.push_back(0);
////        }
////        else if (dx > 0 && dy < 0) {
////            code.push_back(1);
////        }
////        else if (dx > 0 && dy == 0) {
////            code.push_back(2);
////        }
////        else if (dx > 0 && dy > 0) {
////            code.push_back(3);
////        }
////        else if (dx == 0 && dy > 0) {
////            code.push_back(4);
////        }
////        else if (dx < 0 && dy > 0) {
////            code.push_back(5);
////        }
////        else if (dx < 0 && dy == 0) {
////            code.push_back(6);
////        }
////        else if (dx < 0 && dy < 0) {
////            code.push_back(7);
////        }
////    }
////    return code;
////}
////
////int main() {
////    // Load image and find contours
////    cv::Mat image = cv::imread("contour.png", cv::IMREAD_GRAYSCALE);
////    std::vector<std::vector<cv::Point>> contours;
////    cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
////
////    // Get the 4-chain code representation for each contour
////    for (const auto& contour : contours) {
////        std::vector<int> code = get_4_chain_code(contour);
////        // Print the code
////        std::cout << "4-chain code: ";
////        for (const auto& c : code) {
////            std::cout << c << " ";
////        }
////        std::cout << std::endl;
////    }
////
////    return 0;
////}
////
////
////
////
////
////
////
////
////
////
////
////
////
////
////
////
////
////
////
//////#include <stdio.h>
//////#include <opencv2/opencv.hpp>
//////#include<iostream>
//////#include <opencv2/imgproc.hpp>
//////#include <cmath>
//////#include <cstdio>
//////#include <vector>
//////#define _USE_MATH_DEFINES
//////
//////
//////using namespace std;
//////using namespace cv;
//////
//////#include <vector>
//////
//////#include <vector>
//////
//////std::vector<int> chainCode4(int* x_points, int* y_points, int points_n) {
//////    std::vector<int> code;
//////
//////    int current_x = x_points[0];
//////    int current_y = y_points[0];
//////
//////    for (int i = 1; i < points_n; i++) {
//////        int dx = x_points[i] - current_x;
//////        int dy = y_points[i] - current_y;
//////
//////        if (dx == 0 && dy == -1) { // up
//////            code.push_back(0);
//////        }
//////        else if (dx == 0 && dy == 1) { // down
//////            code.push_back(2);
//////        }
//////        else if (dx == -1 && dy == 0) { // left
//////            code.push_back(1);
//////        }
//////        else if (dx == 1 && dy == 0) { // right
//////            code.push_back(3);
//////        }
//////
//////        current_x = x_points[i];
//////        current_y = y_points[i];
//////    }
//////
//////    return code;
//////}
//////
//////
//////
//////#include <iostream>
//////#include <vector>
//////
//////// Function to get 4-chain code representation of a contour
//////std::vector<int> chainCode4(int* x_points, int* y_points, int points_n);
//////
//////int main() {
//////    int x_points[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
//////    int y_points[] = { 0, 0, 0, 1, 1, 1, 2, 2, 3, 3 };
//////    int points_n = sizeof(x_points) / sizeof(x_points[0]);
//////
//////    // Get the 4-chain code representation of the contour
//////    std::vector<int> code = chainCode4(x_points, y_points, points_n);
//////
//////    // Print the 4-chain code representation
//////    std::cout << "4-chain code representation: ";
//////    for (int i = 0; i < code.size(); i++) {
//////        std::cout << code[i] << " ";
//////    }
//////    std::cout << std::endl;
//////
//////    return 0;
//////}
////////cv::Mat toGreyScale(cv::Mat image);
////////cv::Mat readImage(string path);
////////cv::Mat masking(Mat image, int mask[3][3]);
////////int(*(getArray)(string mode, string direction))[3];
////////Mat calc_histogram(Mat image);
////////void plot_histogram(Mat histogram);
////////void plot_distribution_curve(Mat histogram);
////////void plot_rgb_distribution_function(Mat image, string mode);
////////void non_max_suppression(const Mat& magnitude, const Mat& direction, Mat& result);
////////void convertToGrayscale(Mat& input_image, Mat& output_image);
////////
///////////////// new part /////////
////////
////////cv::Mat CannyEdgeDetection(cv::Mat image, int segma, int lowThreshold, int highThreshold, int KernalSize);
////////Mat padding(Mat img, int k_width, int k_height);
////////void convolve_gaussian(Mat scr, Mat& dst, int k_w, int k_h);
////////Mat define_kernel_gaussian(int k_width, int k_height);
///////////////////////// Task 2 Trying ////////////////////////////
////////
////////float deg2rad(int deg);
////////cv::Mat houghLine(cv::Mat image);
////////void nonMaxSuppression(const cv::Mat& src, cv::Mat& dst, int windowSize);
//////cv::Mat ChathoughLine(cv::Mat Cannyimage, cv::Mat Orignalimage);
////////#include <opencv2/core.hpp>
////////#include <opencv2/imgcodecs.hpp>
////////#include <opencv2/highgui.hpp>
////////#include <opencv2/imgproc.hpp>
////////#include <iostream>
////////#include <cmath>
////////
//////using namespace cv;
//////using namespace std;
//////
////////int main()
////////{
////////    // Load the input image
////////    Mat inputImg = imread("D:/pexels-kaique-rocha-65438.png", IMREAD_GRAYSCALE);
////////    resize(inputImg, inputImg, cv::Size(500, 500));
////////
////////    // Apply Canny edge detection
////////    Mat edges;
////////    Canny(inputImg, edges, 126 , 130);
////////
////////    // Define the Hough Transform parameters
////////    double rhoRes = 1;
////////    double thetaRes = CV_PI / 180;
////////    int threshold = 170;
////////
////////    // Compute the Hough Transform
////////    vector<Vec2f> lines;
////////    HoughLines(edges, lines, rhoRes, thetaRes, threshold);
////////
////////    // Draw the detected lines on the input image
////////    Mat outputImg;
////////    cvtColor(inputImg, outputImg, COLOR_GRAY2BGR);
////////    for (size_t i = 0; i < lines.size(); i++)
////////    {
////////        float rho = lines[i][0];
////////        float theta = lines[i][1];
////////        Point pt1, pt2;
////////        double a = cos(theta), b = sin(theta);
////////        double x0 = a * rho, y0 = b * rho;
////////        pt1.x = cvRound(x0 + 1000 * (-b));
////////        pt1.y = cvRound(y0 + 1000 * (a));
////////        pt2.x = cvRound(x0 - 1000 * (-b));
////////        pt2.y = cvRound(y0 - 1000 * (a));
////////        line(outputImg, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);
////////    }
////////
////////    // Show the output image
////////    imshow("Output", outputImg);
////////    waitKey(0);
////////    return 0;
////////}
//////
//////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
//////
////////#include <iostream>
////////#include <vector>
////////
////////using namespace std;
////////
////////vector<int> chainCode(int x[], int y[], int n) {
////////    vector<int> code;
////////    int dir[8][2] = { {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1} };
////////
////////    for (int i = 0; i < n; i++) {
////////        int min_dist = INT_MAX;
////////        int min_index = -1;
////////
////////        // Find the closest point to the current point
////////        for (int j = 0; j < n; j++) {
////////            int dx = x[j] - x[i];
////////            int dy = y[j] - y[i];
////////            int dist = dx*dx + dy*dy;
////////
////////            if (dist < min_dist) {
////////                min_dist = dist;
////////                min_index = j;
////////            }
////////        }
////////
////////        // Calculate the direction between the current point and the closest point
////////        int dx = x[min_index] - x[i];
////////        int dy = y[min_index] - y[i];
////////        int next_dir = -1;
////////
////////        for (int j = 0; j < 8; j++) {
////////            if (dir[j][0] == dx && dir[j][1] == dy) {
////////                next_dir = j;
////////                break;
////////            }
////////        }
////////
////////        if (next_dir == -1) {
////////            cerr << "Error: Invalid contour!" << endl;
////////            exit(1);
////////        }
////////
////////        code.push_back(next_dir);
////////    }
////////
////////    return code;
////////}
//////
//////
////////int main() {
////////    // Sample contour points
////////int x[] = {1, 3, 6, 8, 9, 8, 6, 3, 1};
////////int y[] = {4, 6, 7, 6, 4, 2, 1, 2, 4};
////////int n = sizeof(x) / sizeof(int);
////////
////////    // Compute the 8-chain code
////////    vector<int> code = chainCode(x, y, n);
////////
////////    // Print the chain code
////////    cout << "8-chain code: ";
////////    for (size_t i = 0; i < code.size(); i++) {
////////        cout << code[i];
////////    }
////////    cout << endl;
////////
////////    return 0;
////////}
//////
////////int main(int argc, char** argv)
////////{    
////////    string path = "D:/pexels-kaique-rocha-65438.png";
////////    Mat image =imread(path, IMREAD_GRAYSCALE);
////////    resize(image, image, cv::Size(500, 500));
////////
////////    Mat edges;
////////    Canny(image, edges, 100, 200);
////////
////////    Mat accumlator,accumlatorSuppresed,result;
////////    result = ChathoughLine(edges, image);
////////
////////    vector<Vec2f> lines;
////////    HoughLines(edges, lines, 1, CV_PI / 180, 170);
////////
////////    // Draw detected lines on the original image
////////    Mat result1 = image.clone();
////////    for (size_t i = 0; i < lines.size(); i++)
////////    {
////////        float rho = lines[i][0], theta = lines[i][1];
////////        Point pt1, pt2;
////////        double a = cos(theta), b = sin(theta);
////////        double x0 = a * rho, y0 = b * rho;
////////        pt1.x = cvRound(x0 + 1000 * (-b));
////////        pt1.y = cvRound(y0 + 1000 * (a));
////////        pt2.x = cvRound(x0 - 1000 * (-b));
////////        pt2.y = cvRound(y0 - 1000 * (a));
////////        line(result1, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);
////////    }
////////
////////
////////    imshow("after", result);
////////    imshow("2", result1);
////////    waitKey(0);
////////    return 0;
////////}
////////// Task 2 //
////////
////////
////////
//////////
//////////
////////cv::Mat ChathoughLine(cv::Mat Cannyimage , cv::Mat Orignalimage) {
////////    // Define range of angles to scan
////////    std::vector<double> angles;
////////    for (int i = -90; i <= 90; i++) {
////////        angles.push_back(i * CV_PI / 180.0);
////////    }
////////
////////    // Create accumulator array
////////    int max_dist = cvRound(sqrt(pow(Cannyimage.rows, 2) + pow(Cannyimage.cols, 2)));
////////    cv::Mat accumulator = cv::Mat::zeros(max_dist * 2 + 1, angles.size(), CV_32SC1);
////////
////////    // Loop through image pixels
////////    for (int y = 0; y < Cannyimage.rows; y++) {
////////        for (int x = 0; x < Cannyimage.cols; x++) {
////////            if (Cannyimage.at<uchar>(y, x) > 0) {
////////                // Calculate rho for each angle
////////                for (int a = 0; a < angles.size(); a++) {
////////                    int rho = cvRound(x * cos(angles[a]) + y * sin(angles[a])) + max_dist;
////////                    accumulator.at<int>(rho, a)++;
////////                }
////////            }
////////        }
////////    }
////////
////////    // Find local maxima in accumulator
////////    std::vector<cv::Vec2i> lines;
////////    int threshold = 170;
////////    for (int r = 0; r < accumulator.rows; r++) {
////////        for (int a = 0; a < accumulator.cols; a++) {
////////            if (accumulator.at<int>(r, a) > threshold) {
////////                lines.push_back(cv::Vec2i(r, a));
////////            }
////////        }
////////    }
////////
////////    // Draw lines on image
////////    cv::Mat output = Cannyimage.clone();
////////    cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
////////    for (int i = 0; i < lines.size(); i++) {
////////        float rho = lines[i][0] - max_dist;
////////        float theta = angles[lines[i][1]];
////////        double a = cos(theta), b = sin(theta);
////////        double x0 = a * rho, y0 = b * rho;
////////        cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
////////        cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
////////        cv::line(output, pt1, pt2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
////////    }
////////
////////    return output;
////////}
//////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////// Hough line detection //
////////
////////cv::Mat houghLine(cv::Mat image) {
////////    int rows = image.rows;
////////    int cols = image.cols;
////////    int MaxDistance = int((round(sqrt(pow(rows, 2) + pow(cols, 2)))));  // which is diagonal distance
////////    std::vector<double> ang;
////////    for (int i = -90; i <= 90; i++) {
////////        ang.push_back(deg2rad(i));
////////    }
////////    int numPoints = 2 * MaxDistance + 1;
////////    std::vector<double> arr(numPoints);
////////    double increment = (2 * MaxDistance) / static_cast<double>(numPoints - 1);
////////    for (int i = 0; i < numPoints; ++i) {
////////        arr[i] = -MaxDistance + i * increment;
////////    }
////////    cv::Mat accumlator = cv::Mat::zeros(2 * MaxDistance + 1, ang.size(), CV_64F);
////////    for (int i = 0; i < rows; i++) {
////////        for (int j = 0; j < cols; j++) {
////////            if (image.at<uchar>(i, j) > 0) {
////////                for (int k = 0; k < ang.size(); k++) {
////////                    double r = j * cos(ang[k]) + i * sin(ang[k]);
////////                    accumlator.at<double>(int(r + MaxDistance), k) += 1;
////////                }
////////            }
////////        }
////////    }
////////
////////    // Scale and convert the Hough accumulator to an image
////////    double minVal, maxVal;
////////    cv::minMaxLoc(accumlator, &minVal, &maxVal);
////////    cv::Mat accImage;
////////    cv::convertScaleAbs(accumlator, accImage, 255.0 / (maxVal - minVal));
////////
////////    // Display the Hough accumulator image
////////    cv::imshow("Hough Accumulator", accImage);
////////    cv::waitKey(0);
////////
////////    double thresh = 0.5 * cv::mean(accumlator)[0];
////////    std::vector<cv::Vec2f> lines;
////////    for (int r = 0; r < accumlator.rows; ++r) {
////////        for (int c = 0; c < accumlator.cols; ++c) {
////////            if (accumlator.at<double>(r, c) > thresh) {
////////                cv::Vec2f line(c - MaxDistance, arr[r]);
////////                lines.push_back(line);
////////            }
////////        }
////////    }
////////
////////    for (int i = 0; i < lines.size(); ++i) {
////////        float rho = lines[i][1];
////////        float theta = lines[i][0];
////////        cv::Point pt1, pt2;
////////        double a = cos(theta), b = sin(theta);
////////        double x0 = a * rho, y0 = b * rho;
////////        pt1.x = cvRound(x0 + 1000 * (-b));
////////        pt1.y = cvRound(y0 + 1000 * (a));
////////        pt2.x = cvRound(x0 - 1000 * (-b));
////////        pt2.y = cvRound(y0 - 1000 * (a));
////////        cv::line(image, pt1, pt2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
////////    }
////////
////////    return image;
////////}
////////
////////float deg2rad(int deg) {
////////    return deg * 3.14159265358979323846 / 180.0;
////////}
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////
////////cv::Mat readImage(string path) {
////////    cv::Mat image = cv::imread(path,1);
////////    if (!image.data) {
////////        printf("No image Data \n");
////////        return image;
////////    };
////////    cv::resize(image, image, cv::Size(500, 500));
////////    return image;
////////}
//////////
//////////cv::Mat toGreyScale(cv::Mat image) {
//////////    for (int i = 0; i < image.rows; i++) {
//////////        for (int j = 0; j < image.cols; j++) {
//////////            cv::Vec3b rgbPixel = image.at<cv::Vec3b>(i, j); // we get pixel at each postion
//////////            unsigned char grayScale = (rgbPixel[2] + rgbPixel[1] + rgbPixel[0]) / 3;
//////////            image.at<uchar>(i, j) = grayScale;
//////////        }
//////////    }
//////////    return image;
//////////}
//////////
//////////cv::Mat changeQuantisationGrey(cv::Mat& image, int num_bits) {
//////////    CV_Assert((image.type() == CV_8UC1) && (num_bits >= 1) && (num_bits <= 8));
//////////    uchar mask = 0xFF << (8 - num_bits);
//////////    for (int row = 0; row < image.rows; row++)
//////////        for (int col = 0; col < image.cols; col++)
//////////            image.at<uchar>(row, col) = image.at<char>(row, col) & mask;
//////////    return image;
//////////}
//////////
//////////
//////////
//////////
//////////cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel)
//////////{
//////////    //    declare variables
//////////    Mat_<float> dst;
//////////    Mat_<float> flipped_kernel;
//////////    float tmp = 0.0;
//////////
//////////    //    flip kernel
//////////    flip(kernel, flipped_kernel, -1);
//////////
//////////    //    multiply and integrate
//////////    // input rows
//////////    for (int i = 0; i < src.rows; i++) {
//////////        // input columns
//////////        for (int j = 0; j < src.cols; j++) {
//////////            // kernel rows
//////////            for (int k = 0; k < flipped_kernel.rows; k++) {
//////////                // kernel columns
//////////                for (int l = 0; l < flipped_kernel.cols; l++) {
//////////                    tmp += src.at<float>(i, j) * flipped_kernel.at<float>(k, l);
//////////                }
//////////            }
//////////            dst.at<float>(i, j) = tmp;
//////////        }
//////////    }
//////////    return dst.clone();
//////////}
////////int(*(getArray)(string mode, string direction))[3]{
////////    if (mode == "sobel") {
////////        if (direction == "horizontal") {
////////            static int mask[3][3] = { {-1,-2,-1},{0,0,0},{1,2,1} };
////////            return mask;
////////        }
////////        else {
////////            static int mask[3][3] = { {-1,0,1},{-2,0,2},{-1,0,1} };
////////            return mask;
////////        }
////////    }
////////if (mode == "roberts") {
////////    if (direction == "horizontal") {
////////        static int mask[3][3] = { {0,0,0},{0,-1,0},{0,0,1} };
////////        return mask;
////////    }
////////    else {
////////        static int mask[3][3] = { {0,0,0},{0,1,0},{0,0,-1} };
////////        return mask;
////////    }
////////}
////////  if (mode == "sobel") {
////////        if (direction == "horizontal") {
////////            static int mask[3][3] = { {-1,-2,-1},{0,0,0},{1,2,1} };
////////            return mask;
////////        }
////////        else {
////////            static int mask[3][3] = { {-1,0,1},{-2,0,2},{-1,0,1} };
////////            return mask;
////////        }
////////    }
////////else if(mode =="Prewitt")  {
////////        if (direction == "horizontal") {
////////            static int mask[3][3] = { {-1,-1,-1},{0,0,0},{1,1,1} };
////////            return mask;
////////        }
////////        else {
////////            static int mask[3][3] = { {-1,0,1},{-1,0,1},{-1,0,1} };
////////            return mask;
////////        }
////////    }
////////}
////////
////////Mat masking(Mat image,int mask[3][3]) 
////////{
////////        Mat temImage = image.clone();
////////        for (int i = 1; i < image.rows - 1; i++)
////////        {
////////            for (int j = 1; j < image.cols - 1; j++)
////////            {
////////                
////////                  int pixel1 = image.at<uchar>(i - 1, j - 1) * mask[0][0];
////////                    int pixel2 = image.at<uchar>(i, j - 1) * mask[0][1];
////////                    int pixel3 = image.at<uchar>(i + 1, j - 1) * mask[0][2];
////////
////////                    int pixel4 = image.at<uchar>(i - 1, j) * mask[1][0];
////////                   int pixel5 = image.at<uchar>(i, j) * mask[1][1];
////////                   int pixel6 = image.at<uchar>(i + 1, j) * mask[1][2];
////////
////////                    int pixel7 = image.at<uchar>(i - 1, j + 1) * mask[2][0];
////////                    int pixel8 = image.at<uchar>(i, j + 1) * mask[2][1];
////////                    int pixel9 = image.at<uchar>(i + 1, j + 1) * mask[2][2];
////////
////////                    int sum = pixel1 + pixel2 + pixel3 + pixel4 + pixel5 + pixel6 + pixel7 + pixel8 + pixel9;
////////                    if (sum < 0)
////////                        sum = 0;
////////                    if (sum > 255)
////////                        sum = 255;
////////
////////                    temImage.at<uchar>(i, j) = sum;
////////                
////////            }
////////        }
////////        return temImage;
////////}
////////
////////Mat calc_histogram(Mat image) {
////////    Mat hist;
////////    hist = Mat::zeros(256, 1, CV_32F);
////////    image.convertTo(image, CV_32F);
////////    double value = 0;
////////    for (int i = 0; i < image.rows; i++)
////////    {
////////        for (int j = 0; j < image.cols; j++)
////////        {
////////            value = image.at<float>(i, j);
////////            hist.at<float>(value) = hist.at<float>(value) + 1;
////////        }
////////    }
////////    return hist;
////////};
////////void plot_histogram(Mat histogram) {
////////    Mat histogram_image(400, 512, CV_8UC3, Scalar(0, 0, 0));
////////    Mat normalized_histogram;
////////    normalize(histogram, normalized_histogram, 0, 400, NORM_MINMAX, -1, Mat());
////////
////////    for (int i = 0; i < 256; i++)
////////   {
////////        rectangle(histogram_image, Point(2 * i, histogram_image.rows - normalized_histogram.at<float>(i)),
////////            Point(2 * (i + 1), histogram_image.rows), Scalar(255, 0, 0));
////////    }
////////
////////    namedWindow("Histogram", WINDOW_NORMAL);
////////    imshow("Histogram", histogram_image);
////////};
////////
////////void plot_distribution_curve(Mat histogram) {
////////    int num_bins = histogram.rows;
////////    Mat curve_image(400, 512, CV_8UC3, Scalar(0, 0, 0));
////////    Mat normalized_histogram;
////////    normalize(histogram, normalized_histogram, 0, 400, NORM_MINMAX, -1, Mat());
////////
////////    vector<Point> curve_points(num_bins);
////////    for (int i = 0; i < num_bins; i++) {
////////        curve_points[i] = Point(2 * i, curve_image.rows - normalized_histogram.at<float>(i));
////////    }
////////    const Point* pts = (const Point*)Mat(curve_points).data;
////////    int npts = Mat(curve_points).rows;
////////
////////    polylines(curve_image, &pts, &npts, 1, false, Scalar(255, 0, 0), 2);
////////
////////    namedWindow("Distribution Curve", WINDOW_NORMAL);
////////    imshow("Distribution Curve", curve_image);
////////};
////////void plot_rgb_distribution_function(Mat image, string mode) {
////////    vector<Mat> bgr_planes;
////////    split(image, bgr_planes);
////////
////////    const int num_bins = 256;
////////    const int hist_height = 5000;
////////    const int hist_width = 512;
////////    const int bin_width = cvRound(static_cast<double>(hist_width) / num_bins);
////////
////////    // Create histograms for each color channel
////////    Mat b_hist, g_hist, r_hist;
////////    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &num_bins, 0);
////////    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &num_bins, 0);
////////    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &num_bins, 0);
////////
////////    // Create separate images for each histogram
////////    Mat b_DF(hist_height, hist_width, CV_8UC3, Scalar(0, 0, 0));
////////    Mat g_DF(hist_height, hist_width, CV_8UC3, Scalar(0, 0, 0));
////////    Mat r_DF(hist_height, hist_width, CV_8UC3, Scalar(0, 0, 0));
////////
////////    if (mode == "cumulative") {
////////        Mat cumulative_r, cumulative_b, cumulative_g;
////////        cumulative_r = r_DF.clone();
////////        cumulative_g = g_DF.clone();
////////        cumulative_b = b_DF.clone();
////////
////////
////////        for (int j = 0; j < num_bins - 1; j++) {
////////
////////
////////            cumulative_r.at<float >(j) += cumulative_r.at<float>(j - 1);
////////            cumulative_b.at<float >(j) += cumulative_b.at<float>(j - 1);
////////            cumulative_g.at<float >(j) += cumulative_g.at<float>(j - 1);
////////        }
////////        normalize(cumulative_b, b_hist, 0, hist_height, NORM_MINMAX, -1, Mat());
////////        normalize(cumulative_g, g_hist, 0, hist_height, NORM_MINMAX, -1, Mat());
////////        normalize(cumulative_r, r_hist, 0, hist_height, NORM_MINMAX, -1, Mat());
////////    }
////////    else {
////////
////////        // Normalize the histograms
////////        normalize(b_hist, b_hist, 0, hist_height, NORM_MINMAX, -1, Mat());
////////        normalize(g_hist, g_hist, 0, hist_height, NORM_MINMAX, -1, Mat());
////////        normalize(r_hist, r_hist, 0, hist_height, NORM_MINMAX, -1, Mat());
////////
////////    }
////////
////////
////////    // Plot the histograms
////////    for (int i = 1; i < num_bins; i++) {
////////        line(b_DF, Point(bin_width * (i - 1), hist_height - cvRound(b_hist.at<float>(i - 1))),
////////            Point(bin_width * i, hist_height - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, LINE_AA);
////////        line(g_DF, Point(bin_width * (i - 1), hist_height - cvRound(g_hist.at<float>(i - 1))),
////////            Point(bin_width * i, hist_height - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, LINE_AA);
////////        line(r_DF, Point(bin_width * (i - 1), hist_height - cvRound(r_hist.at<float>(i - 1))),
////////            Point(bin_width * i, hist_height - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, LINE_AA);
////////    }
////////
////////    namedWindow("Blue Histogram", WINDOW_NORMAL);
////////    imshow("Blue Histogram", b_hist);
////////    namedWindow("Green Histogram", WINDOW_NORMAL);
////////    imshow("Green Histogram", g_hist);
////////    namedWindow("Red Histogram", WINDOW_NORMAL);
////////    imshow("Red Histogram", r_hist);
////////}
//////////////////////////MAGDY CODE ////////////////////////////////////////////////////////////////////////
////////Mat padding(Mat img, int k_width, int k_height)
////////{
////////    Mat scr;
////////    img.convertTo(scr, CV_64FC1); // converting the image pixels to 64 bits 
////////    int pad_rows, pad_cols;
////////    pad_rows = (k_height - 1) / 2; // = 1
////////    pad_cols = (k_width - 1) / 2;  // = 1
////////    Mat pad_image(Size(scr.cols + 2 * pad_cols, scr.rows + 2 * pad_rows), CV_64FC1, Scalar(0)); // resizing the image with the padding
////////    scr.copyTo(pad_image(Rect(pad_cols, pad_rows, scr.cols, scr.rows))); // creating new padded image
////////    return pad_image;
////////}
////////
////////// function to define kernels for gaussian convolution
////////Mat define_kernel_gaussian(int k_width, int k_height)
////////{
////////    // I will assume k = 1 and sigma = 1
////////    int pad_rows = (k_height - 1) / 2; // = 1
////////    int pad_cols = (k_width - 1) / 2;  // = 1
////////    Mat kernel(k_height, k_width, CV_64FC1); // creates 3x3 matrix
////////    for (int i = -pad_rows; i <= pad_rows; i++)
////////    {
////////        for (int j = -pad_cols; j <= pad_cols; j++)
////////        {
////////            kernel.at<double>(i + pad_rows, j + pad_cols) = exp(-(i * i + j * j) / 2.0);
////////        }
////////    }
////////    kernel = kernel / sum(kernel); //normalization
////////    return kernel;
////////}
////////
////////// function to implement convolution of gaussian filter
////////void convolve_gaussian(Mat scr, Mat& dst, int k_w, int k_h)
////////{
////////    Mat pad_img, kernel;
////////    pad_img = padding(scr, k_w, k_h);
////////    kernel = define_kernel_gaussian(k_w, k_h);
////////
////////    Mat output = Mat::zeros(scr.size(), CV_64FC1);
////////
////////    for (int i = 0; i < scr.rows; i++)
////////        for (int j = 0; j < scr.cols; j++)
////////            output.at<double>(i, j) = sum(kernel.mul(pad_img(Rect(j, i, k_w, k_h)))).val[0];
////////
////////    output.convertTo(dst, CV_8UC1);
////////}
////////
////////
////////
////////
//////////cannnnnnnnnnnny////////////////////////////////////////////////////////////////////////////////////
////////Mat cannyEdgeDetection(Mat image) {
////////    // Gussian Filter
////////    Mat kernel,dst;
////////    int k_w = 3;  // kernel width
////////    int k_h = 3;  // kernel height
////////
////////    int(*maskH)[3];
////////    int(*maskV)[3];
////////
////////    convolve_gaussian(image, dst, k_w, k_h);
////////    imshow("After Gauss Filtering", dst);
////////
////////    maskH = getArray("sobel", "horizontal");
////////    maskV = getArray("sobel", "vertical");
////////    Mat gradientx = masking(dst, maskH);
////////    Mat gradienty = masking(dst, maskV);
////////    Mat magnitude, direction;
////////    cartToPolar(gradientx, gradienty, magnitude, direction, true);
////////
////////    return gradientx;
////////
////////}
////////
////////void non_max_suppression(const Mat& magnitude, const Mat& direction, Mat& result) {
////////    // Create a copy of the magnitude matrix
////////    result = magnitude.clone();
////////
////////    // Suppress non-maximum points
////////    for (int y = 1; y < magnitude.rows - 1; y++) {
////////        for (int x = 1; x < magnitude.cols - 1; x++) {
////////            // Calculate the angle of the gradient at this pixel
////////            float angle = direction.at<float>(y, x) * 180.0 / CV_PI;
////////
////////            // Wrap the angle around 180 degrees
////////            if (angle < 0) {
////////                angle += 180;
////////            }
////////
////////            // Find the two neighboring pixels along the gradient direction
////////            int x1, y1, x2, y2;
////////            if (angle < 22.5 || angle >= 157.5) {
////////                x1 = x2 = x;
////////                y1 = y - 1;
////////                y2 = y + 1;
////////            }
////////            else if (angle < 67.5) {
////////                x1 = x - 1;
////////                y1 = y - 1;
////////                x2 = x + 1;
////////                y2 = y + 1;
////////            }
////////            else if (angle < 112.5) {
////////                x1 = x - 1;
////////                y1 = y;
////////                x2 = x + 1;
////////                y2 = y;
////////            }
////////            else {
////////                x1 = x - 1;
////////                y1 = y + 1;
////////                x2 = x + 1;
////////                y2 = y - 1;
////////            }
////////
////////            // Suppress the point if its magnitude is smaller than either of its neighbors
////////            float mag = magnitude.at<float>(y, x);
////////            float mag1 = magnitude.at<float>(y1, x1);
////////            float mag2 = magnitude.at<float>(y2, x2);
////////            if (mag < mag1 || mag < mag2) {
////////                result.at<float>(y, x) = 0;
////////            }
////////        }
////////    }
////////}
////////void convertToGrayscale(Mat& input_image, Mat& output_image)
////////{
////////    if (input_image.empty())
////////    {
////////        cout << "Input image is empty";
////////        return;
////////    }
////////
////////    output_image.create(input_image.rows, input_image.cols, CV_8UC1);
////////
////////    for (int i = 0; i < input_image.rows; i++)
////////    {
////////        for (int j = 0; j < input_image.cols; j++)
////////        {
////////            Vec3b pixel = input_image.at<Vec3b>(i, j);
////////
////////            int gray_value = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0];
////////
////////            output_image.at<uchar>(i, j) = gray_value;
////////        }
////////    }
////////}
////////cv::Mat CannyEdgeDetection(cv::Mat image, int segma, int lowThreshold, int highThreshold, int KernalSize) {
////////    cv::Mat Blured, magnitude, direction, result;
////////    int(*maskH)[3];
////////    int(*maskV)[3];
////////    //Gussian Bluring
////////    GaussianBlur(image, Blured, cv::Size(KernalSize, KernalSize), segma, segma);
////////    maskH = getArray("sobel", "horizontal");
////////    maskV = getArray("sobel", "vertical");
////////    //Sobel Edge detection in both vertical and horizontal directions
////////    cv::Mat gradientx = masking(Blured, maskH);
////////    cv::Mat gradienty = masking(Blured, maskV);
////////    gradientx.convertTo(gradientx, CV_32F);
////////    gradienty.convertTo(gradienty, CV_32F);
////////    cartToPolar(gradientx, gradienty, magnitude, direction, true);
////////    non_max_suppression(magnitude, direction, result);
////////    imshow("heyyy",result);
////////    inRange(result, cv::Scalar(lowThreshold), cv::Scalar(highThreshold), result);
////////    return result;
////////};
////////// Non-maximum suppression function
////////void nonMaxSuppression(const cv::Mat& src, cv::Mat& dst, int windowSize) {
////////    // Initialize the output matrix with zeros
////////    dst = cv::Mat::zeros(src.size(), CV_64F);
////////
////////    // Define the window centered at each pixel of the source image
////////    int w = (windowSize - 1) / 2;
////////    for (int y = 0; y < src.rows; ++y) {
////////        for (int x = 0; x < src.cols; ++x) {
////////            // Skip non-maximum values
////////            if (src.at<double>(y, x) < 1e-5) continue;
////////
////////            // Find the maximum value within the window
////////            double maxValue = 0.0;
////////            for (int dy = -w; dy <= w; ++dy) {
////////                for (int dx = -w; dx <= w; ++dx) {
////////                    int j = y + dy;
////////                    int i = x + dx;
////////                    if (j >= 0 && j < src.rows && i >= 0 && i < src.cols) {
////////                        maxValue = std::max(maxValue, src.at<double>(j, i));
////////                    }
////////                }
////////            }
////////
////////            // Keep only the maximum value
////////            if (src.at<double>(y, x) == maxValue) {
////////                dst.at<double>(y, x) = maxValue;
////////            }
////////        }
////////    }
////////}
