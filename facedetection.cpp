#include "facedetection.h"
//#include "ui_mainwindow.h"



cv::Mat findFacesInImage(cv::Mat img){

    if (img.empty()) {
          throw std::invalid_argument("No image");
      }

      cv::CascadeClassifier faceCascade;
      if (!faceCascade.load("D:/3rd Year 2st Term/Computer Vision/Task 5 Repository/Face_Recognition/xmlFiles/haarcascade_frontalface_default.xml")) {
          throw std::invalid_argument("Failed to load face cascade classifier.");
      }

      cv::Mat grayImage;
      cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);

      // Increase scaleFactor, adjust minSize and maxSize for better face detection
      double scaleFactor = 1.2;
      cv::Size minSize(30, 30);
      cv::Size maxSize(200, 200);

      std::vector<cv::Rect> faces;
      faceCascade.detectMultiScale(grayImage, faces, scaleFactor, 5, 0, minSize, maxSize);

      // Draw rectangles around detected faces
      for (const auto& face : faces) {
          cv::rectangle(img, face, cv::Scalar(0, 255, 0), 2);
      }

      return img;
}

