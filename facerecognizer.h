#ifndef FACE_RECOGNIZER_H
#define FACE_RECOGNIZER_H

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

class FaceRecognizer {
public:
    FaceRecognizer(string img_path, Mat _avgVec, Mat _eigenVec, Mat _facesInEigen, vector<string>& _loadedFacesID, int threshold);
    void prepareFace(Mat _testImg);
    void projectFace(Mat testVec, Mat _avgVec, Mat _eigenVec);
    void recognize(Mat testPrjFace, Mat _facesInEigen, vector<string>& _loadedFacesID, int threshold);
    string getClosetFaceID();
    double getClosetDist();
    ~FaceRecognizer();
    int  test_model(string class1);
private:
    Mat testVec;
    Mat testPrjFace;
    string closetFaceID = "None";
    double closetFaceDist = 10000;
    string imgLbl;
};

#endif
