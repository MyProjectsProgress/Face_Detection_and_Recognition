#include "faceRecognizer.h"
#include <filesystem>

FaceRecognizer::FaceRecognizer(string img_path, Mat _avgVec, Mat _eigenVec, Mat _facesInEigen, vector<string>& _loadedFacesID, int threshold) {
    Mat _testImg = imread(img_path,0);
    prepareFace(_testImg);
    projectFace(testVec, _avgVec, _eigenVec);
    recognize(testPrjFace, _facesInEigen, _loadedFacesID, threshold);
    std::filesystem::path path = img_path;
    string fileName = path.filename().string();
    int pos = fileName.find("-");
    this->imgLbl = fileName.substr(0, pos);
}

void FaceRecognizer::prepareFace(Mat _testImg)
{
    _testImg.convertTo(_testImg, CV_32FC1);
    _testImg.reshape(0, _testImg.rows*_testImg.cols).copyTo(testVec);
}

void FaceRecognizer::projectFace(Mat testVec, Mat _avgVec, Mat _eigenVec){
    Mat tmpData;
    cout << testVec.type()<<endl;
    cout << _avgVec.type() << endl;
    cout << testVec.rows << "X" << testVec.cols <<endl;
    cout << _eigenVec.rows << "X" << _eigenVec.cols <<endl;
    subtract(testVec, _avgVec, tmpData);
    testPrjFace = _eigenVec * tmpData;

}
//Find the closet Euclidean Distance between input and database
void FaceRecognizer::recognize(Mat testPrjFace, Mat _facesInEigen, vector<string>& _loadedFacesID, int threshold)
{
    int minDist = 10000;
    int min_index = -1;
    int i = 0;
    for (i =0; i < _loadedFacesID.size(); i++) {
        Mat src1 = _facesInEigen.col(i);
        Mat src2 = testPrjFace;

        double dist = norm(src1, src2, NORM_L2);
        cout << dist << endl ;
        if (dist < minDist) {
            minDist = dist;
            min_index = i;
        }
    }
    if(minDist<threshold){
        this->closetFaceID = _loadedFacesID[min_index];
    }
    else{
        this->closetFaceID = "Unknown";
    }
    this->closetFaceDist = minDist;


}

string FaceRecognizer::getClosetFaceID()
{
    return closetFaceID;
}

double FaceRecognizer::getClosetDist()
{
    return closetFaceDist;
}

FaceRecognizer::~FaceRecognizer() {}


int FaceRecognizer:: test_model(string class1){
    if(this->closetFaceID == this->imgLbl && this->imgLbl == class1){
        return 1;
    }
    else if(this->closetFaceID != this->imgLbl && this->imgLbl == class1){
        return 2;
    }
    else if(this->closetFaceID == class1 && this->imgLbl != class1){
        return 3;
    }
    else if(this->closetFaceID != class1 && this->imgLbl != class1){
        return 4;
    }
    return 0;
}
