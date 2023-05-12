#ifndef FACEDETECTION_H
#define FACEDETECTION_H


#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QImage>
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;


Mat findFacesInImage(Mat img);
#endif // FACEDETECTION_H
