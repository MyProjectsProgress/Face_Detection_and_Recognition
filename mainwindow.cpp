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
#include <facedetection.h>
#include<facerecognizer.h>
#include<pca.h>
#include<ReadFile.h>
#include<writetraindata.h>
using namespace std;
using namespace cv;
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}
QString imgPath;


void MainWindow::on_pushButton_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void MainWindow::on_pushButton_3_clicked()
{
    ui->stackedWidget->setCurrentIndex(1);
}

void MainWindow::on_pushButton_2_clicked()
{
    ui->stackedWidget->setCurrentIndex(2);
}
void showImg(cv::Mat& img, QLabel* imgLbl, enum QImage::Format imgFormat, int width , int hieght, bool colorTransform)
{
    if(colorTransform){
        cvtColor(img, img, COLOR_BGR2RGB);
    }
    QImage image((uchar*)img.data, img.cols, img.rows, imgFormat);
    QPixmap pix = QPixmap::fromImage(image);
    imgLbl->setPixmap(pix.scaled(width, hieght, Qt::KeepAspectRatio));
}



Mat img;
void MainWindow::on_actionUpload_triggered()
{
    ui->imginput1->clear();
    ui->imgOutput1->clear();
    imgPath = QFileDialog::getOpenFileName(this, "Open an Image", "..", "Images (*.png *.xpm *.jpg *.bmp)");

    if(imgPath.isEmpty())
        return;
    img = imread(imgPath.toStdString());
    cv::resize(img, img, Size(512, 512));
    showImg(img, ui->imginput1, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height(),1);
    showImg(img, ui->imginput2, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height(),0);



}



void MainWindow::on_submitBtn_clicked()
{
    Mat output;
    output=findFacesInImage(img.clone());


    showImg(output, ui->imgOutput1, QImage::Format_RGB888, ui->imgOutput1->width(), ui->imgOutput1->height(),0);

}


void MainWindow::on_predictBtn_clicked()
{
        vector<string> trainFacesPath;
         vector<string> trainFacesID;
         vector<string> loadedFacesID;
        //read training list and ID from txt file
        //read training data(faces, eigenvector, average face) from txt file
        string trainListFilePath="C:/Users/sata/Documents/GitHub/Face_Recognition/list/train_list.txt";
        readList(trainListFilePath, trainFacesPath, trainFacesID);
        Mat avgVec, eigenVec, facesInEigen;
        facesInEigen = readFaces(int(trainFacesID.size()), loadedFacesID);
        avgVec = readMean();
        eigenVec = readEigen(int(trainFacesID.size()));
        string testImgPath;
        Mat frame, processed, testImg;
//        do PCA analysis for training faces
        std::string TestPath = imgPath.toStdString();
        cout<<TestPath;

        FaceRecognizer faceRecognizer = FaceRecognizer(TestPath, avgVec, eigenVec, facesInEigen, loadedFacesID,3000);
        // Show Result
        QString closestID = QString::fromStdString(faceRecognizer.getClosetFaceID());
        ui->prediction_label->setText(closestID);

}

