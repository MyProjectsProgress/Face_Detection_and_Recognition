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
    QString imgPath = QFileDialog::getOpenFileName(this, "Open an Image", "..", "Images (*.png *.xpm *.jpg *.bmb)");

    if(imgPath.isEmpty())
        return;
    img = imread(imgPath.toStdString());
    cv::resize(img, img, Size(512, 512));
    showImg(img, ui->imginput1, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height(),1);


}



void MainWindow::on_submitBtn_clicked()
{
    Mat output;
    output=findFacesInImage(img.clone());


    showImg(output, ui->imgOutput1, QImage::Format_BGR888, ui->imgOutput1->width(), ui->imgOutput1->height(),1);

}

