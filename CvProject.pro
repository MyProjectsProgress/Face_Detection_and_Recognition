QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
INCLUDEPATH += C:\opencv\build\include
LIBS += C:\opencv\open-cv-build\bin\libopencv_calib3d460.dll
LIBS += C:\opencv\open-cv-build\bin\libopencv_core460.dll
LIBS += C:\opencv\open-cv-build\bin\libopencv_highgui460.dll
LIBS += C:\opencv\open-cv-build\bin\libopencv_imgcodecs460.dll
LIBS += C:\opencv\open-cv-build\bin\libopencv_imgproc460.dll
LIBS += C:\opencv\open-cv-build\bin\libopencv_objdetect460.dll



SOURCES += \
    facedetection.cpp \
    facerecognizer.cpp \
    main.cpp \
    mainwindow.cpp \
    pca.cpp \
    roc.cpp \
    writetraindata.cpp

HEADERS += \
    ReadFile.h \
    facedetection.h \
    facerecognizer.h \
    mainwindow.h \
    pca.h \
    roc.h \
    writetraindata.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
