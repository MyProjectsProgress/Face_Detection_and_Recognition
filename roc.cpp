#include "roc.h"



//int* threshold_range(string folder_path){
//        std::vector<cv::Mat> images;
//        cv::String folder(folder_path);
//        std::vector<cv::String> file_names;
//        cv::glob(folder, file_names);
//        vector<string> trainFacesPath;
//        vector<string> trainFacesID;
//        vector<string> loadedFacesID;
//        //read training list and ID from txt file
//        //read training data(faces, eigenvector, average face) from txt file
//        string trainListFilePath="C:/Users/sata/Documents/GitHub/Face_Recognition/list/train_list.txt";
//        readList(trainListFilePath, trainFacesPath, trainFacesID);
//        Mat avgVec, eigenVec, facesInEigen;
//        facesInEigen = readFaces(int(trainFacesID.size()), loadedFacesID);
//        avgVec = readMean();
//        eigenVec = readEigen(int(trainFacesID.size()));
//        vector<int> distances;
//        int euclidean_dist = 0;
//        for (auto& file_name : file_names) {
//            FaceRecognizer obj1 =FaceRecognizer(file_name, avgVec, eigenVec, facesInEigen, loadedFacesID,3000);

//            // caclculate distance for each test image
//             euclidean_dist=obj1.getClosetDist();
//             distances.push_back(euclidean_dist);
//         }







//    // get min and max value in the vector
//    int min = *min_element(distances.begin(), distances.end());
//    int max = *max_element(distances.begin(), distances.end());

//    int range[2] = {min , max};

//    return range;
//}


//int* confusionMatrix(string folder_path, int threshold, string class_name){

////    int tp, tn, fp, fn = 0;

//    std::vector<cv::Mat> images;
//    cv::String folder(folder_path);
//    std::vector<cv::String> file_names;
//    cv::glob(folder, file_names);
//    vector<string> trainFacesPath;
//    vector<string> trainFacesID;
//    vector<string> loadedFacesID;
//    //read training list and ID from txt file
//    //read training data(faces, eigenvector, average face) from txt file
//    string trainListFilePath="C:/Users/sata/Documents/GitHub/Face_Recognition/list/train_list.txt";
//    readList(trainListFilePath, trainFacesPath, trainFacesID);
//    Mat avgVec, eigenVec, facesInEigen;
//    facesInEigen = readFaces(int(trainFacesID.size()), loadedFacesID);
//    avgVec = readMean();
//    eigenVec = readEigen(int(trainFacesID.size()));



//    int values[4];
//    for (int i = 0; i < 4; i++){
//        values[i] = 0;
//    }

//    // Run through all test images
//    for (auto& file_name : file_names){
//        FaceRecognizer obj1 =FaceRecognizer(file_name, avgVec, eigenVec, facesInEigen, loadedFacesID,threshold);

//        int prediction_result = obj1.test_model(class_name);

//        switch(prediction_result){
//        case 1:
////            tp++;
//            values[0]++;
//            break;

//        case 2:
////            fn++;
//            values[1]++;
//            break;

//        case 3:
////            fp++;
//            values[3]++;
//            break;

//        case 4:
////            tn++;
//            values[2]++;
//            break;

//        }

//    }

//    return values;
//}



//vector<vector<float>> roc(string folder_path, int min_thresh, int max_thresh, string class_name){

//    vector<float> tpr_list;
//    vector<float> fpr_list;
//    float tpr, fpr;

////    int values[4];
//    vector<vector<float>> roc_values;

//    int step = max_thresh / 100;

//    for (int thresh; thresh <= max_thresh; thresh + step){
//        int* values = confusionMatrix(folder_path, thresh, class_name);
//        tpr = values[0] / values[0] + values[1];
//        fpr = values[3] / values[3] + values[2];

//        tpr_list.push_back(tpr);
//        fpr_list.push_back(fpr);
//    }

//    roc_values = {tpr_list, fpr_list};

//    return roc_values;
//}
