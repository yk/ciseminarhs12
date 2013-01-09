////
////  main.cpp
////  fundamentalMatrix
////
////  Created by Alex Attinger on 10.12.12.
////  Copyright (c) 2012 Alex Attinger. All rights reserved.
////
//
//#include <iostream>
//#include "opencv2/opencv.hpp"
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/legacy/legacy.hpp"
//
//
//using namespace cv;
//using namespace std;
//
//std::vector<Point2f> matchedPoints_1,matchedPoints_2;
//std::vector<KeyPoint> matchedKeyPoints_1,matchedKeyPoints_2;
//Mat view1,view2,outliers;
//
//
////void showandsave(string name, Mat img) {
////	//imwrite("data/res/" + name + ".png", img);
////	imshow(name, img);
////	waitKey();
////}
//
//Mat getHomogeneousMat(const Point2f &x){
//    double pointData[] = {x.x,x.y,1.0};
//    Mat point = Mat(3,1,CV_64FC1,pointData).clone();
//    return point;
//
//}
//
//
//
//
//
//
//Mat calculateFundamental(){
//    Mat fundamentalRan;
//
//
//    fundamentalRan = findFundamentalMat(matchedPoints_1, matchedPoints_2,CV_FM_RANSAC,1.,.99,outliers);
//
//    return fundamentalRan;
//}
//
//
//
//void match() {
//
////	view1 = imread("/Users/alexattinger_3/Dropbox/5_sem/ki_seminar/7/data/johnHunter/001.png", CV_LOAD_IMAGE_GRAYSCALE);
////	view2 = imread("/Users/alexattinger_3/Dropbox/5_sem/ki_seminar/7/data/johnHunter/002.png", CV_LOAD_IMAGE_GRAYSCALE);
////
////
////	view1 = imread("/Users/alexattinger_3/Dropbox/5_sem/ki_seminar/7/data/car/input1.png", CV_LOAD_IMAGE_GRAYSCALE);
////	view2 = imread("/Users/alexattinger_3/Dropbox/5_sem/ki_seminar/7/data/car/input2.png", CV_LOAD_IMAGE_GRAYSCALE);
////
//            //
////
//
//
//	std::vector<KeyPoint> keypoints_1, keypoints_2;
//    SurfFeatureDetector detector(900);
//
//
//
//	detector.detect(view1, keypoints_1);
//	detector.detect(view2, keypoints_2);
//
//
//
//	//-- Step 2: Calculate descriptors (feature vectors)
//	SurfDescriptorExtractor extractor;
//    //	SiftDescriptorExtractor extractor;
//
//	Mat descriptors_1, descriptors_2;
//
//	extractor.compute(view1, keypoints_1, descriptors_1);
//	extractor.compute(view2, keypoints_2, descriptors_2);
//
//
//	//-- Step 3: Matching descriptor vectors with a brute force matcher
//	BruteForceMatcher<L2<float> > matcher;
//	std::vector<DMatch> matches;
//	matcher.match(descriptors_1, descriptors_2, matches);
//
//	float thresh = 0.24;
//
//	auto newend =
//    remove_if(matches.begin(), matches.end(),
//              [&](DMatch& match)->bool {/*cout<<match.distance<<endl;*/return (abs(match.distance) > thresh);});
//	auto good_matches = vector<DMatch>(matches.begin(), newend);
//	//-- Draw matches
//	Mat img_matches;
//	drawMatches(view1, keypoints_1, view2, keypoints_2, good_matches, img_matches);
//    cout<<keypoints_1[1].pt.x<<endl;
//    for(int i = 0;i<good_matches.size();i++){
//        matchedKeyPoints_1.push_back(keypoints_1[good_matches[i].queryIdx]);
//        matchedKeyPoints_2.push_back(keypoints_2[good_matches[i].trainIdx]);
//        matchedPoints_1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
//        matchedPoints_2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
//    }
//
//	//-- Show detected matches
//	showandsave("Matches_surf_02", img_matches);
//}
//
//int minDisp = 64;
//int numDisp = 128;
//int SADSize = 3;
//int P1 = 5400;
//int P2 = 21600;
//string winname = "Disparity Map";
//Mat dest1,dest2;
//Mat dispMap;
//
//void processInput(int, void*){
//    int _minDisp = -minDisp;
//    int _numDisp = numDisp-numDisp%16;
//    int _SADsize = SADSize;
//    StereoSGBM bm(_minDisp, _numDisp, _SADsize,P1,P2);
//    bm(dest1,dest2,dispMap);
//    normalize(dispMap,dispMap, 0,256,CV_MINMAX,NULL);
//    //applyColorMap(dispMap, dispMap, cv::COLORMAP_JET);
//    //Mat disp_vis = Mat(dispMap.size(), CV_8U);
//    //convertScaleAbs(dispMap, disp_vis);
//    imshow(winname,dispMap);
//
//}
//
//void setUpTrackbars(){
//    namedWindow(winname,1);
//    createTrackbar("minimum", winname, &minDisp, 128,processInput);
//    createTrackbar("num Disp",winname,&numDisp,500,processInput);
//    createTrackbar("Patch size",winname,&SADSize,21,processInput);
//    createTrackbar("P1",winname,&P1,7000,processInput);
//    createTrackbar("p2", winname, &P2, 30000,processInput);
//}
//
//
//
//int main2(int argc, const char * argv[])
//{
//    //EX 1
//    match();
//    for(int i = 0; i<matchedPoints_1.size();i++){
//        cout<<matchedPoints_1[i].y-matchedPoints_2[i].y<<endl;
//    }
//    Mat f = calculateFundamental();
//    Mat h1,h2;
//
//
//    //rectify images
//    stereoRectifyUncalibrated(matchedPoints_1, matchedPoints_2, f, view1.size(), h1, h2);
//    //apply the transformation
//    warpPerspective(view1, dest1, h1, view1.size());
//    warpPerspective(view2, dest2, h2, view1.size());
//    //show images
//    imshow("Rectified 1",dest1);
//    imshow("Rectified 2",dest2);
//    //do the matching
//    setUpTrackbars();
//    processInput(0, 0);
//
//
//    waitKey();
//    return 0;
//}
