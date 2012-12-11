//
//  main.cpp
//  fundamentalMatrix
//
//  Created by Alex Attinger on 10.12.12.
//  Copyright (c) 2012 Alex Attinger. All rights reserved.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"


using namespace cv;
using namespace std;

std::vector<Point2f> matchedPoints_1,matchedPoints_2;
std::vector<KeyPoint> matchedKeyPoints_1,matchedKeyPoints_2;
Mat view1,view2;


void showandsave(string name, Mat img) {
	//imwrite("data/res/" + name + ".png", img);
	imshow(name, img);
	waitKey();
}

Mat getHomogeneousMat(const Point2f &x){
    double pointData[] = {x.x,x.y,1.0};
    Mat point = Mat(3,1,CV_64FC1,pointData).clone();
    return point;
    
}

void checkFundamentalMatrix(Point2f &x,Point2f &xprime,Mat &f){

    Mat left = getHomogeneousMat(xprime);
    Mat right = getHomogeneousMat(x);
    
    
    Mat prod;
    prod = f*right;
    prod = left.t()*prod;
  
    cout<<endl;
    cout<<"Product x'T*F*x: "<<prod<<endl;
    //prod = left*prod;
    //cout<<prod;
}




Mat calculateFundamental(){
    Mat fundamental8,fundamentalRan;
    Mat out;
    fundamental8=findFundamentalMat(matchedPoints_1, matchedPoints_2,CV_FM_8POINT);
    cout<<"Fundamental by 8 point"<<endl<<fundamental8<<endl;
    
    fundamentalRan = findFundamentalMat(matchedPoints_1, matchedPoints_2,CV_FM_RANSAC,1.,.99,out);
 
    cout<<"Fundamental by Ransac"<<endl<<fundamentalRan<<endl;
    cout<<out;
    for (int i = 0; i<matchedPoints_1.size(); i++) {
        //should all be very close to 0!
        checkFundamentalMatrix(matchedPoints_1[i], matchedPoints_2[i], fundamentalRan);

    }
        return fundamentalRan;
}


vector<KeyPoint> getHarrisPoints(Mat& img,int thresh=95) {
	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(img.size(), CV_32FC1);
    //	int thresh = 95;
    
	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;
    
	/// Detecting corners
	cornerHarris(img, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
    
	/// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);
    cout<<dst_norm.rows<<endl;
	
    vector<KeyPoint> v;
	/// Drawing a circle around corners
	for (int j = 0; j < dst_norm.rows; j++) {
		for (int i = 0; i < dst_norm.cols; i++) {
			if ((int) dst_norm.at<float>(j, i) > thresh) {
				v.push_back(KeyPoint(i,j,1));
			}
		}
	}
	cout << "got " << v.size() << " harris points" << endl;
	return v;
}

void match() {
	view1 = imread("/Users/alexattinger_3/Dropbox/5_sem/ki_seminar/6/johnHunter/001.png", CV_LOAD_IMAGE_GRAYSCALE);
	view2 = imread("/Users/alexattinger_3/Dropbox/5_sem/ki_seminar/6/johnHunter/002.png", CV_LOAD_IMAGE_GRAYSCALE);

  
    //	SiftFeatureDetector detector(0.14,0.14);
    
	std::vector<KeyPoint> keypoints_1, keypoints_2;
    SurfFeatureDetector detector(2500);
    //	SiftFeatureDetector detector(0.14,0.14);
    
	    
	detector.detect(view1, keypoints_1);
	detector.detect(view2, keypoints_2);

    
    
	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;
    //	SiftDescriptorExtractor extractor;
    
	Mat descriptors_1, descriptors_2;
    
	extractor.compute(view1, keypoints_1, descriptors_1);
	extractor.compute(view2, keypoints_2, descriptors_2);
    
    
	//-- Step 3: Matching descriptor vectors with a brute force matcher
	BruteForceMatcher<L2<float> > matcher;
	std::vector<DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);
    
	float thresh = 0.15;
    
	auto newend =
    remove_if(matches.begin(), matches.end(),
              [&](DMatch& match)->bool {/*cout<<match.distance<<endl;*/return (abs(match.distance) > thresh);});
	auto good_matches = vector<DMatch>(matches.begin(), newend);
	//-- Draw matches
	Mat img_matches;
	drawMatches(view1, keypoints_1, view2, keypoints_2, good_matches, img_matches);
    cout<<keypoints_1[1].pt.x<<endl;
    for(int i = 0;i<good_matches.size();i++){
        matchedKeyPoints_1.push_back(keypoints_1[good_matches[i].queryIdx]);
        matchedKeyPoints_2.push_back(keypoints_2[good_matches[i].trainIdx]);
        matchedPoints_1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
        matchedPoints_2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
    }
        
	//-- Show detected matches
	showandsave("Matches_surf_02", img_matches);
}

void drawEpilines(Mat &view,Mat &lines){
    float a,b,c;
    float m;
    Point2f x1,x2;
    for(int i = 0;i<lines.rows;i++){
        a=lines.at<float>(i,0);
        b=lines.at<float>(i,1);
        c=lines.at<float>(i,2);
        m=-a/b;
        x1.x=0;
        x1.y=-c/b;
        x2.x=view.rows;
        x2.y=m*view.rows-c/b;
        line(view,x1,x2,CV_RGB(0,255,0));
        //cout<<"Result x^2+y^2 = "<<a*a+b*b<<endl;
        
    }

    
}

Mat getLineVector(Mat f, Point2f x){

    double pointData[] = {x.x,x.y,1.0};
    Mat point = Mat(3,1,CV_64F,pointData).clone();
   
    return point.t()*f;
    
}


double calculateDistance(const Mat &line,const Point2f &x){
    Mat point = getHomogeneousMat(x);
    Mat prod = point.t()*line.t();
       return prod.at<double>(0,0);
}



int main(int argc, const char * argv[])
{
    //EX 1
    match();
    Mat f = calculateFundamental();
    
    
    //EX 2
    Mat lines1,lines2;
    //get the Epipolar lines, a*x+b*y+c
    //computeCorrespondingEpilines: gives me:
    //l'=F*x the lines on image 2
    computeCorrespondEpilines(matchedPoints_1, 1, f, lines1);
    //l=FT*x' the lines on image 1
    computeCorrespondEpilines(matchedPoints_2, 2, f, lines2);
    //draw the lines and keypoints for each image
    drawEpilines(view1, lines2);
    drawKeypoints(view1, matchedKeyPoints_1, view1);
    imshow("epilines in view 1",view1);
    drawEpilines(view2, lines1);
    drawKeypoints(view2, matchedKeyPoints_2, view2);
    imshow("epilines in view 2",view2);
  
    for(int i = 0;i<matchedPoints_1.size();i++){
        Mat line = Mat::ones(1,3,CV_64FC1);
        for(int j = 0;j<3;j++){
             line.at<double>(0,j)=lines2.at<float>(i,j);
        }
       
       
        double d = calculateDistance(line, matchedPoints_1[i]);
        cout<<"Distance between Point "<<i<<" and corresponding line: "<<d<<endl;
    }
    
    
    waitKey();
    return 0;
}
