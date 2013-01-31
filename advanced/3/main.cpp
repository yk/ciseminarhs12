//
//  main.cpp
//  GeneralizedHoughTransfrom
//
//  Created by Alex Attinger on 09.01.13.
//  Copyright (c) 2013 Alex Attinger. All rights reserved.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "dirent.h"
#include <boost/algorithm/string.hpp>



using namespace cv;
using namespace std;




Mat gradDirMap(Mat &templ){
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;
    Mat blurred;
    GaussianBlur(templ, blurred, Size(3,3), 0,0,BORDER_DEFAULT);
    Mat grad_x,grad_y,abs_grad_x,abs_grad_y;
    Sobel(blurred, grad_x, ddepth, 1, 0,3,scale,delta,BORDER_DEFAULT); //approximate gradient in x directio
    Sobel(blurred,grad_y,ddepth,0,1,3,scale,delta,BORDER_DEFAULT); //approximate in y direction

//        convertScaleAbs(grad_x, abs_grad_x);
//        convertScaleAbs(grad_y,abs_grad_y);
//        imshow("Grad y",abs_grad_y);
//        imshow("Grad x",abs_grad_x);

    Mat dir = Mat::zeros(grad_x.size(),CV_32F);
    
    cout<<grad_x.size()<<endl;
    Point curr(0,0);
    
    for(int r = 0; r<dir.rows;r++){
        for(int c = 0; c<dir.cols;c++){
            curr.x=c;
            curr.y=r;
            
            float x = grad_x.at<float>(curr);
            float y = grad_y.at<float>(curr);
           //direction: arctan of the two gradient vectors
            float res = fastAtan2(y,x);
          
            dir.at<float>(curr)=res;
            
        }
    }
   // cout<<"direction type "<<dir.type()<<" grad_x "<<grad_x.channels()<<endl;
    return dir;
}

vector< vector<Point> > generateLookUpTable(Mat &gradientMap, int nBins, Point referencePoint, Mat &cannyFiltered){
    //look up table
    vector<vector<Point> >table(nBins);
    int binsize = 180/nBins;
    
    for(int r=0;r<gradientMap.rows;r++){
        for(int c = 0; c<gradientMap.cols;c++){
            
            Point curr(c,r);
            float gradDir = gradientMap.at<float>(curr);
            //to improve performance, we only look at pixels which are detected by a canny edge detector as edge pixels
            unsigned char gradMag = cannyFiltered.at<unsigned char>(curr);
           
            if(gradMag > 0){
                //normalize direction
                if(gradDir >= 180){
                    gradDir = gradDir - 180;
                }
                
                // get the appropriate bin for the direction
                int bin = gradDir/binsize;
               //save a vector pointing from the current point to the reference point in the table
                table[bin].push_back(referencePoint-curr);
            }
        }
    }
    return table;
    
}



bool pointInsideImage(Point &p, Mat &img){
    //check if given point is inside image
    if(p.x>=0 && p.x<img.cols && p.y>=0 && p.y< img.rows){
        return true;
    }
    return false;
}

void evalPointList(vector<Point> & list, Mat &points,Point &current){
    for (Point p: list) {
        Point reference = p+current;
        if(pointInsideImage(reference,points)){
            
            points.at<unsigned short>(reference)+=1;
            
        }
    }
}

Mat findPoints(Mat &gradMap,vector<vector<Point> > lookUpTable,int nBins){
    int binsize = 180/nBins;
    Mat points = Mat::zeros(gradMap.size(),CV_16U);
    for(int r=0;r<gradMap.rows;r++){
        for(int c = 0; c<gradMap.cols;c++){
            
            Point curr(c,r);
            //get the gradient direction at current location
            float gradDir = gradMap.at<float>(curr);
            
     
                vector<Point> pointList;
                if(gradDir>=180){
                    gradDir = gradDir -180;
                }
            //get the correponding bin in the table
                int bin = gradDir/binsize;
            //get all vectors corresponding to this bin
                pointList = lookUpTable[bin];
               
                evalPointList(pointList, points,curr);
          
        }
    }
   
    return points;
    
}


void displayScoreMap(Mat &map){
    Mat normMap;
    normalize(map,normMap,255,0,CV_MINMAX,CV_8UC1);
    imshow("Score Map",normMap);
}

void drawCirclesAroundMatches(Mat &img,Mat &map){
    Point curr;
    for(int r= 0; r<img.rows;r++){
        for (int c = 0; c<img.cols;c++){
            curr.x = c;
            curr.y = r;
            if(map.at<float>(curr)>0){
                circle(img,curr,40,Scalar(50));
            }
        }
    }
}

void findAndDisplayMatches(Mat &points,Mat &testImage, double thresh){
    Mat thresholded;
    points.convertTo(thresholded, CV_32FC1);
    
    threshold(thresholded,thresholded,thresh,1.0,THRESH_BINARY);
    
    drawCirclesAroundMatches(testImage,thresholded);
    imshow("threshold",thresholded);
    
    imshow("TestImage",testImage);

}

int main(int argc, const char * argv[])
{
    
    //load template
    Mat templ = imread(argv[1],0);
    Mat cannyFiltered,testCannyFiltered;
    //filter template to get a mask for the edges
    Canny(templ, cannyFiltered, 80, 240);
    
    //imshow("Canny Output", cannyFiltered);
    imshow("Template",templ);
 
    int nBins = 20;
    
    //generate the gradient map
    Mat map = gradDirMap(templ);
    

    cout<<"number of nonzero in map"<<countNonZero(map)<<endl;
    //generate the look up table
    vector< vector<Point> > table = generateLookUpTable(map, nBins, Point(30,30     ),cannyFiltered);
    int sum = 0;
    for(int i = 0; i<table.size();i++){
        sum += table[i].size();
    }
    cout<<sum<<endl;
    cout<<"number in first bin "<<table[0].size()<<endl;
    waitKey();
    
    
    //test image
    Mat gradMagTest;
    Mat testImage = imread(argv[2],0);
    //GaussianBlur(testImage, testImage, Size(3,3),0,0,BORDER_DEFAULT);
    Mat testmap = gradDirMap(testImage);
    
    Mat points;
    points = findPoints(testmap, table, nBins);

    
    displayScoreMap(points);
    //normalize the points for display
    
    Point maxLoc;
    //find the maximum
    double max;
    minMaxLoc(points,NULL,&max,NULL,&maxLoc);
    //cout<<maxLoc<<"max: "<<max<<endl;

    findAndDisplayMatches(points,testImage,max-30);
    
        
    
    
   
    waitKey(); 
    return 0;
}

