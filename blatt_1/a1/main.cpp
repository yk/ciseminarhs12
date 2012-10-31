//
//  main.cpp
//  computer_vision
//
//  Created by Alexander Attinger on 27.09.12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//
#include <iostream>
#include "opencv2/opencv.hpp"
using namespace cv;

void drawHist(const Mat src, const string &winname,const int nBins);

int main(int argc, const char * argv[])
{
    
    Mat src, dst;
    Mat dstcop;
    string source_window = "Source image";
    string equalized_window = "Equalized Image";
    
    src = imread(argv[1],0);
    namedWindow(source_window);
    namedWindow(equalized_window);
    src.convertTo(src, CV_8UC1);
    dst = src.clone();
    imshow(source_window,src);

    
    
    equalizeHist(src, dst);
    
   
    imshow(equalized_window,dst);
    
    
    drawHist(src,"Histogram original",50);
    drawHist(dst,"Histogram equalized",256);
    waitKey();
    cvDestroyWindow( "mywindow" );
    
    
    return 0;
}

void drawHist(const Mat src, const string &winname,const int nBins){
    //calculating the hist
    Mat hist;
    float range[] = {0,256};
    const float* histRange = {range};
    bool uniform = true, accumulate = false;
    
    calcHist(&src,1,0,Mat(),hist,1,&nBins,&histRange,uniform,accumulate);
    
    //drawing the hist
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/nBins );
    
    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 155,155,155) );
    
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    
    for( int i = 1; i < nBins; i++ ) {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) , Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
             Scalar( 255,0, 0), 2, 8, 0 );
       
    }
    
    namedWindow(winname, CV_WINDOW_AUTOSIZE );
    imshow(winname, histImage );
    
    
}
