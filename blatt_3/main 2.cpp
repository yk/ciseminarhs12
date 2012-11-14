//
//  main.cpp
//  a1
//
//  Created by Alex Attinger on 12.11.12.
//  Copyright (c) 2012 Alex Attinger. All rights reserved.
//

#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

Mat src,dst,detected,orig;
int edgeThresh = 1;
int lowThresh;
int const max_lowThresh = 100;
int ratio=3;
int kernel_size = 3;
int gaussKernel = 1;
int sobelThres = 20;
string winname = "Canny Egde";
string origPic = "Original";
string sobelPic = "Sobel";
string threswin = "Min Threshold Canny";
string kernelwin = "Gaussian Kernel size";

Mat sob_x,sob_y,sobel;

void cannySlider(int,void*){
  
   
    Canny(src,detected,lowThresh,lowThresh*ratio,kernel_size);
    dst = Scalar::all(0);
    
    src.copyTo(dst,detected);
    imshow(winname,dst);
    moveWindow(winname, src.cols+10, 0);

    
}

void cannyEdgeDetection(){
    
    dst.create(src.size(),src.type());
       
    cannySlider(0,0);
    
    
}

void sobelEdgeDetection(){
    Sobel(orig,sob_x,CV_16S,1,0,3,1,0,BORDER_DEFAULT);
    Sobel(orig,sob_y,CV_16S,0,1,3,1,0,BORDER_DEFAULT);
    convertScaleAbs(sob_y, sob_y);
    convertScaleAbs(sob_x,sob_x);
    
    addWeighted(sob_x, .5, sob_y, .5, 0, sobel);
    threshold(sobel,sobel,sobelThres,255,0);
    imshow(sobelPic, sobel);
    moveWindow(sobelPic, sobel.cols*2+40, 0);
    
    
}

void processImage(int,void*){
    
    if (gaussKernel%2 == 0) {
        gaussKernel = gaussKernel+1;
    };
    GaussianBlur(orig, src, Size(gaussKernel,gaussKernel), 0,0);
    
    imshow(origPic,src);
    
    cannyEdgeDetection();
    
    sobelEdgeDetection();

}
void setupTrackbars(){
    namedWindow(threswin);
    resizeWindow(threswin, 600, 100);
    createTrackbar( "Canny lower", threswin, &lowThresh, max_lowThresh, processImage);
    createTrackbar("Gaussian Blur Kernel",threswin,&gaussKernel,15,processImage);
    createTrackbar("Sobel Threshold",threswin,&sobelThres,200,processImage);

    
}

int main(int argc, const char * argv[])
{
    string dirname = argv[1];
        
    string files[] = {"butterfly.jpg","cells.jpg","outdoor.jpg","stairs.png","wheel.png"};
    setupTrackbars();
    for (int i =  0;i<5;i++){
        orig = imread(dirname+files[i],0);
        processImage(0, 0);
        
        
                waitKey(0);
    }
   
    
    return 0;
}

