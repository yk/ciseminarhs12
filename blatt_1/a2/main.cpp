//
//  main.cpp
//  ex_2
//
//  Created by Alex Attinger on 27.10.12.
//  Copyright (c) 2012 Alex Attinger. All rights reserved.
//


#include <iostream>
#include "opencv2/opencv.hpp"
#include <sstream>
using namespace cv;

int main(int argc, const char * argv[])
{
    
    int max_kernel_size = 21;
    
    string win_orig[] = {"Office 1","office 2"};
    string win_blur[] = {"Blur Filered Image 1","BlurFiltered Image 2"};
    string win_gblur[] = {"Gauss Filtered Image 1","Gauss Filtered Image 2"};
    string win_mblur[] = {"Median Filtered Image 1", "Median Filtered Image 2"};
    
    vector<Mat> dst = vector<Mat>(2);
    vector<Mat> src = vector<Mat>(2);
    int im_width = 300;
    
    for (int i = 0; i<2;i++){
      
        src[i] = imread(argv[i+1],0);
        
        dst[i] = src[i].clone();
        
        namedWindow(win_orig[i]);
        moveWindow(win_orig[i],i*im_width,0);
        imshow(win_orig[i],src[i]);

    }
    
    for (int i = 1; i< max_kernel_size;i+=2){
        for (int j = 0; j<2;j++){
            blur(src[j], dst[j], Size(i,i),Point(-1,-1));
            namedWindow(win_blur[j]);
            moveWindow(win_blur[j],j*im_width,300);
            imshow(win_blur[j],dst[j]);
            
            GaussianBlur(src[j], dst[j], Size(i,i),0,0);
            namedWindow(win_gblur[j]);
            moveWindow(win_gblur[j],j*im_width,600);
            imshow(win_gblur[j],dst[j]);
            
            medianBlur(src[j], dst[j], i);
            namedWindow(win_mblur[j]);
            moveWindow(win_mblur[j],(j+2)*im_width,300);
            imshow(win_mblur[j],dst[j]);
            
            
            

        }
        std::cout<<"Kernel size "<< i;
        waitKey();
    }        
    
    destroyAllWindows();
    
    Mat src_orig = imread("/Users/alexattinger_3/Dropbox/5_sem/ki_seminar/blatt_1/office.jpg",0);
    namedWindow("Original office");
    imshow("Original office",src_orig);
    Mat dst_laplace;
    
    Laplacian(src_orig, dst_laplace, CV_8U);
    convertScaleAbs(dst_laplace, dst_laplace);
    namedWindow("lablace");
    imshow("laplace",dst_laplace);
    Mat dif = src_orig-dst_laplace;
    namedWindow("dif");
    imshow("dif", dif);
    
    waitKey();
    
    
    
    return 0;
}

