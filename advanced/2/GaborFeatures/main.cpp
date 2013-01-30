//
//  main.cpp
//  GaborFeatures
//
//  Created by Alex Attinger on 14.12.12.
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


int kernel_size=31;
int pos_sigma= 5; //width of gaussian
int pos_lm = 200; // frequency of co
int pos_th = 0; //angle
int pos_psi = 90; // shift from center
cv::Mat src_f;
cv::Mat dest;
cv::Mat srcImageBW;
 vector<Mat> featureVectors;


cv::Mat gabor;
cv::Mat jet;
int color = 0;
Point referencePixel(0,0);


cv::Mat mkKernel(int ks, double sig, double th, double lm, double ps)
{


    Mat kernel = getGaborKernel(Size(ks,ks), sig, th, lm, 1.0,ps,CV_32F);

       return kernel;
}



void Process(int , void *)
{

    double sig = pos_sigma; // width of the gaussian

    double lm = 2+pos_lm/100.0; // frequency of the cosine
    double th = pos_th*CV_PI/180.0; // orientation of the filter; usually 8 orientations used
    double ps = pos_psi*CV_PI/180.0;//phase
    
    cv::Mat kernel = mkKernel(kernel_size, sig, th, lm, ps);
 
    cv::filter2D(src_f, dest, CV_32F, kernel);
    normalize(dest, dest,1.0,0.0,CV_MINMAX);
    cv::imshow("Process window", dest);
    cv::Mat Lkernel(kernel_size*10, kernel_size*10, CV_32F);
        cv::resize(kernel, Lkernel, Lkernel.size());
    normalize(Lkernel, Lkernel,0.0,1.0,CV_MINMAX);
    Mat jet;
    normalize(Lkernel,jet,0,255,CV_MINMAX,CV_8UC1);
    applyColorMap(jet, jet, COLORMAP_JET);
    
    cv::imshow("Kernel", jet);

    cv::Mat mag;
    cv::pow(dest, 2.0, mag);
    cv::imshow("Mag", mag);
}

void visualizeFilter(Mat src){
    src.convertTo(src_f,CV_32F,1.0,0.0);
    if (!kernel_size%2)
    {
        kernel_size+=1;
    }
    cv::namedWindow("Process window", 1);
    cv::createTrackbar("Sigma", "Process window", &pos_sigma, kernel_size, Process);
    cv::createTrackbar("Lambda", "Process window", &pos_lm, 800, Process);//the lambda passed to getGaborKernel should be at least two, smaller values -> numerical artifacts!
    cv::createTrackbar("Theta", "Process window", &pos_th, 360, Process);
    cv::createTrackbar("Psi", "Process window", &pos_psi, 360, Process);
    Process(0,0);
    cv::waitKey(0);
    
}
void initFeatureVectors(vector<Mat> &fv,Size const &img_size,int n){
    //for every pixel, create a nx1 vector with n being the number of gabor kernels
    int nvecs = img_size.area();
    Mat initVec=Mat::zeros(n, 1, CV_32F);
    for(int i = 0;i<nvecs;i++){
        fv.push_back(initVec.clone());
    }
}
void fillFeatureVectors(vector<Mat> &fv,Mat res,int n_currentFeature){
    //De
    for (int r = 0; r<res.rows; r++) {
        for(int c = 0; c<res.cols;c++){
            fv[(r+1)*c].at<float>(n_currentFeature)=res.at<float>(r,c);
        }
    }
    
}

vector<Mat> generateFilters(){
    //generate a filter bank
    vector<Mat> kernels(27);
    
    double thetas[] = {0.0,CV_PI/4.0,CV_PI/2.0};
    double sigmas[] = {kernel_size/5.0,kernel_size/3.0,kernel_size/1.0};
    double lambdas[] = {4.0,6.0,8.0};
    int i = 0;
    for (double &th: thetas){
        for( double &sig: sigmas){
            for(double &lam: lambdas){
                kernels[i++]=mkKernel(kernel_size, sig, th, lam, 0);
            }
        }
    }
    
    return kernels;
}

Mat getDistanceMap(vector<Mat> & featureVectors){
    Mat map=Mat::zeros(srcImageBW.size(),CV_32F);
    Mat reference = featureVectors[(referencePixel.x+1)*referencePixel.y];
    
    for(int i = 0; i<featureVectors.size();i++){
        Mat diff = reference-featureVectors[i];
        
        double sum = diff.dot(diff);
        sum = sqrt(sum);
        map.at<float>(Point(i/map.rows,i%map.rows))=sum;
    }
    return map;
}

void displayDistances(vector<Mat> & featureVectors){
    Mat map = getDistanceMap(featureVectors);
    normalize(map,map,0.0,1.0,CV_MINMAX);
    imshow("Distance",map);
    
}

void A2a(Mat &src){
    //generate 20 gabor kernels
    vector<Mat> kernels = generateFilters();
    
   
    initFeatureVectors(featureVectors,src.size(),kernels.size());
    Mat res;
    Mat fvec;
    Mat disp;
    for(int i = 0;i<kernels.size();i++){
        
        filter2D(src, res, CV_32F, kernels[i],Point(-1,-1),BORDER_CONSTANT);
      //  normalize(res, disp,1.0,0.0,CV_MINMAX);
       // cv::imshow("Process window", disp);
        //waitKey();
        
        fillFeatureVectors(featureVectors,res,i);
        
    }
    displayDistances(featureVectors);
    waitKey();
}

void onMouse(int event, int x, int y, int,void*){
    if(event!=CV_EVENT_LBUTTONDOWN){
        return;
    }
    referencePixel = Point(x,y);
    displayDistances(featureVectors);
}





int main(int argc, char** argv)
{
    cv::Mat image = cv::imread(argv[1],1);
 
    cv::imshow("Src", image);
    
    cv::cvtColor(image, srcImageBW, CV_BGR2GRAY);
    setMouseCallback("Src", onMouse);

    //visualizeFilter(srcImageBW);
    
    A2a(srcImageBW);

    
    return 0;
}




