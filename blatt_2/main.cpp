#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>
using namespace cv;
using namespace std;


void a1a(){
    cv::Mat img = cv::imread("../data/striped-lines.png",CV_LOAD_IMAGE_GRAYSCALE);
    Mat planes[] = {Mat_<float>(img),Mat::zeros(img.size(),CV_32F)};
    Mat complexI;
    merge(planes,2,complexI);
    dft(complexI,complexI);
    split(complexI,planes);
    Mat magI;
    magnitude(planes[0],planes[1],magI);
    log(magI,magI);
    normalize(magI,magI,0,1,CV_MINMAX);
    Mat phaI;
    phase(planes[0],planes[1],phaI);
    normalize(phaI,phaI,0,1,CV_MINMAX);
    cv::imshow("Log Magnitude",magI);
    cv::imshow("Phase",phaI);
    cv::waitKey();
}

Mat spaceConv(Mat& h, Mat& v){
    Mat_<float> hf(h), vf(v);
    filter2D(hf,hf,-1,vf);
    Mat r(hf);
    return r;
}

Mat freqConv(Mat& h, Mat& v){
    Mat_<float> hf(h), vf(v);
    dft(hf,hf);
    dft(vf,vf);
    mulSpectrums(hf,vf,hf,0);
    idft(hf,hf);
    Mat r(hf);
    return r;
}

void a1b(){
    Mat h = imread("../data/h.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat v = imread("../data/v.png",CV_LOAD_IMAGE_GRAYSCALE);
    Mat r = spaceConv(h,v);
    imshow("Result",r);
    waitKey();
}

void a1c(){
    Mat h = imread("../data/h.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat v = imread("../data/v.png",CV_LOAD_IMAGE_GRAYSCALE);
    Mat r = freqConv(h,v);
    imshow("Result",r);
    waitKey();
}

void a1d(){
    Mat h = imread("../data/h.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat h1k = imread("../data/h-1000.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat h10k = imread("../data/h-10000.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat v = imread("../data/v.png",CV_LOAD_IMAGE_GRAYSCALE);
    Mat v1k = imread("../data/v-1000.png",CV_LOAD_IMAGE_GRAYSCALE);
    Mat v10k = imread("../data/v-10000.png",CV_LOAD_IMAGE_GRAYSCALE);
    cout << "spacial convolution" << endl;
    clock_t begin = clock();
    Mat rs = spaceConv(h,v);
    cout << double(clock()-begin)/CLOCKS_PER_SEC << endl;
    begin = clock();
    Mat rs1k = spaceConv(h1k,v1k);
    cout << double(clock()-begin)/CLOCKS_PER_SEC << endl;
    begin = clock();
    Mat rs10k = spaceConv(h10k,v10k);
    cout << double(clock()-begin)/CLOCKS_PER_SEC << endl;
    cout << "frequency convolution" << endl;
    begin = clock();
    Mat rf = freqConv(h,v);
    cout << double(clock()-begin)/CLOCKS_PER_SEC << endl;
    begin = clock();
    Mat rf1k = freqConv(h1k,v1k);
    cout << double(clock()-begin)/CLOCKS_PER_SEC << endl;
    begin = clock();
    Mat rf10k = freqConv(h10k,v10k);
    cout << double(clock()-begin)/CLOCKS_PER_SEC << endl;
}

void a1e(){
    Mat img = imread("../data/striped-lines.png",CV_LOAD_IMAGE_GRAYSCALE);
    Mat imgf;
    img.convertTo(imgf,CV_32F);
    dft(imgf,imgf);

    idft(imgf,imgf);
    imshow("Result1",imgf);//strange
    waitKey();

}


int main ( int argc, char *argv[] ) {
    a1e();
    return 0;
}
