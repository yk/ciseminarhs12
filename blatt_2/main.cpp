#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>
using namespace cv;
using namespace std;

Mat shift(Mat& magI){
    magI = magI(Rect(0, 0, magI.cols, magI.rows));
    int cx = magI.cols/2;
    int cy = magI.rows/2;
    Mat q0(magI, Rect(0, 0, cx, cy)); // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy)); // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy)); // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    return magI;
}


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
    magI=shift(magI);
    
    Mat phaI;
    phase(planes[0],planes[1],phaI);
    normalize(phaI,phaI,0,1,CV_MINMAX);
    phaI=shift(phaI);
    cv::imshow("Log Magnitude",magI);
    cv::imshow("Phase",phaI);
    cv::waitKey();
}

Mat spaceConv(Mat& h, Mat& v){
    Mat_<float> hf(h), vf(v);
    filter2D(hf,hf,-1,vf,Point(-1,-1),0,BORDER_CONSTANT);
    
    normalize(hf,hf,0,1,CV_MINMAX);
    return hf;
}

Mat freqConv(Mat& h, Mat& v){
    Mat_<float> hf(h), vf(v);
    dft(hf,hf);
    dft(vf,vf);
    mulSpectrums(hf,vf,hf,0);
    idft(hf,hf,DFT_SCALE);

    return hf;
}

Mat convolveDFT(Mat& A, Mat& B)
{
  
    Mat C(A.rows,A.cols,A.type());
    // reallocate the output array if needed
    
    Size dftSize;
    // calculate the size of DFT transform
    dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
    dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);
    
    // allocate temporary buffers and initialize them with 0's
    Mat tempA(dftSize, A.type(), Scalar::all(0));
    Mat tempB(dftSize, B.type(), Scalar::all(0));
    
    // copy A and B to the top-left corners of tempA and tempB, respectively
    Mat roiA(tempA, Rect(0,0,A.cols,A.rows));
    A.copyTo(roiA);
    Mat roiB(tempB, Rect(0,0,B.cols,B.rows));
    B.copyTo(roiB);
    
    // now transform the padded A & B in-place;
    // use "nonzeroRows" hint for faster processing
    dft(tempA, tempA, 0, A.rows);
    dft(tempB, tempB, 0, B.rows);
    
    // multiply the spectrums;
    // the function handles packed spectrum representations well
    mulSpectrums(tempA, tempB, tempA,0);
    
    // transform the product back from the frequency domain.
 
    //dft(tempA, tempA, DFT_INVERSE + DFT_SCALE);
    idft(tempA,tempA,DFT_SCALE);
    // now copy the result back to C.
    
    //int offsetR = 0;
    //int offsetC = 0;
    int offsetC = tempA.cols/4;
    int offsetR = tempA.rows/4;
    tempA(Rect(Point(offsetR, offsetC), Size(C.cols, C.rows))).copyTo(C);
    
    return C;
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
    //Mat r = freqConv(h,v);
    Mat h32,v32;
    h.convertTo(h32,CV_32FC1);
    
    v.convertTo(v32,CV_32FC1);
    
    //Mat r = freqConv(h,v);
    
    Mat r = convolveDFT(h32, v32);
    cout<<r.rows<<r.cols;
    normalize(r,r,0,1,CV_MINMAX);
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

Mat generateLowPass(Size_<int>& s, int cutoff ){
    Mat filter(s,CV_32FC1,Scalar::all(0));
    int c_0=s.width/2;
    int r_0 = s.height/2;
    cout<<c_0;
    for (int r = 0; r<s.height;r++){
        for (int c = 0; c<s.width; c++) {
            if(sqrt((r-r_0)^2+(c-c_0)^2)<cutoff){
                filter.at<float>(r,c)=1;
            }else{
                filter.at<float>(r,c)=0;
            }
            
        }
    }
    
    return filter;
}

void a1e(){
    Mat img = imread("../data/striped-lines.png",CV_LOAD_IMAGE_GRAYSCALE);
    Mat imgf;
    img.convertTo(imgf,CV_32F);
    dft(imgf,imgf);
    
    //create filter
    Size s(imgf.cols,imgf.rows);
    Mat f = generateLowPass(s, 13);
    idft(imgf,imgf,DFT_SCALE);
    imshow("filter",f);
    
    imshow("Result1",imgf);//strange
    waitKey();
    
}


int main ( int argc, char *argv[] ) {
    //chdir("/Users/alexattinger_3/Dropbox/5_sem/ki_seminar/2/blatt2");
    a1e();
    return 0;
}

