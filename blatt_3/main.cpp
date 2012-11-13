#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void canny(string& name,Mat& img){
	Mat edges;
	Canny(img,edges,100,200);
	imshow("Candy: " + name,edges);
	waitKey();
}

void sobel(string& name, Mat& img){
	Mat edges;
	Sobel(img,edges,img.depth(),1,1,3);
	imshow("Sobel: " + name,edges);
	waitKey();
}

void sobelSmooth(string& name, Mat& img){
	Mat blurred;
	Size ksize(0,0);
	GaussianBlur(img,blurred,ksize,1.0);
	sobel(name,blurred);
}

void main1(){
	vector<string> v {"butterfly.jpg","cells.jpg","outdoor.jpg","stairs.png","wheel.png"};
	string path = "data/edge/";
	for(string& s: v){
		string fileName = path + s;
		Mat img = imread(fileName,CV_LOAD_IMAGE_GRAYSCALE);
		//canny(s,img);
		//sobel(s,img);
		sobelSmooth(s,img);
	}
}

void hough(string& name, Mat& img){
	vector<Vec2f> lines;
	HoughLines(img,lines,5,0.1,10);
	//TODO
}

void houghP(string& name, Mat& img){
	vector<Vec2f> lines;
	HoughLinesP(img,lines,5,0.1,10);
	//TODO
}



void main2(){
	vector<string> v {"corridor.jpg","outdoor.jpg","room_1.jpg","room_2.jpg","stairs.png"};
	string path = "data/hough/";
	for(string& s: v){
		string fileName = path + s;
		Mat img = imread(fileName,CV_LOAD_IMAGE_GRAYSCALE);
		hough(s,img); //FIXME: says that img must have 1, 3 or 4 chanenls, but has 1 channel ???
	}
}

int main(){
	main2();
}
