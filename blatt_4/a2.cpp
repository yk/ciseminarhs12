#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void a1(){
	Mat element = getStructuringElement(MORPH_RECT,Size(3,3));
	Mat img = imread("data/africa.png",CV_LOAD_IMAGE_GRAYSCALE);
	imshow("original",img);
	waitKey();
	Mat eroded;
	erode(img,eroded,element);
	imshow("eroded",eroded);
	waitKey();
	Mat dilated;
	dilate(img,dilated,element);
	dilate(dilated,dilated,element);
	dilate(dilated,dilated,element);
	imshow("dilated",dilated);
	waitKey();
	Mat outline;
	dilate(img,outline,element);
	dilate(outline,outline,element);
	dilate(outline,outline,element);
	erode(outline,outline,element);
	erode(outline,outline,element);
	erode(outline,outline,element);
	Mat outline_er;
	erode(outline,outline_er,element);
	imshow("border",outline-outline_er);
	waitKey();
}

void b1(){
	Mat dotsandlines = imread("data/dotsandlines.png",CV_LOAD_IMAGE_GRAYSCALE);
	imshow("original",dotsandlines);
	waitKey();
	Mat disk = getStructuringElement(MORPH_ELLIPSE,Size(9,9));
	Mat dots;
	morphologyEx(dotsandlines,dots,MORPH_OPEN,disk);
	imshow("dots",dots);
	waitKey();
	Mat cells = imread("data/cells.png",CV_LOAD_IMAGE_GRAYSCALE);
	imshow("cells",cells);
	waitKey();
	threshold(cells,cells,210,255,THRESH_BINARY_INV);
	imshow("cells_thresh",cells);
	waitKey();
	Mat dots1;
	morphologyEx(cells,dots1,MORPH_OPEN,disk);
	imshow("dots1",dots1);
	waitKey();
	Mat circles = imread("data/circles.png",CV_LOAD_IMAGE_GRAYSCALE);
	imshow("circles",circles);
	waitKey();
	Mat circles_closed;
	morphologyEx(circles,circles_closed,MORPH_CLOSE,disk);
	imshow("circles_closed",circles_closed);
	waitKey();

}

void c1(){
	Mat house = imread("data/house.png",CV_LOAD_IMAGE_GRAYSCALE);
	imshow("house",house);
	waitKey();
	Mat rect = getStructuringElement(MORPH_RECT,Size(3,3));
	Mat grad;
	morphologyEx(house,grad,MORPH_GRADIENT,rect);
	imshow("grad",grad);
	waitKey();
	Mat canny;
	Canny(house,canny,100,50);
	imshow("canny",canny);
	waitKey();
}





int main(){
//	a1();
//	b1();
	c1();
}
