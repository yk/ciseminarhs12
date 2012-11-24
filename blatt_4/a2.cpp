#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void showandsave(string name, Mat img){
	imwrite("data/res/"+name + ".png",img);
	imshow(name,img);
	waitKey();
}

void a1(){
	Mat element = getStructuringElement(MORPH_RECT,Size(3,3));
	Mat img = imread("data/africa.png",CV_LOAD_IMAGE_GRAYSCALE);
	imshow("original",img);
	waitKey();
	Mat eroded;
	erode(img,eroded,element);
	showandsave("eroded",eroded);
	Mat dilated;
	dilate(img,dilated,element);
	dilate(dilated,dilated,element);
	dilate(dilated,dilated,element);
	showandsave("dilated",dilated);
	Mat outline;
	dilate(img,outline,element);
	dilate(outline,outline,element);
	dilate(outline,outline,element);
	erode(outline,outline,element);
	erode(outline,outline,element);
	erode(outline,outline,element);
	Mat outline_er;
	erode(outline,outline_er,element);
	showandsave("border",outline-outline_er);
}

void b1(){
	Mat dotsandlines = imread("data/dotsandlines.png",CV_LOAD_IMAGE_GRAYSCALE);
	showandsave("original",dotsandlines);
	Mat disk = getStructuringElement(MORPH_ELLIPSE,Size(9,9));
	Mat dots;
	morphologyEx(dotsandlines,dots,MORPH_OPEN,disk);
	showandsave("dots",dots);
	Mat cells = imread("data/cells.png",CV_LOAD_IMAGE_GRAYSCALE);
	showandsave("cells",cells);
	threshold(cells,cells,210,255,THRESH_BINARY_INV);
	showandsave("cells_thresh",cells);
	Mat dots1;
	disk = getStructuringElement(MORPH_ELLIPSE,Size(7,7));
	morphologyEx(cells,dots1,MORPH_OPEN,disk);
	showandsave("dots1",dots1);
	Mat circles = imread("data/circles.png",CV_LOAD_IMAGE_GRAYSCALE);
	showandsave("circles",circles);
	Mat circles_closed;
	disk = getStructuringElement(MORPH_ELLIPSE,Size(9,9));
	morphologyEx(circles,circles_closed,MORPH_CLOSE,disk);
	showandsave("circles_closed9",circles_closed);
	disk = getStructuringElement(MORPH_ELLIPSE,Size(17,17));
	morphologyEx(circles,circles_closed,MORPH_CLOSE,disk);
	showandsave("circles_closed17",circles_closed);
	disk = getStructuringElement(MORPH_ELLIPSE,Size(33,33));
	morphologyEx(circles,circles_closed,MORPH_CLOSE,disk);
	showandsave("circles_closed33",circles_closed);

}

void c1(){
	Mat house = imread("data/house.png",CV_LOAD_IMAGE_GRAYSCALE);
	showandsave("house",house);
	Mat rect = getStructuringElement(MORPH_CROSS,Size(3,3));
	Mat grad;
	morphologyEx(house,grad,MORPH_GRADIENT,rect);
	threshold(grad,grad,60,255,CV_THRESH_BINARY);
	showandsave("grad",grad);
	Mat canny;
	Canny(house,canny,210,150);
	showandsave("canny",canny);
	morphologyEx(house,grad,MORPH_GRADIENT,rect);
	Mat rh = getStructuringElement(MORPH_RECT,Size(1,3));
	Mat rv = getStructuringElement(MORPH_RECT,Size(3,1));
	Mat gh,gv;
	morphologyEx(house,gh,MORPH_GRADIENT,rh);
	morphologyEx(house,gv,MORPH_GRADIENT,rv);
	Mat corners = abs(gh) + abs(gv);
	threshold(corners,corners,190,510,CV_THRESH_BINARY);
	showandsave("corners",corners);
}





int main(){
	a1();
	b1();
	c1();
}
