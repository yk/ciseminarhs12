#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

using namespace std;
using namespace cv;

void showandsave(string name, Mat img) {
	imwrite("data/res/" + name + ".png", img);
	imshow(name, img);
	waitKey();
}

void a1() {
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat img = imread("data/africa.png", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("original", img);
	waitKey();
	Mat eroded;
	erode(img, eroded, element);
	showandsave("eroded", eroded);
	Mat dilated;
	dilate(img, dilated, element);
	dilate(dilated, dilated, element);
	dilate(dilated, dilated, element);
	showandsave("dilated", dilated);
	Mat outline;
	dilate(img, outline, element);
	dilate(outline, outline, element);
	dilate(outline, outline, element);
	erode(outline, outline, element);
	erode(outline, outline, element);
	erode(outline, outline, element);
	Mat outline_er;
	erode(outline, outline_er, element);
	showandsave("border", outline - outline_er);
}

void b1() {
	Mat dotsandlines = imread("data/dotsandlines.png", CV_LOAD_IMAGE_GRAYSCALE);
	showandsave("original", dotsandlines);
	Mat disk = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
	Mat dots;
	morphologyEx(dotsandlines, dots, MORPH_OPEN, disk);
	showandsave("dots", dots);
	Mat cells = imread("data/cells.png", CV_LOAD_IMAGE_GRAYSCALE);
	showandsave("cells", cells);
	threshold(cells, cells, 210, 255, THRESH_BINARY_INV);
	showandsave("cells_thresh", cells);
	Mat dots1;
	disk = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
	morphologyEx(cells, dots1, MORPH_OPEN, disk);
	showandsave("dots1", dots1);
	Mat circles = imread("data/circles.png", CV_LOAD_IMAGE_GRAYSCALE);
	showandsave("circles", circles);
	Mat circles_closed;
	disk = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
	morphologyEx(circles, circles_closed, MORPH_CLOSE, disk);
	showandsave("circles_closed9", circles_closed);
	disk = getStructuringElement(MORPH_ELLIPSE, Size(17, 17));
	morphologyEx(circles, circles_closed, MORPH_CLOSE, disk);
	showandsave("circles_closed17", circles_closed);
	disk = getStructuringElement(MORPH_ELLIPSE, Size(33, 33));
	morphologyEx(circles, circles_closed, MORPH_CLOSE, disk);
	showandsave("circles_closed33", circles_closed);

}

void c1() {
	Mat house = imread("data/house.png", CV_LOAD_IMAGE_GRAYSCALE);
	showandsave("house", house);
	Mat rect = getStructuringElement(MORPH_CROSS, Size(3, 3));
	Mat grad;
	morphologyEx(house, grad, MORPH_GRADIENT, rect);
	threshold(grad, grad, 60, 255, CV_THRESH_BINARY);
	showandsave("grad", grad);
	Mat canny;
	Canny(house, canny, 210, 150);
	showandsave("canny", canny);
	morphologyEx(house, grad, MORPH_GRADIENT, rect);
	Mat rh = getStructuringElement(MORPH_RECT, Size(1, 3));
	Mat rv = getStructuringElement(MORPH_RECT, Size(3, 1));
	Mat gh, gv;
	morphologyEx(house, gh, MORPH_GRADIENT, rh);
	morphologyEx(house, gv, MORPH_GRADIENT, rv);
	Mat corners = abs(gh) + abs(gv);
	threshold(corners, corners, 190, 510, CV_THRESH_BINARY);
	showandsave("corners", corners);
}

void watershit() {
	int diskSize = 50;
	Mat img = imread("data/apples.png", CV_LOAD_IMAGE_GRAYSCALE);
	showandsave("apples", img);
	Mat gaussed;
	GaussianBlur(img, gaussed, Size2i(5, 5), 2);
	Mat histEqGau;
	equalizeHist(gaussed, histEqGau);
	Mat disk20 = getStructuringElement(MORPH_ELLIPSE, Size(diskSize, diskSize));
	Mat disk10 = getStructuringElement(MORPH_ELLIPSE,
			Size(diskSize / 2, diskSize / 2));
	Mat tmp;
	morphologyEx(gaussed, tmp, MORPH_OPEN, disk20);
	morphologyEx(tmp, tmp, MORPH_CLOSE, disk20);
	tmp += histEqGau;
	normalize(tmp, tmp, 0.0, 1.0);
	threshold(tmp, tmp, 0.5, 1.0, THRESH_BINARY);
	morphologyEx(gaussed, tmp, MORPH_ERODE, disk10);
	morphologyEx(gaussed, tmp, MORPH_OPEN, disk20);
	Mat fgLabelMap;
	tmp.copyTo(fgLabelMap);
	double fgThresh = 160;
	threshold(fgLabelMap, fgLabelMap, fgThresh, 255.0, THRESH_BINARY);
	double bgThresh = 60;
	Mat bgLabelMap;
	threshold(gaussed, bgLabelMap, bgThresh, 155.0, THRESH_BINARY_INV);
	tmp = fgLabelMap + bgLabelMap;
	showandsave("label_map_" + to_string(diskSize), tmp);
	Mat markers;
	tmp.convertTo(markers, CV_32S);
	watershed(imread("data/apples.png", CV_LOAD_IMAGE_COLOR), markers);
	Mat ms;
	markers.convertTo(ms, CV_8U);
	showandsave("markers_after_" + to_string(diskSize), ms);
}

#define use_toucan 0

void ml() {
#if use_toucan
Point toucP1(455, 185); //toucan
Point toucP2(810, 690);
string imname = "toucan";
float thresh = 1.0;
#else
	Point toucP1(90, 70); //kitty
	Point toucP2(240, 450);
	string imname = "grabcut-dataset/388016";
	float thresh = 5.0;
#endif
	Mat img = imread("data/" + imname + ".jpg", CV_LOAD_IMAGE_COLOR);
	blur(img,img,Size(25,25));
//	blur(img,img,Size(20,20));
//	blur(img,img,Size(20,20));
//	blur(img,img,Size(20,20));
//	blur(img,img,Size(20,20));
	Mat tmp;
	img.copyTo(tmp);
	rectangle(tmp, toucP1, toucP2, Scalar(255, 255, 255));
	showandsave(imname + "_bounding", tmp);
	vector<Mat> fgp, bgp;
	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {
			Vec3b vpix = img.at<Vec3b>(j, i);
			Mat pix(3, 1, CV_64F);
			pix.at<double>(0, 0) = (double)vpix[0];
			pix.at<double>(1, 0) = (double)vpix[1];
			pix.at<double>(2, 0) = (double)vpix[2];
			if (i < toucP1.x || i > toucP2.x || j < toucP1.y || j > toucP2.y) {
				bgp.push_back(pix);
			} else {
				fgp.push_back(pix);
			}
		}
	}
	Mat meanfg, meanbg, covfg, covbg;
	calcCovarMatrix(&(fgp[0]), fgp.size(), covfg, meanfg, CV_COVAR_NORMAL | CV_COVAR_SCALE);
	calcCovarMatrix(&(bgp[0]), bgp.size(), covbg, meanbg, CV_COVAR_NORMAL | CV_COVAR_SCALE);
	Mat icfg, icbg;
	invert(covfg, icfg);
	invert(covbg, icbg);
	cout << "mufg" << endl;
	cout << meanfg << endl;
	cout << "covfg" << endl;
	cout << covfg << endl;
	cout << "mubg" << endl;
	cout << meanbg << endl;
	cout << "covbg" << endl;
	cout << covbg << endl;
	double d = 3;
	double prefg = 1.0 / (pow(2.0 * M_PI, d / 2.0) * sqrt(determinant(covfg)));
	double prebg = 1.0 / (pow(2.0 * M_PI, d / 2.0) * sqrt(determinant(covbg)));
	auto t = [&](Mat x)->double {
		Mat_<double> xf(x);
		Mat fgdiff = xf - meanfg;
		Mat tfgdiff;
		transpose(fgdiff,tfgdiff);
		Mat fgexp = tfgdiff*icfg*fgdiff;
		double f = prefg*exp(-0.5*fgexp.at<double>(0,0));
		Mat bgdiff = xf - meanbg;
		Mat tbgdiff;
		transpose(bgdiff,tbgdiff);
		Mat bgexp = tbgdiff*icbg*bgdiff;
		double b = prebg*exp(-0.5*bgexp.at<double>(0,0));
		return f/b;
	};



	Mat out = Mat::zeros(img.rows, img.cols, CV_8U);
	for (int x = 0; x < out.cols; x++) {
		for (int y = 0; y < out.rows; y++) {
			Vec3b vpix = img.at<Vec3b>(y, x);
			Mat pix(3, 1, CV_8U);
			pix.at<uchar>(0, 0) = vpix[0];
			pix.at<uchar>(1, 0) = vpix[1];
			pix.at<uchar>(2, 0) = vpix[2];
			float tp = t(pix);
			if (tp >= thresh) {
				out.at<uchar>(y, x) = 255;
			} else {
				out.at<uchar>(y, x) = 0;
			}
		}
	}
	showandsave(imname + "_output_" + to_string(thresh), out);

}

int main() {
//	a1();
//	b1();
//	c1();
//	watershit();
	ml();
}
