#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


void drawHough(string& name, vector<Vec2f>& lines, Mat& img) {
	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(img, pt1, pt2, 255, 1, CV_AA);
	}
	imshow(name, img);
	imwrite("data/hough/res/" + name, img);
	waitKey();
}

void hough(string& name, Mat& img, double theta = 0.75,
		double rho = CV_PI / 180, int numVotes = 100) {
	Mat cimg;
	Canny(img, cimg, 50, 200, 3);
	vector<Vec2f> lines;
	HoughLines(cimg, lines, theta, rho, numVotes);
	drawHough(name, lines, img);
}

void drawHoughP(string& name, vector<Vec4i>& lines, Mat& img) {
	for (size_t i = 0; i < lines.size(); i++) {
		Vec4i l = lines[i];
		line(img, Point(l[0], l[1]), Point(l[2], l[3]), 255, 1,
				CV_AA);
	}
	imshow(name+"P", img);
	imwrite("data/hough/res/P" + name, img);
	waitKey();
}

void houghP(string& name, Mat& img,double theta, double rho, int numVotes) {
	Mat cimg;
	Canny(img, cimg, 50, 200, 3);
	vector<Vec4i> lines;
	HoughLinesP(cimg, lines, theta,rho,numVotes);
	drawHoughP(name, lines, img);
}

void main2() {
	vector<string> v { "corridor.jpg", "outdoor.jpg", "room_1.jpg",
			"room_2.jpg", "stairs.png" };
	vector<double> theta { 1, 0.75, 0.75, 0.75, 1 };
	vector<double> rho { CV_PI / 180, 0.75*CV_PI / 180, 0.75*CV_PI / 180, 0.75*CV_PI / 180,
		CV_PI / 180 };
	vector<int> votes { 100, 140, 77, 105, 130 };
	string path = "data/hough/";
	for (int i = 0; i < 5; i++) {
		string s = v[i];
		string fileName = path + s;
		Mat img = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
		hough(s, img, theta[i], rho[i], votes[i]);
		houghP(s,img,theta[i],rho[i],votes[i]/3);
	}
}

int main() {
	main2();
}
