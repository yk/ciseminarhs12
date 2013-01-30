#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

void showandsave(string name, Mat img) {
	imwrite("data/res/" + name + ".png", img);
	imshow(name, img);
	waitKey();
}

void siftandsurf() {
	Mat edges = imread("data/library.jpg", CV_LOAD_IMAGE_GRAYSCALE);

//	SiftFeatureDetector detector(0.15,0.15);
	SurfFeatureDetector detector(4000);
	vector<KeyPoint> keypoints;
	detector.detect(edges, keypoints);
//	SiftDescriptorExtractor extractor;
	SurfDescriptorExtractor extractor;
	Mat descriptors;
	extractor.compute(edges, keypoints, descriptors);
	drawKeypoints(edges, keypoints, edges);
	showandsave("surfedges", edges);
}

vector<KeyPoint> getHarrisPoints(Mat& img, int thresh = 95) {
	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(img.size(), CV_32FC1);
//	int thresh = 95;

/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	/// Detecting corners
	cornerHarris(img, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	vector<KeyPoint> v;
	/// Drawing a circle around corners
	for (int j = 0; j < dst_norm.rows; j++) {
		for (int i = 0; i < dst_norm.cols; i++) {
			if ((int) dst_norm.at<float>(j, i) > thresh) {
				v.push_back(KeyPoint(i, j, 1));
			}
		}
	}
	cout << "got " << v.size() << " harris points" << endl;
	return v;
}

void harris() {
	Mat src_gray = imread("data/bookT.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src_gray.size(), CV_32FC1);
//	int thresh = 95; //libr
//	int thresh = 200; //r
//	int thresh = 190; //p
	int thresh = 130; //t

	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	/// Detecting corners
	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	/// Drawing a circle around corners
	for (int j = 0; j < dst_norm.rows; j++) {
		for (int i = 0; i < dst_norm.cols; i++) {
			if ((int) dst_norm.at<float>(j, i) > thresh) {
				circle(src_gray, Point(i, j), 5, Scalar(0), 2, 8, 0);
//				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
			}
		}
	}
	/// Showing the result
	showandsave("harris_bookt", src_gray);
}

pair<pair<vector<KeyPoint>, vector<KeyPoint> >, vector<DMatch>> match(Mat& img,
		Mat& book, string bookname) {

//	int minHessian = 400;

	SurfFeatureDetector detector(2500);
//	SiftFeatureDetector detector(0.14,0.14);

	std::vector<KeyPoint> keypoints_1, keypoints_2;

	detector.detect(img, keypoints_1);
	detector.detect(book, keypoints_2);

//	keypoints_1 = getHarrisPoints(img);
//	keypoints_2 = getHarrisPoints(book,130);

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;
//	SiftDescriptorExtractor extractor;

	Mat descriptors_1, descriptors_2;

	extractor.compute(img, keypoints_1, descriptors_1);
	extractor.compute(book, keypoints_2, descriptors_2);

//	cout << "d1 " << descriptors_1 << " d2 " << descriptors_2 << endl;

	//-- Step 3: Matching descriptor vectors with a brute force matcher
	BruteForceMatcher<L2<float> > matcher;
	std::vector<DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	float thresh = 0.15;

	auto newend =
			remove_if(matches.begin(), matches.end(),
					[&](DMatch& match)->bool {/*cout<<match.distance<<endl;*/return (abs(match.distance) > thresh);});
	auto good_matches = vector<DMatch>(matches.begin(), newend);
	//-- Draw matches
	Mat img_matches;
	drawMatches(img, keypoints_1, book, keypoints_2, good_matches, img_matches);

	//-- Show detected matches
	showandsave("matches_" + bookname, img_matches);
	return make_pair(make_pair(keypoints_1, keypoints_2), good_matches);
}

float distPointLine(float a, float b, float x, float y) {
	return abs(a * x - y + b) / sqrt(a * a + 1);
}

float ran(float min, float max) {
	float r3 = min + (float) rand() / ((float) RAND_MAX / (max - min));
	return r3;
}

void ransac() {
#include "data/data.hpp"
	float a, b;
	float mina = -1.0, maxa = 1.0, minb = 5.0, maxb = 10.0;
	int numCons;
	float boundary = 2.7;
	float consRatio = 0.7;
	do {
		a = ran(mina, maxa);
		b = ran(minb, maxb);
		numCons =
				count_if(data.begin(), data.end(),
						[&](Point2f& p)->bool {return (distPointLine(a,b,p.x,p.y)<boundary);});
		cout << "nc " << numCons << " ds " << data.size() << " r "
				<< numCons * 1.0 / data.size() << " cr " << consRatio << " b "
				<< (numCons * 1.0 / data.size() < consRatio) << endl;
	} while ((numCons * 1.0 / data.size()) < consRatio);
	cout << "a = " << a << ", b = " << b << endl;
}

void ransacmatch() {
	string booknames[] = {"book","bookP","bookR","bookT"};
	Mat img = imread("data/library.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	for(string& bookname : booknames){
	Mat book = imread("data/" + bookname + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
	auto kpms = match(img, book,bookname);
	auto matches = kpms.second;
	auto qdesc = kpms.first.first;
	auto tdesc = kpms.first.second;
	auto tries = 10000;
	Mat bestMat;
	double bestScore = numeric_limits<double>::max();
	for (int i = 0; i < tries; i++) {
		vector<DMatch> selected;
		Point2f src[3];
		Point2f dst[3];
		for (int j = 0; j < 3; j++) {
			auto s = matches[rand() % matches.size()];
//			if (find(selected.begin(), selected.end(), s) != selected.end()) {
//				cout << "duplicate" << endl;
//				j--;
//				continue;
//			}
			selected.push_back(s);
			src[j] = qdesc[s.queryIdx].pt;
			dst[j] = tdesc[s.trainIdx].pt;
		}
		//calc affine transformation
		Mat warp_mat = getAffineTransform(src, dst);
		double error = 0;
		for (auto it = matches.begin();it!=matches.end();it++) {
			auto match = *it;
			auto pointInOneImage = qdesc[match.queryIdx].pt;
			auto pointInOtherImage = tdesc[match.trainIdx].pt;
			cv::Mat point(3, 1, CV_64FC1, cv::Scalar(0));
			point.at<double>(0) = pointInOneImage.x;
			point.at<double>(1) = pointInOneImage.y;
			point.at<double>(2) = 1.f;

			cv::Mat calculatedPosition = warp_mat * point;

			cv::Mat truePosition(2, 1, CV_64FC1);
			truePosition.at<double>(0) = pointInOtherImage.x;
			truePosition.at<double>(1) = pointInOtherImage.y;
			error += cv::norm(truePosition, calculatedPosition);
		}
		if(error < bestScore){
			bestScore = error;
			bestMat = warp_mat;
		}
	}
	cout << bookname << ": " << bestScore << endl;
	Mat M;
	invertAffineTransform(bestMat,M);
	Mat wrpdst;
	warpAffine(book,wrpdst,M, img.size());
	Mat imcopy;
	img.copyTo(imcopy);
	imcopy -= wrpdst;
	showandsave("warped " + bookname, imcopy);
	}
}

vector<Mat> getGaussPyramid(Mat img, double minsigma, double maxsigma, int numsteps){
	vector<Mat> pyr;
	for(int step=0;step<numsteps;step++){
		double sigma = minsigma + step*maxsigma/(numsteps-1);
		Mat blrd;
		GaussianBlur(img,blrd,Size2f(0,0),sigma);
		pyr.push_back(blrd);
	}
	return pyr;
}

vector<Mat> getDiffPyramid(vector<Mat> pyramid){
	vector<Mat> pyr;
	for(int i=0;i<pyramid.size()-1;i++){
		pyr.push_back(pyramid[i+1] - pyramid[i]);
	}
	return pyr;
}

#define easysift_global_maxima 1

vector<Point> getExtrema(vector<Mat> diffPyr){
	vector<Point> v;
	for(auto& m : diffPyr){
#if easysift_global_maxima
		Point maxLoc;
		minMaxLoc(abs(m),NULL,NULL,NULL,&maxLoc);
		v.push_back(maxLoc);
#else
		Mat ma = abs(m);
		for(int i=0;i<m.cols;i++){
			for(int j=0;j<m.rows;j++){
				bool max = true;
				for(int x=-1;x<=1;x++){
					for(int y=-1;y<=1;y++){
						if(x != 0 && y != 0){
							if(ma.at<float>(i+x,j+y) > ma.at<float>(i,j)){
								max = false;
							}
						}
					}
				}
				if(max){
					Point2i p;
					p.x = i;
					p.y = j;
					v.push_back(p);
				}
			}
		}
#endif
	}
	return v;
}

void easySIFT(){
	Mat img = imread("data/library.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	auto gp = getGaussPyramid(img,0.01,10.0,100);
	auto dp = getDiffPyramid(gp);
	auto ex = getExtrema(dp);
	Mat out;
	vector<KeyPoint> keyPoints;
	for(auto& p : ex){
		KeyPoint kp;
		kp.pt = p;
		keyPoints.push_back(kp);
	}
	drawKeypoints(img,keyPoints,out);
	showandsave("easysift",out);
}

int main() {
//	siftandsurf();
//	harris();
//	match();
//	ransac();
//	ransacmatch();
	easySIFT();
}

