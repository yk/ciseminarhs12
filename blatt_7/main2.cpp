//
//  main.cpp
//  fundamentalMatrix
//
//  Created by Alex Attinger on 10.12.12.
//  Copyright (c) 2012 Alex Attinger. All rights reserved.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/impl/point_types.hpp>
#include "boost/thread.hpp"
#include <math.h>

using namespace cv;
using namespace std;
using namespace pcl;

std::vector<Point2f> matchedPoints_1, matchedPoints_2;
std::vector<KeyPoint> matchedKeyPoints_1, matchedKeyPoints_2;
Mat view1, view2, outliers;

void showandsave(string name, Mat img) {
	//imwrite("data/res/" + name + ".png", img);
	imshow(name, img);
	waitKey();
}

Mat getHomogeneousMat(const Point2f &x) {
	double pointData[] = { x.x, x.y, 1.0 };
	Mat point = Mat(3, 1, CV_64FC1, pointData).clone();
	return point;

}

Mat calculateFundamental(vector<Point2f> matchedPoints_1,
		vector<Point2f> matchedPoints_2) {
	Mat fundamentalRan;

	fundamentalRan = findFundamentalMat(matchedPoints_1, matchedPoints_2,
			CV_FM_RANSAC, 1., .99, outliers);

	return fundamentalRan;
}

pair<pair<vector<KeyPoint>, vector<KeyPoint> >,
		pair<vector<Point2f>, vector<Point2f> > > match(Mat view1, Mat view2) {

	equalizeHist(view1, view1);
	equalizeHist(view2, view2);

	std::vector<KeyPoint> keypoints_1, keypoints_2;
	SurfFeatureDetector detector(900);

	detector.detect(view1, keypoints_1);
	detector.detect(view2, keypoints_2);

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;
	//	SiftDescriptorExtractor extractor;

	Mat descriptors_1, descriptors_2;

	extractor.compute(view1, keypoints_1, descriptors_1);
	extractor.compute(view2, keypoints_2, descriptors_2);

	//-- Step 3: Matching descriptor vectors with a brute force matcher
	BruteForceMatcher < L2<float> > matcher;
	std::vector < DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	float thresh = 0.24;

	auto newend =
			remove_if(matches.begin(), matches.end(),
					[&](DMatch& match)->bool {/*cout<<match.distance<<endl;*/return (abs(match.distance) > thresh);});
	auto good_matches = vector < DMatch > (matches.begin(), newend);
	//-- Draw matches
	Mat img_matches;
	drawMatches(view1, keypoints_1, view2, keypoints_2, good_matches,
			img_matches);
	for (int i = 0; i < good_matches.size(); i++) {
		matchedKeyPoints_1.push_back(keypoints_1[good_matches[i].queryIdx]);
		matchedKeyPoints_2.push_back(keypoints_2[good_matches[i].trainIdx]);
		matchedPoints_1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
		matchedPoints_2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
	}

	return make_pair(make_pair(matchedKeyPoints_1, matchedKeyPoints_2),
			make_pair(matchedPoints_1, matchedPoints_2));

	//-- Show detected matches
//	showandsave("Matches_surf_02", img_matches);
}

int minDisp = 64;
int numDisp = 128;
int SADSize = 3;
int P1 = 5400;
int P2 = 21600;
string winname = "Disparity Map";
Mat dest1, dest2;
Mat dispMap;

void processInput(int, void*) {
	int _minDisp = -minDisp;
	int _numDisp = numDisp - numDisp % 16;
	int _SADsize = SADSize;
	StereoSGBM bm(_minDisp, _numDisp, _SADsize, P1, P2);
	bm(dest1, dest2, dispMap);
	normalize(dispMap, dispMap, 0, 256, CV_MINMAX, NULL);
	//applyColorMap(dispMap, dispMap, cv::COLORMAP_JET);
	//Mat disp_vis = Mat(dispMap.size(), CV_8U);
	//convertScaleAbs(dispMap, disp_vis);
	imshow(winname, dispMap);

}

void setUpTrackbars() {
	namedWindow(winname, 1);
	createTrackbar("minimum", winname, &minDisp, 128, processInput);
	createTrackbar("num Disp", winname, &numDisp, 500, processInput);
	createTrackbar("Patch size", winname, &SADSize, 21, processInput);
	createTrackbar("P1", winname, &P1, 7000, processInput);
	createTrackbar("p2", winname, &P2, 30000, processInput);
}

void show(Mat depth, Mat rectifiedImage) {
	int npixels = depth.rows * depth.cols;
	PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
	cloud->width = npixels;
	cloud->height = 1;
	cloud->is_dense = false;
	cloud->points.resize(cloud->width * cloud->height);

	for (size_t m = 0; m < rectifiedImage.rows; ++m) {
		for (size_t n = 0; n < rectifiedImage.cols; ++n) {
			cloud->points[m * depth.cols + n].x = n;
			cloud->points[m * depth.cols + n].y = m;
//	      cloud->points[m*depth.cols+n].z = (float)depth.at<short>(m,n);
			cloud->points[m * depth.cols + n].z = ((float) depth.at<short>(m, n))/255.0;

//	      cv::Vec3b col = rectifiedImage.at<cv::Vec3b>(m,n);
//	      // pack r/g/b into rgb
//	      uint8_t r = col[2], g = col[1], b = col[0];    // Example: Red color
//	      uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
//	      cloud->points[m*depth.cols+n].rgb = *reinterpret_cast<float*>(&rgb);
			//FIXME rectifiedImage is grayscale, not 3-channel
			uint8_t r = (float) rectifiedImage.at<short>(m, n), g =
					(float) rectifiedImage.at<short>(m, n), b =
					(float) rectifiedImage.at<short>(m, n); // Example: Red color
			uint32_t rgb = ((uint32_t) r << 16 | (uint32_t) g << 8
					| (uint32_t) b);
			cloud->points[m * depth.cols + n].rgb =
					*reinterpret_cast<float*>(&rgb);
		}
	}

	boost::shared_ptr<visualization::PCLVisualizer> viewer(
			new visualization::PCLVisualizer("3D Viewer"));
	visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb(cloud);
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPointCloud < PointXYZRGB > (cloud, rgb, "sample cloud");
	viewer->setPointCloudRenderingProperties(
			visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	while (!viewer->wasStopped()) {
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

int main3() {
	//EX 1
	view1 = imread("data/johnHunter/001.png", CV_LOAD_IMAGE_GRAYSCALE);
	view2 = imread("data/johnHunter/002.png", CV_LOAD_IMAGE_GRAYSCALE);
//	view1 = imread("data/table/input1.png", CV_LOAD_IMAGE_GRAYSCALE);
//	view2 = imread("data/table/input2.png", CV_LOAD_IMAGE_GRAYSCALE);
	showandsave("johnHunter1", view1);
	auto ps = match(view1, view2);
	auto mkps = ps.first;
	matchedKeyPoints_1 = mkps.first;
	matchedKeyPoints_2 = mkps.second;
	auto mps = ps.second;
	matchedPoints_1 = mps.first;
	matchedPoints_2 = mps.second;
//    for(int i = 0; i<matchedPoints_1.size();i++){
//        cout<<matchedPoints_1[i].y-matchedPoints_2[i].y<<endl;
//    }
	Mat f = calculateFundamental(matchedPoints_1, matchedPoints_2);
	Mat h1, h2;

	//rectify images
	stereoRectifyUncalibrated(matchedPoints_1, matchedPoints_2, f, view1.size(),
			h1, h2);
	//apply the transformation
	warpPerspective(view1, dest1, h1, view1.size());
	warpPerspective(view2, dest2, h2, view1.size());
	//show images
//    imshow("Rectified 1",dest1);
//    imshow("Rectified 2",dest2);
	//do the matching
	setUpTrackbars();
	processInput(0, 0);

//	PointCloud<point> pc;

	waitKey();
	showandsave("dest1", dest1);
	normalize(dispMap, dispMap, 0, 255, NORM_MINMAX, CV_8UC1);
	show(dispMap, dest1);
	return 0;
}

int main4() {
	Mat car1 = imread("data/car/input1.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat car2 = imread("data/car/input2.png", CV_LOAD_IMAGE_GRAYSCALE);
//	auto ps = match(car1, car2);
//	auto mkps =  ps.first;
//	auto matchedKeyPoints_1 = mkps.first;
//	auto matchedKeyPoints_2 = mkps.second;
//	auto mps = ps.second;
//	auto matchedPoints_1 = mps.first;
//	auto matchedPoints_2 = mps.second;
//	Mat prevPts = Mat::zeros(car1.rows, car1.cols, CV_32FC2);
	vector < Point2f > prevPts;
	for (int i = 0; i < car1.rows; i++) {
		for (int j = 0; j < car1.cols; j++) {
//			prevPts.at < Point2f > (i, j) = Point2f(i, j);
			prevPts.push_back(Point2f(j, i));
		}
	}
	vector < Point2f > nextPts;
	Mat status, err;
	calcOpticalFlowPyrLK(car1, car2, prevPts, nextPts, status, err);
//	Mat out = Mat::zeros(car1.rows, car1.cols, CV_32FC2);
//	for (int i = 0; i < car1.rows; i++) {
//		for (int j = 0; j < car1.cols; j++) {
//			out.at < Point2f > (i, j) = nextPts.at < Point2f > (i, j);
//		}
//	}
	Mat out = Mat::zeros(car1.rows, car1.cols, CV_32F);
	for (int i = 0; i < car1.rows; i++) {
		for (int j = 0; j < car1.cols; j++) {
			Point2f p = nextPts[i * car1.cols + j];
			out.at<float>(i, j) = (atan2(p.y, p.x) + M_PI) / (2 * M_PI);
		}
	}
	//now apply color map
	Mat out_cm;
	normalize(out, out_cm, 0, 255, NORM_MINMAX, CV_8UC1);
	showandsave("optical_flow_direction", out_cm);
	for (int i = 0; i < car1.rows; i++) {
		for (int j = 0; j < car1.cols; j++) {
			Point2f p = nextPts[i * car1.cols + j];
			out.at<float>(i, j) = sqrt(p.x * p.x + p.y * p.y);
		}
	}
	//now apply color map
	normalize(out, out_cm, 0, 255, NORM_MINMAX, CV_8UC1);
	showandsave("optical_flow_norm", out_cm);
	return 0;
}

int main() {
	main3();
//	main4();
	return 0;
}
