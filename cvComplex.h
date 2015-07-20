//cvComplex.h

#include <time.h>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <dirent.h>
#include <fstream>
#include <vector>

#if !defined(CVCOMPLEX_CONSTANTS_H)
#define CVCOMPLEX_CONSTANTS_H 1

static const int16_t SHOW_COMPLEX_MAG = 0;
static const int16_t SHOW_COMPLEX_COMPONENTS = 1; 
static const int16_t SHOW_COMPLEX_REAL = 2; 
static const int16_t SHOW_COMPLEX_IMAGINARY = 3; 

void circularShift(cv::Mat img, cv::Mat result, int x, int y);
void maxComplexReal(cv::Mat& m, std::string label);
void complexConj(const cv::Mat& m, cv::Mat& output);
void complexAbs(const cv::Mat& m, cv::Mat& output);
void complexMultiply(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& output);
void complexScalarMultiply(double scaler, cv::Mat& m, cv::Mat output);
void complexDivide(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& output);
void complexInverse(const cv::Mat& m, cv::Mat& inverse);
cv::Mat fftShift(cv::Mat m);
cv::Mat ifftShift(cv::Mat m);
void fft2(cv::Mat& input, cv::Mat& output);
void ifft2(cv::Mat& input, cv::Mat& output);
void complex_imwrite(std::string fname, cv::Mat& m1);
void onMouse( int event, int x, int y, int, void* param);
void showImgMag(cv::Mat m, std::string windowTitle);
void showImg(cv::Mat m, std::string windowTitle);
void showImgObject(cv::Mat m, std::string windowTitle);
void showImgFourier(cv::Mat m, std::string windowTitle);
void showComplexImg(cv::Mat m, int16_t displayFlag, std::string windowTitle);

#endif
