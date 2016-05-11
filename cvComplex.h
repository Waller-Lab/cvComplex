//cvComplex.h

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>

#if !defined(CVCOMPLEX_H_1)
#define CVCOMPLEX_H_1

// CONSTANT DEFINITIONS
extern const int16_t SHOW_COMPLEX_MAG;
extern const int16_t SHOW_COMPLEX_COMPONENTS;
extern const int16_t SHOW_COMPLEX_REAL;
extern const int16_t SHOW_COMPLEX_IMAGINARY;
extern const int16_t SHOW_AMP_PHASE;

extern const int16_t RW_MODE_AMP_PHASE;
extern const int16_t RW_MODE_REAL_IMAG;

extern const int16_t CMAP_MIN;
extern const int16_t CMAP_MAX;
extern const int16_t COLORMAP_NONE;

// METHODS
void circularShift(cv::Mat img, cv::Mat result, int x, int y);
void maxComplexReal(cv::Mat& m, std::string label);
void complexConj(const cv::Mat& m, cv::Mat& output);
void complexAbs(const cv::Mat& m, cv::Mat& output);
void complexMultiply(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& output);
void complexScalarMultiply(std::complex<double> scaler, cv::Mat& m, cv::Mat output);
void complexDivide(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& output);
void complexInverse(const cv::Mat& m, cv::Mat& inverse);
cv::Mat fftShiftS(cv::Mat m);
cv::Mat ifftShiftS(cv::Mat m);
void fftShift(const cv::Mat& input, cv::Mat& output);
void ifftShift(const cv::Mat& input, cv::Mat& output);
void fft2(cv::Mat& input, cv::Mat& output);
void ifft2(cv::Mat& input, cv::Mat& output);
void complex_imread(std::string fNameAmp, std::string fNamePhase, cv::Mat& output, int16_t rwMode);
void complex_imwrite(cv::Mat& m1,std::string fname, int16_t rwMode);
void onMouse( int event, int x, int y, int, void* param);
void showImg(cv::Mat m, std::string windowTitle);
void showComplexImg(cv::Mat m, int16_t displayFlag, std::string windowTitle);
void setColorMap(int16_t cMap);
void printMat(cv::Mat m, std::string title);

#endif
