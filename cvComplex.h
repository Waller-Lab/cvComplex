//cvComplex.h

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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

extern const int16_t COLORIMAGE_REAL;
extern const int16_t COLORIMAGE_COMPLEX;

// METHODS
void circularShift(const cv::UMat& input, cv::UMat& output, int16_t x, int16_t y);
//void circularShift(const cv::Mat& input, cv::Mat& output, int16_t x, int16_t y);
void maxComplexReal(cv::UMat& m, std::string label);
void complexConj(const cv::UMat& input, cv::UMat& output);
void complexAbs(const cv::UMat& input, cv::UMat& output);
void complexMultiply(const cv::UMat& input1, const cv::UMat& input2, cv::UMat& output);
void complexScalarMultiply(std::complex<double> scaler, cv::UMat& input, cv::UMat output);
void complexDivide(const cv::UMat& input1, const cv::UMat& input2, cv::UMat& output);
void complexInverse(const cv::UMat& input, cv::UMat& output);
void fftShift(const cv::UMat& input, cv::UMat& output);
void ifftShift(const cv::UMat& input, cv::UMat& output);
void fft2(cv::UMat& input, cv::UMat& output);
void ifft2(cv::UMat& input, cv::UMat& output);
void complex_imread(std::string fNameAmp, std::string fNamePhase, cv::UMat& output, int16_t rwMode);
void complex_imwrite(cv::UMat& m1,std::string fname, int16_t rwMode);
void onMouse( int event, int x, int y, int, void* param);
void showImg(cv::UMat m, std::string windowTitle, int16_t gv_cMap);
void showImg(cv::UMat m, std::string windowTitle);
void showComplexImg(cv::UMat m, int16_t displayFlag, std::string windowTitle, int16_t gv_cMap);
void showComplexImg(cv::UMat m, int16_t displayFlag, std::string windowTitle);
void showImgC(cv::UMat* ImgC, std::string windowTitle, int16_t REAL_COMPLEX);
void setColorMap(int16_t cMap);
void printMat(cv::UMat m, std::string title);
void splitUMat( cv::UMat& input, cv::UMat * output);
void printOclPlatformInfo();
void mergeUMat( cv::UMat* input, int8_t channels, cv::UMat& output);
void splitUMat( cv::UMat& input, cv::UMat * output);

/******************************************************************************
  Compatability functions for older applicaitons which only use cv::Mat class
******************************************************************************
void showComplexImg(cv::Mat m, int16_t displayFlag, std::string windowTitle, int16_t gv_cMap)
{showComplexImg(m.getUMat(cv::ACCESS_READ), displayFlag,windowTitle,gv_cMap);}

void maxComplexReal(cv::Mat& m, std::string label)
{maxComplexReal(m.getUMat(cv::ACCESS_READ),label);}

void complexConj(const cv::Mat& input, cv::Mat& output)
{complexConj(const cv::UMat& input.getUMat(cv::ACCESS_READ), output.getUMat(cv::ACCESS_READ));}
*/

/*
void circularShift(const cv::Mat& input, cv::Mat& output, int16_t x, int16_t y)
{circularShift(&input.getUMat(cv::ACCESS_READ),&output.getUMat(cv::ACCESS_RW),x,y);}
*/

#endif
