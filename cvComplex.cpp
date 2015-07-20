/*
 * cvComplex.cpp
 * A set of functions for dealing with two channel complex matricies inside
 * the OpenCV 2.4.* Imaging processing library. Complex numbers are generally
 * dealt with using a 2 color channel Mat object rather than std::complex. 
 * This library provides several common functions for manipulating matricies
 * in this format.
 *
 * Maintained by Z. Phillips, Computational Imaging Lab, UC Berkeley
 * Report bugs directly to zkphil@berkeley.edu
 */
 
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <string>
#include <stdio.h>
#include <dirent.h>
#include <fstream>
#include <vector>

#include "fpmUtils.h"

using namespace cv;
void circularShift(cv::Mat img, cv::Mat result, int x, int y){
    int w = img.cols;
    int h = img.rows;

    int shiftR = x % w;
    int shiftD = y % h;
    
    if (shiftR < 0)//if want to shift in -x direction
        shiftR += w;
    
    if (shiftD < 0)//if want to shift in -y direction
        shiftD += h;

    cv::Rect gate1(0, 0, w-shiftR, h-shiftD);//rect(x, y, width, height)
    cv::Rect out1(shiftR, shiftD, w-shiftR, h-shiftD);
    
    cv::Rect gate2(w-shiftR, 0, shiftR, h-shiftD); 
    cv::Rect out2(0, shiftD, shiftR, h-shiftD);
    
    cv::Rect gate3(0, h-shiftD, w-shiftR, shiftD);
    cv::Rect out3(shiftR, 0, w-shiftR, shiftD);
    
    cv::Rect gate4(w-shiftR, h-shiftD, shiftR, shiftD);
    cv::Rect out4(0, 0, shiftR, shiftD);
   
    cv::Mat shift1 = img ( gate1 );
    cv::Mat shift2 = img ( gate2 );
    cv::Mat shift3 = img ( gate3 );
    cv::Mat shift4 = img ( gate4 );

    shift1.copyTo(cv::Mat(result, out1));
    shift2.copyTo(cv::Mat(result, out2));
    shift3.copyTo(cv::Mat(result, out3));
    shift4.copyTo(cv::Mat(result, out4));
}

void maxComplexReal(cv::Mat& m, std::string label)
{
      cv::Mat planes[] = {cv::Mat::zeros(m.rows, m.cols, m.type()), cv::Mat::zeros(m.rows, m.cols, m.type())};
      cv::split(m,planes);
      double minVal, maxVal;
      cv::minMaxLoc(planes[0], &minVal, &maxVal);
      std::cout << "Max/Min values of " <<label << " are: " << maxVal << ", " << minVal << std::endl;
}

void complexConj(const cv::Mat& m, cv::Mat& output)
{
   if (output.empty())
   	output = cv::Mat::zeros(m.rows, m.cols, CV_64FC2);
   	
	for(int i = 0; i < m.rows; i++) // loop through y
	{
    const double* m_i = m.ptr<double>(i);  // Input
    double* o_i = output.ptr<double>(i);   // Output
    
    for(int j = 0; j < m.cols; j++)
    {
        o_i[j*2] = (double) m_i[j*2];
        o_i[j*2+1] = (double) -1.0 * m_i[j*2+1];
    }
	}
}

void complexAbs(const cv::Mat& m, cv::Mat& output)
{
    
   if (output.empty())
   	output = cv::Mat::zeros(m.rows, m.cols, CV_64FC2);
   	
	for(int i = 0; i < m.rows; i++) // loop through y
	{
    const double* m_i = m.ptr<double>(i);  // Input
    double* o_i = output.ptr<double>(i);   // Output
    
    for(int j = 0; j < m.cols; j++)
    {
        o_i[j*2] = (double)std::sqrt(m_i[j*2] * m_i[j*2] + m_i[j*2+1] * m_i[j*2+1]);
        o_i[j*2+1] = 0.0;
    }
	}
}

/* complexMultiply(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& output)
 * Multiplies 2 complex matricies where the first two color channels are the
 * real and imaginary coefficents, respectivly. Uses the equation:
 *         (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
 *
 * INPUTS:
 *   const cv::Mat& m1:  Complex Matrix 1
 *   const cv::Mat& m2:  Complex Matrix 2
 * OUTPUT:
 *   cv::Mat& outpit:    Complex Product of m1 and m2 
 */
void complexMultiply(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& output)
{
   if (output.empty())
   	output = cv::Mat::zeros(m1.rows, m1.cols, CV_64FC2);
   // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
	for(int i = 0; i < m1.rows; i++) // loop through y
	{
    const double* m1_i = m1.ptr<double>(i);   // Input 1
    const double* m2_i = m2.ptr<double>(i);   // Input 2
    double* o_i = output.ptr<double>(i);      // Output
    for(int j = 0; j < m1.cols; j++)
    {
        o_i[j*2] = (m1_i[j*2] * m2_i[j*2]) - (m1_i[j*2+1] * m2_i[j*2+1]);    // Real
        o_i[j*2+1] = (m1_i[j*2] * m2_i[j*2+1]) + (m1_i[j*2+1] * m2_i[j*2]);  // Imaginary
    }
	}
}

void complexScalarMultiply(double scaler, cv::Mat& m, cv::Mat output)
{
   if (output.empty())
   	output = cv::Mat::zeros(m.rows, m.cols, CV_64FC2);
   // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
	for(int i = 0; i < m.rows; i++) // loop through y
	{
    const double* m_i = m.ptr<double>(i);   // Input 1
    double* o_i = output.ptr<double>(i);      // Output
    for(int j = 0; j < m.cols; j++)
    {
        o_i[j*2] = scaler * m_i[j*2]; // Real
        o_i[j*2+1] = scaler * m_i[j*2+1]; // Real
    }
	}
}

void complexDivide(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& output)
{
   if (output.empty())
   	output = cv::Mat::zeros(m1.rows, m1.cols, CV_64FC2);
   // (a+bi) / (c+di) = (ac+bd) / (c^2+d^2) + (bc-ad) / (c^2+d^2) * i
	for(int i = 0; i < m1.rows; i++) // loop through y
	{
    const double* m1_i = m1.ptr<double>(i);   // Input 1
    const double* m2_i = m2.ptr<double>(i);   // Input 2
    double* o_i = output.ptr<double>(i);      // Output
    for(int j = 0; j < m1.cols; j++)
    {
        o_i[j*2] = ((m1_i[j*2] * m2_i[j*2]) + (m1_i[j*2+1] * m2_i[j*2+1])) / (m2_i[j*2] * m2_i[j*2] + m2_i[j*2+1] * m2_i[j*2+1]); // Real
        o_i[j*2+1] = ((m1_i[j*2+1] * m2_i[j*2]) - (m1_i[j*2] * m2_i[j*2+1])) / (m2_i[j*2] * m2_i[j*2] + m2_i[j*2+1] * m2_i[j*2+1]);  // Imaginary
    }
	}
}

void complexInverse(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& output)
{
   if (output.empty())
   	output = cv::Mat::zeros(m1.rows, m1.cols, CV_64FC2);
   // (a+bi) / (c+di) = (ac+bd) / (c^2+d^2) + (bc-ad) / (c^2+d^2) * i
	for(int i = 0; i < m1.rows; i++) // loop through y
	{
    const double* m1_i = m1.ptr<double>(i);   // Input 1
    const double* m2_i = m2.ptr<double>(i);   // Input 2
    double* o_i = output.ptr<double>(i);      // Output
    for(int j = 0; j < m1.cols; j++)
    {
        o_i[j*2] = ((m2_i[j*2]) + ( m2_i[j*2+1])) / (m2_i[j*2] * m2_i[j*2] + m2_i[j*2+1] * m2_i[j*2+1]); // Real
        o_i[j*2+1] = (( m2_i[j*2]) - (m2_i[j*2+1])) / (m2_i[j*2] * m2_i[j*2] + m2_i[j*2+1] * m2_i[j*2+1]);  // Imaginary
    }
	}
}

cv::Mat fftShift(cv::Mat m)
{
      cv::Mat shifted = cv::Mat(m.cols,m.rows,m.type());
      circularShift(m, shifted, std::ceil((double) m.cols/2), std::ceil((double) m.rows/2));
      return shifted;
}

cv::Mat ifftShift(cv::Mat m)
{
      cv::Mat shifted = cv::Mat(m.cols,m.rows,m.type());
      circularShift(m, shifted, std::floor((double) m.cols/2), std::floor((double)m.rows/2));
      return shifted;
}

// Opencv fft implimentation
void fft2(cv::Mat& input, cv::Mat& output)
{
   cv::Mat paddedInput;
   int m = cv::getOptimalDFTSize( input.rows );
   int n = cv::getOptimalDFTSize( input.cols ); 
   
   // Zero pad for Speed
   cv::copyMakeBorder(input, paddedInput, 0, m - input.rows, 0, n - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
   cv::dft(paddedInput, output, cv::DFT_COMPLEX_OUTPUT);
}
 
// Opencv ifft implimentation
void ifft2(cv::Mat& input, cv::Mat& output)
{
   cv::Mat paddedInput;
   int m = cv::getOptimalDFTSize( input.rows );
   int n = cv::getOptimalDFTSize( input.cols ); 
   
   // Zero pad for speed
   cv::copyMakeBorder(input, paddedInput, 0, m - input.rows, 0, n - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
   cv::dft(paddedInput, output, cv::DFT_INVERSE | cv::DFT_COMPLEX_OUTPUT | cv::DFT_SCALE); // Real-space of object
}

void complex_imwrite(std::string fname, cv::Mat& m1)
{
   cv::Mat outputPlanes[] = {cv::Mat::zeros(m1.rows, m1.cols, m1.type()), cv::Mat::zeros(m1.rows, m1.cols, m1.type()),cv::Mat::zeros(m1.rows, m1.cols, m1.type())};
   cv::Mat inputPlanes[] = {cv::Mat::zeros(m1.rows, m1.cols, m1.type()), cv::Mat::zeros(m1.rows, m1.cols, m1.type())};
   
   cv::split(m1,inputPlanes);
   outputPlanes[0] = inputPlanes[0];
   outputPlanes[1] = inputPlanes[1];
   cv::Mat outMat;
   cv::merge(outputPlanes,3,outMat);
   cv::imwrite(fname,outMat);
}

void onMouse( int event, int x, int y, int, void* param )
{

    cv::Mat* imgPtr = (cv::Mat*) param;
    cv::Mat image;
    imgPtr->copyTo(image);

    
    switch (event)
    {
       case ::CV_EVENT_LBUTTONDOWN:
       {
          cv::Mat planes[] = {cv::Mat::zeros(image.rows, image.cols, image.type()), cv::Mat::zeros(image.rows, image.cols, image.type())};
          cv::split(image, planes);    
        
          printf("x:%d y:%d: %f + %fi\n", 
          x, y, 
          planes[0].at<double>(y,x), 
          planes[1].at<double>(y,x));
          break;
       }
       case ::CV_EVENT_RBUTTONDOWN:
       {
          cv::Mat planes[] = {cv::Mat::zeros(image.rows, image.cols, image.type()), cv::Mat::zeros(image.rows, image.cols, image.type())};
          cv::split(image, planes);      
          double minVal, maxVal;
          cv::minMaxLoc(planes[0], &minVal, &maxVal);
          std::cout << "Max/Min values of real part are: " << maxVal << ", " << minVal << std::endl;
          cv::minMaxLoc(planes[1], &minVal, &maxVal);
          std::cout << "Max/Min values of imaginary part are: " << maxVal << ", " << minVal << std::endl << std::endl;
       break;
       }
       default:
         return;
   }
}

void showComplexImg(cv::Mat m, int16_t displayFlag, std::string windowTitle)
{
   cv::Mat planes[] = {cv::Mat::zeros(m.rows, m.cols, m.type()), cv::Mat::zeros(m.rows, m.cols, m.type())};
   cv::split(m, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
   cv::Mat displayMat = cv::Mat::zeros(m.rows, m.cols, CV_32FC3);
   Mat displayMatR = cv::Mat::zeros(m.rows, m.cols, CV_32FC3);
   Mat displayMatI = cv::Mat::zeros(m.rows, m.cols, CV_32FC3);
   
   switch(displayFlag)
   {
   	case (SHOW_COMPLEX_MAG):
   	{
   		cv::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
   		cv::Mat magI = planes[0];
			magI += cv::Scalar::all(1);                    // switch to logarithmic scale
			cv::log(magI, magI);
			magI.convertTo(magI,CV_32FC1);
			cvtColor(magI, displayMat, CV_GRAY2RGB);
			cv::applyColorMap(displayMat, displayMat, COLORMAP_JET);
			break;
		}
		case (SHOW_COMPLEX_REAL):
		{
			cv::Mat scaledReal;
			cv::normalize(planes[0], scaledReal, 0, 1, CV_MINMAX);
			scaledReal.convertTo(scaledReal,CV_32FC1);
			cvtColor(scaledReal, displayMat, CV_GRAY2RGB);
			cv::applyColorMap(displayMat, displayMat, COLORMAP_JET);
		   
		   std::string reWindowTitle = windowTitle + " Real";
			cv::namedWindow(reWindowTitle, cv::WINDOW_NORMAL);
			cv::setMouseCallback(reWindowTitle, onMouse, &planes[0]);
			cv::imshow(reWindowTitle, displayMat);
			break;
		}
		case (SHOW_COMPLEX_IMAGINARY):
		{
			cv::Mat scaledImag;
			cv::normalize(planes[1], scaledImag, 0, 1, CV_MINMAX);
			scaledImag.convertTo(scaledImag,CV_32FC1);
			cvtColor(scaledImag, displayMat, CV_GRAY2RGB);
			cv::applyColorMap(displayMat, displayMat, COLORMAP_JET);
		   
		   std::string imWindowTitle = windowTitle + " Imaginary";
			cv::namedWindow(imWindowTitle, cv::WINDOW_NORMAL);
			cv::setMouseCallback(imWindowTitle, onMouse, &planes[0]);
			cv::imshow(imWindowTitle, displayMat);
			break;
		}
		default:
		{
		   // Show complex components
			cv::Mat scaledReal; cv::Mat scaledImag;
			
			cv::normalize(planes[0], scaledReal, 0, 1, CV_MINMAX);
			cv::normalize(planes[1], scaledImag, 0, 1, CV_MINMAX);
			scaledReal.convertTo(scaledReal,CV_32FC1);
			scaledImag.convertTo(scaledImag,CV_32FC1);
			cv::cvtColor(scaledReal, displayMatR, CV_GRAY2RGB);
			cv::cvtColor(scaledImag, displayMatI, CV_GRAY2RGB);
			cv::applyColorMap(displayMatR, displayMatR, COLORMAP_JET);
			cv::applyColorMap(displayMatI, displayMatI, COLORMAP_JET);
		   
		   std::string reWindowTitle = windowTitle + " Real";
		   std::string imWindowTitle = windowTitle + " Imaginary";
			cv::namedWindow(reWindowTitle, cv::WINDOW_NORMAL);
			cv::setMouseCallback(reWindowTitle, onMouse, &planes[0]);
			cv::imshow(reWindowTitle, displayMatR);
			cv::namedWindow(imWindowTitle, cv::WINDOW_NORMAL);
			cv::setMouseCallback(imWindowTitle, onMouse, &planes[1]);
			cv::imshow(imWindowTitle, displayMatI);
			break;
		}
	}
	cv::waitKey();
	cv::destroyAllWindows();	
}
