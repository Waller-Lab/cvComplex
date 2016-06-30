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

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/ocl.hpp>
//#include "opencv2/contrib.hpp"
#include "cvComplex.h"
#include <stdio.h>
#include <cmath>

int16_t gv_cMap = -1; // Global Colormap Setting

const int16_t SHOW_COMPLEX_MAG = 0;
const int16_t SHOW_COMPLEX_COMPONENTS = 1;
const int16_t SHOW_COMPLEX_REAL = 2;
const int16_t SHOW_COMPLEX_IMAGINARY = 3;
const int16_t SHOW_AMP_PHASE = 4;

const int16_t RW_MODE_AMP_PHASE = 1;
const int16_t RW_MODE_REAL_IMAG = 2;

const int16_t CMAP_MIN = 0;
const int16_t CMAP_MAX = 11;
const int16_t COLORMAP_NONE = -1;

const int16_t COLORIMAGE_REAL = 0;
const int16_t COLORIMAGE_COMPLEX = 1;

using namespace cv;
using namespace std;

void circularShift(const cv::UMat& input, cv::UMat& output, int16_t x, int16_t y)
{
  
  if (output.empty())
    output = cv::UMat::zeros(input.rows, input.cols, input.type());

  cv::Mat input1 = input.getMat(ACCESS_READ);
  cv::Mat output1 = output.getMat(ACCESS_RW);

  int16_t w = input.cols;
  int16_t h = input.rows;

  int16_t shiftR = x % w;
  int16_t shiftD = y % h;

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

  // Generate pointers
  cv::Mat shift1, shift2, shift3, shift4;
  
  if (&input == &output) // Check if matricies have same pointer
  {
    shift1 = input1(gate1).clone();
  	shift2 = input1(gate2).clone();
  	shift3 = input1(gate3).clone();
  	shift4 = input1(gate4).clone();
  }
  else // safe to shallow copy
  {
    shift1 = input1(gate1);
    shift2 = input1(gate2);
    shift3 = input1(gate3);
    shift4 = input1(gate4);
  }

  /*
  shift1 = input1(gate1).clone();
  shift2 = input1(gate2).clone();
  shift3 = input1(gate3).clone();
  shift4 = input1(gate4).clone();
  */

  // Copy to result
	shift1.copyTo(cv::Mat(output1, out1));
	shift2.copyTo(cv::Mat(output1, out2));
	shift3.copyTo(cv::Mat(output1, out3));
	shift4.copyTo(cv::Mat(output1, out4));
}

void maxComplexReal(cv::UMat& m, std::string label)
{
      cv::UMat planes[] = {cv::UMat::zeros(m.rows, m.cols, m.type()), cv::UMat::zeros(m.rows, m.cols, m.type())};
      splitUMat(m,2,planes);
      double minVal, maxVal;
      cv::minMaxLoc(planes[0], &minVal, &maxVal);
      std::cout << "Max/Min values of " <<label << " are: " << maxVal << ", " << minVal << std::endl;
}

void splitUMat( cv::UMat& input, int8_t channels, cv::UMat * output)
{
  if ((*output).channels() != (int) channels) {
    std::cout << "splitUMat: output matrix channels does not match channels argument" << std::endl;
    std::cout << "output channels: " << (*output).channels() << " channels arg: " << (int) channels << std::endl;
  }
  // Use standard cv::split for now, parallelize later
  cv::Mat* outputPlanes = new Mat[channels];
  for (int i = 0; i < channels; i++) {
    outputPlanes[i] = cv::Mat::zeros(input.rows, input.cols, input.type());
  }

  cv::split(input.getMat(ACCESS_RW),outputPlanes);
  for (int8_t ch=0; ch<input.channels(); ch++)
    output[ch] = outputPlanes[ch].getUMat(ACCESS_RW);

  delete[] outputPlanes;
}

void mergeUMat( cv::UMat* input, int8_t channels, cv::UMat& output)
{
  if ((*input).channels() != (int) channels) {
    std::cout << "mergeUMat: input matrix channels does not match channels argument" << std::endl;
    std::cout << "input channels: " << (*input).channels() << " channels arg: " << (int) channels << std::endl;
  }
  // Use standard cv::merge for now, parallelize later 
  cv::Mat* inputPlanes = new Mat[channels];
  for (int i = 0; i < channels; i++) {
    inputPlanes[i] = cv::Mat::zeros(input[0].rows, input[0].cols, input[0].type());
  }

  for (int8_t ch=0; ch<channels; ch++)
    inputPlanes[ch] = input[ch].getMat(ACCESS_RW);

  cv::Mat outputMat;
  cv::merge(inputPlanes,channels,outputMat);
  output = outputMat.getUMat(ACCESS_RW);

  delete[] inputPlanes;
}

/*
void complexConj(const cv::UMat& input, cv::UMat& output)
{
  if (output.empty())
    output = cv::UMat::zeros(input.rows, input.cols, CV_64FC2);

  double real, imag;
	for(int i = 0; i < input.rows; i++) // loop through y
	{
    const double* m_i = input.getMat(ACCESS_RW).ptr<double>(i);  // Input
    double* o_i = output.getMat(ACCESS_RW).ptr<double>(i);   // Output

    for(int j = 0; j < input.cols; j++)
    {
        real = (double) m_i[j*2];
        imag = (double) -1.0 * m_i[j*2+1];

        o_i[j*2] = real;
        o_i[j*2+1] = imag;
    }
	}
}*/

void complexConj(const cv::UMat& input, cv::UMat& output)
{
  if (output.empty())
    output = cv::UMat::zeros(input.rows, input.cols, CV_64FC2);
  output = input.mul(cv::Scalar(1,-1));
}


void complexAngle(const cv::UMat& input, cv::UMat& output)
{
  if (output.empty())
    output = cv::UMat::zeros(input.rows, input.cols, CV_64FC2);

  for(int i = 0; i < input.rows; i++) // loop through y
 	{
     const double* m_i = input.getMat(ACCESS_RW).ptr<double>(i);  // Input
     double* o_i = output.getMat(ACCESS_RW).ptr<double>(i);   // Output

     for(int j = 0; j < input.cols; j++)
     {
         o_i[2*j] = (double) atan2(m_i[j*2+1],m_i[j*2]);
         o_i[j*2+1] = 0.0;
     }
 	}
}

void complexAbs(const cv::UMat& input, cv::UMat& output)
{
  // Ensure output is not empty
  if (output.empty())
      output = cv::UMat::zeros(input.rows, input.cols, CV_64FC2);

	for(int i = 0; i < input.rows; i++) // loop through y
	{
    const double* m_i = input.getMat(ACCESS_RW).ptr<double>(i);  // Input
    double* o_i = output.getMat(ACCESS_RW).ptr<double>(i);   // Output

    for(int j = 0; j < input.cols; j++)
    {
        o_i[j*2] = (double) std::sqrt(m_i[j*2] * m_i[j*2] + m_i[j*2+1] * m_i[j*2+1]);
        o_i[j*2+1] = 0.0;
    }
	}
}

void complexAmpPhaseToRealImag(const cv::UMat& input, cv::UMat& output)
{

  if (output.empty())
    output = cv::UMat::zeros(input.rows, input.cols, CV_64FC2);

  double real,imag;
	for(int i = 0; i < input.rows; i++) // loop through y
	{
    const double* m_i = input.getMat(ACCESS_RW).ptr<double>(i);  // Input
    double* o_i = output.getMat(ACCESS_RW).ptr<double>(i);   // Output

    for(int j = 0; j < input.cols; j++)
    {
        real = (double)m_i[j*2]*sin((double)m_i[j*2+1]);
        imag = (double)m_i[j*2]*cos((double)m_i[j*2+1]);
        o_i[j*2]  = real; o_i[j*2+1] = imag;
    }
	}
}

/* complexMultiply(const cv::UMat& m1, const cv::UMat& m2, cv::UMat& output)
 * Multiplies 2 complex matricies where the first two color channels are the
 * real and imaginary coefficents, respectivly. Uses the equation:
 *         (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
 *
 * INPUTS:
 *   const cv::UMat& m1:  Complex Matrix 1
 *   const cv::UMat& m2:  Complex Matrix 2
 * OUTPUT:
 *   cv::UMat& output:    Complex Product of m1 and m2
 */
void complexMultiply(const cv::UMat& input1, const cv::UMat& input2, cv::UMat& output)//elementwise multiplies
{
  // Check if matricies are of same size and type
  if (!((input1.size() == input2.size()) && (input1.type() == input2.type())))
  {
    std::cout << "ERROR - matricies are of different size!" << std::endl;
    return;
  }
  // Ensure output is not empty
  if (output.empty())
      output = cv::UMat::zeros(input1.rows, input1.cols, CV_64FC2);

  // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
  double real,imag;
	for(int i = 0; i < input1.rows; i++) // loop through y
	{
    const double* m1_i = input1.getMat(ACCESS_RW).ptr<double>(i);   // Input 1
    const double* m2_i = input2.getMat(ACCESS_RW).ptr<double>(i);   // Input 2
    double* o_i = output.getMat(ACCESS_RW).ptr<double>(i); // Output
    for(int j = 0; j < input1.cols; j++)
    {
        real = (m1_i[j*2] * m2_i[j*2]) - (m1_i[j*2+1] * m2_i[j*2+1]);    // Real
        imag = (m1_i[j*2] * m2_i[j*2+1]) + (m1_i[j*2+1] * m2_i[j*2]);  // Imaginary
        o_i[j*2]  = real; o_i[j*2+1] = imag;
    }
	}
}

/* complexScalarMultiply(const cv::UMat& m1, const cv::UMat& m2, cv::UMat& output)
 * Multiplies the real and imaginary parts of a complex matrix by a scalar.
 *
 * INPUTS:
 *   double scalar:      Scalar to multiply
 *   const cv::UMat& m1:  Complex matrix input
 * OUTPUT:
 *   cv::UMat& output:    Complex product of m1 and scalar
 */
void complexScalarMultiply(std::complex<double> scaler, cv::UMat& input, cv::UMat output)
{
  if (output.empty())
    output = cv::UMat::zeros(input.rows, input.cols, CV_64FC2);

   // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
   double real, imag;
	for(int i = 0; i < input.rows; i++) // loop through y
	{
    const double* m_i = input.getMat(ACCESS_RW).ptr<double>(i);   // Input 1
    double* o_i = output.getMat(ACCESS_RW).ptr<double>(i);      // Output
    for(int j = 0; j < input.cols; j++)
    {
        real = scaler.real() * m_i[j*2] - scaler.imag() * m_i[j*2+1]; // Real
        imag= scaler.imag() * m_i[j*2] + scaler.real() * m_i[j*2+1]; // Imaginary
        o_i[j*2]  = real; o_i[j*2+1] = imag;
    }
	}
}

/* complexDivide(const cv::UMat& m1, const cv::UMat& m2, cv::UMat& output)
 * Divides one matrix by another where the first two color channels are the
 * real and imaginary coefficents, respectivly. Uses the equation:
 *   (a+bi) / (c+di) = (ac+bd) / (c^2+d^2) + (bc-ad) / (c^2+d^2) * i
 *
 * INPUTS:
 *   const cv::UMat& m1:  Complex Matrix 1
 *   const cv::UMat& m2:  Complex Matrix 2
 * OUTPUT:
 *   cv::UMat& output:    Complex Product of m1 and m2
 */
void complexDivide(const cv::UMat& input1, const cv::UMat& input2, cv::UMat& output)
{
  // Check if matricies are of same size and type
  if (!((input1.size() == input2.size()) && (input1.type() == input2.type())))
  {
    std::cout << "ERROR - matricies are of different size!" << std::endl;
    return;
  }

  // Ensure output is not empty
  if (output.empty())
      output = cv::UMat::zeros(input1.rows, input1.cols, CV_64FC2);

   // (a+bi) / (c+di) = (ac+bd) / (c^2+d^2) + (bc-ad) / (c^2+d^2) * i
   double real, imag;
	for(int i = 0; i < input1.rows; i++) // loop through y
	{
    const double* m1_i = input1.getMat(ACCESS_RW).ptr<double>(i);   // Input 1
    const double* m2_i = input2.getMat(ACCESS_RW).ptr<double>(i);   // Input 2
    double* o_i = output.getMat(ACCESS_RW).ptr<double>(i);      // Output
    for(int j = 0; j < input1.cols; j++)
    {
        real = ((m1_i[j*2] * m2_i[j*2]) + (m1_i[j*2+1] * m2_i[j*2+1])) / (m2_i[j*2] * m2_i[j*2] + m2_i[j*2+1] * m2_i[j*2+1]); // Real
        imag = ((m1_i[j*2+1] * m2_i[j*2]) - (m1_i[j*2] * m2_i[j*2+1])) / (m2_i[j*2] * m2_i[j*2] + m2_i[j*2+1] * m2_i[j*2+1]);  // Imaginary
        o_i[j*2]  = real; o_i[j*2+1] = imag;
    }
	}
}

void complexInverse(const cv::UMat& input, cv::UMat& output)
{

  if (output.empty())
    output = cv::UMat::zeros(input.rows, input.cols, CV_64FC2);

   // (a+bi) / (c+di) = (ac+bd) / (c^2+d^2) + (bc-ad) / (c^2+d^2) * i
  double real, imag;
	for(int i = 0; i < input.rows; i++) // loop through y
	{
    //const double* m1_i = m1.getMat(ACCESS_RW).ptr<double>(i);   // Input 1
    const double* m2_i = input.getMat(ACCESS_RW).ptr<double>(i);   // Input 2
    double* o_i = output.getMat(ACCESS_RW).ptr<double>(i);      // Output
    for(int j = 0; j < input.cols; j++)
    {
        real = ((m2_i[j*2]) + ( m2_i[j*2+1])) / (m2_i[j*2] * m2_i[j*2] + m2_i[j*2+1] * m2_i[j*2+1]); // Real
        imag = (( m2_i[j*2]) - (m2_i[j*2+1])) / (m2_i[j*2] * m2_i[j*2] + m2_i[j*2+1] * m2_i[j*2+1]);  // Imaginary
        o_i[j*2]  = real; o_i[j*2+1] = imag;
    }
	}
}

void fftShift(const cv::UMat& input, cv::UMat& output)
{
	 	circularShift(input, output, std::ceil((double) input.cols/2), std::ceil((double) input.rows/2));
}

void ifftShift(const cv::UMat& input, cv::UMat& output)
{
	 	circularShift(input, output, std::floor((double) input.cols/2), std::ceil((double) input.rows/2));
}
// Opencv fft implimentation
void fft2(cv::UMat& input, cv::UMat& output)
{
   //cv::UMat paddedInput;
   //int m = cv::getOptimalDFTSize( input.rows );
   //int n = cv::getOptimalDFTSize( input.cols );

   // Zero pad for Speed
   //cv::copyMakeBorder(input, paddedInput, 0, m - input.rows, 0, n - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
   cv::dft(input, output, cv::DFT_COMPLEX_OUTPUT);
}

// Inverse Fourier Transform
void ifft2(cv::UMat& input, cv::UMat& output)
{
   //cv::UMat paddedInput;
   //int m = cv::getOptimalDFTSize( input.rows );
   //int n = cv::getOptimalDFTSize( input.cols );

   // Zero pad for speed
   //cv::copyMakeBorder(input, paddedInput, 0, m - input.rows, 0, n - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
   cv::dft(input, output, cv::DFT_INVERSE | cv::DFT_COMPLEX_OUTPUT | cv::DFT_SCALE); // Real-space of object
}

// Write complex matrix to file
void complex_imwrite(cv::UMat& m1,std::string fname, int16_t rwMode)
{
    cv::UMat complexPlanes[] = {cv::UMat::zeros(m1.rows, m1.cols, m1.type()), cv::UMat::zeros(m1.rows, m1.cols, m1.type())};
    std::string typeStr1;
    std::string typeStr2;
    if (rwMode & RW_MODE_AMP_PHASE)
    {
        complexAbs(m1,complexPlanes[0]);
        complexAngle(m1,complexPlanes[1]);
        typeStr1 = "Amp"; typeStr2 = "Phase";
    }
    else if (rwMode & RW_MODE_REAL_IMAG)
    {
        splitUMat(m1,2,complexPlanes);
        typeStr1 = "Real"; typeStr2 = "Imag";
    }
    cv::imwrite(fname+'_'+typeStr1,complexPlanes[0]);
    cv::imwrite(fname+'_'+typeStr2,complexPlanes[1]);
}

// Read a complex image from two matricies
void complex_imread(std::string fNameAmp, std::string fNamePhase, cv::UMat& output, int16_t rwMode)
{
    Mat m1 = cv::imread(fNameAmp,-1*CV_LOAD_IMAGE_ANYDEPTH); // Any depth, any type
    Mat m2 = cv::imread(fNamePhase,-1*CV_LOAD_IMAGE_ANYDEPTH); // Any depth, any type
    if (m1.rows ==0 || m2.rows==0)
    {
        std::cout << "ERROR - images not found!"<<std::endl;
        return;
    }
    cv::UMat complexPlanes[] = {cv::UMat::zeros(m1.rows, m1.cols, m1.type()), cv::UMat::zeros(m1.rows, m1.cols, m1.type())};
    complexPlanes[0] = m1.getUMat(ACCESS_RW); complexPlanes[1] = m2.getUMat(ACCESS_RW);
    mergeUMat(complexPlanes,2,output);
    cv::UMat tmpMat = cv::UMat::zeros(output.rows,output.cols,output.type());
    if (rwMode & RW_MODE_AMP_PHASE)
    {
        complexAmpPhaseToRealImag(output,tmpMat);
        tmpMat.copyTo(output);
    }


}
// Mouse callback for showImg
void onMouse( int event, int x, int y, int, void* param )
{

	cv::UMat* imgPtr = (cv::UMat*) param;
	cv::UMat image;
	imgPtr->copyTo(image);
	image.convertTo(image,CV_64F); // to keep types consitant

	// Split image into channels
	cv::UMat *  planes= new cv::UMat[image.channels()];
	for (int16_t ch=0; ch < image.channels(); ch++)
		planes[ch] = cv::UMat(image.rows, image.cols, image.type());
	splitUMat(image,image.channels(),planes);

	switch (event)
	{
		case ::CV_EVENT_LBUTTONDOWN:
		{
			// Pretty printing of complex matricies
			if (image.channels() ==2)
			{
				std::printf("x:%d y:%d: \n", x, y);
				std::printf("  %.4f + %.4fi\n",
				planes[0].getMat(ACCESS_RW).at<double>(y,x),
				planes[1].getMat(ACCESS_RW).at<double>(y,x));
			}
			else //All other color channels
			{
				std::printf("x:%d y:%d: \n", x, y);
				for (int16_t ch=0; ch < image.channels(); ch++)
				{
					std::printf("  Channel %d: %.4f \n",ch+1,planes[ch].getMat(ACCESS_RW).at<double>(y,x));
				}
			}
			std::cout<<std::endl;
			break;
		}
		case ::CV_EVENT_RBUTTONDOWN:
		{
			double minVal, maxVal;
			std::printf("x:%d y:%d: \n", x, y);
			for (int16_t ch=0; ch < image.channels(); ch++)
			{
				cv::minMaxLoc(planes[ch], &minVal, &maxVal);
				std::printf("  Channel %d: min: %.4f, max: %.4f \n",ch+1,minVal,maxVal);
			}
			std::cout<<std::endl;
			break;
		}
	default:
		return;
	}
}

// Compatability with previous versions
void showComplexImg(cv::UMat m, int16_t displayFlag, std::string windowTitle)
{
  showComplexImg(m, displayFlag, windowTitle,-1);
}

void showImg(cv::UMat m, std::string windowTitle)
{
  showImg(m,windowTitle,-1);
}

// Display a complex image
void showComplexImg(cv::UMat m, int16_t displayFlag, std::string windowTitle, int16_t gv_cMap)
{
   if (m.channels() == 2) // Ensure Complex Matrix
   {
    //Mat m1 = m.getMat(ACCESS_READ);
		cv::UMat planes[] = {cv::UMat::zeros(m.rows, m.cols, m.type()), cv::UMat::zeros(m.rows, m.cols, m.type())};
		//cv::split(m1, planes);
    splitUMat(m,2,planes);                   // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))

		switch(displayFlag)
		{
			case (SHOW_COMPLEX_MAG):
			{
				cv::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
            windowTitle = windowTitle + " Magnitude";
            cv::log(planes[0],planes[0]);
            showImg(planes[0], windowTitle, gv_cMap);
				break;
			}
			case (SHOW_COMPLEX_REAL):
			{
			   std::string reWindowTitle = windowTitle + " Real";
            showImg(planes[0], reWindowTitle, gv_cMap);
				break;
			}
			case (SHOW_COMPLEX_IMAGINARY):
			{
				std::string imWindowTitle = windowTitle + " Imaginary";
            showImg(planes[1], imWindowTitle, gv_cMap);
				break;
			}
      case (SHOW_AMP_PHASE):
      {
        cv::UMat m2;
        cv::UMat planes2[] = {cv::UMat::zeros(m.rows, m.cols, m.type()), cv::UMat::zeros(m.rows, m.cols, m.type())};
        complexAngle(m,m2);
        splitUMat(m2,2,planes2);

        std::string amWindowTitle = windowTitle + " Amplitude";
        std::string phWindowTitle = windowTitle + " Phase";
        complexAbs(m,m);
        splitUMat(m,2,planes);

        showImg(planes[0], amWindowTitle, gv_cMap);
        showImg(planes2[0], phWindowTitle, gv_cMap);
        break;
      }
			default:
			{

				std::string reWindowTitle = windowTitle + " Real";
				std::string imWindowTitle = windowTitle + " Imaginary";

				showImg(planes[0], reWindowTitle, gv_cMap);
				showImg(planes[1], imWindowTitle, gv_cMap);

				break;
			}
		}
		//cv::waitKey();
		//cv::destroyAllWindows();
	}
	else
		std::cout << "ERROR ( cvComplex::shotComplexImg ) : Input Mat is not complex (m.channels() != 2)" << std::endl;
}

// Print information about a matrix
void printMat(cv::UMat m, std::string title)
{
	std::cout << "cv::UMat " << title<<" properties:"<<std::endl;
	std::cout << "  sz: " <<m.cols<< " x " <<m.rows<<std::endl;
	std::cout << "  depth: "<<m.depth()<<", channels: " <<m.channels()<<std::endl;
	std::cout << "  type: "<<m.type()<<std::endl;
}

// Show a single-channel image
void showImg(cv::UMat m, std::string windowTitle, int16_t gv_cMap)
{

	cv::UMat displayMat;

   if (gv_cMap >= cv::COLORMAP_AUTUMN && gv_cMap <= cv::COLORMAP_HOT)
   {
	   cv::UMat scaledMat;
	   cv::normalize(m, scaledMat, 0, 255, CV_MINMAX);
	   scaledMat.convertTo(displayMat, CV_8U);
	   cv::applyColorMap(displayMat, displayMat, gv_cMap);
   }
   else
   {
	   cv::UMat scaledImg;
	   cv::normalize(m, scaledImg, 0, 1, CV_MINMAX);
	   scaledImg.convertTo(displayMat,CV_32FC1);
	   cvtColor(displayMat, displayMat, CV_GRAY2RGB);
   }

    cv::startWindowThread();
    cv::namedWindow(windowTitle, cv::WINDOW_NORMAL);
    cv::setMouseCallback(windowTitle, onMouse, &m);;
	cv::imshow(windowTitle, displayMat);
	cv::waitKey();
	cv::destroyAllWindows();
}

void showImgC(cv::UMat* ImgC, std::string windowTitle, int16_t REAL_COMPLEX)
{
	cv::UMat displayMat, ImgCC;

	if(REAL_COMPLEX == COLORIMAGE_REAL)
	{
		mergeUMat(ImgC,3,ImgCC);
	}
	else
	{
	   cv::UMat ImgCC_buff[] = {cv::UMat::zeros(ImgC[0].rows, ImgC[0].cols, CV_64FC1),
			                    cv::UMat::zeros(ImgC[0].rows, ImgC[0].cols, CV_64FC1),
		    					cv::UMat::zeros(ImgC[0].rows, ImgC[0].cols, CV_64FC1)};
		cv::UMat ImgCC_buff2[] = {cv::UMat::zeros(ImgC[0].rows, ImgC[0].cols, CV_64FC1),
				  		         cv::UMat::zeros(ImgC[0].rows, ImgC[0].cols, CV_64FC1)};

		splitUMat(ImgC[0],2,ImgCC_buff2);
		ImgCC_buff[0] = ImgCC_buff2[0].clone();
		splitUMat(ImgC[1],2,ImgCC_buff2);
		ImgCC_buff[1] = ImgCC_buff2[0].clone();
		splitUMat(ImgC[2],2,ImgCC_buff2);
		ImgCC_buff[2] = ImgCC_buff2[0].clone();
		mergeUMat(ImgCC_buff,3,ImgCC);
	}

	  cv::normalize(ImgCC, displayMat, 0, 1, CV_MINMAX);
      displayMat.convertTo(displayMat,CV_64FC3);

      cv::startWindowThread();
      cv::namedWindow(windowTitle, cv::WINDOW_NORMAL);
      cv::setMouseCallback(windowTitle, onMouse, ImgC);;
  	  cv::imshow(windowTitle, displayMat);
  	  cv::waitKey();
  	  cv::destroyAllWindows();

}

// OpenCL-specific functions

// Get platform list


void printOclPlatformInfo()
{
  //OpenCV: Platform Info
  std::vector<cv::ocl::PlatformInfo> platforms;
  cv::ocl::getPlatfomsInfo(platforms);

  // FOR CPU - run this in terminal
  // export OPENCV_OPENCL_DEVICE=":CPU:0"

  // FOR GPU - run this in terminal
  // export OPENCV_OPENCL_DEVICE=":GPU:0"

  //OpenCV Platforms
  std::cout << "OpenCL Platforms:" <<std::endl;
  for (size_t i = 0; i < platforms.size(); i++)
  {
      const cv::ocl::PlatformInfo* platform = &platforms[i];

          //Platform Name
      std::cout << "Platform Name: " << platform->name().c_str() << "\n";

          //Access known device
      cv::ocl::Device current_device;

      for (int j = 0; j < platform->deviceNumber(); j++)
      {
              //Access Device
          platform->getDevice(current_device, j);
          std::cout << "Device Name: " << current_device.name().c_str() << "\n";
      }
  }
}

/*void setColorMap(int16_t cMap)
{
	if (cMap >= CMAP_MIN && cMap <= CMAP_MAX)
		gv_cMap = cMap;
	else
		std::cout << "ERROR ( cvComplex::setColorMap )  : Invalid Color Map (Valid Values are between " << CMAP_MIN <<" and " << CMAP_MAX << std::endl;
}*/
