// Example Program to perform fresnel propigation of a hologram

#include <stdio.h>
#include <complex.h>
#include "../../cvComplex.h" 
#include <math.h>

using namespace std; using namespace cv;

double objectiveMag = 1.0;
double pixelSz = 5e-6;
double ps_eff = pixelSz / objectiveMag;
double lambda = 6.32e-7;
double k = 2 * M_PI / lambda;
std::complex<double> ii = std::complex<double>(0,1);

void genFresnelKernel(const Mat& input, Mat& kernelMat, double zDist, double du, double dv){
   double fx, fy;
   Mat complexPlanes[] = {Mat::zeros(input.cols,input.rows, CV_64F), Mat::zeros(input.cols,input.rows, CV_64F)};
	for (uint16_t i = 0; i < input.cols; i++){ 
		for (uint16_t j = 0; j < input.rows; j++){
		   fx = (i - floor(input.cols/2.)) * du;
		   fy = (j - floor(input.rows/2.)) * dv;
			std::complex<double> cVal = exp(ii * k *zDist) * exp(-ii * M_PI * lambda * zDist * (fx*fx + fy*fy));
			complexPlanes[0].at<double>(i,j) = cVal.real();
			complexPlanes[1].at<double>(i,j) = cVal.imag();
		}
	}
   merge(complexPlanes, 2, kernelMat);
}  

int main(int argc, char** argv ){

	std::string imgFname = argv[1];
	
	// Load Image
	Mat img = imread(imgFname); // Load image as 64-bit double
	img.convertTo(img,CV_8U);
	setColorMap(COLORMAP_AUTUMN);
	showImg(img,"Hologram");

   
} 
