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
	if (argc < 1){
			cout << "Error: Not enough inputs.\nUSAGE: ./fresnelProp image_filename refocus_distance" << endl; return 0;}

	std::string imgFname = argv[1];
	double zDist = atof(argv[2]);
	
	// Load Image
	Mat img = imread(imgFname, CV_64FC1); // Load image as 64-bit double
	img.convertTo(img,CV_64FC1);

	// Generate Kernel
	Mat kernel = Mat::zeros(img.rows,img.cols,img.type());
	genFresnelKernel(img, kernel, zDist, 1.0/(img.cols*ps_eff), 1.0/(img.rows*ps_eff));
	//showComplexImg(kernel, SHOW_COMPLEX_COMPONENTS, "Fresnel Kernel");
	
	showComplexImg(kernel, -2, "Fresnel Kernel");
	

	// Convolve with FT of image
	Mat img_Ft;
	fft2(img, img_Ft);
	fftShift(img_Ft,img_Ft);

	Mat output_Ft;
	complexMultiply(img_Ft, kernel, output_Ft);
	fftShift(output_Ft,output_Ft);

	Mat output;
	ifft2(output_Ft, output );

	showComplexImg(output, SHOW_COMPLEX_REAL, "Refocused Bug Hologram");
} 
