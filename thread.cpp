// @author Jacques-Olivier Lachaud, LAMA, CNRS, Univ. Savoie Mont Blanc
// @date Mai 9, 2025

#include <iostream>
#include <fstream>
#include <chrono>
#include <limits>
#include <numeric>

// We use CLI11 to manage options
// see https://github.com/CLIUtils/CLI11
// and https://cliutils.gitlab.io/CLI11Tutorial/
#include "CLI11.hpp"

// We use opencv for display and image representation and i/o
#include <opencv2/opencv.hpp>

#include "Threader.hpp"

// using namespace cv;

cv::Mat Gx( cv::Mat input )
{
  cv::Mat Sx =  (1./4.) * ( cv::Mat_<float>( 3, 3 ) << -1, 0, 1, -2, 0, 2, -1, 0, 1 );
  cv::Mat output;
  cv::filter2D( input, output, -1, Sx, cv::Point(-1,-1), 128.0, cv::BORDER_DEFAULT );
  return output;
}
cv::Mat Gxx( cv::Mat input )
{
  cv::Mat Sx =  (1./4.) * ( cv::Mat_<float>( 3, 3 ) << -1, 0, 1, -2, 0, 2, -1, 0, 1 );
  cv::Mat tmp, output;
  cv::filter2D( input, tmp, -1, Sx, cv::Point(-1,-1), 128.0, cv::BORDER_DEFAULT );
  cv::filter2D( tmp, output, -1, Sx, cv::Point(-1,-1), 128.0, cv::BORDER_DEFAULT );
  return output;
}
cv::Mat Gy( cv::Mat input )
{
  cv::Mat Sy =  (1./4.) * ( cv::Mat_<float>( 3, 3 ) << 1, 2, 1, 0, 0, 0, -1, -2, -1 );
  cv::Mat output;
  cv::filter2D( input, output, -1, Sy, cv::Point(-1,-1), 128.0, cv::BORDER_DEFAULT );
  return output;
}

cv::Mat Gyy( cv::Mat input )
{
  cv::Mat Sy =  (1./4.) * ( cv::Mat_<float>( 3, 3 ) << 1, 2, 1, 0, 0, 0, -1, -2, -1 );
  cv::Mat tmp, output;
  cv::filter2D( input, tmp, -1, Sy, cv::Point(-1,-1), 128.0, cv::BORDER_DEFAULT );
  cv::filter2D( tmp, output, -1, Sy, cv::Point(-1,-1), 128.0, cv::BORDER_DEFAULT );
  return output;
}
cv::Mat Gxy( cv::Mat input )
{
  return Gx( Gy( input ) );
}
cv::Mat Rehausseur( cv::Mat input, double a )
{
  cv::Mat L  = ( cv::Mat_<float>( 3, 3 ) << 0, -a, 0, -a, 1.+4.*a, -a, 0, -a, 0 );
  cv::Mat output;
  cv::filter2D( input, output, -1, L, cv::Point(-1,-1), 0.0, cv::BORDER_DEFAULT );
  return output;
}
cv::Mat Diag( cv::Mat input )
{
  cv::Mat L  = (1./4.) * ( cv::Mat_<float>( 3, 3 ) << 1, -1, 1, -1, 4, -1, 1, -1, 1 );
  cv::Mat output;
  cv::filter2D( input, output, -1, L, cv::Point(-1,-1), 128.0, cv::BORDER_DEFAULT );
  return output;
}

cv::Mat Lap( cv::Mat input )
{
  cv::Mat L  =            ( cv::Mat_<float>( 3, 3 ) << 0, 1, 0, 1, -4, 1, 0, 1, 0 );
  cv::Mat output;
  cv::filter2D( input, output, -1, L, cv::Point(-1,-1), 128.0, cv::BORDER_DEFAULT );
  return output;
}
cv::Mat SobelLap( cv::Mat input )
{
  cv::Mat Ixx = Gxx( input );
  cv::Mat Iyy = Gyy( input );
  cv::Mat R;
  cv::addWeighted( Ixx, 1.0, Iyy, 1.0, -128.0, R );
  return R;
}

cv::Mat norm_gradient( cv::Mat input )
{
  cv::Mat Sx =  (1./4.) * ( cv::Mat_<float>( 3, 3 ) << -1, 0, 1, -2, 0, 2, -1, 0, 1 );
  cv::Mat Sy =  (1./4.) * ( cv::Mat_<float>( 3, 3 ) << 1, 2, 1, 0, 0, 0, -1, -2, -1 );
  cv::Mat output = input.clone();
  cv::Mat tmp1, tmp2;
  cv::filter2D( input, tmp1, -1, Sx, cv::Point(-1,-1), 128.0, cv::BORDER_DEFAULT );
  cv::filter2D( input, tmp2, -1, Sy, cv::Point(-1,-1), 128.0, cv::BORDER_DEFAULT );
  auto it1 = tmp1.begin< uchar >();
  auto it2 = tmp2.begin< uchar >();
  for ( auto it = output.begin< uchar >(), itE = output.end< uchar >();
        it != itE; ++it ) {
    double gx = ( (double) *it1++ ) - 128.0;
    double gy = ( (double) *it2++ ) - 128.0;
    *it = cv::saturate_cast< uchar >( sqrt( gx*gx + gy*gy ) );
  }
  return output;
}

cv::Mat blur( cv::Mat input, double sigma )
{
  cv::Mat output;
  int    ksize = (int) (round( 2.5*sigma ) );
  cv::GaussianBlur( input, output, cv::Size(2*ksize+1,2*ksize+1), sigma, sigma);
  return output;
}

void clamp01f( cv::Mat input )
{
  for ( auto it = input.begin<float>(), ite = input.end<float>(); it != ite; ++it )
    *it = std::min( 1.0f, std::max( 0.0f, *it ) );
}

template <typename T>
struct Memorizer {
  std::vector<T> _data;
  void add( const T& v ) { _data.push_back( v ); }
  T averageLastValues( std::size_t nb ) const
  {
    const std::size_t  last = _data.size();
    const std::size_t first = nb > last ? 0 : last - nb;
    T value = T(0.0);
    for ( auto k = first; k < last; k++ ) value += _data[ k ];
    return value / (last - first);
  }
};

double
getPSNR(const cv::Mat& I1, const cv::Mat& I2)
{
  cv::Mat s1;
  cv::absdiff(I1, I2, s1);       // |I1 - I2|
  // s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
  s1 = s1.mul(s1);           // |I1 - I2|^2
  
  cv::Scalar s = sum(s1);        // sum elements per channel
  
  double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
  
  if( sse <= 1e-10) // for small values return zero
    return std::numeric_limits<double>::infinity();
  else
    {
      double mse  = sse / (double)(I1.channels() * I1.total());
      double psnr = 10.0 * log10((1.*1.) / mse);
      return psnr;
    }
}


int main(int argc, char** argv )
{
    // parse command line using CLI
  CLI::App app;
  
  app.description("Draws a grayscale image with a (very long) thread.\n");

  std::string image_fname;
  int         zoom = 4;
  double thickness = 0.02;
  int           nb = 80;
  double        lp = 2.0;
  double      coef = 2.0;
  bool        view = false;
  int         stop = 5;
  std::string output = "output";
  double sigma    = 0.5;
  double sigmag   = 0.5;
  double blend    = 0.95;
  double contrast = 0.5;
  
  app.add_option("-i,--image", image_fname, "the input (grayscale) image that will be threaded.")->required();
  app.add_option("-o,--output", output, "the output base filename.")->capture_default_str();
  app.add_flag("-v,--viz", view, "displays the threading during computation.");
  app.add_option("-t,--thickness", thickness, "the thread thickness (the smaller, the finer is the result), in [0.001,1.].")->capture_default_str();
  app.add_option("-n,--nails", nb, "the number of nails on each side of the frame, in [25,500].")->capture_default_str();
  app.add_option("-z,--zomm", zoom, "the zoom factor used for the bitmap image where computation are done, in [1,16].")->capture_default_str();
  app.add_option("-p,--lpnorm", lp, "the l_p-norm used in error computations (l_2-norm is standard), in [2,16].")->capture_default_str();
  app.add_option("-c,--lpcoef", coef, "the amplifying coefficient in errors when the value is already too dark, in [1,100].")->capture_default_str();
  app.add_option("-s,--stop", stop, "stops the process and outputs the result when the PSNR has not increased for this number of iterations, in [1,oo].")->capture_default_str();
  app.add_option("--sigma", sigma, "the standard deviation for blurring the input.")->capture_default_str();
  app.add_option("--sigma-G", sigma, "the standard deviation for blurring the norm of the input gradient.")->capture_default_str();
  app.add_option("--blend", blend, "tells how much the input image is taken into account.")->capture_default_str();
  app.add_option("--contrast", contrast, "tells how much we subtract the gradient norm from the input image.")->capture_default_str();

  app.get_formatter()->column_width(30);
  CLI11_PARSE(app, argc, argv);
  // END parse command line using CLI
  
  cv::Mat image, timage;
  int width, height;
  int twidth, theight;
  image = cv::imread( image_fname, cv::IMREAD_GRAYSCALE );
  if ( !image.data )
    {
      printf("No image data \n");
      return -1;
    }
  int isigma    = int( sigma * 10);
  int isigmag   = int( sigmag * 10 );
  int iblend    = int( blend * 100 );
  int icontrast = int( contrast * 10 );
  if ( view ) {
    cv::namedWindow("Input Image",   cv::WINDOW_AUTOSIZE );
    cv::namedWindow("Average Image", cv::WINDOW_AUTOSIZE );
    cv::namedWindow("Thread Image",  cv::WINDOW_NORMAL );
    cv::imshow     ( "Input Image",  image );
    cv::createTrackbar("sigma I (1/10px)", "Input Image", &isigma, 100, NULL );
    cv::createTrackbar("sigma G (1/10px)", "Input Image", &isigmag, 100, NULL );
    cv::createTrackbar("contrast  (1/10)", "Input Image", &icontrast, 100, NULL );
    cv::createTrackbar("blend    (1/100)", "Input Image", &iblend, 100, NULL );
  }
  width   = image.cols;
  height  = image.rows;
  if ( view ) {
    cv::Mat inputf, ngradf, blurf, outputf;
    image.convertTo( outputf, CV_32FC1, 1.0/255.0);
    while ( true )
      {
	char c = cv::waitKey( 50 );
	if ( c == ' ' ) break;
	double sigma    = isigma * 0.1;
	double sigmag   = isigmag * 0.1;
	double blend    = iblend * 0.01;
	double contrast = icontrast * 0.1;
	image.convertTo( inputf, CV_32FC1, 1.0/255.0);
	cv::Mat ngrad = norm_gradient( image );
	ngrad.convertTo( ngradf, CV_32FC1, 1.0/255.0);
	outputf = (1.0-blend) + blend * blur( inputf, sigma )
	  - contrast * blur( ngradf, sigmag );
	clamp01f( outputf );
	cv::imshow( "Input Image", outputf );
      }
    outputf.convertTo( image, CV_8UC1, 255.0);
  }
  Threader threader( image, zoom, thickness, nb, 3.0 );
  threader.setError( lp, coef );
  bool  random = false;
  bool compute = true;
  std::size_t current = 0;
  Memorizer<double> error;
  const std::size_t nb_iterations = 20;
  cv::Mat target_clone;
  double      psnr           = 0.0;
  double      best_psnr      = 0.0;
  std::size_t best_iteration = 0;
  double      avg_error      = 0.0;
  
  while ( true )
    {
      char c = 0;
      if ( view ) char c = cv::waitKey( 50 );
      if ( c == 'q' ) break;
      if ( c == 'b' ) random = false;
      if ( c == 'r' ) random = true;
      if ( c == ' ' ) compute = ! compute;
      if ( compute )
	{
	  // if ( random ) current = rand() % threader._nails.size();
	  if ( random ) 
	    for ( auto k = 0; k < nb_iterations; k++ )
	      {
		current = threader.drawBestThread( rand() % threader._nails.size(),
						   thickness );
		error.add( threader._last_error );
	      }
	  else
	    for ( auto k = 0; k < nb_iterations; k++ )
	      {
		current = threader.drawBestThread( current, thickness );
		error.add( threader._last_error );
	      }		
	  //}
	}
      avg_error = error.averageLastValues( nb_iterations );
      //threader._target.convertTo( target_clone, CV_8UC3, 255.0);
      // target_clone = threader._target.clone();
      psnr = getPSNR( threader._target,threader._average );
      std::ostream& out = std::cout;
      out << "[#Size=" << threader._threads.size()
	  << " PSNR=" << std::fixed << std::setw(7) << std::setprecision(3) << psnr
	  << " Aerr=" << avg_error;
      out << "\r";
      out.flush();
      if ( psnr >= best_psnr )
	{
	  best_psnr = psnr;
	  best_iteration = threader._threads.size();
	}
      else
	{
	  if ( threader._threads.size() - best_iteration > stop )
	    {
	      out << "\nCannot increase PSNR anymore. Exiting" << std::endl;
	      break;
	    }
	}
      
      cv::cvtColor( image, target_clone, cv::COLOR_GRAY2RGB );
      std::stringstream sstr;
      sstr << "err=" << avg_error;
      std::string s;
      sstr >> s;
      cv::putText( target_clone, //target image
		   s.c_str(), //text
		   cv::Point(10, 10), //top-left position
		   cv::FONT_HERSHEY_DUPLEX,
		   0.5,
		   avg_error < 0.0 ? CV_RGB(225, 25, 25) : CV_RGB(25, 225, 25), //font color
		   2);
      std::stringstream sstr2;
      sstr2 << "psnr=" << psnr;
      std::string s2;
      sstr2 >> s2;
      cv::putText( target_clone, //target image
		   s2.c_str(), //text
		   cv::Point(10, target_clone.rows - 10), //top-left position
		   cv::FONT_HERSHEY_DUPLEX,
		   0.5,
		   CV_RGB(25, 225, 225 ),
		   2);
      if ( view )
	{
	  cv::imshow( "Input Image",  target_clone );
	  cv::imshow( "Average Image",  threader._average );
	  cv::imshow( "Thread Image", threader._timage );
	}
    }

  cv::imwrite( "input.png", image );
  cv::imwrite( output + "-target.png", target_clone );
  cv::Mat avg, thr;
  threader._average.convertTo( avg, CV_8UC1, 255.0);
  threader._timage.convertTo ( thr, CV_8UC1, 255.0);
  cv::imwrite( output + "-avg.png",    avg );
  cv::imwrite( output + "-thread.png", thr );
  std::ofstream f( (output + ".svg").c_str() );
  threader.exportAsSVG( f );
  f.close();
  std::cout << "last-PSNR=" << psnr << " last-error=" << avg_error << std::endl;
  return 0;
}
