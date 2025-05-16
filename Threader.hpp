#pragma once

// We use opencv for display and image representation and i/o
#include <opencv2/opencv.hpp>

// opencv does not provide a comparator for points.
struct PointComparator {
  bool
  operator()( const cv::Point2i& p, const cv::Point2i& q ) const
  {
    return ( p.x < q.x ) || ( ( p.x == q.x ) && ( p.y < q.y ) );
  }
};

/// Draws a grayscale image as a long black thread.
///
/// The coordinate system start from (0,0) in the upper left corner to
/// (w,h) in the lower right corner. Input image pixels have width and
/// height 1, so the last pixel correspond to the domain (w-1,h-1) to
/// (w,h). Sometimes you have negative coordinates because the
/// thread(s) go(es) around nails at distance bw from the side of the
/// image. 
///
/// In the rectangle frame, there are nb+1 nails per side.
struct Threader {
  typedef cv::Point2d RealPoint;
  typedef cv::Point2i LatticePoint;
  typedef std::size_t Index;
  typedef std::pair< LatticePoint, float > PixelFraction;
  
  cv::Mat _input;  ///< input image (gray scale)
  cv::Mat _target; ///< target image (same as input but as a float image)
  int     _width;  ///< input width
  int     _height; ///< input height
  int     _zoom;   ///< zoom factor for computations
  cv::Mat _average;///< average image (current average of _timage, same size as _input
  cv::Mat _timage; ///< thread image
  int     _twidth; ///< width of thread image
  int     _theight;///< height of thread image
  double  _tarea;  ///< area of pixel in _timage
  double  _thickness; ///< thread thickness
  double  _expand; ///< (_zoom+1)/_zoom
  double  _last_error; ///< last error
  double  _lp_error; ///< type of error
  double  _error_coef; ///< coefficient of error if too dark
  std::vector< RealPoint > _nails; ///< the list of nails.
  /// for each nail, gives the threadable nails
  std::vector< std::vector< Index > > _threadables; 
  std::vector< std::pair<Index,Index> > _threads; ///< drawn threads 

  std::vector<double> _asin; ///< asin precomputation
  
  Threader( cv::Mat input, int zoom, double thickness, int nb, double border_width )
  {
    initRectangleFrame( input, zoom, thickness, nb, border_width );
  }
  void setError( double lp, double coef )
  {
    _lp_error   = lp;
    _error_coef = coef;
  }
  
  void initRectangleFrame( cv::Mat input, int zoom, double thickness,
			   int nb, double border_width )
  {
    _input   = input;
    _input.convertTo( _target, CV_32FC1, 1.0/255.0);
    _width   = input.cols;
    _height  = input.rows;
    _twidth  = zoom * _width;
    _theight = zoom * _height;
    _zoom    = zoom;
    _expand  = (zoom+sqrt(2.0))/zoom;
    _tarea   = 1.0 / double(zoom*zoom);
    _timage  = cv::Mat( _theight, _twidth, CV_32FC1, 1.0 );
    _average = cv::Mat( _height,  _width,  CV_32FC1, 1.0 );
    _thickness = thickness;
    const int n = 256;
    _asin.resize( 2*n+1 );
    for ( auto i = -n; i <= n; i++ ) _asin[ i+n ] = asin( double(i)/double(n) );
    // compute nails, nb per side at distance
    double rnb = nb;
    _nails.clear();
    for ( auto k = 0; k <= nb; k++ )
      {
	double x = ( double(_width)  * k ) / rnb;
	double y = ( double(_height) * k ) / rnb;
	_nails.push_back( RealPoint( x, -border_width ) );
	_nails.push_back( RealPoint( x, double(_height)+border_width ) );
	_nails.push_back( RealPoint( -border_width, y ) );
	_nails.push_back( RealPoint( double(_width)+border_width, y ) );
      }
    computeThreadableNails();
  }

  // precomputed asin
  double myArcSin( double t ) const
  {
    int i = round( t * 256 ) + 256;
    if ( i < 0 ) return _asin[ 0 ];
    else if ( i > 512 ) return _asin[ 512 ];
    return _asin[ i ];
  }
  
  void computeThreadableNails()
  {
    const double eps = 1e-6;
    const Index    n = _nails.size();
    _threadables     = std::vector< std::vector< Index > >( n );
    Index         nt = 0;
    Index   min_size = ( _width + _height ) / 4.0;
    for ( auto k = 0; k < n; ++k )
      for ( auto l = k+1; l < n; ++l )
	{
	  const RealPoint pq = _nails[ l ] - _nails[ k ];
	  if ( ( ( fabs( pq.x ) > eps ) && ( fabs( pq.y ) > eps ) )
	       || ( ( fabs( pq.x ) <= eps ) && ( fabs( pq.y ) > _height ) )
	       || ( ( fabs( pq.y ) <= eps ) && ( fabs( pq.x ) > _width  ) ) )
	    {
	      auto TP = getThreadPixels( _nails[ k ], _nails[ l ], 1./_zoom );
	      auto IP = getImagePixels( TP );
	      if ( IP.size() >= min_size ) 
		{
		  _threadables[ k ].push_back( l );
		  _threadables[ l ].push_back( k );
		  nt += 1;
		}
	    }
	}
    std::cout << "There are " << nt << " threadable nail pairs" << std::endl;
  }

  /// Main method. Given a starting index, find the best thread, draws
  /// it, and returns the new starting point.
  Index drawBestThread( Index start, double w, bool enhancement = true )
  {
    if ( start >= _nails.size() ) return start;
    // Compute error for each possible thread from start.
    Index  best_k     = _nails.size();
    double best_error = std::numeric_limits<double>::infinity();
    const RealPoint   p = _nails[ start ];
    //Assume A is a given vector with N elements
    std::vector< double > E( _threadables[ start ].size() );
    for ( auto i = 0; i < _threadables[ start ].size(); i++ )
      {
	const Index      k = _threadables[ start ][ i ];
	const RealPoint  q = _nails[ k ];
	std::vector< PixelFraction > TP = getThreadPixels( p, q, w );
	std::vector< PixelFraction > IP = getImagePixels( TP );
	const double error = enhancement
	  ? getEnhancement( IP )
	  : getError( IP );
	// const RealPoint   pq = q - p;
	E[ i ]             = error; // / sqrt( pq.ddot( pq ) );
      }
    std::vector< Index > V( E.size() );
    std::iota( V.begin(), V.end(), 0 ); //Initializing
    std::sort( V.begin(), V.end(), [&](int i,int j){return E[i]<E[j];} );
    Index best_i = E.size() - 1;
    // Use a little bit of randomness
    for ( auto i = 0; i < E.size(); i++ )
      if ( rand() % 2 == 0 )
	{ best_i = i; break; }
    best_error  = E[ V[ best_i ] ];
    _last_error = best_error;
    best_k = _threadables[ start ][ V[ best_i ] ];
    // Update all images
    Index      next = best_k;
    const RealPoint q = _nails[ next ];
    std::vector< PixelFraction > TP = getThreadPixels( p, q, w );
    std::vector< PixelFraction > IP = getImagePixels( TP );
    for ( const auto& pf : TP )
      _timage.at<float>( pf.first.y, pf.first.x ) =
	std::max( 0.0f, _timage.at<float>( pf.first.y, pf.first.x ) - pf.second );
    updateAverageImage( IP );
    // Remove choice
    for ( auto i = 0; i < _threadables[ start ].size(); i++ )
      if ( _threadables[ start ][ i ] == next )
	{
	  std::swap( _threadables[ start ][ i ], _threadables[ start ].back() );
	  _threadables[ start ].pop_back();
	  break;
	}
    _threads.push_back( std::make_pair( start, next ) );
    return next;
  }
  
  std::pair< RealPoint, RealPoint > randomThread() const
  {
    Index k = rand() % _threadables.size();
    Index l = rand() % _threadables[ k ].size();
    return std::make_pair( _nails[ k ], _nails[ _threadables[k][l] ] );
  }
  
  void stupidDraw( RealPoint p, RealPoint q, double w )
  {
    std::chrono::high_resolution_clock::time_point
      t1 = std::chrono::high_resolution_clock::now();
    for ( auto y = 0; y < _theight; y++ )
      for ( auto x = 0; x < _twidth; x++ )
	{
	  const RealPoint c = t2r( LatticePoint( x, y ) );
	  if ( ! isInsideBox( p, q, c ) ) continue;
	  const float   f = areaThread( c, _tarea, p, q, w );
	  if ( f > 0.0 )
	    _timage.at<float>( y, x ) = std::max( 0.0f, _timage.at<float>( y, x ) - f );
	}
    std::chrono::high_resolution_clock::time_point
      t2 = std::chrono::high_resolution_clock::now();
    double t = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    std::cout << "in " << int(round(t*1e-3)) << " mus.\n";
  }

  void smartDraw( RealPoint p, RealPoint q, double w )
  {
    std::chrono::high_resolution_clock::time_point
      t1 = std::chrono::high_resolution_clock::now();
    std::vector< PixelFraction > R = getThreadPixels( p, q, w);
    for ( const auto& pf : R )
      {
	_timage.at<float>( pf.first.y, pf.first.x ) =
	  std::max( 0.0f, _timage.at<float>( pf.first.y, pf.first.x ) - pf.second );
      }
    std::chrono::high_resolution_clock::time_point
      t2 = std::chrono::high_resolution_clock::now();
    double t = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    std::cout << "in " << int(round(t*1e-3)) << " mys.\n";
  }

  float localError( float av, float iv ) const
  {
    float dv = fabs( av - iv );
    float e  = pow( dv, _lp_error );
    return ( av > iv ) ? e : 2.0 * e;
  }
  
  /// Compare the updated average image with the input image along \a ipixels.
  double getError( const std::vector< PixelFraction >& ipixels ) const
  {
    double error = 0.0;
    double    nb = 0.0; 
    for ( const auto& pf : ipixels )
      {
	const LatticePoint p = pf.first;
	const float   f = pf.second;
	const float  uv = std::max( 0.0f, _average.at<float>( p.y, p.x ) - f );
	const float  iv = _target .at<float>( p.y, p.x );
	const double dv = fabs(uv-iv);
	error += localError( uv, iv );
	nb    += 1.0;
      }
    return error;
  }
  /// Compare the updated average image with the input image along \a ipixels.
  double getEnhancement( const std::vector< PixelFraction >& ipixels ) const
  {
    double error_before = 0.0;
    double error_after  = 0.0;
    double    nb = 0.0; 
    for ( const auto& pf : ipixels )
      {
	const LatticePoint p = pf.first;
	const float   f = pf.second;
	const float  av = _average.at<float>( p.y, p.x );
	const float  uv = std::max( 0.0f, _average.at<float>( p.y, p.x ) - f );
	const float  iv = _target .at<float>( p.y, p.x );
	error_before   += localError( av, iv );
	error_after    += localError( uv, iv );
      }
    return error_after - error_before;
  }

  /// Once a thread line has been chosen, update the average.
  void updateAverageImage( const std::vector< PixelFraction >& ipixels )
  {
    for ( const auto& pf : ipixels )
      {
	const LatticePoint p = pf.first;
        const float   f = pf.second;
        _average.at<float>( p.y, p.x ) =
	  std::max( 0.0f, _average.at<float>( p.y, p.x ) - f );
      }
  }
  /// @return the image pixels traversed by the thread pixels (and the area fraction).
  std::vector< PixelFraction >
  getImagePixels( const std::vector< PixelFraction >& tpixels ) const
  {
    std::map< LatticePoint, float, PointComparator > M;
    for ( const auto& pf : tpixels )
      {
	const LatticePoint tp = pf.first;
	const LatticePoint ip = t2i( tp );
	const auto    it = M.find( ip );
	const float    f = pf.second;
	const float    v = _timage.at<float>( tp.y, tp.x );
	const float   ff = std::max( 0.0f, std::min( f, v ) );
	if ( it == M.end() ) M[ ip ]     = ff;
	else                 it->second += ff;
      }
    std::vector< PixelFraction > P;
    P.reserve( M.size() );
    for ( const auto& q : M )
      P.push_back( { q.first, q.second * _tarea } );
    return P;
  }

  
  /// @return the pixels traversed by the straight segment [pq] of width w
  std::vector< PixelFraction >
  getThreadPixels( RealPoint p, RealPoint q, double w ) const
  {
    RealPoint pq = q - p;
    return ( fabs( pq.x ) >= fabs( pq.y ) )
      ? getThreadPixelsDrawnAlongX( p, q, w )
      : getThreadPixelsDrawnAlongY( p, q, w );
  }
  
  /// @return the pixels traversed by the straight segment [pq] of width w
  std::vector< PixelFraction >
  getThreadPixelsDrawnAlongX( RealPoint p, RealPoint q, double w ) const
  {
    if ( q.x < p.x ) std::swap( p, q );
    std::vector< PixelFraction > R;
    // Anticipate the number of pixels
    RealPoint pq = q - p;
    // needs to be slightly expanded since the area is computed as if
    // pixels have no width.
    double total_area = sqrt( pq.ddot( pq ) ) * w * _expand;
    int    nb_pixels  = ceil( total_area * _zoom * _zoom );
    R.reserve( nb_pixels );
    const auto f_x = [&] ( double x ) -> double
    { return p.y + ( x - p.x ) * pq.y / pq.x; };
    LatticePoint tp = r2t( p );
    LatticePoint tq = r2t( q );
    int   left = std::max( 0, tp.x );
    int   right= std::min( _twidth - 1, tq.x );
    int    nbv = 0;
    for ( int ix = left; ix <= right; ix++ )
      {
	double mx = t2r( ix );
	double my = f_x( mx );
	int    ly = std::max( 0, r2t( my - w ) );
	int    uy = std::min( _theight-1, r2t( my + w ) );
	for ( int iy = ly; iy <= uy; iy++ )
	  {
	    nbv += 1;
	    const LatticePoint ic = LatticePoint( ix, iy );
	    const RealPoint  c = t2r( ic );
	    const auto     f = areaThread( c, _tarea, p, q, w );
	    if ( f > 0.0 ) R.push_back( { ic, f } );
	  }
      }
    // std::cout << "#reserved=" << nb_pixels << " #R=" << R.size()
    // 	      << " #V=" << nbv << std::endl;
    return R;
  }

  /// @return the pixels traversed by the straight segment [pq] of width w
  std::vector< PixelFraction >
  getThreadPixelsDrawnAlongY( RealPoint p, RealPoint q, double w ) const
  {
    if ( q.y < p.y ) std::swap( p, q );
    std::vector< PixelFraction > R;
    // Anticipate the number of pixels
    RealPoint pq = q - p;
    // needs to be slightly expanded since the area is computed as if
    // pixels have no width.
    double total_area = sqrt( pq.ddot( pq ) ) * w * _expand;
    int    nb_pixels  = ceil( total_area * _zoom * _zoom );
    R.reserve( nb_pixels );
    const auto f_y = [&] ( double y ) -> double
    { return p.x + ( y - p.y ) * pq.x / pq.y; };
    LatticePoint tp = r2t( p );
    LatticePoint tq = r2t( q );
    int   lower= std::max( 0, tp.y ); 
    int   upper= std::min( _theight - 1, tq.y );
    int    nbv = 0;
    for ( int iy = lower; iy <= upper; iy++ )
      {
	double my = t2r( iy );
	double mx = f_y( my );
	int    lx = std::max( 0, r2t( mx - w ) );
	int    ux = std::min( _twidth-1, r2t( mx + w ) );
	for ( int ix = lx; ix <= ux; ix++ )
	  {
	    nbv += 1;
	    const LatticePoint ic = LatticePoint( ix, iy );
	    const RealPoint  c = t2r( ic );
	    const auto     f = areaThread( c, _tarea, p, q, w );
	    if ( f > 0.0 ) R.push_back( { ic, f } );
	  }
      }
    // std::cout << "#reserved=" << nb_pixels << " #R=" << R.size()
    // 	      << " #V=" << nbv << std::endl;
    return R;
  }

  // ------------------------------ coordinates services ------------------------------
public:
  /// thread image int coordinate --> input image int coordinate
  double t2i( int t ) const             { return t / _zoom; }
  /// thread image LatticePoint --> input image LatticePoint
  LatticePoint t2i( const LatticePoint& t ) const { return t / _zoom; }
  /// thread image int coordinate --> real coordinate
  double t2r( int t ) const             { return (double(t)+0.5) / _zoom; }
  /// thread image LatticePoint --> RealPoint coordinates
  RealPoint t2r( const LatticePoint& t ) const { return RealPoint( t2r( t.x ), t2r( t.y ) ); }
  /// real coordinate --> thread image int coordinate
  int r2t( double r ) const             { return int( round( _zoom * r - 0.5 ) ); }
  /// RealPoint coordinates --> thread image LatticePoint
  LatticePoint r2t( const RealPoint& r ) const { return LatticePoint( r2t( r.x ), r2t( r.y ) ); }

  
  // ---------------------- geometry services ---------------------------------------
public:
  /// @return 'true' iff [p1,p2] \cap [p3,p4] is not empty
  static bool isSegmentIntersection( RealPoint p1, RealPoint p2,
				     RealPoint p3, RealPoint p4 ) 
  {
    const double d3 = orientation( p1, p2, p3 );
    const double d4 = orientation( p1, p2, p4 );
    const double d1 = orientation( p3, p4, p1 );
    const double d2 = orientation( p3, p4, p2 );
    if ( ( ( ( d3 > 0.0 ) && ( d4 < 0.0 ) )
	   || ( ( d3 < 0.0 ) && ( d4 > 0.0 ) ) )
	 && ( ( ( d1 > 0.0 ) && ( d2 < 0.0 ) )
	      || ( ( d1 < 0.0 ) && ( d2 > 0.0 ) ) ) )
      return true;
    else if ( d3 == 0.0 ) return isInsideBox( p1, p2, p3 );
    else if ( d4 == 0.0 ) return isInsideBox( p1, p2, p4 );
    else if ( d1 == 0.0 ) return isInsideBox( p3, p4, p1 );
    else if ( d2 == 0.0 ) return isInsideBox( p3, p4, p2 );
    return false;
  }
  
  /// Given a pixel centered on c and given area tells the approximate area fraction of the
  /// intersection of this pixel with the thread from to p to q
  /// and width w.
  double areaThread( RealPoint c, double area,
		     RealPoint p, RealPoint q, double w ) const
  {
    // // We cheat a little bit by computing the intersection of [pq]
    // // with the dilated rectangle [l,u] \oplus [-f*w,f*w]^2.
    // static const double f = 0.5 / sqrt(2.0);
    // const double   fw = f*w;
    // const RealPoint wll = RealPoint( l.x-fw, l.y-fw );
    // const RealPoint wlr = RealPoint( u.x+fw, l.y-fw );
    // const RealPoint wul = RealPoint( l.x-fw, u.y+fw );
    // const RealPoint wur = RealPoint( u.x+fw, u.y+fw );
    
    // // Checks the intersection with the left side.
    // bool left  = isSegmentIntersection( wll, wul, p, q ); //< check with left side
    // bool right = isSegmentIntersection( wlr, wur, p, q ); //< check with right side
    // bool lower = isSegmentIntersection( wll, wlr, p, q ); //< check with lower side
    // bool upper = isSegmentIntersection( wul, wur, p, q ); //< check with upper side
    // int  nbi   = (left?1:0)+(right?1:0)+(lower?1:0)+(upper?1:0);
    // if ( nbi == 1 ) std::cout << "Weird intersection" << std::endl;

    // Rectangle should be a square.
    double  r    = sqrt( area / M_PI );
    RealPoint pq   = q-p;
    RealPoint n( -pq.y, pq.x );
    RealPoint nn   = n / sqrt( n.ddot( n ) );
    double  a    =  std::min(  p.ddot( nn ) + 0.5*w - c.ddot( nn ), r );
    double  b    = -std::min( -p.ddot( nn ) + 0.5*w + c.ddot( nn ), r );
    if ( a <= -r ) return 0.0;
    if ( b >=  r ) return 0.0;
    if ( b < a ) std::swap( a, b );
    double  r2   = r*r;
    // gives the area of the band intersected with a disk centered on c and radius r.
    // double cl_ar = std::min( 1.0, std::max( a/r, -1.0 ) );
    // double cl_br = std::min( 1.0, std::max( b/r, -1.0 ) );
    // double  A    = -r2*asin( cl_ar ) + r2*asin( cl_br )
    //   - sqrt( std::max( r2 - a*a, 0.0 ) )*a
    //   + sqrt( std::max( r2 - b*b, 0.0 ) )*b;
    double  A    = -r2*myArcSin( a/r ) + r2*myArcSin( b/r )
      - sqrt( std::max( r2 - a*a, 0.0 ) )*a
      + sqrt( std::max( r2 - b*b, 0.0 ) )*b;
    // std::cout << " " << -r << " <= " << a << " <= " << b << " <= " << r
    // 	      << " f=" << ( A / area ) << std::endl;
    return  std::min( 1.0, A / area );
  }

  static double det( const RealPoint& u, const RealPoint& v ) 
  {
    return u.x * v.y - u.y * v.x;
  }

  /// @return a positive number is r is to the left of [pq], 0.0 on
  /// the line, negative otherwise.
  static double orientation ( const RealPoint& p, const RealPoint& q, const RealPoint& r ) 
  {
    const RealPoint u = q - p;
    const RealPoint v = r - p;
    return det( u, v );
  }

  /// @return 'true' iff p3 is inside the box p1,p2
  static bool isInsideBox( RealPoint p1, RealPoint p2, RealPoint p3 )
  {
    double minx = std::min( p1.x, p2.x );
    double miny = std::min( p1.y, p2.y );
    double maxx = std::max( p1.x, p2.x );
    double maxy = std::max( p1.y, p2.y );
    return ( minx <= p3.x ) && ( p3.x <= maxx )
      &&   ( miny <= p3.y ) && ( p3.y <= maxy );
  }

  void exportAsSVG( std::ostream& out )
  {
    if ( _threads.empty() ) return; 
    RealPoint l = _nails[ 0 ];
    RealPoint u = _nails[ 0 ];
    for ( auto p : _nails ) {
      if ( p.x < l.x ) l.x = p.x;
      if ( p.y < l.y ) l.y = p.y;
      if ( p.x > u.x ) u.x = p.x;
      if ( p.y > u.y ) u.y = p.y;
    }
    double w = u.x - l.x;
    double h = u.y - l.y;
    double r = w / h;
    // format is 20cm
    const double format = 20.0;
    // resolution is 4096
    const double resolution = 4096;
    double rw, rh;
    if ( w > h ) {
      rw = format;
      rh = rw / r;
    }
    else {
      rh = format;
      rw = rh * r;
    }
    out << "<?xml version=\"1.0\" standalone=\"no\"?>\n"
      // << "<svg width=\"" << rw << "cm\" height=\"" << rh << "cm\""
	<< "<svg width=\"" << w << "\" height=\"" << h << "\""
	<< " viewBox=\"" << l.x << " " << l.y << " " << (w+l.x) << " " << (h+l.y) << "\""
	<< " xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n"
	<< "<desc>Thread image</desc>\n";
    out << "<g stroke=\"black\" stroke-width=\"" << _thickness << "\">\n";

    Index previous = _threads[ 0 ].first;
    for ( auto t : _threads )
      {
	out << "<line x1=\"" << _nails[ t.first ].x << "\"";
	out << " y1=\"" << _nails[ t.first ].y << "\"";
	out << " x2=\"" << _nails[ t.second ].x << "\"";
	out << " y2=\"" << _nails[ t.second ].y << "\" />\n";
      }
    out << "</g>\n";
    out << "</svg>\n";
  }
  
};
