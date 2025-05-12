# ImageThreader

This program aims at rendering an image with a long black thread joining nails surrounding the image frame. It mimicks an artistic view of an image with threads.

<table>
<tr>
<td>Input image</td>
<td>Threaded image</td>
</tr>
<tr>
<td> <image src="Input/closeup-face.png" width=200> </td>
<td> <image src="Examples/closeup-face-avg.png" width=200> </td>
</tr>
<tr>
<td></td>
<td>
<a href="Examples/closeup-face-target.png">Info</a> |
<a href="Examples/closeup-face-thread.png">Zoomed</a> |
<a href="Examples/closeup-face.svg">SVG</a>
</td>
</tr>
<tr>
<td> <image src="Input/two-women.png" width=200> </td>
<td> <image src="Examples/two-women-avg.png" width=200> </td>
</tr>
<tr>
<td></td>
<td>
<a href="Examples/two-women-target.png">Info</a> |
<a href="Examples/two-women-thread.png">Zoomed</a> |
<a href="Examples/two-women.svg">SVG</a>
</td>
<tr>
<td> <image src="Input/elegant-woman.png" width=200> </td>
<td> <image src="Examples/elegant-woman-avg.png" width=200> </td>
</tr>
<tr>
<td></td>
<td>
<a href="Examples/elegant-woman-target.png">Info</a> |
<a href="Examples/elegant-woman-thread.png">Zoomed</a> |
<a href="Examples/elegant-woman.svg">SVG</a>
</td>
</tr>
<tr>
<td> <image src="Input/side-face.png" width=200> </td>
<td> <image src="Examples/side-face-t0_005-avg.png" width=200> </td>
</tr>
<tr>
<td></td>
<td>
<a href="Examples/side-face-t0_005-target.png">Info</a> |
<a href="Examples/side-face-t0_005-thread.png">Zoomed</a> |
<a href="Examples/side-face-t0_005.svg">SVG</a>
</td>
</tr>
</table>

All these examples were produced with these parameters;
```
./thread -i INPUT -z 4 -t 0.025 -n 150 -p 2.0 -c 2.0 -o OUTPUT
```

## Compiling instructions

Requirement:
- [OpenCV](https://opencv.org) (tested with OpenCV4, but should work with older version)
- [cmake](https://cmake.org) (tested with cmake 3.30.4,, but should work with older version)

After cloning the directory, just type in these commands:

```
cd ImageThreader
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

You may check that the program work with:
```
./thread -i ../Input/side-face.png -z 4 -t 0.05 -n 100 -p 2.0 -c 2.0 -o "side-face"
open side-face.svg
```

## User instructions

The program can be runned without interface or with an interface (with
option `-v` or `--view`). The advantage of the interface version is
that you can modify a little bit the input image before processing (by
smoothing it and forcing the contrast around strong contours). When
you are finished with changing the input, just press ' ' (spacebar) to
let the threader weaves the thread around the nails.

Just type `./thread --help` to get all the possible parameters.

```
Draws a grayscale image with a (very long) thread.

Usage: ./thread [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -i,--image TEXT REQUIRED    the input (grayscale) image that will be threaded.
  -o,--output TEXT=output     the output base filename.
  -v,--viz                    displays the threading during computation.
  -t,--thickness FLOAT=0.02   the thread thickness (the smaller, the finer is the result), in [0.001,1.].
  -n,--nails INT=80           the number of nails on each side of the frame, in [25,500].
  -z,--zomm INT=4             the zoom factor used for the bitmap image where computation are done, in [1,16].
  -p,--lpnorm FLOAT=2         the l_p-norm used in error computations (l_2-norm is standard), in [2,16].
  -c,--lpcoef FLOAT=2         the amplifying coefficient in errors when the value is already too dark, in [1,100].
  -s,--stop INT=5             stops the process and outputs the result when the PSNR has not increased for this number of iterations, in [1,oo].
  --sigma FLOAT=0.5           the standard deviation for blurring the input.
  --sigma-G FLOAT=0.5         the standard deviation for blurring the norm of the input gradient.
  --blend FLOAT=0.95          tells how much the input image is taken into account.
  --contrast FLOAT=0.5        tells how much we subtract the gradient norm from the input image.
```

## Credits

Idea is from Guglielmo Fadabini. The code is written by Jacques-Olivier Lachaud.
This is just a demo tool and a work in progress, and hopefully it can be enhanced.