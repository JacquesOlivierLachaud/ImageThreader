# ImageThreader

This program aims at rendering an image with a long black thread joining nails surrounding the image frame. It mimicks an artistic view of an image with threads.

<table>
<tr>
<td>Input image</td>
<td>Threaded image</td>
</tr>
<tr>
<td> <image src="Input/closeup-face.png"> </td>
<td> <image src="Examples/closeup-face.svg"> </td>
</tr>
</table>

## Compiling instruction

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

