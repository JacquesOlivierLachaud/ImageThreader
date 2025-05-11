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
</table>

All these examples were produced with these parameters;
```
./thread -i INPUT -z 4 -t 0.025 -n 150 -p 2.0 -c 2.0 -o OUTPUT
```

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

