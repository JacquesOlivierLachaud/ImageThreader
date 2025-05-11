# ImageThreader

Artistic view of an image with threads

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
./thread
```

