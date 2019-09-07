# Sample code 

## windows

### How to build

```cmake
$ git clone https://github.com/garethwang/LPM_OpenCV.git
$ cd LPM_OpenCV
$ mkdir build
$ cd build
$ cmake-gui ../
```

- Click  on Configure to process CMakeLists.txt
- Set the OpenCV_DIR to find OpenCV library.
- Click on Configure again.
- Click on Generate.
- Close the cmake-gui.

```cmake
$ cd ..
$ cmake --build build
```

### How to run

```cmake
$ ./build/Debug/demo_lpm.exe ./data/matches.txt ./data/retina0.jpg ./data/retina1.jpg
```

## Result

![](https://github.com/garethwang/LPM_OpenCV/blob/master/data/result.jpg)

