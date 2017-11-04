
WEBSITES TO CONSIDER WHILE INSTALLING:

-Linux OpenCV install : <https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html>
-Using OpenCV with gcc and Cmake : <https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_gcc_cmake/linux_gcc_cmake.html>

STEP 1: Transfer OpenCV files into putty
Unzip the the opencv.zip file I emailed you guys and create a folder in putty where they can reside

STEP 2: Build OpenCV from source using cmake
$cd <myOpenCVDirectory>
$mkdir release
$cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..

If above statement does not work:
cake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local ..

*You may not be able to install from relase and must step back into <myOpenCVDirectory>

