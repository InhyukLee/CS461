
WEBSITES TO CONSIDER WHILE INSTALLING:

-Linux OpenCV install : <https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html>
-Using OpenCV with gcc and Cmake : <https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_gcc_cmake/linux_gcc_cmake.html>

*If you do utilize these resources, ignore any sudo, su -, prompts the websites say to execute. the sudo or  su - commands
prompt for the user to act as root which we are not allowed to do as students on OSU's network. *

STEP 1: Transfer OpenCV files into putty
Unzip the the opencv.zip file I emailed you guys and create a folder in putty where they can reside

STEP 2: Build OpenCV from source using cmake
$cd <myOpenCVDirectory>
$mkdir release
$cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..

If above statement does not work:
cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local ..

*You may not be able to install from the release folder and must step back into <myOpenCVDirectory>*
THEN
$make

STEP 3
Add to your directory the files DisplayImage.cpp and CMakeLists.txt

STEP 4
while in the directory:
$cmake .
$make
./DisplayImage bellman.jpg

If everything up to this point is correct, you should get the following error message

(DisplayImage:******) Gtk-WARNING** : could not open display

This is because the gtk software, which is part of OpenCV, utilizes a graphical interface called Gtk+ which is based on a
framework called X11, which in turns needs its own display server. This server for windows is called Xming 
(Extremely confusing i know)

STEP 4 Install Xming
I provided the program Xming.exe in github. This must always be running when working on the project. 
NOTE: it might not show up as an icon once it is running, check task manager to ensure it actually is.

STEP 5 configure putty
On the left side of the Putty configuration window( the main one that pops up when its started)
  ->SSH
    ->X11
      ->Enable X11 Forwarding (CHECK)
      ->X display location = :0.0
      ->MIT-Magic-Cookie-1 (CHECK)
      ->X authority file for local display
        ->browse to Xming.exe and select it
  ~~~~ MAKE SURE TO SAVE SESSION ~~~~
  
STEP 6: Putting it all together:
Follow these last steps:

Xming.exe -> run
putty.exe -> run
(Check settings one last time and make sure they are the same as the aforementioned ones.)
$cd <myOpenCVDirectory>
$./DisplayImage bellman.jpg (or any other picture you have saved in this directory)

This should then pop up an xming window containing the picture selected.

*if you get a similar warning as the one above, (Gtk-WARNING) saying that the connection was refused, 
look over your X11 settings, that is where the problem resides.*

