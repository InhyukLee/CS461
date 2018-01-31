/////////////////////////////////////////////////////////
This is the architecture I have so far regarding the Convolutional Neural Network
There's no need for you guys to fully understand each method, but I do recommend taking a look at the main method
since it calls all the different parts of a CNN in the correct order.
To run this code:
--->Download ConvNet.cpp
--->Download CMakeFiles.txt
///** MAKE SURE TO PLACE THESE IN THE SAME DIRECTORY AS YOUR RUNNING VERSION OF OPENCV **///
--->Run the following commands:
    ---> cmake .
    ---> make
Once this is done you can run the ConvNet by invoking:
    ---> ./ConvNet
The first thing the program will ask you is how many iterations you would like to run, try 100 so it generates some files.
It will generate files w0.txt - w7.txt, but these will be far from optimized. The current cost value I achieved is .15, which is very good. 
Visualizing these files is the next step and requires a couple more programs, but just by running the neural net
you can see the cost function value decreasing, meaning its getting better at identifying the data (just pictures of
handwritten data at this point). 
I will post the optimized Kernel files as well as pictures in the morning after I run the net all night again so that
we can show the TA and Chris. Next step is to aquire as many signatures as possible so lets see what we can do about that. 

Cheers. 
Juan
