#include <stdio.h>
#include <string.h>
#include <iostream>
#include <tesseract/baseapi.h>
#include <tesseract/strngs.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <fstream>
#include <stdlib.h>
#include "OCR_SigLoc.cpp"
#include "ConvNet.cpp"
using namespace std;
using namespace cv;


int main(int argc, char** argv){
   if(argc <2){
      cout << "Need png file" << endl;
      return -1;
   }

   string result_image;

   result_image = runMain(argv[1]);  
   RunNetwork(result_image);

   return 0;    
}
