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

#include "SigLoc_Struct.h"

int main(int argc, char** argv){
   if(argc <2){
      cout << "Need png file" << endl;
      return -1;
   }

   sigloc result_images;

   result_images = OCR(argc,argv);  
   if(result_images.Sig_Paths.size()==0){
	   return -1;
   }
   /*
   for(int i=0;i<result_images.Sig_Paths.size();i++){
      cout << result_images.Sig_Paths.at(i) << endl;
	  cout << result_images.Sig_coordinates.at(i) << endl;
   }
   */
   forwardPass(result_images,argv[1]);

   return 0;    
}
