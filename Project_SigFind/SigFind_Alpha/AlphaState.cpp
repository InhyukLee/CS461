/****************************************************************************
Program description: This is main cpp file for our SigFind program. It takes
pnd file name that needs to be processed from the bash script. Then, it pass
the path to OCR function in OCR_SigLoc.cpp. After it gets result from OCR
function, it pass the sigloc struct to function forwardPass in ConvNet,cpp.

Input: This program can only process one png file at a time, so the argument must be the
path of pdf file.

Output: forwardPass will detect signatue from signature box. If there is signature,
forward pass will color the signature box as green, and if there is no signature, it
will color the signatue box as red. The result will be saved in contains_signature
directory.
*****************************************************************************/

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

   //Process OCR function to get file paths and locations of the signature box.
   result_images = OCR(argc,argv);  
   if(result_images.Sig_Paths.size()==0){
	   return -1;
   }
   
   //Process forwardPass function to detect signature and save the result in contains_signature directory
   forwardPass(result_images,argv[1]);

   return 0;    
}
