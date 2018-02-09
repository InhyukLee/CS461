/*
 * OCR_tesseract.cpp
 *
 *  Created on: Jan 24, 2018
 *      Author: inhyuk
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <tesseract/baseapi.h>
#include <tesseract/strngs.h>
#include <leptonica/allheaders.h>

using namespace cv;

int main(int argc, char** argv) {

   if (argc < 2) {
      std::cout << "empty img_path" << std::endl;
      return -1;
   }

   //read image
   Mat image;
   image = imread( argv[1], 1 );

   if( argc != 2 || !image.data )
      {
         printf( "No image data \n" );
         return -1;
      }

   //display image
   namedWindow( "Display Image", WINDOW_AUTOSIZE );
   imshow( "Display Image", image );

   
   //set langueage and image
   const char* lan = "eng";
   const char* img_path = argv[1];

   //build tesseract API
   tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();

   //display version of tesseract and leptonica
   std::cout << "tesseract-ocr-" << ocr->Version() << std::endl;
   std::cout << getLeptonicaVersion() << std::endl;

   if (ocr->Init(NULL, lan, tesseract::OEM_DEFAULT)) {
      std::cout << "tesseract-ocr initialize error" << std::endl;
      return -1;
   }


   FILE* in = fopen(img_path, "rb");
   if (in == NULL) {
      std::cout << "can not open > " << img_path << std::endl;
      return -1;
   }
   fclose(in);

   //read image
   Pix *img = pixRead(img_path);
   ocr->SetImage(img);
   ocr->SetSourceResolution(70);
   
   //convert image to text
   char *text;
   text = ocr->GetUTF8Text();
   std::cout << text << std::endl;
   
   //free the spaces
   delete [] text;
   pixDestroy(&img);
   ocr->End();

   //wait to close the display image
   waitKey(0);
   return 0;
}
