#include <stdio.h>
#include <string.h>
#include <iostream>
#include <tesseract/baseapi.h>
#include <tesseract/strngs.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <vector>

using namespace cv;
using namespace std;

void reverse(char s[]){
   int i, j;
   char c;

   for (i = 0, j = strlen(s)-1; i<j; i++, j--) {
      c = s[i];
      s[i] = s[j];
      s[j] = c;
   }
}

void itoa(int n, char s[]){
   int i, sign;

   if ((sign = n) < 0)  /* record sign */
      n = -n;          /* make n positive */
   i = 0;
   do {       /* generate digits in reverse order */
      s[i++] = n % 10 + '0';   /* get next digit */
   } while ((n /= 10) > 0);     /* delete it */
   if (sign < 0)
      s[i++] = '-';
   s[i] = '\0';
   reverse(s);
}  

int main(int argc, char** argv) {
   if(argc <2){
      cout << "Need png file" << endl;
      return -1;
   }
   char* img_path = argv[1];
   
   //create border
   cout << "Creating border ... " << endl;
   Mat src, border, gray;
   src = imread(img_path);

   //set the size of top, bottom, left, and right
   int top = (int) (0.001*src.rows);
   int bottom = (int) (0.001*src.rows);
   int left = (int) (0.001*src.cols);
   int right = (int) (0.001*src.cols);
   
   copyMakeBorder( src, border, top, bottom, left, right, BORDER_CONSTANT );
   cout << "Border successfully created" << endl << endl;
   
   src = border;
   
   //char save_path[30];
   char num_char[3];
   Pix *segbox;
   Pix *image = pixRead(img_path);
   tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
   api->Init(NULL, "eng");
   api->SetPageSegMode(tesseract::PSM_AUTO);
   api->SetImage(image);
   api->SetSourceResolution(70);
   Boxa* boxes = api->GetComponentImages(tesseract::RIL_BLOCK, true, NULL, NULL);
   printf("Found %d textline image components.\n", boxes->n);
   api->SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
   for (int i = 0; i < boxes->n; i++) {
      BOX* box = boxaGetBox(boxes, i, L_CLONE);
	  Mat page_segment;
	  src.copyTo(page_segment);
	  line( page_segment, Point(box->x, box->y), Point(box->x+box->w, box->y), Scalar(0,0,255), 3, 8 );
	  line( page_segment, Point(box->x, box->y+box->h), Point(box->x+box->w, box->y+box->h), Scalar(0,0,255), 3, 8 );
	  line( page_segment, Point(box->x, box->y), Point(box->x, box->y+box->h), Scalar(0,0,255), 3, 8 );
	  line( page_segment, Point(box->x+box->w, box->y), Point(box->x+box->w, box->y+box->h), Scalar(0,0,255), 3, 8 );
      namedWindow( "page_segment", WINDOW_NORMAL );
      imshow("page_segment", page_segment);
      waitKey(0);
	  
	  
	  Mat segmented_page;
	  page_segment(Rect(box->x,box->y,box->w,box->h)).copyTo(segmented_page);
	  
	  
      segbox = pixClipRectangle(image, box, NULL);
      api->Init(NULL, "eng");
      api->SetImage(segbox);
	  api->SetSourceResolution(70);
      api->Recognize(0);
	  tesseract::ResultIterator* ri = api->GetIterator();
	  tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
	  if (ri != 0) {
		do {
		  const char* word = ri->GetUTF8Text(level);
		  float conf = ri->Confidence(level);
		  int x1, y1, x2, y2;
		  ri->BoundingBox(level, &x1, &y1, &x2, &y2);
		  //printf("word: '%s';  \tconf: %.2f; BoundingBox: %d,%d,%d,%d;\n", word, conf, x1, y1, x2, y2);
		  line( segmented_page, Point(x1, y1), Point(x2, y1), Scalar(0,0,255), 3, 8 );
		  line( segmented_page, Point(x1, y2), Point(x2, y2), Scalar(0,0,255), 3, 8 );
		  line( segmented_page, Point(x1, y1), Point(x1, y2), Scalar(0,0,255), 3, 8 );
		  line( segmented_page, Point(x2, y1), Point(x2, y2), Scalar(0,0,255), 3, 8 );
		  delete[] word;
		} while (ri->Next(level));
	  }
	  namedWindow( "page_segment", WINDOW_NORMAL );
	  imshow("page_segment", segmented_page);
	  waitKey(0);
   }

   //pixDestroy(&image);
   api->End();
   return 0;
}

