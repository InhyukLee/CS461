#include <stdio.h>
#include <string.h>
#include <iostream>
#include <tesseract/baseapi.h>
#include <tesseract/strngs.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include <math.h>

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

vector<string> OCR(int argc, char** argv) {
   cout << "----------------Starting SigLoc ----------------" << endl;
   if(argc <2){
      cout << "Need png file" << endl;
   }
   char* img_path = argv[1];
   
   vector<string> SigPaths;   

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
   
   //namedWindow( "border", 1 );
   //imshow("border", border);
   //waitKey(0);
   
   src = border;
   //namedWindow( "src", 1 );
   //imshow("src", src);
   //waitKey(0);
   
   //Line detection
   cout << "Detecting lines ... " << endl;
   int Maxline =  src.cols/8;
   cvtColor( src, gray, COLOR_BGR2GRAY );
   
   //adaptive mean thresholding
   Mat amth;
   adaptiveThreshold(~gray, amth, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
	
   // extracted horizontal lines will be saved in horiz mat
   Mat horiz = amth.clone();
   int horiz_size = horiz.cols / 8;

   // Create structure element for extracting horizontal lines through morphology operations
   Mat horiz_struct = getStructuringElement(MORPH_RECT, Size(horiz_size,1));

   // Apply morphology operations
   erode(horiz, horiz, horiz_struct, Point(-1, -1));
   dilate(horiz, horiz, horiz_struct, Point(-1, -1));
   
   vector<Vec4i> lines;
   HoughLinesP( horiz, lines, 1, CV_PI/180, 80, Maxline, 10 );
   if(lines.size()>0){
      cout << "Lines are detected" << endl << endl;
      //namedWindow( "horizontal", WINDOW_NORMAL );
      //imshow("horizontal", horiz);
      //waitKey(0);
   }else{
      cout << "Fail to detect lines" << endl << endl;
   }
   
   //namedWindow( "horizontal", 1 );
   //imshow("horizontal", horiz);
   //waitKey(0);
   
   cout << "Running OCR" << endl;
   char save_path[100];
   char num_char[3];
   Pix *segbox;
   Pix *image = pixRead(img_path);
   tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
   api->Init(NULL, "eng");
   api->SetPageSegMode(tesseract::PSM_AUTO);
   api->SetImage(image);
   api->SetSourceResolution(70);
   cout << "Analysing page" << endl;
   Boxa* boxes = api->GetComponentImages(tesseract::RIL_BLOCK, true, NULL, NULL);
   printf("Found %d textline image components.\n", boxes->n);
   api->SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
   cout << "Page analysis is done" << endl;

   cout << "Converting images to text, and search for signature box" << endl;
   int sigcounter = 1;
   for (int i = 0; i < boxes->n; i++) {
      BOX* box = boxaGetBox(boxes, i, L_CLONE);
      api->SetRectangle(box->x, box->y, box->w, box->h);
      char* ocrResult = api->GetUTF8Text();
      char* sigfind = strstr(ocrResult, "SIGNATURE");
      if(sigfind){
		 cout << "Signature box has been detected" << endl;
         int line_distence = 5*box->h;
         int line_loc = 0;
		 cout << "Searching for the signature line ... " << endl;
         for( size_t i = 0; i < lines.size(); i++ ){
            if(lines[i][0]<=box->x && lines[i][2]>=box->x+box->w){
               if(lines[i][1]>=box->y-5*box->h && lines[i][1]<=box->y){
                  int new_line_dist = box->y - lines[i][1];
                  if(new_line_dist<line_distence){
                     line_distence = new_line_dist;
                     line_loc = i;
                  }
               } 
            } 
         }
		 cout << "Setting the size of signature box" << endl;
         //cut the signature box based on line location
         box->x = lines[line_loc][0];
         box->y = lines[line_loc][1]-7*box->h;
         box->w = lines[line_loc][2] - lines[line_loc][0];
         box->h = 8*box->h;
         
         //naming and saving signature box
         segbox = pixClipRectangle(image, box, NULL);
         //removing path components
         string infile = argv[1];
         string remove_pre = "./png_bin/";
         string remove_pro = ".png";
         int i = infile.find(remove_pre);
         infile.erase(i,remove_pre.length());
         i = infile.find(remove_pro);
         infile.erase(i,remove_pro.length());
         //adding new path and file name
         strcpy (save_path, "./image/");
         strncat(save_path, infile.c_str(),infile.length());
         strncat(save_path, "_sig",4);
         itoa(sigcounter,num_char);
         strncat (save_path, num_char,3);
         strncat (save_path,".png",4);
		 cout << "Saving signature box" << endl;
         pixWrite(save_path, segbox, IFF_PNG);
		 cout << "Signature box saved" << endl << endl;
         //cout << save_path << endl;
         SigPaths.push_back(save_path);
         sigcounter++;
      }
   }

   //for(int i=0;i<SigPaths.size();i++){
     // cout << SigPaths.at(i) << endl;
   //}
   cout << "----------------SigLoc Completed----------------" << endl<< endl;
   pixDestroy(&image);
   api->End();
   return SigPaths;
}


