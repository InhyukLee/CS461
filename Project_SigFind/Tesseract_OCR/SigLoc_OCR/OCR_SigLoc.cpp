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

int main(int argc, char** argv) {
   if(argc <2){
      cout << "Need png file" << endl;
      return -1;
   }
   char* img_path = argv[1];
   
   //Line detection
   
   Mat src, dst, color_dst;
   src = imread(img_path, 0);
   int Minline =  src.cols/8;
   vector<Vec4i> lines;
   Canny( src, dst, 50, 200, 3 );
   //cvtColor( dst, color_dst, CV_GRAY2BGR );
   HoughLinesP( dst, lines, 1, CV_PI/180, 80, Minline, 10 );
   
   char save_path[30];
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

   int sigcounter = 1;
   for (int i = 0; i < boxes->n; i++) {
      BOX* box = boxaGetBox(boxes, i, L_CLONE);
      api->SetRectangle(box->x, box->y, box->w, box->h);
      char* ocrResult = api->GetUTF8Text();
      char* sigfind = strstr(ocrResult, "CUSTOMER SIGNATURE");
      if(sigfind){
         int line_distence = 5*box->h;
         int line_loc = 0;
         for( size_t i = 0; i < lines.size(); i++ ){
            cout << "Lines" <<endl;
            cout << "S x: "<<lines[i][0] << " S y: "<< lines[i][1] << " E x: "<<lines[i][2] << " E y: "<< lines[i][3] << endl;
            if(lines[i][0]<=box->x && lines[i][2]>=box->x+box->w){
               if(lines[i][1]>=box->y-5*box->h && lines[i][1]<=box->y){
                  int new_line_dist = box->y - lines[i][1];
                  if(new_line_dist<line_distence){
                     line_distence = new_line_dist;
                     line_loc = i;
                  }
                  cout << "Hi"<< endl;
                  cout << "x: "<<lines[i][0] << " y: "<< lines[i][1] << endl;
               } 
            } 
         }
         cout << "BOX" <<endl;
         cout << "x: "<<box->x << " y: " << box->y << endl;
         cout << "h: "<<box->h << "w: "<<box->w <<endl;
         box->x = lines[line_loc][0];
         box->y = lines[line_loc][1]-355;
         box->w = lines[line_loc][2] - lines[line_loc][0];
         box->h = 355+box->h;

         //box->x = box->x - 315;
         //box->y = box->y-320;
         //box->w = 2671;
         //box->h = 355;
         
         segbox = pixClipRectangle(image, box, NULL);
         string infile = argv[1];
         string remove_pre = "./png_bin/";
         string remove_pro = ".png";
         int i = infile.find(remove_pre);
         infile.erase(i,remove_pre.length());
         i = infile.find(remove_pro);
         infile.erase(i,remove_pro.length());
         strcpy (save_path, "./image/");
         strncat(save_path, infile.c_str(),infile.length());
         strncat(save_path, "_sig",4);
         itoa(sigcounter,num_char);
         strncat (save_path, num_char,3);
         strncat (save_path,".png",4);
         pixWrite(save_path, segbox, IFF_PNG);
         sigcounter++;
      }
   }

   pixDestroy(&image);
   api->End();
   return 0;
}


