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
   cout << "----------------Starting SigLoc ----------------" << endl;
   if(argc <2){
      cout << "Need png file" << endl;
      return -1;
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
   
   src = border;
   /*
   //display border
   namedWindow( "src", 1 );
   imshow("src", src);
   waitKey(0);
   */
   
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

   //merging lines
   vector<int> line_elem;
   vector<Vec4i> lines_c = lines;
   vector<Vec4i> group;
   vector<Vec4i> merged_lines;
   //create tracking vector
   for(int i = 0; i<lines_c.size(); i++){
      line_elem.push_back(i);
   }
   Vec4i temp = lines_c[line_elem.front()];
   line_elem.erase(line_elem.begin());
   group.push_back(temp);
   //start merging
   while(!line_elem.empty()){
      for(int i = 0; i<line_elem.size();i++){
         Vec4i temp2 = lines_c[line_elem.at(i)];
         if(abs(temp[1]-temp2[1])<3){
            //temp = temp2;
            group.push_back(temp2);
            line_elem.erase(line_elem.begin()+i);
         }
      }
      //merge group
      for(int k = 0; k<group.size();k++){
         for(int l = 0; l<group.size();l++){
            if(!((group[k][0]<group[l][0])&&(group[k][2]<group[l][0]))||!((group[l][0]<group[k][0])&&(group[l][2]<group[k][0]))){
               if(group[k][0]>group[l][0]){
                  group[k][0] = group[l][0];
               }else{
                  group[l][0] = group[k][0];
               }
               if(group[k][2]>group[l][2]){
                  group[l][2] = group[k][2];
               }else{
                  group[k][2] = group[l][2];
               }
            }
         }
      }
      //copy the merged lines
      for(int k = 0; k<group.size();k++){
         merged_lines.push_back(group[k]);
      }
      group.clear();
      if(!line_elem.empty()){
         temp = lines_c[line_elem.front()];
         line_elem.erase(line_elem.begin());
         group.push_back(temp);
      }
   }
   
   
   if(merged_lines.size()>0){
      cout << "Lines are detected" << endl << endl;
	  /*
	  //display merged_lines
	  for( size_t i = 0; i < merged_lines.size(); i++ ){
         line( src, Point(merged_lines[i][0], merged_lines[i][1]), Point(merged_lines[i][2], merged_lines[i][3]), Scalar(0,0,255), 3, 8 );
      }
      namedWindow( "merged_lines", WINDOW_NORMAL );
      imshow("merged_lines", src);
      waitKey(0);
	  */
   }else{
      cout << "Fail to detect lines" << endl << endl;
   }
   
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
	  bool sign_line = false;
      BOX* box = boxaGetBox(boxes, i, L_CLONE);
      api->SetRectangle(box->x, box->y, box->w, box->h);
      char* ocrResult = api->GetUTF8Text();
	  //lowercase to uppercase
      for(int k = 0; k< strlen(ocrResult); k++){
         ocrResult[k] = toupper(ocrResult[k]);
      }
      char* sigfind = strstr(ocrResult, "SIGNATURE");
      if(sigfind){
         cout << "Signature box has been detected" << endl;
		 
		 /*
		 //cout box coordinates
		 cout << "Signature Box" <<endl;
		 cout << "X: "<<box->x<<" X2: "<<box->x+box->w<<" Y: "<<box->y<<" H: "<<box->h<<endl;
		 cout << endl;
		 */
		 
         int line_distence = box->h;
         //int line_length = box->w;
         int line_loc = 0;
         cout << "Searching for the signature line ... " << endl;
         for( size_t j = 0; j < merged_lines.size(); j++ ){
			
			/*
			//cout line coordinates
			cout << j <<" line" << endl;
			cout << "X: "<<merged_lines[j][0]<<" X2: "<<merged_lines[j][2]<<" Y: "<<merged_lines[j][1]<< endl;
			cout << endl;
			*/
            if(merged_lines[j][0]<=box->x+5 && merged_lines[j][2]+5>=box->x+box->w){
               if(merged_lines[j][1]>=box->y-box->h && merged_lines[j][1]<=box->y){
				  sign_line = true;
                  int new_line_dist = box->y - merged_lines[j][1];
                  if(new_line_dist<line_distence){
                     line_distence = new_line_dist;
                     line_loc = j;
                  }
               } 
            } 
         }
		 if(sign_line == true){
			 cout << "Setting the size of signature box" << endl;
			 //cut the signature box based on line location
			 box->x = merged_lines[line_loc][0];
			 box->y = merged_lines[line_loc][1]-7*box->h;
			 box->w = merged_lines[line_loc][2] - merged_lines[line_loc][0];
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
   }

   for(int i=0;i<SigPaths.size();i++){
      cout << SigPaths.at(i) << endl;
   }
   cout << "----------------SigLoc Completed----------------" << endl<< endl;
   pixDestroy(&image);
   api->End();
   
   return 0;
}


