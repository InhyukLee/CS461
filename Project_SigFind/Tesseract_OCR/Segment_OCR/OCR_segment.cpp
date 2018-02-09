#include <stdio.h>
#include <string.h>
#include <iostream>
#include <tesseract/baseapi.h>
#include <tesseract/strngs.h>
#include <leptonica/allheaders.h>

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
   for (int i = 0; i < boxes->n; i++) {
      BOX* box = boxaGetBox(boxes, i, L_CLONE);

      segbox = pixClipRectangle(image, box, NULL);
      strcpy (save_path, "./image/segmentation_");
      itoa(i,num_char);
      strncat (save_path, num_char,3);
      pixWrite(save_path, segbox, IFF_PNG);

      api->SetRectangle(box->x, box->y, box->w, box->h);
      char* ocrResult = api->GetUTF8Text();
      int conf = api->MeanTextConf();
      fprintf(stdout, "Box[%d]: x=%d, y=%d, w=%d, h=%d, confidence: %d, text: %s", i, box->x, box->y, box->w, box->h, conf, ocrResult);
   }

   pixDestroy(&image);
   api->End();
   return 0;
}


