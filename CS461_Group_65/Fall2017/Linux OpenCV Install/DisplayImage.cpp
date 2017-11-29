#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void windowRemoveGreen(Mat &image, int w_width, int w_height,int start_x, int start_y){

    // for every x in pixel array
    for( int i=start_x; i < w_width + start_x; i++){

	//for every y in pixel array
        for(int a=start_y; a < w_height + start_y; a++){
		
                image.at<Vec3b>(a,i)[1] = 0;
        }

    }
}

//creates a sliding window of given widht and height
void slideWindow(Mat image,int w_width, int w_height){
	int i=0,
	    a=0;
	while(i < image.cols - w_width){
		
		//resets height counter so window starts back at the top
		a=0;
		
		//debug check
		cout<< "slide window i = "<< i <<endl;	
		while( a < image.rows - w_height){
			cout<< a<<endl;	
			//use windowRemoveGreen to remove the green RGB value 
			//in OpenCV instead of RGB its BGR			
			windowRemoveGreen(image,w_width,w_height,i,a);
			
			//creates a window to hold our picture
			namedWindow("IMAGELEL",WINDOW_AUTOSIZE);
			
			//anchors the window so it always pops up in the same spot on desktop
			moveWindow("IMAGELEL",100,100);
			
			//display image on the named window
			imshow("IMAGELEL",image);
			
			//waits 800 ms
			waitKey(600);
			
			//destorys all current windows
			destroyAllWindows();
			
			//moves window down 10 pixels
			a+=10;
			}
		
		//moves window sideways 10 pixels
		i+=10;
		
	}
		

}

int main(int argc, char** argv )
{
    //if no argument is given or not enough arguments are given
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    //Mat is the imge type for OpenCV
    Mat image;
	
    //method that reads a file type, supported types i know so far are PNG and JPG but im sure most are supported
    image = imread( argv[1], 1 );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
	
  
    //call slideWindow
    slideWindow(image,50,50);
   

    return 0;
}
