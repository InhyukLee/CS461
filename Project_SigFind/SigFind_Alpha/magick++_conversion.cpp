#include <Magick++.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

using namespace std;
using namespace Magick;


int main(int argc, char** argv){
	Image image;
	try{
		image.density("300");
		image.read(argv[1]);
		image.write("final_form.png");
	}catch(WarningCoder &e){
		cout << "Warning: " << e.what() << endl;
		return 1;
	}	
	
	return 0;
}
