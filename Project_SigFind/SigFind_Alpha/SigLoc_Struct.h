/****************************************************************************
This struct is used for saving the file paths and coordinates
of the located signature box. It will be used between passing
paths and coordinates from OCR_SigLoc.cpp to ConvNet.cpp
*****************************************************************************/


#ifndef SIGLOC
#define SIGLOC
struct sigloc{
	vector<string> Sig_Paths; 
	vector<Vec4i> Sig_coordinates;
};
#endif