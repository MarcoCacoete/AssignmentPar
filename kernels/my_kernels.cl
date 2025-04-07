
// Histogram using image unsigned character array as input and an int array as output which will hold the intensity values and bin size I picked in host code.
kernel void hist_atom(global const uchar* inputImage, global int* histogramOutput, int imageSize){
	int id = get_global_id(0); // Gets work item id
	if (id < imageSize) {
	int intensityValue = inputImage[id];  // This assigns the intensity value of the pixel that matches the id to a variable.
	atomic_inc(&histogramOutput[intensityValue]);
	}  // Increments the corresponding bin each time by using the intensity value as the index number.
}

// Code adapted and modified from the workshop materials for tutorial 3, more specifically the reduce_add_4 kernel.
kernel void hist_local(global const uchar* inputImage, global int* histogramOutput, local int* localHist, int binNumber,int imageSize){ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	if(id>=imageSize) return; // This is a bounds check, it works with my kernel code global work size padding, 
							  // only found out that workgroup size was working previously because global worksize was divisible by 256 by chance.

	//cache all N values from global memory to local memory
	for(int i = lid; i<binNumber; i+=N){
		localHist[i] = 0;
	};

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	uchar intensityValue = inputImage[id];

	atomic_add(&localHist[intensityValue],1);

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish updating local memory

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	for(int i = lid; i<binNumber; i+=N){
		atomic_add(&histogramOutput[i], localHist[i]);
	};
}

// Just a simple normaliser that divides value by max value for histogram. Required step for back projection.
kernel void hist_normal(global float* cumHist,float maxBin){
	int id = get_global_id(0);
	cumHist[id] = (float)cumHist[id] / maxBin;		
}

//Hillis-Steele basic inclusive scan adapted from workshop materials for tutorial 3, made iterative changes necessary to get it to work for my purposes.
// It was difficult to debug without the debugger, due to my limited experience with opencl, tried different things until it worked.
kernel void cum_hist(global int* A, global int* B) {
    int id = get_global_id(0);
    int N = get_global_size(0);
    B[id] = A[id]; //Copies histogram to B 
    barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

    for (int stride = 1; stride < N; stride *= 2) { // Peforms scan using stride, is inclusive contrary to the Blelloch method
        int memHolder = B[id];						// Temp memory storage for value
        barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
        if (id >= stride)				// Bounds check and checks neighbours value
            memHolder = B[id - stride];            
        barrier(CLK_GLOBAL_MEM_FENCE); //sync the step    
        if (id >= stride)
            B[id] += memHolder;        // Adds it up
        barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
    }
}

//Blelloch basic exclusive scan cumulative histogram based on algorithm found on tutorial 3 workshop materials
kernel void scan_bl(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;
	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) {// Stride to "select" threads 
		if (((id + 1) % (stride*2)) == 0)
			A[id] += A[id - stride]; // Adds the values 

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
	//down-sweep
	if (id == 0)
		A[N-1] = 0;//exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	for (int stride = N/2; stride > 0; stride /= 2) { // Downsweep replaces the value with the cumulative sum
		if (((id + 1) % (stride*2)) == 0) {
			t = A[id];
			A[id] += A[id - stride]; //reduce 
			A[id - stride] = t;		 //move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
}

// Back projection kernel, used this website for information - https://www.songho.ca/dsp/histogram/histogram.html
kernel void back_projector(global const uchar* inputImage, global  uchar* outputImage, global const float* LUT, int imageSize){
	int id = get_global_id(0);
	if (id < imageSize) {  
		int value = inputImage[id];
		outputImage[id] = LUT[value]*255; // Multiplies normalised value by 255 for a balanced result of values.
	}
} 

// Attempt at an rgb histogram maker kernel, iteration of local histogram learned in workshops, using offsets for different colour channels.
kernel void hist_rgb(global const uchar* inputImage, global int* histR,global int* histG,global int* histB, int rgbImageSize){
	int id = get_global_id(0); // Gets work item id
	int lid = get_local_id(0);
	int n = rgbImageSize;		 // N to offset pixel colour channels
	local int localHistR[256]; //Defines local histograms for 256 bins, 1 per colour channel.
	local int localHistG[256];
	local int localHistB[256];

	 // Initialises local histograms to zero
	if (lid < 256) {
        localHistR[lid] = 0;
        localHistG[lid] = 0;
        localHistB[lid] = 0;
    }
	barrier(CLK_LOCAL_MEM_FENCE); // syncs the step
	if (id < rgbImageSize) {
		int intensityValueR = inputImage[id];  
		int intensityValueG = inputImage[n+id];  // Assigns values to RGB using offset.
		int intensityValueB = inputImage[2*n+id];  
		atomic_inc(&localHistR[intensityValueR]);  // Increments atomically local hist.
		atomic_inc(&localHistG[intensityValueG]); 
		atomic_inc(&localHistB[intensityValueB]); 
	}	
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < 256) {
        atomic_add(&histR[lid], localHistR[lid]); // Then updates global histogram.
        atomic_add(&histG[lid], localHistG[lid]);
        atomic_add(&histB[lid], localHistB[lid]);
    }	
}


// This is again an iteration of the same kernel used previously but adapted as needed for the different types of image.
// I tried to run all 65k bins but could not output the histograms properly with CImg.
kernel void hist_rgb_16bit(
    global const ushort* inputImage, global int* histR, global int* histG, global int* histB,int rgbImageSize, int binSize) {

    // This is the definition of the local histograms, need to use the hard coded value because opencl doesnt allow for dynamic values.
    local int localHistR[1024];
    local int localHistG[1024];
    local int localHistB[1024];

    int localId = get_local_id(0);
    int id = get_global_id(0);
    int localSize = get_local_size(0);
	int n = rgbImageSize; 		 // N to offset pixel colour channels


    for (int i = localId; i < binSize; i += localSize) { // I was having issues with populating all bins so initialised all bins using a stride.
        localHistR[i] = 0;
        localHistG[i] = 0;
        localHistB[i] = 0;
    }
	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

    if (id < rgbImageSize) {  // Stride n to offset for colour channels
        int r = inputImage[id] / 64;  // Scaling from original values.
        int g = inputImage[id+n] / 64; // Assings to RGB using the addition of the offset.
        int b = inputImage[2*n+id] / 64;

        atomic_inc(&localHistR[r]);// Atomic incrementation as before.
        atomic_inc(&localHistG[g]);
        atomic_inc(&localHistB[b]);
    }
	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

    for (int i = localId; i < binSize; i += localSize) { // again using stride, updates global histogram.
        atomic_add(&histR[i], localHistR[i]);
        atomic_add(&histG[i], localHistG[i]);
        atomic_add(&histB[i], localHistB[i]);
    }
}

// Building upon the previous examples, it scales the bin number to 1024, 
// I tried to run all 65k bins but could not output the histograms properly with CImg.
kernel void hist_greyscale_16bit(global const ushort* inputImage, global int* outputHist,int imageSize, int binSize) {

    // This is the definition of the local histograms, need to use the hard coded value because opencl doesnt allow for dynamic values.
    local int localHist[1024];    

    int localId = get_local_id(0);
    int id = get_global_id(0);
    int localSize = get_local_size(0);

    for (int i = localId; i < binSize; i += localSize) { // I was having issues with populating all bins so initialised all bins using a stride.
        localHist[i] = 0;        
    }
	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

    if (id < imageSize) {  
        int intensity= inputImage[id] / 64;  // Scaling from original values.
        atomic_inc(&localHist[intensity]);// Atomic incrementation as before.        
    }
	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
    for (int i = localId; i < binSize; i += localSize) { // again using stride, updates global histogram.
        atomic_add(&outputHist[i], localHist[i]);        
    }
}

// Back projection kernel attempt at making this based on the previous attempt but with the offsets for the channels.
kernel void back_projectorRgb(global const uchar* inputImage, global  uchar* outputImage, global const float* LUTr,global const float* LUTg,global const float* LUTb,int rgbImageSize){
	int id = get_global_id(0);
	if (id < rgbImageSize) {
	int n = rgbImageSize;
	outputImage[id] = LUTr[inputImage[id]]*rgbImageSize-1; // Similar to the other one changed to dynamic bin number. Also 3 channels.
	outputImage[n+id] = LUTg[inputImage[n+id]]*rgbImageSize-1;
	outputImage[2*n+id] = LUTb[inputImage[2*n+id]]*rgbImageSize-1;	
	}
} 


kernel void back_projector_rgb_16bit(global const ushort* inputImage,global ushort* outputImage,global const float* lutR,global const float* lutG,global const float* lutB,int rgbImageSize,float scale) 
{
	    int id = get_global_id(0);    
    if (id >= rgbImageSize) return;
    int n = rgbImageSize;
    int indexR = (int)(inputImage[id]*scale); // Main difference is the need to scale the values back to create 16bit image.
    int indexG = (int)(inputImage[n+id]*scale);
    int indexB = (int)(inputImage[2*n+id]*scale);
    outputImage[id] = (ushort)(lutR[indexR]*65535.0f);
    outputImage[n+id] = (ushort)(lutG[indexG]*65535.0f);
    outputImage[2*n+id] = (ushort)(lutB[indexB]*65535.0f);
}
kernel void back_projector_grayscale_16bit(global const ushort* inputImage,global ushort* outputImage,global const float* LUT,int ImageSize,float scale) 
{
	int id = get_global_id(0);    
    if (id >= ImageSize) return;	

	int intensity = (int)(inputImage[id]*scale); // Main difference is the need to scale the values back to create 16bit image.
    
    outputImage[id] = (ushort)(LUT[intensity]*65535.0f);
    
}
