
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
							  // only found out that workgroup size was working previously because gworksize was divisible by 256 by chance.

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

//Hillis-Steele basic inclusive scan adapted from workshop materials for tutorial 3
kernel void cum_hist(global int* A, global int* B) {
    int id = get_global_id(0);
    int N = get_global_size(0);
    B[id] = A[id];

    barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

    for (int stride = 1; stride < N; stride *= 2) { 
        int memHolder = B[id];
        barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
        if (id >= stride)
            memHolder = B[id - stride];            
        barrier(CLK_GLOBAL_MEM_FENCE); //sync the step    
        if (id >= stride)
            B[id] += memHolder;
        barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
    }
}

//Blelloch basic exclusive scan cumulative histogram based on algorithm found on tutorial 3 workshop materials

kernel void scan_bl(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;
	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride*2)) == 0)
			A[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
	//down-sweep
	if (id == 0)
		A[N-1] = 0;//exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	for (int stride = N/2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride*2)) == 0) {
			t = A[id];
			A[id] += A[id - stride]; //reduce 
			A[id - stride] = t;		 //move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
}

// Back projection kernel blind attempt at making from scratch.
kernel void back_projector(global const uchar* inputImage, global  uchar* outputImage, global const float* LUT, int imageSize){
	int id = get_global_id(0);
	if (id < imageSize) {  
		int value = inputImage[id];
		outputImage[id] = LUT[value]*255; // Multiplies normalised value by 255 for a balanced result of values.
	}
} 

// Attempt at an rgb histogram maker kernel.
kernel void hist_rgb(global const uchar* inputImage, global int* histR,global int* histG,global int* histB, int rgbImageSize){
	int id = get_global_id(0); // Gets work item id
	int lid = get_local_id(0);
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
		int rgbId = id*3; // Offsets the workitem id by 3 to skip the GB indexes in RGB.
		int intensityValueR = inputImage[rgbId];  
		int intensityValueG = inputImage[rgbId+1];  // Assigns values to RGB using offset.
		int intensityValueB = inputImage[rgbId+2];  
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


// Local memory version of the histogram kernel, it scales the bin number to 1024, 
// I tried to run all 65k bins but could not output the histograms properly with CImg.
kernel void hist_rgb_16bit(
    global const ushort* inputImage, global int* histR, global int* histG, global int* histB,int rgbImageSize, int binSize) {

    // This is the definition of the local histograms, need to use the hard coded value because opencl doesnt allow for dynamic values.
    local int localHistR[1024];
    local int localHistG[1024];
    local int localHistB[1024];

    int localId = get_local_id(0);
    int globalId = get_global_id(0);
    int localSize = get_local_size(0);

    for (int i = localId; i < binSize; i += localSize) { // I was having issues with populating all bins so initialised all bins using a stride.
        localHistR[i] = 0;
        localHistG[i] = 0;
        localHistB[i] = 0;
    }
	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

    if (globalId < rgbImageSize) {  
        int rgbId = globalId * 3;  // Assigns ID per group of 3 RGB values. Allows skipping to correct index.
        int r = inputImage[rgbId] / 64;  // Scaling from original values.
        int g = inputImage[rgbId + 1] / 64; // Assings to RGB using the addition of the offset.
        int b = inputImage[rgbId + 2] / 64;

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

// Local memory version of the histogram kernel, it scales the bin number to 1024, 
// I tried to run all 65k bins but could not output the histograms properly with CImg.
kernel void hist_greyscale_16bit(global const ushort* inputImage, global int* outputHist,int imageSize, int binSize) {

    // This is the definition of the local histograms, need to use the hard coded value because opencl doesnt allow for dynamic values.
    local int localHist[1024];    

    int localId = get_local_id(0);
    int globalId = get_global_id(0);
    int localSize = get_local_size(0);

    for (int i = localId; i < binSize; i += localSize) { // I was having issues with populating all bins so initialised all bins using a stride.
        localHist[i] = 0;        
    }
	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

    if (globalId < imageSize) {  
        int intensity= inputImage[globalId] / 64;  // Scaling from original values.
        atomic_inc(&localHist[intensity]);// Atomic incrementation as before.        
    }
	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
    for (int i = localId; i < binSize; i += localSize) { // again using stride, updates global histogram.
        atomic_add(&outputHist[i], localHist[i]);        
    }
}

// Back projection kernel blind attempt at making from scratch.
kernel void back_projectorRgb(global const uchar* inputImage, global  uchar* outputImage, global const float* LUTr,global const float* LUTg,global const float* LUTb,int binNumber){
	int id = get_global_id(0);
	int rgbId = id*3;
	outputImage[rgbId] = LUTr[inputImage[rgbId]]*binNumber-1; // Similar to the other one changed to dynamic bin number. Also 3 channels.
	outputImage[rgbId+1] = LUTg[inputImage[rgbId+1]]*binNumber-1;
	outputImage[rgbId+2] = LUTb[inputImage[rgbId+2]]*binNumber-1;	
} 


kernel void back_projector_rgb_16bit(global const ushort* inputImage,global ushort* outputImage,global const float* lutR,global const float* lutG,global const float* lutB,int rgbImageSize,float scale) 
{
	    int id = get_global_id(0);    
    if (id >= rgbImageSize) return;
    int rgbId = id * 3;
    int indexR = (int)(inputImage[rgbId]*scale); // Main difference is the need to scale the values back to create 16bit image.
    int indexG = (int)(inputImage[rgbId+1]*scale);
    int indexB = (int)(inputImage[rgbId+2]*scale);
    outputImage[rgbId] = (ushort)(lutR[indexR]*65535.0f);
    outputImage[rgbId + 1] = (ushort)(lutG[indexG]*65535.0f);
    outputImage[rgbId + 2] = (ushort)(lutB[indexB]*65535.0f);
}
kernel void back_projector_grayscale_16bit(global const ushort* inputImage,global ushort* outputImage,global const float* LUT,int ImageSize,float scale) 
{
	int id = get_global_id(0);    
    if (id >= ImageSize) return;	

	int intensity = (int)(inputImage[id]*scale); // Main difference is the need to scale the values back to create 16bit image.
    
    outputImage[id] = (ushort)(LUT[intensity]*65535.0f);
    
}
