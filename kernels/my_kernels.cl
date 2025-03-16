// Histogram using image unsigned character array as input and an int array as output which will hold the intensity values and bin size I picked in host code.
kernel void hist_Atom(global const uchar* inputImage, global int* histogramOutput){
	int id = get_global_id(0); // Gets work item id
	int intensityValue = inputImage[id];  // This assigns the intensity value of the pixel that matches the id to a variable.
	atomic_inc(&histogramOutput[intensityValue]);  // Increments the corresponding bin each time by using the intensity value as the index number.
}

// Code adapted and modified from the workshop materials for tutorial 3, more specifically the reduce_add_4 kernel.
kernel void hist_Local(global const uchar* inputImage, global int* histogramOutput, local int* localHist, int binNumber){ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

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

kernel void histNormal(global float* comHist,float maxBin){
	int id = get_global_id(0);

	comHist[id] = (float)comHist[id] / maxBin;
		
}

//Hillis-Steele basic inclusive scan adapted from workshop materials for tutorial 3
kernel void com_Hist(global int* A, global int* B) {
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
kernel void proj(global const uchar* inputImage, global  uchar* outputImage, global const float* LUT){

	int id = get_global_id(0);
	int value = inputImage[id];

	outputImage[id] = LUT[value]*255;
	
} 