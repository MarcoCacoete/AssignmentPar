#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"


// This host code includes host code provided for the Tutorial 2 tasks in the workshops. Mostly the openCL setup.

using namespace cimg_library;
using namespace std;

// Two functions to print out the images, intput and output. Greyscale 8 and 16 bit.
CImg<unsigned char> picture_output(const std::string& image_filename){
	CImg<unsigned char> image_input(image_filename.c_str());
	

	int pic_width = image_input.width();  // Assigns various image attirubtes to variables.
	int window_width = image_input.width();  
	int window_height = image_input.height(); 
	if (pic_width > 1080) {
		window_width = image_input.width()/3;  // This conditional is here because I was having trouble fitting 
		window_height = image_input.height()/3; // some images on my screen otherwise.
	}	
	const char* image_name = image_filename.c_str(); 
	
	CImgDisplay disp_input(window_width, window_height, image_name, 0); // cimg image display object and arguments.

	disp_input.display(image_input); // Call to display.

	disp_input.resize(window_width, window_height); // Resizes output windows.

	while (!disp_input.is_closed()) {  // keeps them open until user input.
		disp_input.wait();
	}
	return image_input;
}

void input16(const std::string& image_filename) { // Same as above pretty much. but for 16bit.
    // Load the 16-bit image
    CImg<unsigned short> img16(image_filename.c_str());

    int pic_width = img16.width();
    int window_width = img16.width();
    int window_height = img16.height();

    if (pic_width > 1080) {
        window_width = img16.width() / 3;
        window_height = img16.height() / 3;
    }
    CImg<unsigned char> img8 = img16.get_normalize(0, 255);

    CImgDisplay disp_input(window_width, window_height, "input (16-bit normalised)", 0);
    disp_input.display(img8);
    disp_input.resize(window_width, window_height);

    while (!disp_input.is_closed()) {
        disp_input.wait();
    }
}

void print_help() { // Code provided for workshops. lets user select parameters for running program.
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	std::cout<<"Enter image name."<<endl; //User input to enter prefered image.	
	string imageName;	
	cin>>imageName;
	string image_filename = imageName ;
	// string image_filename = "test_large.pgm";

	for (int i = 1; i < argc; i++) { // More code from workshops, accepts options selected by user.
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	} 

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input; // Defines empty input image char vector.
		CImg<unsigned short> img16(image_filename.c_str()); // Always creates 16bit short vector, to check if it's a 
		unsigned short max = img16.max();					// 16 bit image. Couldnt find a better way.

		bool is16Bit = false;// Boolean to flip if it is a 16bit image.
	
		if (max <= 255) {
			std::cout << "8-bit image detected." << std::endl;// Small block to show image type and dispaly it.
			image_input = picture_output(image_filename); 
	
		} else {
			std::cout << "16-bit image detected." << std::endl;
			input16(image_filename); 
			is16Bit = true; //Flips boolean to true.
		}
		
		//Part 3 - host operations 
		//3.1 Select computing devices. More code provided from workshops.
		cl::Context context = GetContext(platform_id, device_id);// Defines platform and device to be used.

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE); // Added option for profiling.

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - device operations		

		int width = is16Bit? img16.width() : image_input.width(); // Creates various necessary variables holding image metadata.
		int height = is16Bit? img16.height() : image_input.height(); // Picks appropriate value if 16bit or not.
		int spectrum = is16Bit? img16.spectrum() : image_input.spectrum();
		int depth = is16Bit? img16.depth() : image_input.depth();
		int imageDimensions = width*height;
		//Image size in bytes.
		size_t image_size = is16Bit ? imageDimensions * spectrum * sizeof(unsigned short) : imageDimensions * spectrum * sizeof(unsigned char);

		int binNumber = 256; //defines bin numbers for greyscale, 8bit or 16bit.
		if(is16Bit){
			binNumber = 65536;
		}
		const size_t localWorkSize = 256; // Local worksize not real information on how to pick a good size.

		std::cout<<"Width:"<<width<<endl; //Some prints of image metadata.
		std::cout<<"Height:"<<height<<endl;
		std::cout<<"Pixel count: "<<width*height<<endl;
		std::cout << "Image size (bytes): " << image_size << endl;
		// Defines globalworksize with padding for cases where it might not be divisible well, depending on image. Used with bounds check.
		size_t globalWorkSize = ((imageDimensions + localWorkSize - 1) / localWorkSize) * localWorkSize; // Adjusted for pixel count
		size_t buffer_Size = binNumber * sizeof(int); // Sizing buffers. 
		size_t buffer_Size_float = binNumber * sizeof(float);

		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_size); // Defining buffers for in and output images.
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_size);
		// Buffers for pretty much everything else.
		cl::Buffer dev_intensityHistogram(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_comHistogram(context, CL_MEM_READ_WRITE, buffer_Size); 
		cl::Buffer dev_histNormal(context, CL_MEM_READ_WRITE, buffer_Size_float);
		cl::Buffer dev_histR(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_histG(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_histB(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_histRcom(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_histGcom(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_histBcom(context, CL_MEM_READ_WRITE, buffer_Size);		

		//4.1 Copy images to device memory
		
		if(!is16Bit){ // queues Write buffers with difference sizes depending on type of image.
			queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_size, &image_input.data()[0]);
			queue.enqueueWriteBuffer(dev_image_output, CL_TRUE, 0, image_size, &image_input.data()[0]);
		}else{
			queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_size, &img16.data()[0]);
			queue.enqueueWriteBuffer(dev_image_output, CL_TRUE, 0, image_size, &img16.data()[0]);
		}	

		//4.2 Setup and execute the kernel (i.e. device code)
		
		vector<int> histogram (binNumber,0); // Histogram for greyscale.	
		queue.enqueueWriteBuffer(dev_intensityHistogram, CL_TRUE, 0, buffer_Size, &histogram.data()[0],nullptr);		

		bool check = false; // Check for if conditions are met to break from while.
		
		while(!check){ // While not check used to make sure user inputs correct options.

			if(spectrum==1){ // Spectrum 1 matches greyscale.
				std::cout<<"What histogram kernel would you like to use. Local or Atom?"<<endl; // Input choice for kernel for histogram.
				string kernelType;
				cin>>kernelType;
				// To lower to prevent capitals in user input.
				std::transform(kernelType.begin(),kernelType.end(),kernelType.begin(),::tolower);

				if(kernelType=="atom"){ // Atomic histogram block.
					std::cout<<"Atom"<<endl;
					check = true; //Triggers check to move on after.
					cl::Kernel kernelAtom = cl::Kernel(program, "hist_atom"); // Kernel argument setup.
					kernelAtom.setArg(0, dev_image_input);
					kernelAtom.setArg(1, dev_intensityHistogram);
					kernelAtom.setArg(2, cl::Local(buffer_Size)); // Local memory for 256 bins
					kernelAtom.setArg(3, imageDimensions);
					// Enqueued with global and local work size for local memory work, for efficiency.
					queue.enqueueNDRangeKernel(kernelAtom, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize),nullptr);
				}
				else if(kernelType=="local"){ // Same as above but for other local kernel.
					std::cout<<"Local"<<endl;
					check = true;
					cl::Kernel kernelLocal = cl::Kernel(program, "hist_local");cl::Kernel kernelHistLocal = cl::Kernel(program, "hist_local");
					kernelHistLocal.setArg(0, dev_image_input);
					kernelHistLocal.setArg(1, dev_intensityHistogram);
					kernelHistLocal.setArg(2, buffer_Size,NULL);
					kernelHistLocal.setArg(3, binNumber);
					kernelHistLocal.setArg(4, image_size);	
		
					queue.enqueueNDRangeKernel(kernelHistLocal, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(binNumber),nullptr);
				}
				else{
					std::cout<<"Invalid input. Please enter either Atom or Local"<<endl;
				}
			}else if(!is16Bit){ // The 16bit and 8bitrgb pictures have their oown different kernels

				check=true;				
				std::cout<<"Colour image detected"<<endl;// Vectors for output to user in CImg. Zeroed.
				vector <int> histR (binNumber,0);
				vector <int> histG (binNumber,0);
				vector <int> histB (binNumber,0);		
				
				queue.enqueueWriteBuffer(dev_histR, CL_TRUE, 0, buffer_Size, histR.data()); // Write buffers for histograms.
				queue.enqueueWriteBuffer(dev_histG, CL_TRUE, 0, buffer_Size, histG.data());
				queue.enqueueWriteBuffer(dev_histB, CL_TRUE, 0, buffer_Size, histB.data());

				cl::Kernel kernelHistRgb = cl::Kernel(program, "hist_rgb"); //Kernel call for rgb histogram kernel.
				kernelHistRgb.setArg(0, dev_image_input);
				kernelHistRgb.setArg(1, dev_histR);
				kernelHistRgb.setArg(2, dev_histG);
				kernelHistRgb.setArg(3, dev_histB);
				kernelHistRgb.setArg(4,imageDimensions);
				
				// Kernel call queued.
				queue.enqueueNDRangeKernel(kernelHistRgb, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize),nullptr);

				queue.enqueueReadBuffer(dev_histR, CL_TRUE, 0, buffer_Size, &histR.data()[0],nullptr); //Queued read buffers for hists.
				queue.enqueueReadBuffer(dev_histG, CL_TRUE, 0, buffer_Size, &histG.data()[0],nullptr);
				queue.enqueueReadBuffer(dev_histB, CL_TRUE, 0, buffer_Size, &histB.data()[0],nullptr);

				vector <vector<int>> histRgb = {histR,histG,histB}; //Vector of the hists to iterate for outputs.

				for (int i = 0; i < histRgb.size(); i++) { //Creates cimg object for histograms.
					CImg<float> histogramGraphRgb(binNumber, 1, 1, 1, 0);
					for (int j = 0; j < binNumber; ++j) {
						histogramGraphRgb(j) = static_cast<float>(histRgb[i][j]); 
					}
					const char* histName;

					switch(i){// Picks relevant name.
						case 0:
							histName = "Red Histogram";
							break;
						case 1:
							histName = "Green Histogram";
							break;
						case 2:
							histName = "Blue Histogram";
							break;
					}					
					
					// Sets histogram window size
					CImgDisplay disp_raw(800, 600, histName);      //Defines window dimensions.

					// Display graph, with argument val 3 for bar, no real way of changing font sizes.				
					histogramGraphRgb.display_graph(disp_raw, 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraphRgb.max(),true);	
				}
			}else{
				check=true;	
			
				std::cout<<"Colour image detected"<<endl; // Same repeated steps as above but for RGB 16bit.
				binNumber = 1024; // Necessary due to the astronomical number of pixel values for 16bit.
				buffer_Size = binNumber * sizeof(int); // Update buffer size to match 1024 bins

				vector <int> histR (binNumber,0);
				vector <int> histG (binNumber,0);
				vector <int> histB (binNumber,0);						
				
				queue.enqueueWriteBuffer(dev_histR, CL_TRUE, 0, buffer_Size, histR.data());
				queue.enqueueWriteBuffer(dev_histG, CL_TRUE, 0, buffer_Size, histG.data());
				queue.enqueueWriteBuffer(dev_histB, CL_TRUE, 0, buffer_Size, histB.data());
				

				cl::Kernel kernel(program, "hist_rgb_16bit");
                kernel.setArg(0, dev_image_input);
                kernel.setArg(1, dev_histR);
                kernel.setArg(2, dev_histG);
                kernel.setArg(3, dev_histB);
                kernel.setArg(4, imageDimensions);
				kernel.setArg(5,binNumber);
				
                queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(imageDimensions), cl::NDRange(localWorkSize), nullptr);

				queue.enqueueReadBuffer(dev_histR, CL_TRUE, 0, buffer_Size, histR.data());
				queue.enqueueReadBuffer(dev_histG, CL_TRUE, 0, buffer_Size, histG.data());
				queue.enqueueReadBuffer(dev_histB, CL_TRUE, 0, buffer_Size, histB.data());

				vector <vector<int>> histRgb = {histR,histG,histB};

				for (int i = 0; i < histRgb.size(); i++) {
					CImg<float> histogramGraphRgb(binNumber, 1, 1, 1, 0);
				
					int maxBinCount = *max_element(histRgb[i].begin(), histRgb[i].end());
				
					for (int j = 0; j < binNumber; ++j) {
						histogramGraphRgb(j) = static_cast<float>(histRgb[i][j]) / maxBinCount; // This displays the histogram normalised.
					}
				
					const char* histName;
					switch (i) {
						case 0:
							histName = "Red Histogram";
							break;
						case 1:
							histName = "Green Histogram";
							break;
						case 2:
							histName = "Blue Histogram";
							break;
					}
				
					CImgDisplay disp_raw(800, 600, histName);
				
					histogramGraphRgb.display_graph(disp_raw, 3, 1, "VALUES", 0, binNumber, "COUNT PER BIN", 0, 1, true);
				}
			}
		}

	
		//4.3 Copy the result from device to host
		
		// Reads for greyscale histogram buffer. Some of the lines before while loop are here so they are visible in following scopes.
		queue.enqueueReadBuffer(dev_intensityHistogram, CL_TRUE, 0, buffer_Size, &histogram.data()[0],nullptr);

		int maxValue = *max_element(histogram.begin(), histogram.end()); // Defines max value for normalisation logic.	 		
		int jobCount=0;

		for (int i=0; i<histogram.size();i++){ // Outputs job count , was used to debug.
			// std::cout<<histogram[i]<<endl;
			jobCount+= histogram[i];
		}		
		std::cout<<"Jobcount:"<<jobCount<<endl;

		vector<int> histogramCom (binNumber,0); //Defines comulative histogram and buffer.
		queue.enqueueWriteBuffer(dev_comHistogram, CL_TRUE, 0, buffer_Size, &histogram.data()[0],nullptr);

		vector<int> histogramComR(binNumber,0);// Same as above but for colour images.
		vector<int> histogramComG(binNumber,0);
		vector<int> histogramComB(binNumber,0);

		queue.enqueueWriteBuffer(dev_histRcom, CL_TRUE, 0, buffer_Size, &histogramComR.data()[0],nullptr);
		queue.enqueueWriteBuffer(dev_histGcom, CL_TRUE, 0, buffer_Size, &histogramComG.data()[0],nullptr);
		queue.enqueueWriteBuffer(dev_histBcom, CL_TRUE, 0, buffer_Size, &histogramComB.data()[0],nullptr);
		vector<cl::Buffer*> rgbBuffers = {&dev_histR, &dev_histG, &dev_histB}; // Some vectors of pointers for indexing and iteration.
		vector<cl::Buffer*> rgbBuffersCom = {&dev_histRcom, &dev_histGcom, &dev_histBcom};		
		vector<vector<int>> histogramComRgb = {histogramComR,histogramComG,histogramComB};

		check = false;

		while(!check){ // Same as above but for comulative histogram kernels.
			std::cout<<"What comulative histogram kernel would you like to use. Hillis or Blelloch?"<<endl;
			string kernelType;
			cin>>kernelType;
			std::transform(kernelType.begin(),kernelType.end(),kernelType.begin(),::tolower);

			//Choices between the hillis adapted kernel and Blelloch from workshops.
			if(spectrum==1){
				if (kernelType=="hillis"){
					std::cout<<"Hillis-Steele"<<endl;
					check = true;
					cl::Kernel kernelCom = cl::Kernel(program, "com_hist"); // Same as before only differences are the names of the kernels picked.
					kernelCom.setArg(0, dev_intensityHistogram);		
					kernelCom.setArg(1, dev_comHistogram);

					// The global work item number is based on bin numbers for the comulative kernels.
					queue.enqueueNDRangeKernel(kernelCom, cl::NullRange, cl::NDRange(binNumber), cl::NDRange(binNumber),nullptr);
				}
				else if(kernelType=="blelloch"){
					std::cout<<"Blelloch"<<endl;
					check = true;			
					cl::Kernel kernelCom = cl::Kernel(program, "scan_bl");
					kernelCom.setArg(0, dev_comHistogram);
					queue.enqueueNDRangeKernel(kernelCom, cl::NullRange, cl::NDRange(binNumber), cl::NDRange(binNumber),nullptr);
				}
				else{
					std::cout<<"Invalid input. Please enter either Scan or Blelloch"<<endl;
				}
				queue.enqueueReadBuffer(dev_comHistogram, CL_TRUE, 0, buffer_Size, &histogramCom.data()[0],nullptr);
			}
			else if(!is16Bit){				

				for(int i=0;i<rgbBuffers.size();i++){ // Same as above but for 8bit RGB runs 3 times once per RGB channel.
					if (kernelType=="hillis"){
						std::cout<<"Hillis-Steele"<<endl;
						check = true;
						cl::Kernel kernelCom = cl::Kernel(program, "com_hist"); // Indexed Histograms get passed as arguments and written on when outputted.
						kernelCom.setArg(0, *rgbBuffers[i]);		// De referenced pointers.
						kernelCom.setArg(1, *rgbBuffersCom[i]);
						queue.enqueueNDRangeKernel(kernelCom, cl::NullRange, cl::NDRange(binNumber), cl::NDRange(binNumber),nullptr);
						queue.enqueueReadBuffer(*rgbBuffersCom[i], CL_TRUE, 0, buffer_Size, &histogramComRgb[i].data()[0], nullptr);
					}
					else if(kernelType=="blelloch"){ //Same as above. Input and output histogram gets overwritten.
						std::cout<<"Blelloch"<<endl;
						check = true;			
						cl::Kernel kernelCom = cl::Kernel(program, "scan_bl");
						kernelCom.setArg(0, *rgbBuffers[i]);
						queue.enqueueNDRangeKernel(kernelCom, cl::NullRange, cl::NDRange(binNumber), cl::NDRange(binNumber),nullptr);
						queue.enqueueReadBuffer(*rgbBuffers[i], CL_TRUE, 0, buffer_Size, &histogramComRgb[i].data()[0], nullptr);
					}
					else{
						std::cout<<"Invalid input. Please enter either Scan or Blelloch"<<endl;
					}
				}
			}
			else{
				for(int i=0;i<rgbBuffers.size();i++){ // Same as before but for 16bit.
					std::cout<<binNumber;
					size_t buffer_Size = sizeof(int) * binNumber; // Ensures this matches the buffer size
					if (kernelType=="hillis"){
						std::cout<<"Hillis-Steele"<<endl;
						check = true;
						cl::Kernel kernelCom = cl::Kernel(program, "com_hist");
						kernelCom.setArg(0, *rgbBuffers[i]);		
						kernelCom.setArg(1, *rgbBuffersCom[i]);
						queue.enqueueNDRangeKernel(kernelCom, cl::NullRange, cl::NDRange(binNumber), cl::NDRange(binNumber),nullptr);
						queue.enqueueReadBuffer(*rgbBuffersCom[i], CL_TRUE, 0, buffer_Size, &histogramComRgb[i].data()[0], nullptr);
					}
					else if(kernelType=="blelloch"){
						std::cout<<"Blelloch"<<endl;
						check = true;			
						cl::Kernel kernelCom = cl::Kernel(program, "scan_bl");
						kernelCom.setArg(0, *rgbBuffers[i]);
						queue.enqueueNDRangeKernel(kernelCom, cl::NullRange, cl::NDRange(binNumber), cl::NDRange(binNumber),nullptr);
						queue.enqueueReadBuffer(*rgbBuffers[i], CL_TRUE, 0, buffer_Size, &histogramComRgb[i].data()[0], nullptr);
					}
					else{
						std::cout<<"Invalid input. Please enter either Scan or Blelloch"<<endl;
					}
				}

			}			
			std::cout<<"test"<<endl;
		}

		// This block is responsible for normalisation and back projection kernel setup and calls.
		if(spectrum==1){// For Greyscale images.

			int maximumValue = histogramCom[binNumber - 1];
			float maximumBinValue = static_cast<float>(maximumValue);

			// Converts intermediate results to floats for normalisation
			vector<float> histogramComFloat(binNumber, 0.0f); // New float vector
			for (int i = 0; i < binNumber; ++i) {
				histogramComFloat[i] = static_cast<float>(histogramCom[i]); // Convert int to float
			}
			
			queue.enqueueWriteBuffer(dev_histNormal, CL_TRUE, 0, buffer_Size_float, &histogramComFloat.data()[0],nullptr);

			cl::Kernel histNormal = cl::Kernel(program, "hist_normal");// Same as all previous kernels. 
			histNormal.setArg(0, dev_histNormal);	
			histNormal.setArg(1, maximumBinValue);		
		
			queue.enqueueNDRangeKernel(histNormal, cl::NullRange, cl::NDRange(binNumber), cl::NullRange,nullptr);
			queue.enqueueReadBuffer(dev_histNormal, CL_TRUE, 0, buffer_Size_float, &histogramComFloat.data()[0],nullptr);		

			cl::Kernel proj = cl::Kernel(program, "back_projector");
			proj.setArg(0, dev_image_input);	
			proj.setArg(1, dev_image_output);	
			proj.setArg(2, dev_histNormal);	
		
			queue.enqueueNDRangeKernel(proj, cl::NullRange, cl::NDRange(image_size), cl::NullRange,nullptr);

			vector<unsigned char> output_buffer(image_size);
			queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

			CImg<int> histogramGraph(binNumber, 1, 1, 1, 0); // Creates a 1D CImg object for the raw histogram
			for (int i = 0; i < binNumber; ++i) {
				// int maxValue = *max_element(histogram.begin(), histogram.end());
				histogramGraph(i) =histogram[i];//maxValue; // Copies raw histogram values
			}

			CImg<float> histogramGraphCom(binNumber, 1, 1, 1, 0); // Creates a 1D CImg object for the raw histogram
			for (int i = 0; i < binNumber; ++i) {
				histogramGraphCom(i) = histogramComFloat[i]; // Copies raw histogram values
			}
			
			// Sets histogram window size
			CImgDisplay disp_raw(800, 600, "Raw Histogram");     
			CImgDisplay disp_com(800, 600, "Cumulative Histogram");

			// Display histograms using the custom display objects for greyscale images.
			histogramGraph.display_graph(disp_raw, 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraph.max(),true);
			histogramGraphCom.display_graph(disp_com, 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraphCom.max(),true);	


			CImg<unsigned char> output_image(output_buffer.data(), width, height, depth, spectrum);
			string output_name = "output_image.pgm";
			output_image.save("output_image.pgm");
			picture_output(output_name);
		}else{// Same as above but for rgb images.
			
			if(is16Bit){ 
				buffer_Size_float= sizeof(float) * 1024;// Different values for 16bit.
				binNumber = 1024;		
			}

			cl::Buffer dev_histNormalR(context, CL_MEM_READ_WRITE, buffer_Size_float); //Buffer definitions.
			cl::Buffer dev_histNormalG(context, CL_MEM_READ_WRITE, buffer_Size_float);
			cl::Buffer dev_histNormalB(context, CL_MEM_READ_WRITE, buffer_Size_float);

			vector<cl::Buffer*> rgbBuffersComNorm = {&dev_histNormalR, &dev_histNormalG, &dev_histNormalB}; // Pointer vector holding the buffers.

			// Converts intermediate results to floats for normalization
			vector<float> histogramComFloatR(binNumber, 0.0f); // New float vectors for normalised comulative histograms which require decimals.
			vector<float> histogramComFloatG(binNumber, 0.0f); 
			vector<float> histogramComFloatB(binNumber, 0.0f); 
			vector <vector<float>*> histogramComRgbFloat = {&histogramComFloatR,&histogramComFloatG,&histogramComFloatB}; //Another pointer vector.
			vector<unsigned char> output_buffer(image_size);

			float maximumValue;
			for(int i=0;i<histogramComRgb.size();i++){				
				for (int j = 0; j < binNumber; ++j) 
				(*histogramComRgbFloat[i])[j] = static_cast<float>(histogramComRgb[i][j]); // Converts int to float, non parallel. It was a small job.
			}							

			for(int i=0;i<histogramComRgb.size();i++){	
				maximumValue = histogramComRgb[i][binNumber-1];
				maximumValue = static_cast<float>(maximumValue); // Max value for each different colour channel.

				queue.enqueueWriteBuffer(*rgbBuffersComNorm[i], CL_TRUE, 0, buffer_Size_float, &(*histogramComRgbFloat[i]).data()[0],nullptr);

				cl::Kernel histNormal = cl::Kernel(program, "hist_normal"); // Sets up normalisation kernel.
				histNormal.setArg(0, *rgbBuffersComNorm[i]);	
				histNormal.setArg(1, maximumValue);		
			
				queue.enqueueNDRangeKernel(histNormal, cl::NullRange, cl::NDRange(binNumber), cl::NullRange,nullptr);
				queue.enqueueReadBuffer(*rgbBuffersComNorm[i], CL_TRUE, 0, buffer_Size_float, &(*histogramComRgbFloat[i]).data()[0],nullptr);		
				
				// //  display_graph call
				// histogramGraph.display_graph("Histogram", 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraph.max(),true);	
				// histogramGraphCom.display_graph("Histogram", 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraphCom.max(),true);	

				// CImg<float> histogramGraph(binNumber, 1, 1, 1, 0); // Create a 1D CImg object for the raw histogram
				// for (int i = 0; i < binNumber; ++i) {
				// 	// int maxValue = *max_element(histogram.begin(), histogram.end());
				// 	histogramGraph(i) = static_cast<float>(histogram[i]);//maxValue; // Copy raw histogram values
				// }

				CImg<float> histogramGraphCom(binNumber, 1, 1, 1, 0); // Create a 1D CImg object for the raw histogram
				for (int j = 0; j < binNumber; ++j) {
					histogramGraphCom(j) = (*histogramComRgbFloat[i])[j]; // Copy raw histogram values
					// std::cout<< (*histogramComRgbFloat[i])[j]<<endl;
				}
				
				// // Sets histogram window size
				// CImgDisplay disp_raw(800, 600, "Raw Histogram");     
				CImgDisplay disp_com(800, 600, "Cumulative Histogram");

				// // Display histograms using the custom display objects
				// histogramGraph.display_graph(disp_raw, 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraph.max(),true);
				histogramGraphCom.display_graph(disp_com, 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraphCom.max(),true);					

			}
			if(!is16Bit){// Same as before but for rgb.
				cl::Kernel proj = cl::Kernel(program, "back_projectorRgb");
				proj.setArg(0, dev_image_input);	
				proj.setArg(1, dev_image_output);	
				proj.setArg(2, *rgbBuffersComNorm[0]);
				proj.setArg(3, *rgbBuffersComNorm[1]);
				proj.setArg(4, *rgbBuffersComNorm[2]);
				proj.setArg(5,binNumber);	
			
				queue.enqueueNDRangeKernel(proj, cl::NullRange, cl::NDRange(imageDimensions), cl::NullRange,nullptr);

				queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);
				//Outputs image using the fucntion at the top.
				CImg<unsigned char> output_image(output_buffer.data(), width, height, depth, spectrum);
				string output_name = "output_image.ppm";
				output_image.save("output_image.ppm");

				picture_output(output_name);
			}
			else{ // Once again same as before but for 16bit.
				
				binNumber = 1024; 
				size_t buffer_Size_float = binNumber * sizeof(float);
				float scale = (float)binNumber / 65536.0f; //Scale factor to be used to restore 16bit values for projection.

				std::cout << "imageDimensions: " << imageDimensions << std::endl;
				std::cout << "image_size: " << image_size << std::endl;
				std::cout << "Expected byte size: " << imageDimensions * 3 * sizeof(unsigned short) << std::endl; //Debug messages, was having issues with correct buffer sizes.

				cl::Kernel proj(program, "back_projector_rgb_16bit");
				proj.setArg(0, dev_image_input);
				proj.setArg(1, dev_image_output);
				proj.setArg(2, dev_histNormalR);
				proj.setArg(3, dev_histNormalG);
				proj.setArg(4, dev_histNormalB);
				proj.setArg(5, imageDimensions);
				proj.setArg(6, scale);

				//Re-defining values for larger 16bit image, unsigned short instead of char.
				size_t localWorkSize = 256;
				size_t globalWorkSize = ((imageDimensions + localWorkSize - 1) / localWorkSize) * localWorkSize; // Again global work with padding.
				vector<unsigned short> output_buffer(imageDimensions * 3); //Output buffer vector.
				size_t output_buffer_16bit = imageDimensions * 3 * sizeof(unsigned short); // Sizes output buffer..
				
				// Kernel execution error checking with some messages to specifically see what was wrong when I had issues with umnatched buffer sizes.
				cl_int err = queue.enqueueNDRangeKernel(proj, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize));
				if (err != CL_SUCCESS) {
					std::cerr << "Kernel execution failed: " << getErrorString(err) << std::endl;
					throw cl::Error(err, "enqueueNDRangeKernel"); //The function returns an error code if not successful.
				}								
				err = queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer_16bit, output_buffer.data()); //Similar to above. 
				if (err != CL_SUCCESS) {
					std::cerr << "clEnqueueReadBuffer failed: " << getErrorString(err) << std::endl;
					throw cl::Error(err, "clEnqueueReadBuffer");
				}
				// Finally the ouput image is produced. Using my function at the top.
				CImg<unsigned short> output_image(output_buffer.data(), width, height, depth, spectrum);
				string output_name = "output_image_16bit.ppm";
				output_image.save(output_name.c_str());				
				input16(output_name);
			}
		}			
	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}