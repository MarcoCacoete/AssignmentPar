#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"


// This host code includes host code provided for the Tutorial 2 tasks in the workshops. Mostly the openCL setup.

using namespace cimg_library;
using namespace std;

void metricsMaker(vector<tuple<string, cl::Event, int, int>> event_log, int spectrum ,int is16bit){

		int eventTime; // Variable for per event execution time.
				double totalTime = 0.0; // This is for the accumulated time of the events.
				std::cout << "\nTiming Results (in milliseconds):\n"; 
				for (int i = 0; i < event_log.size(); i++) { // Iterates through a vector I defined at top of code with the tuples with 4 elements.
					if (get<2>(event_log[i]) == spectrum && get<3>(event_log[i]) == is16bit) { // Crosschecks for 8bit, this is hardcoded at the end of pipeline.
						cl_ulong startTime = get<1>(event_log[i]).getProfilingInfo<CL_PROFILING_COMMAND_START>(); // Extracts beginning and end timestamps to 
						cl_ulong endTime = get<1>(event_log[i]).getProfilingInfo<CL_PROFILING_COMMAND_END>();	// Subtracts their values for the duration.
						double durationMs = (endTime - startTime) / 1e6; // This calculates the duration of an event and converts to milliseconds.
				
						cout << get<0>(event_log[i]) << " time to process: " << durationMs << " ms\n"; // Message output.
						totalTime += durationMs; // Accumulator for the durations to calculate total.
					}
				}
				std::cout << "Total time for pipeline in milliseconds: " << totalTime << " ms\n";// Total time for whole pipeline.
}

// Two functions to print out the images, intput and output. Greyscale 8 and 16 bit.
CImg<unsigned char> picture_output(const std::string& image_filename){
	CImg<unsigned char> image_input(image_filename.c_str());
	

	int pic_width = image_input.width();  // Assigns various image attirubtes to variables.
	int pic_height = image_input.height();
	int window_width = image_input.width();  
	int window_height = image_input.height(); 
    if (pic_width > 5000 || pic_height>2000) {
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
	int pic_height = img16.height();
    int window_width = img16.width();
    int window_height = img16.height();

    if (pic_width > 5000 || pic_height>2000) {
        window_width = img16.width() / 3;
        window_height = img16.height() / 3;
    }
    CImg<unsigned char> img8 = img16.get_normalize(0, 255); //Normalises image for display purposes.

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


	for (int i = 1; i < argc; i++) { // More code from workshops, accepts options selected by user, assigns flags to various variables or calls print help function.
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
															// 16 bit image. Not as efficient as checking header but works for more image types.
		int valCheck=0;
		for (int i=0;i<img16.size();i++){
			if(img16[i]>255)
			valCheck = img16[i];			
		}

		bool is16Bit = false;// Boolean to flip if it is a 16bit image.
	
		if (valCheck <= 255) {
			std::cout << "8-bit image detected." << std::endl;// Small block to show image type and dispaly it.
			image_input = picture_output(image_filename); 
	
		} else {
			std::cout << "16-bit image detected." << std::endl;
			input16(image_filename); 
			is16Bit = true; //Flips boolean to true.
		}
		#define RED     "\033[31m" // Red text so that the user can spot this instruction to proceed.
		#define RESET   "\033[0m"
		std::cout<<"Please close all images and histograms as they pop open to proceed."<<endl;

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
		// This is to keep timing of events for metrics in milliseconds. First element for label, then the event object then image type being processed.
		vector<tuple<string, cl::Event, int, int>> event_log; //  0 = 8-bit Greyscale, 1 = 8-bit RGB, 2 = 16-bit Greyscale, 3 = 16-bit RGB finally 8 or 16 bit.


		int width = is16Bit? img16.width() : image_input.width(); // Creates various necessary variables holding image metadata.
		int height = is16Bit? img16.height() : image_input.height(); // Picks appropriate value if 16bit or not.
		int spectrum = is16Bit? img16.spectrum() : image_input.spectrum();
		int depth = is16Bit? img16.depth() : image_input.depth();
		int imageDimensions = width*height;

		//Image size in bytes.
		size_t image_size = is16Bit ? imageDimensions * spectrum * sizeof(unsigned short) : imageDimensions * spectrum * sizeof(unsigned char);

		int binNumber = 256; // Defines bin numbers for greyscale, 8bit or 16bit.
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
		cl::Buffer dev_cumHistogram(context, CL_MEM_READ_WRITE, buffer_Size); 
		cl::Buffer dev_histNormal(context, CL_MEM_READ_WRITE, buffer_Size_float);
		cl::Buffer dev_histR(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_histG(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_histB(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_histRcum(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_histGcum(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_histBcum(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_histGrey	(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_histGreycum	(context, CL_MEM_READ_WRITE, buffer_Size);


		//4.1 Copy images to device memory
		
		if(!is16Bit){ // Queues Write buffers with different sizes depending on type of image 8 or 16bit.
			cl::Event e_input_write; // Creates new event profile for timestamp subtraction so that execution time can be calculated.
			queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_size, &image_input.data()[0],nullptr, &e_input_write); //Marks event with event name.
			event_log.push_back({"Input Image Write Buffer", e_input_write, 0,8});// Appends event to event_log vector which holds label event spectrum and 8/16-bit info.

			cl::Event e_output_write;
			queue.enqueueWriteBuffer(dev_image_output, CL_TRUE, 0, image_size, &image_input.data()[0],nullptr, &e_output_write);
			event_log.push_back({"Output Image Write Buffer", e_output_write, 0,8});
			
		}else{
			cl::Event e_input_write;
			queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_size, &img16.data()[0],nullptr,&e_input_write);
			event_log.push_back({"Input Image Write Buffer", e_input_write, 0,16});
			cl::Event e_output_write;
			queue.enqueueWriteBuffer(dev_image_output, CL_TRUE, 0, image_size, &img16.data()[0],nullptr,&e_output_write);
			event_log.push_back({"Output Image Write Buffer", e_output_write, 0,16});

		}	
		

		//4.2 Setup and execute the kernel (i.e. device code)
		// Enqueues write buffer for 8bit greyscale histogram, struggled with scope in this project, so the odd structure is due to scope issues and
		// not being able to separate all calls into functions without major refactoring far into the project.
		vector<int> histogram (binNumber,0); // Histogram for greyscale.
		cl::Event e_histogram_write;	
		queue.enqueueWriteBuffer(dev_intensityHistogram, CL_TRUE, 0, buffer_Size, &histogram.data()[0],nullptr,&e_histogram_write);
		event_log.push_back({"Intensity Histogram Write Buffer", e_histogram_write, 0,8});
		

		bool check = false; // Check for if conditions are met to break from while.
		
		while(!check){ // While not check used to make sure user inputs correct options.

			if(spectrum==1 && !is16Bit){ // Spectrum 1 matches greyscale.
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
					kernelAtom.setArg(2, imageDimensions); 
					
					// Enqueued with global and local work size for local memory work, for efficiency.
					cl::Event e_atom_enqueue;
					queue.enqueueNDRangeKernel(kernelAtom, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize),nullptr, &e_atom_enqueue);
					event_log.push_back({"Atomic Histogram kernel", e_atom_enqueue, 0,8});

				}
				else if(kernelType=="local"){ // Same as above but for other local memory kernel.
					std::cout<<"Local"<<endl;
					check = true;
					cl::Kernel kernelLocal = cl::Kernel(program, "hist_local");cl::Kernel kernelHistLocal = cl::Kernel(program, "hist_local");
					kernelHistLocal.setArg(0, dev_image_input);
					kernelHistLocal.setArg(1, dev_intensityHistogram);
					kernelHistLocal.setArg(2, cl::Local(buffer_Size));
					kernelHistLocal.setArg(3, binNumber);
					kernelHistLocal.setArg(4, imageDimensions);	
					cl::Event e_local_enqueue;
					queue.enqueueNDRangeKernel(kernelHistLocal, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize),nullptr, &e_local_enqueue);
					event_log.push_back({"Local memory Histogram kernel", e_local_enqueue, 0,8});

				}
				else{
					std::cout<<"Invalid input. Please enter either Atom or Local"<<endl;
				}
			}
			else if(spectrum == 1 && is16Bit){
				check=true;				

				std::cout<<"16 bit greyscale image detected."<<endl; // Same repeated steps as above but for RGB 16bit.
				binNumber = 1024; // Necessary due to the astronomical number of pixel values for 16bit.
				buffer_Size = binNumber * sizeof(int); // Update buffer size to match 1024 bins

				vector <int> hist16Grey (binNumber,0);			
				cl::Event e_histGrey_write_buffer;
				queue.enqueueWriteBuffer(dev_histGrey, CL_TRUE, 0, buffer_Size, hist16Grey.data(),nullptr,&e_histGrey_write_buffer);
				event_log.push_back({"Intensity histogram buffer write", e_histGrey_write_buffer, 0,16});			

				cl::Kernel kernel(program, "hist_greyscale_16bit");
                kernel.setArg(0, dev_image_input);
                kernel.setArg(1, dev_histGrey);                
                kernel.setArg(2, imageDimensions);
				kernel.setArg(3,binNumber);
				cl::Event e_histogram_grey;
                queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize), nullptr,&e_histogram_grey);
				event_log.push_back({"Local histogram kernel", e_histGrey_write_buffer, 0,16});			

				cl::Event e_histogram_read_buffer;
				queue.enqueueReadBuffer(dev_histGrey, CL_TRUE, 0, buffer_Size, hist16Grey.data(),nullptr,&e_histogram_read_buffer);
				event_log.push_back({"Local histogram read buffer", e_histogram_read_buffer, 0,16});		

				CImg<int> histogramGraph(binNumber, 1, 1, 1, 0); // Creates a 1D CImg object for the raw histogram
				for (int i = 0; i < binNumber; ++i) {
					// int maxValue = *max_element(histogram.begin(), histogram.end());
					histogramGraph(i) =hist16Grey[i]; // Copies raw histogram values
				}			
				// Sets histogram window size
				CImgDisplay disp_raw(800, 600, "Raw Histogram");     

				// Display histograms using the custom display objects for greyscale images.
				histogramGraph.display_graph(disp_raw, 3, 1, "VALUES", 0, binNumber, "COUNT PER BIN", 0, histogramGraph.max(), false);
			}else if(!is16Bit){ // The 16bit and 8bitrgb pictures have their oown different kernels

				check=true;				
				std::cout<<"Colour image detected"<<endl;// Vectors for output to user in CImg. Zeroed.
				vector <int> histR (binNumber,0);
				vector <int> histG (binNumber,0);
				vector <int> histB (binNumber,0);		
				
				cl::Event e_histR_write_buffer;
				queue.enqueueWriteBuffer(dev_histR, CL_TRUE, 0, buffer_Size, histR.data(),nullptr, &e_histR_write_buffer);
				event_log.push_back({"Histogram write buffer Red", e_histR_write_buffer, 1,8});		
				cl::Event e_histG_write_buffer;
				queue.enqueueWriteBuffer(dev_histG, CL_TRUE, 0, buffer_Size, histG.data(),nullptr, &e_histG_write_buffer);
				event_log.push_back({"Histogram write buffer Green", e_histG_write_buffer, 1,8});		
				cl::Event e_histB_write_buffer;
				queue.enqueueWriteBuffer(dev_histB, CL_TRUE, 0, buffer_Size, histB.data(),nullptr, &e_histB_write_buffer);
				event_log.push_back({"Histogram write buffer Blue", e_histB_write_buffer, 1,8});		

				cl::Kernel kernelHistRgb = cl::Kernel(program, "hist_rgb"); //Kernel call for rgb histogram kernel.
				kernelHistRgb.setArg(0, dev_image_input);
				kernelHistRgb.setArg(1, dev_histR);
				kernelHistRgb.setArg(2, dev_histG);
				kernelHistRgb.setArg(3, dev_histB);
				kernelHistRgb.setArg(4,imageDimensions);
				
				cl::Event e_histogram_rgb_kernel;
				queue.enqueueNDRangeKernel(kernelHistRgb, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize),nullptr, &e_histogram_rgb_kernel);
				event_log.push_back({"Local histogram kernel", e_histogram_rgb_kernel, 1,8});		

				cl::Event e_histR_read_buffer;
				queue.enqueueReadBuffer(dev_histR, CL_TRUE, 0, buffer_Size, &histR.data()[0],nullptr, &e_histR_read_buffer);//Queued read buffers for hists.
				event_log.push_back({"Histogram read buffer Red", e_histR_read_buffer, 1,8});		

				cl::Event e_histG_read_buffer;
				queue.enqueueReadBuffer(dev_histG, CL_TRUE, 0, buffer_Size, &histG.data()[0],nullptr, &e_histG_read_buffer);
				event_log.push_back({"Histogram read buffer Green", e_histG_read_buffer, 1,8});		

				cl::Event e_histB_read_buffer;
				queue.enqueueReadBuffer(dev_histB, CL_TRUE, 0, buffer_Size, &histB.data()[0],nullptr, &e_histB_read_buffer);
				event_log.push_back({"Histogram read buffer Blue", e_histB_read_buffer, 1,8});		


				vector <vector<int>*> histRgb = {&histR,&histG,&histB}; //Vector of the hists to iterate for outputs.

				for (int i = 0; i < histRgb.size(); i++) { //Creates cimg object for histograms.
					CImg<float> histogramGraphRgb(binNumber, 1, 1, 1, 0);
					for (int j = 0; j < binNumber; ++j) {
						histogramGraphRgb(j) = static_cast<float>((*histRgb[i])[j]); 
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
					
					// Sets histogram window size and name.
					CImgDisplay disp_raw(800, 600, histName);     

					// Display graph, with argument value 3 for bar chart, no real way of changing font sizes.				
					histogramGraphRgb.display_graph(disp_raw, 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraphRgb.max(),true);	
				}
			}else{
				check=true;	
			
				std::cout<<"Colour image detected"<<endl; // Same repeated steps as above but for RGB 16bit.
				binNumber = 1024; // Necessary due to the astronomical number of pixel values for 16bit, used after scaling down, then later scaled back up
				buffer_Size = binNumber * sizeof(int); // Update buffer size to match 1024 bins

				vector <int> histR (binNumber,0);
				vector <int> histG (binNumber,0);
				vector <int> histB (binNumber,0);		

				cl::Event e_histR_write_buffer;
				queue.enqueueWriteBuffer(dev_histR, CL_TRUE, 0, buffer_Size, histR.data(),nullptr, &e_histR_write_buffer);
				event_log.push_back({"Histogram write buffer Red", e_histR_write_buffer, 1,16});
				cl::Event e_histG_write_buffer;
				queue.enqueueWriteBuffer(dev_histG, CL_TRUE, 0, buffer_Size, histG.data(),nullptr,&e_histG_write_buffer);
				event_log.push_back({"Histogram write buffer Green", e_histG_write_buffer, 1,16});
				cl::Event e_histB_write_buffer;
				queue.enqueueWriteBuffer(dev_histB, CL_TRUE, 0, buffer_Size, histB.data(),nullptr,&e_histB_write_buffer);
				event_log.push_back({"Histogram write buffer Blue", e_histB_write_buffer, 1,16});		


				cl::Kernel kernel(program, "hist_rgb_16bit");
                kernel.setArg(0, dev_image_input);
                kernel.setArg(1, dev_histR);
                kernel.setArg(2, dev_histG);
                kernel.setArg(3, dev_histB);
                kernel.setArg(4, imageDimensions);
				kernel.setArg(5,binNumber);
				cl::Event e_histogram_rgb_kernel;
                queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize), nullptr, &e_histogram_rgb_kernel);
				event_log.push_back({"Local histogram kernel", e_histogram_rgb_kernel, 1,16});		
				
				cl::Event e_histR_read_buffer;
				queue.enqueueReadBuffer(dev_histR, CL_TRUE, 0, buffer_Size, histR.data(),nullptr, &e_histR_read_buffer);
				event_log.push_back({"Histogram read buffer Red", e_histR_read_buffer, 1,16});		

				cl::Event e_histG_read_buffer;
				queue.enqueueReadBuffer(dev_histG, CL_TRUE, 0, buffer_Size, histG.data(),nullptr, &e_histG_read_buffer);
				event_log.push_back({"Histogram read buffer Green", e_histG_read_buffer, 1,16});		

				cl::Event e_histB_read_buffer;
				queue.enqueueReadBuffer(dev_histB, CL_TRUE, 0, buffer_Size, histB.data(),nullptr, &e_histB_read_buffer);
				event_log.push_back({"Histogram read buffer Blue", e_histB_read_buffer, 1,16});		

				vector <vector<int>*> histRgb = {&histR,&histG,&histB};

				for (int i = 0; i < histRgb.size(); i++) { //Creates cimg object for histograms.
					CImg<float> histogramGraphRgb(binNumber, 1, 1, 1, 0);
					for (int j = 0; j < binNumber; ++j) {
						histogramGraphRgb(j) = static_cast<float>((*histRgb[i])[j]); 
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
					
					// Sets histogram window size and name.
					CImgDisplay disp_raw(800, 600, histName);     

					// Display graph, with argument value 3 for bar chart, no real way of changing font sizes.				
					histogramGraphRgb.display_graph(disp_raw, 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraphRgb.max(),true);	
				}
			}
		}

	
		//4.3 Copy the result from device to host
		
		// Reads for greyscale histogram buffer. Some of the lines before while loop are here so they are visible in following scopes.
		cl::Event e_devHist_read_buffer;
		queue.enqueueReadBuffer(dev_intensityHistogram, CL_TRUE, 0, buffer_Size, &histogram.data()[0],nullptr,&e_devHist_read_buffer);
		event_log.push_back({"Histogram read buffer", e_devHist_read_buffer, 0,8});


		int maxValue = *max_element(histogram.begin(), histogram.end()); // Defines max value for normalisation logic.	 		
		
		vector<int> histogramcum (binNumber,0); //Defines cumulative histogram and buffer.
		cl::Event e_cumulative_hist_write_buffer;
		queue.enqueueWriteBuffer(dev_cumHistogram, CL_TRUE, 0, buffer_Size, &histogram.data()[0],nullptr,&e_cumulative_hist_write_buffer);
		event_log.push_back({"Cumulative histogram write buffer", e_devHist_read_buffer, 0,8});

		vector<int> histogramcumR(binNumber,0);// Same as above but for colour images.
		vector<int> histogramcumG(binNumber,0);
		vector<int> histogramcumB(binNumber,0);

		//Cumulative write buffers for cumulative histograms RGB
		cl::Event e_cumulative_hist_write_buffer_R;
		queue.enqueueWriteBuffer(dev_histRcum, CL_TRUE, 0, buffer_Size, &histogramcumR.data()[0],nullptr,&e_cumulative_hist_write_buffer_R);
		event_log.push_back({"Cumulative histogram write buffer Red", e_cumulative_hist_write_buffer_R, 1,16});
		cl::Event e_cumulative_hist_write_buffer_G;
		queue.enqueueWriteBuffer(dev_histGcum, CL_TRUE, 0, buffer_Size, &histogramcumG.data()[0],nullptr,&e_cumulative_hist_write_buffer_G);
		event_log.push_back({"Cumulative histogram write buffer Green", e_cumulative_hist_write_buffer_G, 1,16});
		cl::Event e_cumulative_hist_write_buffer_B;
		queue.enqueueWriteBuffer(dev_histBcum, CL_TRUE, 0, buffer_Size, &histogramcumB.data()[0],nullptr,&e_cumulative_hist_write_buffer_B);
		event_log.push_back({"Cumulative histogram write buffer Blue", e_cumulative_hist_write_buffer_B, 1,16});

		vector<cl::Buffer*> rgbBuffers = {&dev_histR, &dev_histG, &dev_histB}; // Some vectors of pointers for indexing and iteration.
		vector<cl::Buffer*> rgbBufferscum = {&dev_histRcum, &dev_histGcum, &dev_histBcum};		
		vector<vector<int>> histogramcumRgb = {histogramcumR,histogramcumG,histogramcumB};

		check = false;

		while(!check){ // Same as above but for cumulative histogram kernels.
			std::cout<<"What cumulative histogram kernel would you like to use. Hillis or Blelloch?"<<endl;
			string kernelType;
			cin>>kernelType;
			std::transform(kernelType.begin(),kernelType.end(),kernelType.begin(),::tolower);

			//Choices between the hillis adapted kernel and Blelloch from workshops.
			if(spectrum==1 && !is16Bit){
				if (kernelType=="hillis"){
					std::cout<<"Hillis-Steele"<<endl;
					check = true;
					cl::Kernel kernelcum = cl::Kernel(program, "cum_hist"); // Same as before only differences are the names of the kernels picked.
					kernelcum.setArg(0, dev_intensityHistogram);		
					kernelcum.setArg(1, dev_cumHistogram);

					// The global work item number is based on bin numbers for the cumulative kernels.
					cl::Event e_hillis_histcum;
					queue.enqueueNDRangeKernel(kernelcum, cl::NullRange, cl::NDRange(binNumber), cl::NDRange(binNumber),nullptr,&e_hillis_histcum);
					event_log.push_back({"Hillis cumulative histogram", e_hillis_histcum, 0,8});

				}
				else if(kernelType=="blelloch"){
					std::cout<<"Blelloch"<<endl;
					check = true;			
					cl::Kernel kernelcum = cl::Kernel(program, "scan_bl");
					kernelcum.setArg(0, dev_cumHistogram);
					cl::Event e_Blelloch_histcum;
					queue.enqueueNDRangeKernel(kernelcum, cl::NullRange, cl::NDRange(binNumber), cl::NDRange(binNumber),nullptr,&e_Blelloch_histcum);
					event_log.push_back({"Blelloch cumulative histogram", e_Blelloch_histcum, 0,8});
				}
				else{
					std::cout<<"Invalid input. Please enter either Scan or Blelloch"<<endl;
				}
				cl::Event e_cumHistReadBuffer;
				queue.enqueueReadBuffer(dev_cumHistogram, CL_TRUE, 0, buffer_Size, &histogramcum.data()[0],nullptr,&e_cumHistReadBuffer);
				event_log.push_back({"Cumulative histogram read buffer", e_cumHistReadBuffer, 0,8});
				// Cimg output of histograms
				CImg<int> histogramGraph(binNumber, 1, 1, 1, 0); // Creates a 1D CImg object for the raw histogram
				for (int i = 0; i < binNumber; ++i) {
					// int maxValue = *max_element(histogram.begin(), histogram.end());
					histogramGraph(i) =histogram[i];//maxValue; // Copies raw histogram values
				}
					// Sets histogram window size
				CImgDisplay disp_raw(800, 600, "Raw Histogram");     
				histogramGraph.display_graph(disp_raw, 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraph.max(),true); 
				CImg<int> histogramGraphcumUnnorm(binNumber, 1, 1, 1, 0); //Creates cimg object using data from histogram vector.
				for (int i = 0; i < binNumber; ++i) {
					histogramGraphcumUnnorm(i) = histogramcum[i];
				}
				CImgDisplay disp_cum_unnorm(800, 600, "Unnormalised Cumulative Histogram (8-bit Greyscale)"); // Labels and sets window size. And bar type graph value 3 as second argument.
				histogramGraphcumUnnorm.display_graph(disp_cum_unnorm, 3, 1, "VALUES", 0, 255, "COUNT PER BIN", 0, histogramGraphcumUnnorm.max(), true);

			}	
			else if(!is16Bit){				

				for(int i=0;i<rgbBuffers.size();i++){ // Same as above but for 8bit RGB runs 3 times once per RGB channel.
					if (kernelType=="hillis"){
						std::cout<<"Hillis-Steele"<<endl;
						check = true;
						cl::Kernel kernelcum = cl::Kernel(program, "cum_hist"); // Indexed Histograms get passed as arguments and written on when outputted.
						kernelcum.setArg(0, *rgbBuffers[i]);		// De referenced pointers.
						kernelcum.setArg(1, *rgbBufferscum[i]);
						cl::Event e_hillis_histcum;
						queue.enqueueNDRangeKernel(kernelcum, cl::NullRange, cl::NDRange(binNumber), cl::NDRange(binNumber),nullptr,&e_hillis_histcum);
						event_log.push_back({"Hillis cumulative histogram", e_hillis_histcum, 1,8});
						cl::Event e_cumHistReadBuffer;
						queue.enqueueReadBuffer(*rgbBufferscum[i], CL_TRUE, 0, buffer_Size, &histogramcumRgb[i].data()[0], nullptr,&e_cumHistReadBuffer);
						event_log.push_back({"Cumulative histogram read buffer", e_cumHistReadBuffer, 1,8});

					}
					else if(kernelType=="blelloch"){ //Same as above. Input and output histogram gets overwritten.
						std::cout<<"Blelloch"<<endl;
						check = true;			
						cl::Kernel kernelcum = cl::Kernel(program, "scan_bl");
						kernelcum.setArg(0, *rgbBuffers[i]);
						cl::Event e_Blelloch_histcum;
						queue.enqueueNDRangeKernel(kernelcum, cl::NullRange, cl::NDRange(binNumber), cl::NDRange(binNumber),nullptr,&e_Blelloch_histcum);
						event_log.push_back({"Blelloch cumulative histogram", e_Blelloch_histcum, 1,8});
						cl::Event e_cumHistReadBuffer;
						queue.enqueueReadBuffer(*rgbBuffers[i], CL_TRUE, 0, buffer_Size, &histogramcumRgb[i].data()[0], nullptr,&e_cumHistReadBuffer);
						event_log.push_back({"Cumulative histogram read buffer", e_cumHistReadBuffer, 1,8});

					}
					else{
						std::cout<<"Invalid input. Please enter either Scan or Blelloch"<<endl;
					}
				}
				vector<const char*> histNames = {"Red Unnormalised Cumulative Histogram", "Green Unnormalised Cumulative Histogram", "Blue Unnormalised Cumulative Histogram"};
				for (int i = 0; i < 3; i++) {
					CImg<int> histogramGraphcumUnnorm(binNumber, 1, 1, 1, 0);
					for (int j = 0; j < binNumber; ++j) {
						histogramGraphcumUnnorm(j) = histogramcumRgb[i][j];
					}
					CImgDisplay disp_cum_unnorm(800, 600, histNames[i]);
					histogramGraphcumUnnorm.display_graph(disp_cum_unnorm, 3, 1, "VALUES", 0, 255, "COUNT PER BIN", 0, histogramGraphcumUnnorm.max(), true);
				}
			}else if(is16Bit &&spectrum ==1 ){		

				size_t buffer_Size = sizeof(int) * binNumber; 
				if (kernelType=="hillis"){
					std::cout<<"Hillis-Steele"<<endl;
					check = true;
					cl::Kernel kernelcum = cl::Kernel(program, "cum_hist");
					kernelcum.setArg(0, dev_histGrey);		
					kernelcum.setArg(1, dev_histGreycum);
					cl::Event e_hillis_histcum;
					queue.enqueueNDRangeKernel(kernelcum, cl::NullRange, cl::NDRange(binNumber), cl::NDRange(binNumber),nullptr,&e_hillis_histcum);
					event_log.push_back({"Hillis cumulative histogram", e_hillis_histcum, 0,16});
					cl::Event e_cumHistReadBuffer;
					queue.enqueueReadBuffer(dev_histGreycum, CL_TRUE, 0, buffer_Size, &histogramcum.data()[0], nullptr,&e_cumHistReadBuffer);
					event_log.push_back({"Cumulative histogram read buffer", e_cumHistReadBuffer, 0,16});
				}
				else if(kernelType=="blelloch"){
					std::cout<<"Blelloch"<<endl;
					check = true;			
					cl::Kernel kernelcum = cl::Kernel(program, "scan_bl");
					kernelcum.setArg(0, dev_histGrey);
					cl::Event e_Blelloch_histcum;
					queue.enqueueNDRangeKernel(kernelcum, cl::NullRange, cl::NDRange(binNumber), cl::NDRange(binNumber),nullptr,&e_Blelloch_histcum);
					event_log.push_back({"Blelloch cumulative histogram", e_Blelloch_histcum, 0,16});
					cl::Event e_cumHistReadBuffer;
					queue.enqueueReadBuffer(dev_histGrey, CL_TRUE, 0, buffer_Size, &histogramcum.data()[0], nullptr,&e_cumHistReadBuffer);
					event_log.push_back({"Cumulative histogram read buffer", e_cumHistReadBuffer, 0,16});

					
				}
				else{
					std::cout<<"Invalid input. Please enter either Scan or Blelloch"<<endl;
				}

				CImg<int> histogramGraphcumUnnorm(binNumber, 1, 1, 1, 0);
				for (int i = 0; i < binNumber; ++i) {
					histogramGraphcumUnnorm(i) = histogramcum[i];
				}
				CImgDisplay disp_cum_unnorm(800, 600, "Unnormalised Cumulative Histogram (16-bit Greyscale)");
				histogramGraphcumUnnorm.display_graph(disp_cum_unnorm, 3, 1, "VALUES", 0, binNumber - 1, "COUNT PER BIN", 0, histogramGraphcumUnnorm.max(), true);

				int maximumValue = histogramcum[binNumber - 1];
				float maximumBinValue = static_cast<float>(maximumValue);

				// Converts intermediate results to floats for normalisation
				vector<float> histogramcumFloat(binNumber, 0.0f); // New float vector
				for (int i = 0; i < binNumber; ++i) {
					histogramcumFloat[i] = static_cast<float>(histogramcum[i]); // Convert int to float
				}

				buffer_Size_float = binNumber * sizeof(float); 		
				cl::Event e_cum_write_buffer;	
				queue.enqueueWriteBuffer(dev_histNormal, CL_TRUE, 0, buffer_Size_float, &histogramcumFloat.data()[0],nullptr,&e_cum_write_buffer);
				event_log.push_back({"Normalised histogram write buffer", e_cum_write_buffer, 0,16});

				cl::Kernel histNormal = cl::Kernel(program, "hist_normal");// Same as all previous kernels. 
				histNormal.setArg(0, dev_histNormal);	
				histNormal.setArg(1, maximumBinValue);		
				cl::Event e_hist_norm;	
				queue.enqueueNDRangeKernel(histNormal, cl::NullRange, cl::NDRange(binNumber), cl::NullRange,nullptr,&e_hist_norm);
				event_log.push_back({"Histogram normaliser kernel", e_hist_norm, 0,16});				


				cl::Event e_hist_norm_read_buffer;	
				queue.enqueueReadBuffer(dev_histNormal, CL_TRUE, 0, buffer_Size_float, &histogramcumFloat.data()[0],nullptr,&e_hist_norm_read_buffer);
				event_log.push_back({"Normalised histogram read buffer", e_hist_norm_read_buffer, 0,16});				

				CImg<float> histogramGraphcum(binNumber, 1, 1, 1, 0); // Create a 1D CImg object for the raw histogram
				for (int  i= 0;  i< binNumber; ++i) {
					histogramGraphcum(i) = histogramcumFloat[i]; // Copy raw histogram values
				}

				// // Sets histogram window size		   
				CImgDisplay disp_cum(800, 600, "Cumulative Histogram");

				// // Display histograms using the custom display objects
				histogramGraphcum.display_graph(disp_cum, 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraphcum.max());	

				binNumber = 1024; 
				size_t buffer_Size_float = binNumber * sizeof(float);
				float scale = (float)binNumber / 65536.0f; //Scale factor to be used to restore 16bit values for projection.

				std::cout << "imageDimensions: " << imageDimensions << std::endl;
				std::cout << "image_size: " << image_size << std::endl;
				std::cout << "Expected byte size: " << imageDimensions * 3 * sizeof(unsigned short) << std::endl; //Debug messages, was having issues with correct buffer sizes.

				cl::Kernel proj(program, "back_projector_grayscale_16bit");
				proj.setArg(0, dev_image_input);
				proj.setArg(1, dev_image_output);
				proj.setArg(2, dev_histNormal);				
				proj.setArg(3, imageDimensions);
				proj.setArg(4, scale);

				vector<unsigned short> output_buffer(imageDimensions); // 272640 elements, 2 bytes each
				cl::Event e_back_proj;	
				queue.enqueueNDRangeKernel(proj, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize),nullptr,&e_back_proj);
				event_log.push_back({"Image back projection kernel", e_back_proj, 0,16});			
				cl::Event e_out_img_read_buffer;	
				queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, imageDimensions * sizeof(unsigned short), output_buffer.data(),nullptr,&e_out_img_read_buffer);
				event_log.push_back({"Output image read buffer", e_out_img_read_buffer, 0,16});		
				
				//This little block is in charge of outputting metrics for time to execute, for memory and kernel executions. 
				queue.finish();// Make sure all operations are finished.
				metricsMaker(event_log,0,16);		


				CImg<unsigned short> output_image(output_buffer.data(), width, height, depth, spectrum);
				string output_name = "output_image_16bitGreyscale.pgm";
				output_image.save_pnm(output_name.c_str(), 65535); // Saves as 16-bit
				input16(output_name);
			}
			else{
				for(int i=0;i<rgbBuffers.size();i++){ // Same as before but for 16bit.
					size_t buffer_Size = sizeof(int) * binNumber; // Ensures this matches the buffer size
					if (kernelType=="hillis"){
						std::cout<<"Hillis-Steele"<<endl;
						check = true;
						cl::Kernel kernelcum = cl::Kernel(program, "cum_hist");
						kernelcum.setArg(0, *rgbBuffers[i]);		
						kernelcum.setArg(1, *rgbBufferscum[i]);

						cl::Event e_hillis_kernel;	
						queue.enqueueNDRangeKernel(kernelcum, cl::NullRange, cl::NDRange(binNumber), cl::NDRange(binNumber),nullptr,&e_hillis_kernel);
						event_log.push_back({"Hillis Cumulative histogram", e_hillis_kernel, 1,16});		

						cl::Event e_cumHist_read_buffer;	
						queue.enqueueReadBuffer(*rgbBufferscum[i], CL_TRUE, 0, buffer_Size, &histogramcumRgb[i].data()[0], nullptr,&e_cumHist_read_buffer);
						event_log.push_back({"Cumulative histogram read buffer", e_cumHist_read_buffer, 1,16});		

					}
					else if(kernelType=="blelloch"){
						std::cout<<"Blelloch"<<endl;
						check = true;			
						cl::Kernel kernelcum = cl::Kernel(program, "scan_bl");
						kernelcum.setArg(0, *rgbBuffers[i]);
						cl::Event e_blelloch_kernel;	
						queue.enqueueNDRangeKernel(kernelcum, cl::NullRange, cl::NDRange(binNumber), cl::NDRange(binNumber),nullptr,&e_blelloch_kernel);
						event_log.push_back({"Blelloch cumulative histogram", e_blelloch_kernel, 1,16});		

						cl::Event e_cumHist_read_buffer;
						queue.enqueueReadBuffer(*rgbBuffers[i], CL_TRUE, 0, buffer_Size, &histogramcumRgb[i].data()[0], nullptr,&e_cumHist_read_buffer);
						event_log.push_back({"Cumulative histogram read buffer", e_cumHist_read_buffer, 1,16});		

					}
					else{
						std::cout<<"Invalid input. Please enter either Scan or Blelloch"<<endl;
					}
				}
				vector<const char*> histNames = {"Red Unnormalised Cumulative Histogram (16-bit)", "Green Unnormalised Cumulative Histogram (16-bit)", "Blue Unnormalised Cumulative Histogram (16-bit)"};
				for (int i = 0; i < 3; i++) {
					CImg<int> histogramGraphcumUnnorm(binNumber, 1, 1, 1, 0);
					for (int j = 0; j < binNumber; ++j) {
						histogramGraphcumUnnorm(j) = histogramcumRgb[i][j];
					}
					CImgDisplay disp_cum_unnorm(800, 600, histNames[i]);
					histogramGraphcumUnnorm.display_graph(disp_cum_unnorm, 3, 1, "VALUES", 0, binNumber - 1, "COUNT PER BIN", 0, histogramGraphcumUnnorm.max(), true);
				}

			}			
		}
		// This block is responsible for normalisation and back projection kernel setup and calls.
		if(spectrum==1 && !is16Bit){// For Greyscale images.

			int maximumValue = histogramcum[binNumber - 1];
			float maximumBinValue = static_cast<float>(maximumValue);

			// Converts intermediate results to floats for normalisation
			vector<float> histogramcumFloat(binNumber, 0.0f); // New float vector
			for (int i = 0; i < binNumber; ++i) {
				histogramcumFloat[i] = static_cast<float>(histogramcum[i]); // Convert int to float
			}
			cl::Event e_write_buffer_hist_norm;
			queue.enqueueWriteBuffer(dev_histNormal, CL_TRUE, 0, buffer_Size_float, &histogramcumFloat.data()[0],nullptr,&e_write_buffer_hist_norm);
			event_log.push_back({"Histogram normaliser write buffer", e_write_buffer_hist_norm, 0,8});

			cl::Kernel histNormal = cl::Kernel(program, "hist_normal");// Same as all previous kernels. 
			histNormal.setArg(0, dev_histNormal);	
			histNormal.setArg(1, maximumBinValue);		

			cl::Event e_hist_normal_kernel;
			queue.enqueueNDRangeKernel(histNormal, cl::NullRange, cl::NDRange(binNumber), cl::NullRange,nullptr,&e_hist_normal_kernel);
			event_log.push_back({"Histogram normaliser kernel", e_hist_normal_kernel, 0,8});
			cl::Event e_read_buffer_hist_norm;
			queue.enqueueReadBuffer(dev_histNormal, CL_TRUE, 0, buffer_Size_float, &histogramcumFloat.data()[0],nullptr,&e_read_buffer_hist_norm);	
			event_log.push_back({"Histogram normaliser read buffer", e_hist_normal_kernel, 0,8});
	
			
			cl::Kernel proj = cl::Kernel(program, "back_projector");
			proj.setArg(0, dev_image_input);	
			proj.setArg(1, dev_image_output);	
			proj.setArg(2, dev_histNormal);
			proj.setArg(3, imageDimensions); 	

			cl::Event e_back_projector;
			queue.enqueueNDRangeKernel(proj, cl::NullRange, cl::NDRange(image_size), cl::NullRange,nullptr,&e_back_projector);
			event_log.push_back({"Back projection kernel", e_hist_normal_kernel, 0,8});			

			vector<unsigned char> output_buffer(image_size);
			cl::Event e_output_buffer_read;
			queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0],nullptr,&e_output_buffer_read);
			event_log.push_back({"Output buffer read", e_output_buffer_read, 0,8});			

			CImg<float> histogramGraphcum(binNumber, 1, 1, 1, 0); // Creates a 1D CImg object for the raw histogram
			for (int i = 0; i < binNumber; ++i) {
				histogramGraphcum(i) = histogramcumFloat[i]; // Copies raw histogram values
			}			
			
			CImgDisplay disp_cum(800, 600, "Cumulative Histogram");

			// Display histograms using the custom display objects for greyscale images.
			histogramGraphcum.display_graph(disp_cum, 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraphcum.max(),true);	


			CImg<unsigned char> output_image(output_buffer.data(), width, height, depth, spectrum);
			string output_name = "output_image.pgm";
			output_image.save("output_image.pgm");
			picture_output(output_name);

			//This little block is in charge of outputting metrics for time to execute, for memory and kernel executions. 
			queue.finish();// Make sure all operations are finished.
			metricsMaker(event_log,0,8);				


		}else if(spectrum==3){// Same as above but for rgb images.
			
			if(is16Bit){ 
				buffer_Size_float= sizeof(float) * 1024;// Different values for 16bit.
				binNumber = 1024;		
			}

			cl::Buffer dev_histNormalR(context, CL_MEM_READ_WRITE, buffer_Size_float); //Buffer definitions.
			cl::Buffer dev_histNormalG(context, CL_MEM_READ_WRITE, buffer_Size_float);
			cl::Buffer dev_histNormalB(context, CL_MEM_READ_WRITE, buffer_Size_float);

			vector<cl::Buffer*> rgbBufferscumNorm = {&dev_histNormalR, &dev_histNormalG, &dev_histNormalB}; // Pointer vector holding the buffers.

			// Converts intermediate results to floats for normalisation
			vector<float> histogramcumFloatR(binNumber, 0.0f); // New float vectors for normalised cumulative histograms which require decimals.
			vector<float> histogramcumFloatG(binNumber, 0.0f); 
			vector<float> histogramcumFloatB(binNumber, 0.0f); 
			vector <vector<float>*> histogramcumRgbFloat = {&histogramcumFloatR,&histogramcumFloatG,&histogramcumFloatB}; //Another pointer vector.
			vector<unsigned char> output_buffer(image_size);

			float maximumValue;
			for(int i=0;i<histogramcumRgb.size();i++){				
				for (int j = 0; j < binNumber; ++j) 
				(*histogramcumRgbFloat[i])[j] = static_cast<float>(histogramcumRgb[i][j]); // Converts int to float, non parallel. It was a small job.
			}							

			for(int i=0;i<histogramcumRgb.size();i++){	
				maximumValue = histogramcumRgb[i][binNumber-1];
				maximumValue = static_cast<float>(maximumValue); // Max value for each different colour channel.
				cl::Event e_cum_write_buffer;	
				queue.enqueueWriteBuffer(*rgbBufferscumNorm[i], CL_TRUE, 0, buffer_Size_float, &(*histogramcumRgbFloat[i]).data()[0],nullptr,&e_cum_write_buffer);
				event_log.push_back({"Cumulative histogram write buffer", e_cum_write_buffer, 1,8});
				cl::Kernel histNormal = cl::Kernel(program, "hist_normal"); // Sets up normalisation kernel.
				histNormal.setArg(0, *rgbBufferscumNorm[i]);	
				histNormal.setArg(1, maximumValue);		
				cl::Event e_normaliser_kernel;	
				queue.enqueueNDRangeKernel(histNormal, cl::NullRange, cl::NDRange(binNumber), cl::NullRange,nullptr,&e_normaliser_kernel);
				event_log.push_back({"Histogram normaliser kernel", e_normaliser_kernel, 1,8});
				cl::Event e_cum_read_buffer;	
				queue.enqueueReadBuffer(*rgbBufferscumNorm[i], CL_TRUE, 0, buffer_Size_float, &(*histogramcumRgbFloat[i]).data()[0],nullptr,&e_cum_read_buffer);	
				event_log.push_back({"Normalised cumulative histogram read buffer", e_cum_read_buffer, 1,8});

				CImg<float> histogramGraphcum(binNumber, 1, 1, 1, 0); // Create a 1D CImg object for the raw histogram
				for (int j = 0; j < binNumber; ++j) {
					histogramGraphcum(j) = (*histogramcumRgbFloat[i])[j]; // Copy raw histogram values
					// std::cout<< (*histogramcumRgbFloat[i])[j]<<endl;
				}
				
				// // Sets histogram window size
				CImgDisplay disp_cum(800, 600, "Cumulative Histogram");

				// // Display histograms using the custom display objects
				histogramGraphcum.display_graph(disp_cum, 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraphcum.max(),true);					

			}
			if(!is16Bit){// Same as before but for rgb.
				cl::Kernel proj = cl::Kernel(program, "back_projectorRgb");
				proj.setArg(0, dev_image_input);	
				proj.setArg(1, dev_image_output);	
				proj.setArg(2, *rgbBufferscumNorm[0]);
				proj.setArg(3, *rgbBufferscumNorm[1]);
				proj.setArg(4, *rgbBufferscumNorm[2]);
				proj.setArg(5,binNumber);	
				cl::Event e_back_proj_kernel;	
				queue.enqueueNDRangeKernel(proj, cl::NullRange, cl::NDRange(imageDimensions), cl::NullRange,nullptr,&e_back_proj_kernel);
				event_log.push_back({"Image back projection kernel", e_back_proj_kernel, 1,8});
				cl::Event e_output_image_read_kernel;	
				queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0],nullptr,&e_output_image_read_kernel);
				event_log.push_back({"Output image read buffer", e_output_image_read_kernel, 1,8});

				//Outputs image using the fucntion at the top.
				CImg<unsigned char> output_image(output_buffer.data(), width, height, depth, spectrum);
				string output_name = "output_image.ppm";
				output_image.save("output_image.ppm");

				picture_output(output_name);

				//This little block is in charge of outputting metrics for time to execute, for memory and kernel executions. 
				queue.finish();// Make sure all operations are finished.	
				metricsMaker(event_log,1,8);		


				
			}
			else{ // Once again same as before but for 16bit.
				
				binNumber = 1024; 
				size_t buffer_Size_float = binNumber * sizeof(float);
				float scale = (float)binNumber / 65536.0f; //Scale factor to be used to restore 16bit values for projection.

				std::cout << "imageDimensions: " << imageDimensions << std::endl;
				std::cout << "image_size: " << image_size << std::endl;
				std::cout << "Expected byte size: " << imageDimensions * 3 * sizeof(unsigned short) << std::endl; //Debug messages, was having issues with incorrect buffer sizes.

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

				cl::Event e_back_proj_kernel;	
				queue.enqueueNDRangeKernel(proj, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize),nullptr,&e_back_proj_kernel);
				event_log.push_back({"Image back projection kernel", e_back_proj_kernel, 1,16});
				cl::Event e_output_image_read_kernel;
				queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer_16bit, output_buffer.data(),nullptr,&e_output_image_read_kernel); //Similar to above. 
				event_log.push_back({"Output image read buffer", e_output_image_read_kernel, 1,16});


				// Finally the ouput image is produced. Using my function at the top.
				CImg<unsigned short> output_image(output_buffer.data(), width, height, depth, spectrum);
				string output_name = "output_image_16bit.ppm";
				output_image.save(output_name.c_str());				
				input16(output_name);


				//This little block is in charge of outputting metrics for time to execute, for memory and kernel executions. 
				queue.finish();// Make sure all operations are finished.
				metricsMaker(event_log,1,16);		



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