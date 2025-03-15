#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"


// This host code is adapted from the host code provided for the Tutorial 2 tasks in the workshops.

using namespace cimg_library;
using namespace std;


CImg<unsigned char> picture_output(const std::string& image_filename){
	CImg<unsigned char> image_input(image_filename.c_str());
	

	int pic_width = image_input.width();  
	int window_width = image_input.width();  
	int window_height = image_input.height(); 
	if (pic_width > 1080) {
		window_width = image_input.width()/2;  
		window_height = image_input.height()/2; 
	}	
	
	CImgDisplay disp_input(window_width, window_height, "input", 0);

	disp_input.display(image_input);

	disp_input.resize(window_width, window_height);

	while (!disp_input.is_closed()) {
		disp_input.wait();
	}
	return image_input;

}

void print_help() {
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
	// cout<<"Enter image name."<<endl;
	
	// string imageName;
	
	// cin>>imageName;

	// string image_filename = imageName +".pgm";

	string image_filename = "test.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input = picture_output(image_filename);

		
		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

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


		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image

		int bin_number = 256;

		size_t buffer_Size = bin_number * sizeof(int);
		size_t buffer_Size_float = bin_number * sizeof(float);


		cl::Buffer dev_intensityHistogram(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_comHistogram(context, CL_MEM_READ_WRITE, buffer_Size); 
		cl::Buffer dev_histNormal(context, CL_MEM_READ_WRITE, buffer_Size_float * bin_number);
		
		auto beginning = chrono::high_resolution_clock::now(); // Starts measuring whole program execution time.

		//4.1 Copy images to device memory
		
		cl::Event imageBuffer;
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0],nullptr, &imageBuffer);
		queue.enqueueWriteBuffer(dev_image_output, CL_TRUE, 0, image_input.size(), &image_input.data()[0],nullptr, &imageBuffer);


		imageBuffer.wait();

		cl_ulong ibStart = imageBuffer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong ibEnd = imageBuffer.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		double imageBufferTime = static_cast<double>(ibEnd - ibStart) / 1e6;
		cout<<"Image Buffer write duration:"<< imageBufferTime <<" milliseconds"<< endl;
		

		//4.2 Setup and execute the kernel (i.e. device code)
		
		vector<int> histogram (bin_number,0);

		cl::Event histogramBuffer;
		queue.enqueueWriteBuffer(dev_intensityHistogram, CL_TRUE, 0, buffer_Size, &histogram.data()[0],nullptr, &histogramBuffer);
		histogramBuffer.wait();

		cl_ulong hbStart = histogramBuffer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong hbEnd = histogramBuffer.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		double histogramBufferTime = static_cast<double>(hbEnd - hbStart) / 1e6;
		cout<<"Histogram Buffer write duration:"<< histogramBufferTime <<" milliseconds"<< endl;

		cl::Kernel kernelHistLocal = cl::Kernel(program, "hist_Local");
		kernelHistLocal.setArg(0, dev_image_input);
		kernelHistLocal.setArg(1, dev_intensityHistogram);
		kernelHistLocal.setArg(2, buffer_Size,NULL);
		kernelHistLocal.setArg(3, bin_number);

		// cl::Kernel kernelAtom = cl::Kernel(program, "hist_Atom");
		// kernelAtom.setArg(0, dev_image_input);
		// kernelAtom.setArg(1, dev_intensityHistogram);

		cl::Event histogramKernel;

		queue.enqueueNDRangeKernel(kernelHistLocal, cl::NullRange, cl::NDRange(image_input.size()), cl::NDRange(bin_number),nullptr, &histogramKernel);
		histogramKernel.wait();

		// queue.enqueueNDRangeKernel(kernelAtom, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange,nullptr, &histogramKernel);
		// histogramKernel.wait();

		cl_ulong hkStart = histogramKernel.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong hkEnd = histogramKernel.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		double histogramKernelTime = static_cast<double>(hkEnd - hkStart) / 1e6;
		cout<<"Histogram Kernel duration:"<< histogramKernelTime <<" milliseconds"<< endl;

	
		//4.3 Copy the result from device to host

		cl::Event histogramRead;
		queue.enqueueReadBuffer(dev_intensityHistogram, CL_TRUE, 0, buffer_Size, &histogram.data()[0],nullptr, &histogramRead);
		histogramRead.wait();

		int maxValue = *max_element(histogram.begin(), histogram.end());

		
		
		
		cl_ulong hrStart = histogramRead.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong hrEnd = histogramRead.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		double histogramReadTime = static_cast<double>(hrEnd - hrStart) / 1e6;
		cout<<"Histogram Read duration:"<< histogramReadTime <<" milliseconds"<< endl;

		int jobCount=0;

		for (int i=0; i<histogram.size();i++){
			// cout<<histogram[i]<<endl;
			jobCount+= histogram[i];
		}
		int width = image_input.width();
		int height = image_input.height();

		cout<<"Width:"<<width<<endl;
		cout<<"Height:"<<height<<endl;
		cout<<"Pixel count: "<<width*height<<endl;
		cout<<"Jobcount:"<<jobCount<<endl;


		vector<int> histogramCom (bin_number,0);
		queue.enqueueWriteBuffer(dev_comHistogram, CL_TRUE, 0, buffer_Size, &histogram.data()[0],nullptr);

		cl::Kernel kernelCom = cl::Kernel(program, "com_Hist");
		kernelCom.setArg(0, dev_intensityHistogram);		
		kernelCom.setArg(1, dev_comHistogram);
		queue.enqueueNDRangeKernel(kernelCom, cl::NullRange, cl::NDRange(bin_number), cl::NDRange(bin_number),nullptr);


		// cl::Kernel kernelCom = cl::Kernel(program, "scan_bl");
		// kernelCom.setArg(0, dev_comHistogram);
		// queue.enqueueNDRangeKernel(kernelCom, cl::NullRange, cl::NDRange(bin_number), cl::NullRange,nullptr);
		queue.enqueueReadBuffer(dev_comHistogram, CL_TRUE, 0, buffer_Size, &histogramCom.data()[0],nullptr);


		// for (int i = 0;i<histogramCom.size();i++){
		// 	cout<<histogramCom[i]<<endl;
		// }


		int maximumValue = *max_element(histogramCom.begin(), histogramCom.end());
		float maximumBinValue = static_cast<float>(maximumValue);

		// CImg<float> histogramGraphCom(bin_number, 1, 1, 1, 0); // Create a 1D CImg object for the raw histogram
		// for (int i = 0; i < bin_number; ++i) {
		// 	histogramGraphCom(i) = static_cast<float>(histogramCom[i]); // Copy raw histogram values
		// }
		// Convert intermediate results to floats for normalization
		vector<float> histogramComFloat(bin_number, 0.0f); // New float vector
		for (int i = 0; i < bin_number; ++i) {
			histogramComFloat[i] = static_cast<float>(histogramCom[i]); // Convert int to float
		}

		// This finishes the time count and calculates the difference between the 2 registered timestamps so we get the total duration of the events.
		auto ending = chrono::high_resolution_clock::now();
		auto total = chrono::duration<double,milli>(ending-beginning).count() ;

		queue.enqueueWriteBuffer(dev_histNormal, CL_TRUE, 0, buffer_Size_float, &histogramComFloat.data()[0],nullptr);


		cl::Kernel histNormal = cl::Kernel(program, "histNormal");
		histNormal.setArg(0, dev_histNormal);	
		histNormal.setArg(1, maximumBinValue);		
	
		queue.enqueueNDRangeKernel(histNormal, cl::NullRange, cl::NDRange(bin_number), cl::NullRange,nullptr);

		queue.enqueueReadBuffer(dev_histNormal, CL_TRUE, 0, buffer_Size_float, &histogramComFloat.data()[0],nullptr);

		

		cl::Kernel proj = cl::Kernel(program, "proj");
		proj.setArg(0, dev_image_input);	
		proj.setArg(1, dev_image_output);	
		proj.setArg(2, dev_histNormal);	
	
		queue.enqueueNDRangeKernel(proj, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange,nullptr);


		vector<unsigned char> output_buffer(image_input.size());

		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);


		// for (int i = 0;i<histogramComFloat.size();i++){
		// 	cout<<histogramComFloat[i]<<endl;
		// }


		// // Print raw histogram values
		// cout << "Raw Histogram:" << endl;
		// for (int i = 0; i < bin_number; ++i) {
		// 	cout << "Bin " << i << ": " << histogram[i] << endl;
		// }

		// // Print cumulative histogram values
		// cout << "Cumulative Histogram:" << endl;
		// for (int i = 0; i < bin_number; ++i) {
		// 	cout << "Bin " << i << ": " << histogramCom[i] << endl;
		// }

		cout<<"Total time to run program:"<< total <<" milliseconds"<< endl;

		// //  display_graph call
		// histogramGraph.display_graph("Histogram", 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraph.max(),true);	
		// histogramGraphCom.display_graph("Histogram", 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraphCom.max(),true);	

		CImg<float> histogramGraph(bin_number, 1, 1, 1, 0); // Create a 1D CImg object for the raw histogram
		for (int i = 0; i < bin_number; ++i) {
			// int maxValue = *max_element(histogram.begin(), histogram.end());
			histogramGraph(i) = static_cast<float>(histogram[i]);//maxValue; // Copy raw histogram values
		}

		CImg<float> histogramGraphCom(bin_number, 1, 1, 1, 0); // Create a 1D CImg object for the raw histogram
		for (int i = 0; i < bin_number; ++i) {
			histogramGraphCom(i) = histogramComFloat[i]; // Copy raw histogram values
		}
		
		// Sets histogram window size
		CImgDisplay disp_raw(800, 600, "Raw Histogram");     
		CImgDisplay disp_com(800, 600, "Cumulative Histogram");

		// Display histograms using the custom display objects
		histogramGraph.display_graph(disp_raw, 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraph.max(),true);
		histogramGraphCom.display_graph(disp_com, 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraphCom.max(),true);	


		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		string output_name = "output_image.pgm";
		output_image.save("output_image.pgm");
		picture_output(output_name);

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
