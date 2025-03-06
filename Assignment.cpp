#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"


// This code is adapted from the code provided for the Tutorial 2 task in the workshops.

using namespace cimg_library;
using namespace std;

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
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input,"input");

		// //a 3x3 convolution mask implementing an averaging filter
		// std::vector<float> convolution_mask = { 1.f / 9, 1.f / 9, 1.f / 9,
		// 										1.f / 9, 1.f / 9, 1.f / 9,
		// 										1.f / 9, 1.f / 9, 1.f / 9 };

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

		cl::Buffer dev_intensityHistogram(context, CL_MEM_READ_WRITE, buffer_Size); 

//		cl::Buffer dev_convolution_mask(context, CL_MEM_READ_ONLY, convolution_mask.size()*sizeof(float));
		
		auto beginning = chrono::high_resolution_clock::now(); // Starts measuring whole program execution time.

		//4.1 Copy images to device memory
		cl::Event imageBuffer;
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0],nullptr, &imageBuffer);
//		queue.enqueueWriteBuffer(dev_convolution_mask, CL_TRUE, 0, convolution_mask.size()*sizeof(float), &convolution_mask[0]);
		imageBuffer.wait();

		cl_ulong ibStart = imageBuffer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong ibEnd = imageBuffer.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		double imageBufferTime = static_cast<double>(ibEnd - ibStart) / 1e6;
		cout<<"Image Buffer write duration:"<< imageBufferTime <<" milliseconds"<< endl;
		

		//4.2 Setup and execute the kernel (i.e. device code)
// 		cl::Kernel kernel = cl::Kernel(program, "avg_filterND");
// 		kernel.setArg(0, dev_image_input);
// 		kernel.setArg(1, dev_image_output);
		// //		kernel.setArg(2, dev_convolution_mask);
		vector<int> histogram (bin_number,0);

		cl::Event histogramBuffer;
		queue.enqueueWriteBuffer(dev_intensityHistogram, CL_TRUE, 0, buffer_Size, &histogram.data()[0],nullptr, &histogramBuffer);
		histogramBuffer.wait();

		cl_ulong hbStart = histogramBuffer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong hbEnd = histogramBuffer.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		double histogramBufferTime = static_cast<double>(hbEnd - hbStart) / 1e6;
		cout<<"Histogram Buffer write duration:"<< histogramBufferTime <<" milliseconds"<< endl;

		cl::Kernel kernel = cl::Kernel(program, "intensityHistogram");
		kernel.setArg(0, dev_image_input);
		kernel.setArg(1, dev_intensityHistogram);

		cl::Event histogramKernel;

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange,nullptr, &histogramKernel);
		histogramKernel.wait();

		cl_ulong hkStart = histogramKernel.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong hkEnd = histogramKernel.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		double histogramKernelTime = static_cast<double>(hkEnd - hkStart) / 1e6;
		cout<<"Histogram Kernel duration:"<< histogramKernelTime <<" milliseconds"<< endl;

		
		// queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);
		// queue.enqueueNDRangeKernel(kernel, cl::NullRange, 
		// 	cl::NDRange(image_input.width(), image_input.height(), image_input.spectrum()), 
		// 	cl::NullRange);

		// vector<unsigned char> output_buffer(image_input.size());
	
		//4.3 Copy the result from device to host
		// queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		cl::Event histogramRead;
		queue.enqueueReadBuffer(dev_intensityHistogram, CL_TRUE, 0, buffer_Size, &histogram.data()[0],nullptr, &histogramRead);
		histogramRead.wait();

		CImg<int> histogramGraph(bin_number, 1, 1, 1, 0); // Create a 1D CImg object for the raw histogram
		for (int i = 0; i < bin_number; ++i) {
			histogramGraph(i) = histogram[i]; // Copy raw histogram values
		}
		
		
		cl_ulong hrStart = histogramRead.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong hrEnd = histogramRead.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		double histogramReadTime = static_cast<double>(hrEnd - hrStart) / 1e6;
		cout<<"Histogram Read duration:"<< histogramReadTime <<" milliseconds"<< endl;

		// int jobCount;

		// for (int i=0; i<histogram.size();i++){
		// 	// cout<<histogram[i]<<endl;
		// 	jobCount+= histogram[i];
		// }

		// cout<<jobCount<<endl;

		cl::Kernel kernelCom = cl::Kernel(program, "scan_bl");
		kernelCom.setArg(0, dev_intensityHistogram);

		vector<int> histogramScan (bin_number,0);

		queue.enqueueNDRangeKernel(kernelCom, cl::NullRange, cl::NDRange(bin_number), cl::NullRange,nullptr);


		queue.enqueueReadBuffer(dev_intensityHistogram, CL_TRUE, 0, buffer_Size, &histogramScan.data()[0],nullptr);

		CImg<int> histogramGraphCom(bin_number, 1, 1, 1, 0); // Create a 1D CImg object for the raw histogram
		for (int i = 0; i < bin_number; ++i) {
			histogramGraphCom(i) = histogramScan[i]; // Copy raw histogram values
		}

		// This finishes the time count and calculates the difference between the 2 registered timestamps so we get the total duration of the events.
		auto ending = chrono::high_resolution_clock::now();
		auto total = chrono::duration<double,milli>(ending-beginning).count() ;

		cout<<"Total time to run program:"<< total <<" milliseconds"<< endl;




		// CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		// CImgDisplay disp_output(output_image,"output");

 		// while (!disp_input.is_closed() && !disp_output.is_closed()
		// 	&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		//     disp_input.wait(1);
		//     disp_output.wait(1);
	    // }		

		//  display_graph call
		histogramGraph.display_graph("Histogram", 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraph.max(),true);	
		histogramGraphCom.display_graph("Histogram", 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraphCom.max(),true);		
	

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
