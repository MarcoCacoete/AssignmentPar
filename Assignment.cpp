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
		window_width = image_input.width()/3;  
		window_height = image_input.height()/3; 
	}	
	const char* image_name = image_filename.c_str();
	
	CImgDisplay disp_input(window_width, window_height, image_name, 0);

	disp_input.display(image_input);

	disp_input.resize(window_width, window_height);

	while (!disp_input.is_closed()) {
		disp_input.wait();
	}
	return image_input;

}

void input16(const std::string& image_filename) {
    // Load the 16-bit image
    CImg<unsigned short> img16(image_filename.c_str());

    // Calculate image dimensions
    int pic_width = img16.width();
    int window_width = img16.width();
    int window_height = img16.height();

    // Adjust window size if the image is too large
    if (pic_width > 1080) {
        window_width = img16.width() / 3;
        window_height = img16.height() / 3;
    }

    // Normalize the 16-bit image to 8-bit for display
    CImg<unsigned char> img8 = img16.get_normalize(0, 255);

    // Display the normalized image
    CImgDisplay disp_input(window_width, window_height, "input (16-bit normalized)", 0);
    disp_input.display(img8);
    disp_input.resize(window_width, window_height);

    // Wait for the display window to close
    while (!disp_input.is_closed()) {
        disp_input.wait();
    }
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
	std::cout<<"Enter image name."<<endl;
	
	string imageName;
	
	cin>>imageName;

	string image_filename = imageName ;

	// string image_filename = "test_large.pgm";

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
		CImg<unsigned short> img16(image_filename.c_str());
		unsigned short max = img16.max();
	
		if (max <= 255) {
			CImg<unsigned char> image_input(image_filename.c_str());
			std::cout << "8-bit image detected." << std::endl;
			picture_output(image_filename); 
	
		} else {
			std::cout << "16-bit image detected." << std::endl;
			input16(image_filename); 
		}
		
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

		int image_size = image_input.size();		
		int width = image_input.width();
		int height = image_input.height();		
		int spectrum = image_input.spectrum();
		int depth = image_input.depth();
		int rgbImageSize = width*height; 
		int bin_number = 256;
		const size_t localWorkSize = 256;

		size_t globalWorkSize = ((image_size + bin_number - 1) / bin_number) * bin_number;
		size_t buffer_Size = bin_number * sizeof(int);
		size_t buffer_Size_float = bin_number * sizeof(float);

		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_size);
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_size); //should be the same as input image

		
		cl::Buffer dev_intensityHistogram(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_comHistogram(context, CL_MEM_READ_WRITE, buffer_Size); 
		cl::Buffer dev_histNormal(context, CL_MEM_READ_WRITE, buffer_Size_float);
		cl::Buffer dev_histR(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_histG(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_histB(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_histRcom(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_histGcom(context, CL_MEM_READ_WRITE, buffer_Size);
		cl::Buffer dev_histBcom(context, CL_MEM_READ_WRITE, buffer_Size);
		
		auto beginning = chrono::high_resolution_clock::now(); // Starts measuring whole program execution time.

		//4.1 Copy images to device memory
		
		cl::Event imageBuffer;
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_size, &image_input.data()[0],nullptr, &imageBuffer);
		queue.enqueueWriteBuffer(dev_image_output, CL_TRUE, 0, image_size, &image_input.data()[0],nullptr, &imageBuffer);

		imageBuffer.wait();
		cl_ulong ibEnd = imageBuffer.getProfilingInfo<CL_PROFILING_COMMAND_END>();
		cl_ulong ibStart = imageBuffer.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		double imageBufferTime = static_cast<double>(ibEnd - ibStart) / 1e6;
		std::cout<<"Image Buffer write duration:"<< imageBufferTime <<" milliseconds"<< endl;
		

		//4.2 Setup and execute the kernel (i.e. device code)
		
		vector<int> histogram (bin_number,0);

		cl::Event histogramBuffer;
		queue.enqueueWriteBuffer(dev_intensityHistogram, CL_TRUE, 0, buffer_Size, &histogram.data()[0],nullptr, &histogramBuffer);
		histogramBuffer.wait();

		cl_ulong hbStart = histogramBuffer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong hbEnd = histogramBuffer.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		double histogramBufferTime = static_cast<double>(hbEnd - hbStart) / 1e6;
		std::cout<<"Histogram Buffer write duration:"<< histogramBufferTime <<" milliseconds"<< endl;

		cl::Event histogramKernel;

		bool check = false;
		
		while(!check){

			if(spectrum==1){
				std::cout<<"What histogram kernel would you like to use. Local or Atom?"<<endl;
				string kernelType;
				cin>>kernelType;
				std::transform(kernelType.begin(),kernelType.end(),kernelType.begin(),::tolower);

				if(kernelType=="atom"){
					std::cout<<"Atom"<<endl;
					check = true;
					cl::Kernel kernelAtom = cl::Kernel(program, "hist_atom");
					kernelAtom.setArg(0, dev_image_input);
					kernelAtom.setArg(1, dev_intensityHistogram);
					queue.enqueueNDRangeKernel(kernelAtom, cl::NullRange, cl::NDRange(image_size), cl::NullRange,nullptr, &histogramKernel);
					
				}
				else if(kernelType=="local"){
					std::cout<<"Local"<<endl;
					check = true;
					cl::Kernel kernelLocal = cl::Kernel(program, "hist_local");cl::Kernel kernelHistLocal = cl::Kernel(program, "hist_local");
					kernelHistLocal.setArg(0, dev_image_input);
					kernelHistLocal.setArg(1, dev_intensityHistogram);
					kernelHistLocal.setArg(2, buffer_Size,NULL);
					kernelHistLocal.setArg(3, bin_number);
					kernelHistLocal.setArg(4, image_size);	
		
					queue.enqueueNDRangeKernel(kernelHistLocal, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(bin_number),nullptr, &histogramKernel);
				}
				else{
					std::cout<<"Invalid input. Please enter either Atom or Local"<<endl;
				}
			}else{

				check=true;	
			
				std::cout<<"Colour image detected"<<endl;
				vector <int> histR (bin_number,0);
				vector <int> histG (bin_number,0);
				vector <int> histB (bin_number,0);		
				
				queue.enqueueWriteBuffer(dev_histR, CL_TRUE, 0, buffer_Size, histR.data());
				queue.enqueueWriteBuffer(dev_histG, CL_TRUE, 0, buffer_Size, histG.data());
				queue.enqueueWriteBuffer(dev_histB, CL_TRUE, 0, buffer_Size, histB.data());

				cl::Kernel kernelHistRgb = cl::Kernel(program, "hist_rgb");
					kernelHistRgb.setArg(0, dev_image_input);
					kernelHistRgb.setArg(1, dev_histR);
					kernelHistRgb.setArg(2, dev_histG);
					kernelHistRgb.setArg(3, dev_histB);
					kernelHistRgb.setArg(4,rgbImageSize);
						
				queue.enqueueNDRangeKernel(kernelHistRgb, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize),nullptr, &histogramKernel);

				queue.enqueueReadBuffer(dev_histR, CL_TRUE, 0, buffer_Size, &histR.data()[0],nullptr);
				queue.enqueueReadBuffer(dev_histG, CL_TRUE, 0, buffer_Size, &histG.data()[0],nullptr);
				queue.enqueueReadBuffer(dev_histB, CL_TRUE, 0, buffer_Size, &histB.data()[0],nullptr);

				vector <vector<int>> histRgb = {histR,histG,histB};

				for (int i = 0; i < histRgb.size(); i++) {
					CImg<float> histogramGraphRgb(bin_number, 1, 1, 1, 0);
					for (int j = 0; j < bin_number; ++j) {
						histogramGraphRgb(j) = static_cast<float>(histRgb[i][j]); 
					}
					const char* histName;

					switch(i){
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
					CImgDisplay disp_raw(800, 600, histName);     

					// Displays histograms using the custom display objects
					histogramGraphRgb.display_graph(disp_raw, 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraphRgb.max(),true);					
				}
			}
		}

		histogramKernel.wait();

		cl_ulong hkStart = histogramKernel.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong hkEnd = histogramKernel.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		double histogramKernelTime = static_cast<double>(hkEnd - hkStart) / 1e6;
		std::cout<<"Histogram Kernel duration:"<< histogramKernelTime <<" milliseconds"<< endl;
	
		//4.3 Copy the result from device to host

		cl::Event histogramRead;
		queue.enqueueReadBuffer(dev_intensityHistogram, CL_TRUE, 0, buffer_Size, &histogram.data()[0],nullptr, &histogramRead);
		histogramRead.wait();

		int maxValue = *max_element(histogram.begin(), histogram.end());		
		
		cl_ulong hrStart = histogramRead.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong hrEnd = histogramRead.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		double histogramReadTime = static_cast<double>(hrEnd - hrStart) / 1e6;
		std::cout<<"Histogram Read duration:"<< histogramReadTime <<" milliseconds"<< endl;

		int jobCount=0;

		for (int i=0; i<histogram.size();i++){
			// std::cout<<histogram[i]<<endl;
			jobCount+= histogram[i];
		}
		

		std::cout<<"Width:"<<width<<endl;
		std::cout<<"Height:"<<height<<endl;
		std::cout<<"Pixel count: "<<width*height<<endl;
		std::cout<<"Jobcount:"<<jobCount<<endl;

		vector<int> histogramCom (bin_number,0);
		queue.enqueueWriteBuffer(dev_comHistogram, CL_TRUE, 0, buffer_Size, &histogram.data()[0],nullptr);

		vector<int> histogramComR(bin_number,0);
		vector<int> histogramComG(bin_number,0);
		vector<int> histogramComB(bin_number,0);

		queue.enqueueWriteBuffer(dev_histRcom, CL_TRUE, 0, buffer_Size, &histogramComR.data()[0],nullptr);
		queue.enqueueWriteBuffer(dev_histGcom, CL_TRUE, 0, buffer_Size, &histogramComG.data()[0],nullptr);
		queue.enqueueWriteBuffer(dev_histBcom, CL_TRUE, 0, buffer_Size, &histogramComB.data()[0],nullptr);
		vector<cl::Buffer*> rgbBuffers = {&dev_histR, &dev_histG, &dev_histB};
		vector<cl::Buffer*> rgbBuffersCom = {&dev_histRcom, &dev_histGcom, &dev_histBcom};		
		vector<vector<int>> histogramComRgb = {histogramComR,histogramComG,histogramComB};

		check = false;

		while(!check){
			std::cout<<"What comulative histogram kernel would you like to use. Hillis or Blelloch?"<<endl;
			string kernelType;
			cin>>kernelType;
			std::transform(kernelType.begin(),kernelType.end(),kernelType.begin(),::tolower);
			if(spectrum==1){
				if (kernelType=="hillis"){
					std::cout<<"Hillis-Steele"<<endl;
					check = true;
					cl::Kernel kernelCom = cl::Kernel(program, "com_hist");
					kernelCom.setArg(0, dev_intensityHistogram);		
					kernelCom.setArg(1, dev_comHistogram);
					queue.enqueueNDRangeKernel(kernelCom, cl::NullRange, cl::NDRange(bin_number), cl::NDRange(bin_number),nullptr);
				}
				else if(kernelType=="blelloch"){
					std::cout<<"Blelloch"<<endl;
					check = true;			
					cl::Kernel kernelCom = cl::Kernel(program, "scan_bl");
					kernelCom.setArg(0, dev_comHistogram);
					queue.enqueueNDRangeKernel(kernelCom, cl::NullRange, cl::NDRange(bin_number), cl::NDRange(bin_number),nullptr);
				}
				else{
					std::cout<<"Invalid input. Please enter either Scan or Blelloch"<<endl;
				}
				queue.enqueueReadBuffer(dev_comHistogram, CL_TRUE, 0, buffer_Size, &histogramCom.data()[0],nullptr);

			}
			else{				

				for(int i=0;i<rgbBuffers.size();i++){
					if (kernelType=="hillis"){
						std::cout<<"Hillis-Steele"<<endl;
						check = true;
						cl::Kernel kernelCom = cl::Kernel(program, "com_hist");
						kernelCom.setArg(0, *rgbBuffers[i]);		
						kernelCom.setArg(1, *rgbBuffersCom[i]);
						queue.enqueueNDRangeKernel(kernelCom, cl::NullRange, cl::NDRange(bin_number), cl::NDRange(bin_number),nullptr);
						queue.enqueueReadBuffer(*rgbBuffersCom[i], CL_TRUE, 0, buffer_Size, &histogramComRgb[i].data()[0], nullptr);
					}
					else if(kernelType=="blelloch"){
						std::cout<<"Blelloch"<<endl;
						check = true;			
						cl::Kernel kernelCom = cl::Kernel(program, "scan_bl");
						kernelCom.setArg(0, *rgbBuffers[i]);
						queue.enqueueNDRangeKernel(kernelCom, cl::NullRange, cl::NDRange(bin_number), cl::NDRange(bin_number),nullptr);
						queue.enqueueReadBuffer(*rgbBuffers[i], CL_TRUE, 0, buffer_Size, &histogramComRgb[i].data()[0], nullptr);
					}
					else{
						std::cout<<"Invalid input. Please enter either Scan or Blelloch"<<endl;
					}
				}
			}			
		}

		if(spectrum==1){

			int maximumValue = histogramCom[bin_number - 1];
			float maximumBinValue = static_cast<float>(maximumValue);

			// Convert intermediate results to floats for normalization
			vector<float> histogramComFloat(bin_number, 0.0f); // New float vector
			for (int i = 0; i < bin_number; ++i) {
				histogramComFloat[i] = static_cast<float>(histogramCom[i]); // Convert int to float
			}

			// This finishes the time count and calculates the difference between the 2 registered timestamps so we get the total duration of the events.
			auto ending = chrono::high_resolution_clock::now();
			auto total = chrono::duration<double,milli>(ending-beginning).count() ;

			queue.enqueueWriteBuffer(dev_histNormal, CL_TRUE, 0, buffer_Size_float, &histogramComFloat.data()[0],nullptr);

			cl::Kernel histNormal = cl::Kernel(program, "hist_normal");
			histNormal.setArg(0, dev_histNormal);	
			histNormal.setArg(1, maximumBinValue);		
		
			queue.enqueueNDRangeKernel(histNormal, cl::NullRange, cl::NDRange(bin_number), cl::NullRange,nullptr);
			queue.enqueueReadBuffer(dev_histNormal, CL_TRUE, 0, buffer_Size_float, &histogramComFloat.data()[0],nullptr);		

			cl::Kernel proj = cl::Kernel(program, "back_projector");
			proj.setArg(0, dev_image_input);	
			proj.setArg(1, dev_image_output);	
			proj.setArg(2, dev_histNormal);	
		
			queue.enqueueNDRangeKernel(proj, cl::NullRange, cl::NDRange(image_size), cl::NullRange,nullptr);

			vector<unsigned char> output_buffer(image_size);
			queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);


			std::cout<<"Total time to run program:"<< total <<" milliseconds"<< endl;

			CImg<int> histogramGraph(bin_number, 1, 1, 1, 0); // Create a 1D CImg object for the raw histogram
			for (int i = 0; i < bin_number; ++i) {
				// int maxValue = *max_element(histogram.begin(), histogram.end());
				histogramGraph(i) =histogram[i];//maxValue; // Copy raw histogram values
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


			CImg<unsigned char> output_image(output_buffer.data(), width, height, depth, spectrum);
			string output_name = "output_image.pgm";
			output_image.save("output_image.pgm");
			picture_output(output_name);
		}else{

			cl::Buffer dev_histNormalR(context, CL_MEM_READ_WRITE, buffer_Size_float);
			cl::Buffer dev_histNormalG(context, CL_MEM_READ_WRITE, buffer_Size_float);
			cl::Buffer dev_histNormalB(context, CL_MEM_READ_WRITE, buffer_Size_float);

			vector<cl::Buffer*> rgbBuffersComNorm = {&dev_histNormalR, &dev_histNormalG, &dev_histNormalB};

			// Convert intermediate results to floats for normalization
			vector<float> histogramComFloatR(bin_number, 0.0f); // New float vector
			vector<float> histogramComFloatG(bin_number, 0.0f); // New float vector
			vector<float> histogramComFloatB(bin_number, 0.0f); // New float vector
			vector <vector<float>*> histogramComRgbFloat = {&histogramComFloatR,&histogramComFloatG,&histogramComFloatB};
			vector<unsigned char> output_buffer(image_size);

			float maximumValue;

			for(int i=0;i<histogramComRgb.size();i++){
				
				for (int j = 0; j < bin_number; ++j) 
				(*histogramComRgbFloat[i])[j] = static_cast<float>(histogramComRgb[i][j]); // Convert int to float
			}							

			for(int i=0;i<histogramComRgb.size();i++){	
				maximumValue = histogramComRgb[i][bin_number-1];
				maximumValue = static_cast<float>(maximumValue);

				queue.enqueueWriteBuffer(*rgbBuffersComNorm[i], CL_TRUE, 0, buffer_Size_float, &(*histogramComRgbFloat[i]).data()[0],nullptr);

				cl::Kernel histNormal = cl::Kernel(program, "hist_normal");
				histNormal.setArg(0, *rgbBuffersComNorm[i]);	
				histNormal.setArg(1, maximumValue);		
			
				queue.enqueueNDRangeKernel(histNormal, cl::NullRange, cl::NDRange(bin_number), cl::NullRange,nullptr);
				queue.enqueueReadBuffer(*rgbBuffersComNorm[i], CL_TRUE, 0, buffer_Size_float, &(*histogramComRgbFloat[i]).data()[0],nullptr);		
				
				// //  display_graph call
				// histogramGraph.display_graph("Histogram", 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraph.max(),true);	
				// histogramGraphCom.display_graph("Histogram", 3,1,"VALUES",0,255,"COUNT PER BIN",0,histogramGraphCom.max(),true);	

				// CImg<float> histogramGraph(bin_number, 1, 1, 1, 0); // Create a 1D CImg object for the raw histogram
				// for (int i = 0; i < bin_number; ++i) {
				// 	// int maxValue = *max_element(histogram.begin(), histogram.end());
				// 	histogramGraph(i) = static_cast<float>(histogram[i]);//maxValue; // Copy raw histogram values
				// }

				CImg<float> histogramGraphCom(bin_number, 1, 1, 1, 0); // Create a 1D CImg object for the raw histogram
				for (int j = 0; j < bin_number; ++j) {
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

			cl::Kernel proj = cl::Kernel(program, "back_projectorRgb");
				proj.setArg(0, dev_image_input);	
				proj.setArg(1, dev_image_output);	
				proj.setArg(2, *rgbBuffersComNorm[0]);
				proj.setArg(3, *rgbBuffersComNorm[1]);
				proj.setArg(4, *rgbBuffersComNorm[2]);	
			
				queue.enqueueNDRangeKernel(proj, cl::NullRange, cl::NDRange(rgbImageSize), cl::NullRange,nullptr);

				queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

			CImg<unsigned char> output_image(output_buffer.data(), width, height, depth, spectrum);
			string output_name = "output_image.ppm";
			output_image.save("output_image.ppm");

			picture_output(output_name);

			// This finishes the time count and calculates the difference between the 2 registered timestamps so we get the total duration of the events.
			auto ending = chrono::high_resolution_clock::now();
			auto total = chrono::duration<double,milli>(ending-beginning).count() ;

			std::cout<<"Total time to run program:"<< total <<" milliseconds"<< endl;

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
