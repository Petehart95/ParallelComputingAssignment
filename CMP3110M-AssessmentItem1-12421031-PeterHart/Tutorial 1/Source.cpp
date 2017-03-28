/* Source.cpp | CMP3110M Parallel Computing Assignment | 12421031 Peter Hart */

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <CL/cl.hpp>
#include "Utils.h"

using namespace std;

void print_help();
int getAvgTemp(cl::Program, cl::Buffer, cl::Buffer, cl::CommandQueue, size_t, size_t, vector<int>, size_t);
float serial_getAvgTemp(vector<float>, size_t);

int main(int argc, char **argv)
{
	int platform_id = 0;
	int device_id = 0;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////// READ DATA FROM TEXT FILE /////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////

	ifstream weather_data;
	weather_data.open("temp_lincolnshire_short.txt");
	string stationName, line;
	int year, month, day, time;
	float airTemp;
	typedef int mytype;

	// Initialise vectors which will be used for storing each respective column of data from the text file
	vector<string> stationNameVector = {};
	vector<int> yearVector = {};
	vector<int> monthVector = {};
	vector<int> dayVector = {};
	vector<int> timeVector = {};
	vector<mytype> A = {};

	// Check if the file exists, if it doesn't, terminate the program
	if (weather_data.fail())
	{
		cout << "Cannot load file..." << endl;
		getchar();
		return 1;
	}

	// If the file exists, load the data into each of the respective vectors
	while (!weather_data.eof())
	{
		// Temporarily store the data into different variables
		weather_data >> stationName >> year >> month >> day >> time >> airTemp;

		// Add the data contained in each variable to each vector
		stationNameVector.push_back(stationName);
		yearVector.push_back(year);
		monthVector.push_back(month);
		dayVector.push_back(day);
		timeVector.push_back(time);
		A.push_back(airTemp);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////// User Interface for Device Selection //////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////

	for (int i = 1; i < argc; i++)	
	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { cout << ListPlatformsDevices() << endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	try {
		cl::Context context = GetContext(platform_id, device_id);

		// Display the selected device
		cout << "Device Selected: " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << endl; 

		// Queue variable which will be used to push commands to the device
		cl::CommandQueue queue(context); 

		// Load and build the device code
		cl::Program::Sources sources; 
		AddSources(sources, "my_kernels.cl");
		cl::Program program(context, sources);

		// Build and debug the kernel code currently residing on the device
		try 
		{
			program.build();
		}
		catch (const cl::Error& err) 
		{
			cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			throw err;
		}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////// Output Results ////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	

		// Temporary custom vector for initial development of the kernels
		vector<mytype> A = { 9, 6, -10, 4, 3, 2, 1, 15, 1, -2, 4, 2, 4, 1, 9, 8, -1, 6, -10 };
		vector<mytype> B = { 9, 6, -10, 4, 3, 2, 1, 15, 1, -2, 4, 2, 4, 1, 9, 8, -1, 6, -10 };

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 8;

		size_t padding_size = A.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size - padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}
		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size() * sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;

		vector<int> outputList(input_elements);

		// If the input vector is not a multiple of the local_size:
		// Insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) 
		{
			//create an extra vector with neutral values
			vector<int> B_ext(local_size - padding_size, 0);
			//append that extra vector to our input (apply padding to the original vector)
			B.insert(B.end(), B_ext.begin(), B_ext.end());
		}

		//host - output
		size_t output_size = B.size() * sizeof(mytype);//size in bytes

													   //device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

		//Part 5 - device operations

		//Copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

		//Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

		//call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

		//Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		int temp = B.at(0);
		double average = temp / input_elements;

		//std::cout << "A = " << A << std::endl;
		std::cout << "Average =  " << average << std::endl;

	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	getchar();
	return 0;
}

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}


/*end of script*/