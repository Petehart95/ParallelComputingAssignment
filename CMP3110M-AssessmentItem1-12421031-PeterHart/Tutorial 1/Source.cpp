/*CMP3110M Parallel Computing Assignment | 12421031 Peter Hart*/

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <vector>
#include <CL/cl.hpp>
#include "Utils.h"
#include <string>

using namespace std;


void print_help();
int serial_average(float*);

int main(int argc, char **argv)
{
	int platform_id = 0;
	int device_id = 0;
	ifstream weather_data;
	string line;
	weather_data.open("temp_lincolnshire_short.txt");
	string stationName;
	int year, month, day, time;
	float airTemp;


	vector<string> stationNameVector = {};

	vector<int> yearVector = {};
	vector<int> monthVector = {};
	vector<int> dayVector = {};
	vector<int> timeVector = {};

	vector<float> airTempVector = {};

	// Check if the file exists
	if (weather_data.fail())
	{
		cout << "Cannot load file..." << endl;
		getchar();
		return 1;
	}


	// If the file exists, load the data into individual vectors
	while (!weather_data.eof())
	{
		// Store the value stored
		weather_data >> stationName >> year >> month >> day >> time >> airTemp;

		stationNameVector.push_back(stationName);
		yearVector.push_back(year);
		monthVector.push_back(month);
		dayVector.push_back(day);
		timeVector.push_back(time);
		airTempVector.push_back(airTemp);
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

		vector<int> A = { 9, 8, 7, 6, 5, 4, 3, 2, 1 };
		cout << "Before: A = " << A << endl;

		size_t local_size = 16;
		size_t padding_size = A.size() % local_size;

		// If the input vector is not a multiple of the local_size:
		// Insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size - padding_size, 0);
			//append that extra vector to our input (apply padding to the original vector)
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		size_t vector_elements = A.size();//number of elements
		size_t vector_size = A.size()*sizeof(int);//size in bytes

		// Host - output
		std::vector<int> C(vector_elements);

		// Device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);

		// Copy array A to device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0]);

		int minTemp = 0;
		int maxTemp = 0;
		int avgTemp = 0;
		int stdDevTemp = 0;
		int medTemp = 0;

		// Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel_min = cl::Kernel(program, "get_min");

		kernel_min.setArg(0, buffer_A);
		kernel_min.setArg(1, minTemp);

		//kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

		queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange);

		// Copy the result from device to host
		queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0]);

		// Padding applied to ensure sorting algorithm can be applied to values near the boundaries of the vector
		// Remove padding neutral values from original vector to restore vector to original size
		for (int i = (local_size - padding_size); i > 0; i--)
		{
			A.erase(A.begin());
		}

		cout << "After: A = " << A << endl;
		cout << "Maximum Temperature = " << maxTemp << endl;
		cout << "Minimum Temperature = " << minTemp << endl;
		cout << "Average Temperature = " << avgTemp << endl;
		cout << "Standard Deviation = " << stdDevTemp << endl;
		cout << "Median Temperature = " << medTemp << endl;
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

int serial_average(float* A) {
	//*10 to allow atomic, /10 to return decimal


	return 0;
}

/*end of script*/