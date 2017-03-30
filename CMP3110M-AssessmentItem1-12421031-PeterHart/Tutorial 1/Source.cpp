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
#include <algorithm>

using namespace std;

void print_help();
vector<float> quickDelete(vector<float>, float);

int main(int argc, char **argv)
{
	int platform_id = 0;
	int device_id = 0;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////// READ DATA FROM TEXT FILE /////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////

	ifstream weather_data;
	weather_data.open("temp_lincolnshire.txt");
	string stationName, line;
	int year, month, day, time;
	float airTemp;

	// Initialise vectors which will be used for storing each respective column of data from the text file
	vector<string> stationNameVector = {};
	vector<int> yearVector = {};
	vector<int> monthVector = {};
	vector<int> dayVector = {};
	vector<int> timeVector = {};
	vector<float> A = {};

	// Check if the file exists, if it doesn't, terminate the program
	if (weather_data.fail())
	{
		cout << "Cannot load text file..." << endl;
		getchar();
		return 1;
	}

	cout << "Loading data from the text file..." << endl;

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

	cout << "Data was successfully loaded from the text file!" << endl;

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
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);


		// Load and build the device code
		cl::Program::Sources sources; 

		cl::Event prof_event;


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
		//////////////////////////////////// Parallel Statistical Operations ////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Initialise variables
		vector<float> sorted_temp(A.size());
		size_t vector_elements = sorted_temp.size();
		float mean = 0.f;
		float _min = 0.f;
		float _max = 0.f;
		float median = 0.f;
		float firstQuart = 0.f;
		float thirdQuart = 0.f;
		float stdDev = 0.f;
		long mex1 = 0;
		long mex2 = 0;

		cl::Event mem_prof_event_1;

		// Second vector used for the output of the kernels
		vector<float> B(A.size());

		// Define the size of the workgroups in which the data will be sent to on the device
		size_t local_size = 16;

		// Pad the vector with a neutral value, in this case a very high temperature value
		float pad = 300000.f;

		// Pad the vector and ensure that the vector size is divisible by the workgroup size
		float padding_size = A.size() % local_size;
		
		// If the input vector is not currently divisible by the workgroup size:
		// Insert additional neutral elements to the input vector
		if (padding_size)
		{
			// Create an extra temporary vector which stores the required amount of neutral elements
			std::vector<float> A_ext(local_size - padding_size, pad);
			
			// Append this extra temporary vector to the end of the input vector
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		// Total elements within the input vector
		size_t input_elements = A.size();

		// Total size of the input vector, in bytes
		size_t input_size = A.size() * sizeof(float);

		// Total size of the output vector, in bytes
		size_t output_size = B.size() * sizeof(float); 

		// Calculate the total number of workgroups that will be used on the device
		size_t nr_groups = input_elements / local_size;

		// Establish a buffer which will be used for the input vector, ensure read-only to avoid kernels overwriting the original input vector unnecessarily
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);

		// Establish a buffer which will be used for the output vector, ensure write-only to avoid kernels unnecessarily interpreting it as the input vector
		cl::Buffer buffer_B(context, CL_MEM_WRITE_ONLY, output_size);

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////// PARALLEL ADDITION KERNEL /////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Copy the input vector to the device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0], NULL, &mem_prof_event_1);

		// Initialise output vector on the device memory (zero buffer)
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

		// Kernel responsible for the parallel addition involved when calculating the average of the weather dataset
		cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size * sizeof(float))); // local memory size
		kernel_1.setArg(3, pad); // padding size

		cl::Event mem_prof_event_2;

		// Call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);

		// Copy the calculated result from the device back to the host (store the result in the output vector in host)
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0], NULL, &mem_prof_event_2);

		// Store execution time of kernel
		long reduce_add_ns = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		// Store memory transfer time
		mex1 = mem_prof_event_1.getProfilingInfo<CL_PROFILING_COMMAND_END>() - mem_prof_event_1.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		mex2 = mem_prof_event_2.getProfilingInfo<CL_PROFILING_COMMAND_END>() - mem_prof_event_2.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		long reduce_add_mem = mex2 + mex1;
		long reduce_add_op = reduce_add_mem + reduce_add_ns;

		string reduce_add_full = GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US);

		// Copy calculated sum into a new float variable
		float total = B[0];

		// Divide the sum by the total elements in serial to calculate the average
		mean = total / input_elements;

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////// PARALLEL STANDARD DEVIATION KERNEL /////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Copy the input vector to the device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0], NULL, &mem_prof_event_1);

		// Initialise output vector on the device memory (zero buffer)
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

		// Re-use previous kernel
		// Kernel responsible for the parallel standard deviation calculation
		kernel_1 = cl::Kernel(program, "reduce_standard_deviation");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size * sizeof(float))); // local memory size
		kernel_1.setArg(3, mean); // pass previously calculated mean as an argument
		kernel_1.setArg(4, pad); // padding size

		// Call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size),NULL,&prof_event);

		// Copy the calculated result from the device back to the host (store the result in the output vector in host)
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0], NULL, &mem_prof_event_2);

		// Store execution time of kernel
		long reduce_standard_deviation_ns = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		// Store memory transfer time
		mex1 = mem_prof_event_1.getProfilingInfo<CL_PROFILING_COMMAND_END>() - mem_prof_event_1.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		mex2 = mem_prof_event_2.getProfilingInfo<CL_PROFILING_COMMAND_END>() - mem_prof_event_2.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		long reduce_standard_deviation_mem = mex2 + mex1;
		long reduce_standard_deviation_op = reduce_standard_deviation_mem + reduce_standard_deviation_ns;

		string reduce_standard_deviation_full = GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US);

		// Collect sum of the squared differences
		float sumsquaredDifference = B[0];
		float variance = sumsquaredDifference / input_elements;

		// Square-root the average of the square differences in serial
		stdDev = sqrt(variance);

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////// PARALLEL SORT KERNEL /////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Copy the input vector to the device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0], NULL, &mem_prof_event_1);

		// Initialise output vector on the device memory (zero buffer)
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

		// (Re-use previous kernel)
		// Kernel responsible for the sorting required for the median, LQ, UQ, min and max
		kernel_1 = cl::Kernel(program, "parallel_selection_sort");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);

		// Call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		
		// Copy the calculated result from the device back to the host (store the result in the output vector in host)
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0], NULL, &mem_prof_event_2);

		// Store execution time of kernel
		long sort_ns = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		// Store memory transfer time
		mex1 = mem_prof_event_1.getProfilingInfo<CL_PROFILING_COMMAND_END>() - mem_prof_event_1.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		mex2 = mem_prof_event_2.getProfilingInfo<CL_PROFILING_COMMAND_END>() - mem_prof_event_2.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		long sort_mem = mex2 + mex1;
		long sort_op = sort_mem + sort_ns;

		string sort_full = GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US);

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////// OUTPUT RESULTS ///////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		long totalElapsed = 0;
		totalElapsed = reduce_add_op + reduce_standard_deviation_op + sort_op;

		cout << endl;
		cout << "---------------------------------- Performance Results ---------------------------------" << endl;
		cout << "Addition by reduction kernel: " << endl;
		cout << "\tKernel execution time: " << reduce_add_ns << " [ns]" << endl;
		cout << "\tMemory transfer: " << reduce_add_mem << " [ns]" << endl;
		cout << "\tOperation time: " << reduce_add_op << " [ns]" << endl;
		cout << "\t" << reduce_add_full << endl;
		cout << endl;
		cout << "Standard Deviation by a reduction kernel: " << endl;
		cout << "\tKernel execution time: " << reduce_standard_deviation_ns << " [ns]" << endl;
		cout << "\tMemory transfer: " << reduce_standard_deviation_mem << " [ns]" << endl;
		cout << "\tOperation time: " << reduce_standard_deviation_op << " [ns]" << endl;
		cout << "\t" << reduce_standard_deviation_full << endl;

		cout << endl;
		cout << "Parallel selection sort: " << endl;
		cout << "\tKernel execution time: " << sort_ns << " [ns]" << endl;
		cout << "\tMemory transfer: " << sort_mem << " [ns]" << endl;
		cout << "\tOperation time: " << sort_op << " [ns]" << endl;
		cout << "\t" << sort_full << endl;

		cout << endl;
		cout << "Total program execution time: " << totalElapsed << " [ns]" << endl;
		cout << "----------------------- Statistical Analysis Calculation Results -----------------------" << endl;

		sorted_temp = B;
		_min = sorted_temp[0];
		_max = sorted_temp[vector_elements - 1];
		median = sorted_temp[vector_elements / 2 - 1];
		firstQuart = sorted_temp[(vector_elements / 4) - 1];
		thirdQuart = sorted_temp[(vector_elements / 2) + (vector_elements / 4)];

		cout << "Mean: " << mean << endl;
		cout << "Standard Deviation: " << stdDev << endl;
		cout << "Min: " << _min << endl;
		cout << "Max: " << _max << endl;
		cout << "Median: " << median << endl;
		cout << "First Quartile: " << firstQuart << endl;
		cout << "Third Quartile: " << thirdQuart << endl;

		cout << endl;
		cout << "Please enter any key to exit... ";

		/*Output finishes here*/
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

// Function responsible for removing any padded values that have been inserted into the vectors
vector<float> quickDelete(vector<float>vec, float pad)
{
	int vector_elements = vec.size()-1;

	while (vec[vector_elements] == pad)
	{
		vector_elements = vector_elements - 1;
		vec.pop_back();
	}
	return vec;
}

/*end of script*/