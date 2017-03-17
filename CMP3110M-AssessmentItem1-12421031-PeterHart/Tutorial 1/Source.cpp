/*CMP3110M Parallel Computing Assignment | HAR12421031 Peter Hart*/

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <CL/cl.hpp>
#include "Utils.h"

//using namespace std;

void print_help();

int main(int argc, char **argv)
{
	int platform_id = 0;
	int device_id = 0;

	//User-Interface for the device Selection
	for (int i = 1; i < argc; i++)	
	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//TRY-CATCH STATEMENT STARTS HERE (catch exceptions)
	try {
		cl::Context context = GetContext(platform_id, device_id);

		//Display the selected device
		std::cout << "Device Selected: " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl; 

		//Queue variable which will be used to push commands to the device
		cl::CommandQueue queue(context); 

		//Load and build the device code
		cl::Program::Sources sources; 
		AddSources(sources, "my_kernels.cl");
		cl::Program program(context, sources);

		//Build and debug the kernel code currently residing on the device
		try 
		{
			program.build();
		}
		catch (const cl::Error& err) 
		{
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//Load and store the weather dataset
		std::vector<int> A = { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };

		size_t local_size = 16;

		size_t padding_size = A.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size - padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		size_t vector_elements = A.size();//number of elements
		size_t vector_size = A.size()*sizeof(int);//size in bytes

		//host - output
		std::vector<int> C(vector_elements);

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);

		std::cout << "Before: A = " << A << std::endl;

		//Part 5 - device operations

		//5.1 Copy arrays A and B to device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0]);

		//5.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel_sort = cl::Kernel(program, "sort_bitonic");
		kernel_sort.setArg(0, buffer_A);

		queue.enqueueNDRangeKernel(kernel_sort, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange);

		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0]);


		//padding applied to ensure sorting algorithm can be applied to values near the boundaries of the vector
		for (int i = 6; i > 0; i--)
		{
			A.erase(A.begin());
		}


		std::cout << "After: A = " << A << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

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