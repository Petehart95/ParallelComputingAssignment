/*my_kernels.cl*/

//reduce using local memory + accumulation of local sums into a single location
//works with any number of groups - not optimal!
__kernel void reduce_add(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			scratch[lid] += scratch[lid + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[0], scratch[lid]);
	}
}

// Max
__kernel void reduce_max(__global const int* A, __global int* B, __local int* scratch)
{
	// Gets IDs and size
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// Copies values to local memory
	scratch[lid] = A[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loops through values to find the max
	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			if (scratch[lid] < scratch[lid + i])
			{ 
				scratch[lid] = scratch[lid + i];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	// For each workgroup find the max
	if (!lid)
	{
		atomic_max(&B[0], scratch[lid]);
	}
}

// Min
__kernel void reduce_min(__global const int* A, __global int* B, __local int* scratch)
{
	// Gets IDs and size
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// Copies values to local memory
	scratch[lid] = A[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loops through values to find the min
	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			if (scratch[lid] > scratch[lid + i])
			{ 
				scratch[lid] = scratch[lid + i];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	// For each workgroup find the min
	if (!lid)
	{
		atomic_min(&B[0], scratch[lid]);
	}
}
//
//reduce using local memory (so called privatisation)
__kernel void reduce_standard_deviation(__global const int* A, __global int* B, __local int* scratch) 
{
	int mean = 2;
	int var;
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);


	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE); //wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{

		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array	
	if (!lid)
	{
		atomic_add(&B[0], scratch[lid]);
	}
}

//Median


//1st Quartile


//3rd Quartile

