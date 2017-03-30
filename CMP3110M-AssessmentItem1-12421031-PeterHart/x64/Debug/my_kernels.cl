/*my_kernels.cl*/

void atomicAdd(__global float*, float);

// Addition through reduction
__kernel void reduce_add(__global const float* A, __global float* B, __local float* scratch, float pad) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// Skip any neutral elements (pad) added for padding
	if (A[id] != pad)
	{
		// Cache all N elements from global memory to local memory
		scratch[lid] = A[id];
	}

	// Wait for all local threads to finish copying from global to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// Step through the data and accumulate partial sums in each workgroup
	for (int i = 1; i < N; i *= 2) {
		// Stride through the vector
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			// Calculate partial sum
			scratch[lid] += scratch[lid + i];
		}
		// Wait for all local threads to finish partial sum accumulation
		barrier(CLK_LOCAL_MEM_FENCE);
	}


	// Copy the cache to the output vector
	if (!lid) 
	{
		// Accumulate the total of the partial sums (from each local workgroup) to the first element of the output vector
		// (serial operation)
		atomicAdd(&B[0], scratch[lid]);
	}
}

// Standard Deviation through reduction
__kernel void reduce_standard_deviation(__global const float* A, __global float* B, __local float* scratch, float mean, float pad) 
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// Cache all N values from global memory to local memory
	scratch[lid] = A[id];

	// Wait for all local threads to finish copying from global to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	if (scratch[lid] != pad)
	{ 
		// Calculate the squared difference of each element from the mean
		scratch[lid] = ((scratch[lid] - mean) * (scratch[lid] - mean));
	}
	else
	{
		scratch[lid] = pad;
	} 
	// Step through the data and accumulate partial sums in each workgroup
	for (int i = 1; i < N; i *= 2)
	{
		// Stride through the vector
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			if (scratch[lid + i] != pad)
			{ 
				// Calculate partial sum of the squared differences
				scratch[lid] += scratch[lid + i];
			}
		}
		// Wait for all local threads to finish partial sum accumulation
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Copy the cache to the output vector
	if (!lid)
	{
		// Accumulate the total of the partial sums (from each local workgroup) to the first element of the output vector
		// (serial operation)
		atomicAdd(&B[0], scratch[lid]);
	}
}

// Parallel Selection Sort
__kernel void parallel_selection_sort(__global const float* A, __global float* B)
{
	int id = get_global_id(0); 
	int N = get_global_size(0);
	int pos = 0;
	float currentData = 0.f;
	float newData = 0.f;
	bool smaller = 0;

	// Cache all N elements from global memory to local memory
	currentData = A[id];


	// Compute position of A[id] in output
	for (int i = 0; i < N; i++)
	{
		newData = A[i]; // broadcasted
		// If the next element is smaller than the current element
		smaller = (newData < currentData) || (newData == currentData && i < id);  
		
		// Increase position if next element is smaller
		pos += (smaller) ? 1 : 0;
	}
	// Copy the element to a new position
	B[pos] = currentData;
}

// (Atomic) Max
__kernel void reduce_max(__global const int* A, __global int* B, __local int* scratch, float pad)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// Skip any neutral elements (pad) added for padding
	if (A[id] != pad)
	{
		// Cache all N elements from global memory to local memory
		scratch[lid] = A[id];
	}

	// Wait for all local threads to finish copying from global to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// Step through the data and accumulate the max in each workgroup
	for (int i = 1; i < N; i *= 2)
	{
		// Stride through the vector
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			// If this element is greater than the current max of this workgroup
			if (scratch[lid] < scratch[lid + i])
			{
				// Keep this element in this workgroup
				scratch[lid] = scratch[lid + i];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Copy the cache to the output vector
	if (!lid)
	{
		// Accumulate the max of the calculated partial max results to the first element of the output vector
		// (serial operation)
		atomic_max(&B[0], scratch[lid]);
	}
}

// (Atomic) Min
__kernel void reduce_min(__global const int* A, __global int* B, __local int* scratch, float pad)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// Copies values to local memory
	scratch[lid] = A[id];

	// Wait for all local threads to finish copying from global to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// Step through the data and accumulate the min in each workgroup
	for (int i = 1; i < N; i *= 2)
	{
		// Stride through the vector
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			// If this element is less than the current max of this workgroup
			if (scratch[lid] > scratch[lid + i])
			{
				// Keep this element in this workgroup
				scratch[lid] = scratch[lid + i];
			}
		}
		// Wait for all local threads to finish partial sum accumulation
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Copy the cache to the output vector
	if (!lid)
	{
		// Accumulate the min of the calculated partial min results to the first element of the output vector
		// (serial operation)
		atomic_min(&B[0], scratch[lid]);
	}
}

// Custom Atomic Add (for handling floats)
void atomicAdd(__global float *addr, float val)
{
	union 
	{
		unsigned int u32;
		float        f32;
	} next, expected, current;
	current.f32 = *addr;
	do 
	{
		expected.f32 = current.f32;
		next.f32 = expected.f32 + val;
		current.u32 = atomic_cmpxchg((volatile __global unsigned int *)addr, expected.u32, next.u32);
	} while (current.u32 != expected.u32);
}


//Median


//1st Quartile


//3rd Quartile

