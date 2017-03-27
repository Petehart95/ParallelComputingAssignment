/*my_kernels.cl*/
void swap(__global double* A, __global double* B, bool dir)
{
	if ((!dir && *A > *B) || (dir && *A < *B)) 
	{
		int t = *A;
		*A = *B;
		*B = t;
	}
}

void bitonic_merge(int id, __global double* A, int N, bool dir) 
{
	for (int i = N / 2; i > 0; i /= 2) 
	{
		barrier(CLK_GLOBAL_MEM_FENCE);
		if ((id % (i * 2)) < i)
			swap(&A[id], &A[id + i], dir);
			barrier(CLK_GLOBAL_MEM_FENCE);

	}
}

__kernel void sort_bitonic(__global double* A)
{ 
	int id = get_global_id(0);
	int N = get_global_size(0);

	barrier(CLK_GLOBAL_MEM_FENCE);
	for(int i = 1; i < N/2; i*= 2)
	{ 
		if (id % (i*4) < i*2)
			bitonic_merge(id, A, i*2, false);
		else if ((id + i*2) % (i*4) < i*2)
			bitonic_merge(id, A, i*2, true);
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

/*
void swap(__global double* A, __global double* B)
{
	if (*A > *B) 
	{ 
		int t = *A;
		*A = *B;
		*B = t;
	}
}


__kernel void sort_oddeven(__global double* A)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	for (int i = 0; i < N; i+=2)
	{ 
		if (id%2 == 1 && id+1 < N)
		{
			swap(&A[id],&A[id+1]);
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
		if (id%2 == 0 && id+1 < N)
		{
			swap(&A[id],&A[id+1]);
		}
	}
}
*/
//Sum
__kernel void add(__global const int* A, __global int* B, __local int* scratch) 
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
	{
		atomic_add(&B[0], scratch[lid]);
	}
}


//Min
__kernel void get_min(__global int* A, int N)
{ 
	sort_bitonic(A);
	N = A[0];
}

//Max
__kernel void get_max(__global int* A, int N)
{ 
	sort_bitonic(A);
}

//Average


//Standard Deviation


//Median


//1st Quartile


//3rd Quartile

