
void cmpxchg(__global int* A, __global int* B, bool dir)
{
	if ((!dir && *A > *B) || (dir && *A < *B)) {
		int t = *A;
		*A = *B;
		*B = t;
	}
}

void bitonic_merge(int id, __global int* A, int N, bool dir) {
	for (int i = N / 2; i > 0; i /= 2) {
		if ((id % (i * 2)) < i)
			cmpxchg(&A[id], &A[id + i], dir);

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

__kernel void sort_bitonic(__global int* A){ 
	int id = get_global_id(0);
	int N = get_global_size(0);

	for(int i = 1; i < N/2; i*= 2){ 
		if (id % (i*4) < i*2)
			bitonic_merge(id, A, i*2, false);
		else if ((id + i*2) % (i*4) < i*2)
			bitonic_merge(id, A, i*2, true);
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	bitonic_merge(id, A, N, false);
}

//reduce using local memory + accumulation of local sums into a single location
//works with any number of groups - not optimal!

//Min
__kernel void get_min(__global int* A, int N){ 
	sort_bitonic(A);
	N = A[0];
}

//Max
__kernel void get_max(__global int* A, int N){ 
	sort_bitonic(A);
}

//Average


//Standard Deviation


//Median


//1st Quartile


//3rd Quartile

