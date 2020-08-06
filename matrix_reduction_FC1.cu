// C = alpha * A * B + beta * C
__global__ void MatMulKernelAB(const int M,const int K,float *A,
								const int K1, const int N,float *B, 
								const int M1, const int N1,float *C,
								const float alpha,const float beta)
{
	// Each thread computes at most (UNROLL_X * UNROLL_Y) elements of C
	// by accumulating results into Cvalue
	int row = threadIdx.y*2;
	int col = threadIdx.x*128;
	#pragma unroll 
	for(int j = 0; j < 2; ++j){
		#pragma SIMD (i)
		#pragma unroll
		for(int i = 0; i < 128; ++i){
			float Cvalue = C[(row+j)*N+(col+i)];
			for(int e = 0; e < K; ++e){
				#pragma reduction (Cvalue,e,+,1024)
				Cvalue += A[(row+j)*K + e] * B[e*N + (col+i)];
			}
			//C[row][col] = alpha * Cvalue + beta * C[row][col];
			C[(row+j)*N+(col+i)] = Cvalue  ;
		}
	}
}
int main(int argc, char const *argv[])
{
	int M = 32;
	int N = 512;
	int K = 9216;

	float *A;
	float *B;
	float *C;

	dim3 blockDim(4,16,1);
	dim3 gridDim(1,1,1);

	MatMulKernelAB<<<gridDim,blockDim>>>(M,K,A,K,N,B,M,N,C,1.0,0.0);
	return 0;
}
