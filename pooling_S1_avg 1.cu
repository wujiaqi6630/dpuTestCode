
#define bottom_data(n,c,h,w) bottom_data[(n)*H*W*C+(c)*H*W+(h)*W+(w)]
#define top_data(n,c,h,w) top_data[(n)*OC*OH*OW+(c)*OH*OW+(h)*OW+(w)]
#define kernel(n,c,h,w) kernel[(n)*C*FW*FH+(c)*FW*FH+(h)*FW+(w)]
__global__ void DPUPooling(
        int N, int C, int H, int W,float *bottom_data,
        int N1,int OC, int OH, int OW, float *top_data,
        int SH, int SW, int PH, int PW, int FH, int FW)
{
    int init_pw = (blockIdx.x * blockDim.x + threadIdx.x) * 7; 
    int init_ph = (blockIdx.y * blockDim.y + threadIdx.y) * 7 ;
    int init_c = (blockIdx.z * blockDim.z + threadIdx.z)* 1 ;
    #pragma SIMD (n)
    #pragma unroll 
    for(int n = 0; n < N; ++n){
        #pragma unroll
        for (int c = 0; c < 1; ++c) {
            #pragma unroll
            for (int ph = 0; ph < 7; ++ph) {
                #pragma unroll
                for (int pw = 0; pw < 7; ++pw) {
                    if((ph+init_ph) < OH &&  (pw+init_pw) < OW && (init_c+c) < OC){
                        int hs = (ph+init_ph) * SH -PH;
                        int ws = (pw+init_pw) * SW - PW;
                        int hend = min(hs + FH, H);
                        int wend = min(ws + FW, W);
                        int hstart = max(hs, 0);
                        int wstart = max(ws, 0);
                        float sum = 0.0f;
                        //cal
                        
                            for (int h = hstart-hs; h < hend-hs; ++h) {
                                for (int w = wstart-ws; w < wend-ws; ++w) { 
                                    sum += bottom_data(n,init_c+c,h+hs,w+ws) ;            
                                }
                            }
                        //ST
                        top_data(n,init_c+c,ph+init_ph,pw+init_pw) = sum /4;   
                    }               
                }
            }
        }
    }
}

int main(){

    int N = 32 ;            //input & output nums
    int C = 96;           //input_channel
    int H= 55;             //input_height
    int W= 55;              //input 
    int OC = 96;   //output_channel
    int OH = 27;     //OH = (H + PH * 2 - FH)/SH + 1
    int OW = 27;      //OW = (W + PW * 2 - FW)/SW + 1
    int SH = 2;
    int SW = 2;
    int PH = 0;
    int PW = 0;
    int FH = 3;
    int FW = 3;
    
    float *device_bottom_data;
    float *device_top_data;
    
    dim3 grid(1,1,24);//width,height,(channel*num)
    dim3 block(4,4,4);//width,hegiht,channel

   
    DPUPooling<<<grid,block>>>(N, C, H, W,device_bottom_data, N,OC,OH,OW,device_top_data,SH, SW, PH, PW, FH, FW);

    return 0;
}
