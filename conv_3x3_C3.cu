
#define bottom_data(n,c,h,w) bottom_data[(n)*H*W*C+(c)*H*W+(h)*W+(w)]
#define top_data(n,c,h,w) top_data[(n)*OC*OH*OW+(c)*OH*OW+(h)*OW+(w)]
#define kernel(n,c,h,w) kernel[(n)*C*FW*FH+(c)*FW*FH+(h)*FW+(w)]
__global__ void DPUDirectConv(
        int N, int C, int H, int W,float *bottom_data,
        int OC1,int C1,int FH,int FW,float *kernel,
        int N1,int OC, int OH, int OW, float *top_data,
        int SH, int SW, int PH, int PW)
{
    int init_pw = (threadIdx.x) * 7; 
    int init_ph = (blockIdx.y * blockDim.y + threadIdx.y) * 7 ;
    int init_c = (blockIdx.z * blockDim.z + threadIdx.z)* 1;
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
                        int ws = (pw+init_pw) * SW -PW;
                        int hend = min(hs + FH, H);
                        int wend = min(ws + FW, W);
                        int hstart = max(hs, 0);
                        int wstart = max(ws, 0);
                        float sum = 0.0f;
                        //cal
                        #pragma reduction (sum,yc,+,64)
                        for(int yc=0;yc < C; ++yc){
                            for (int h = hstart-hs; h < hend-hs; ++h) {
                                for (int w = wstart-ws; w < wend-ws; ++w) { 
                                    sum += bottom_data(n,yc,h+hs,w+ws) * kernel(init_c+c,yc,h,w);            
                                }
                            }
                        }
                        //ST
                        top_data(n,init_c+c,ph+init_ph,pw+init_pw) = sum;   
                    }               
                }
            }
        }
    }
}

int main(){

    int N = 32 ;            //input & output nums
    int C = 256;           //input_channel
    int H= 13;             //input_height
    int W= 13;              //input 
    int OC = 32;   //output_channel
    int OH = 13;     //OH = (H + PH * 2 - FH)/SH + 1
    int OW = 13;      //OW = (W + PW * 2 - FW)/SW + 1
    int SH = 1;
    int SW = 1;
    int PH = 1;
    int PW = 1;
    int FH = 3;
    int FW = 3;
    
    float *device_bottom_data;
    float *device_top_data;
    float *device_kernel;
    
    dim3 grid(1,1,2);//width,height,(channel*num)
    dim3 block(2,2,16);//width,hegiht,channel

   
    DPUDirectConv<<<grid,block>>>(N, C, H, W,device_bottom_data, OC,C,FH,FW,device_kernel,N,OC,OH,OW,device_top_data,SH, SW, PH, PW);

    return 0;
}
