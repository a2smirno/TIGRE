/*-------------------------------------------------------------------------
 *
 * CUDA functions for Steepest descend in POCS-type algorithms.
 *
 * This file will iteratively minimize by stepest descend the total variation
 * of the input image, with the parameters given, using GPUs.
 *
 * CODE by       Ander Biguri
 *
 * ---------------------------------------------------------------------------
 * ---------------------------------------------------------------------------
 * Copyright (c) 2015, University of Bath and CERN- European Organization for
 * Nuclear Research
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ---------------------------------------------------------------------------
 *
 * Contact: tigre.toolbox@gmail.com
 * Codes  : https://github.com/CERN/TIGRE
 * ---------------------------------------------------------------------------
 */







#define MAXTHREADS 1024
#define MAX_BUFFER 60

#include "POCS_TV.hpp"




#define cudaCheckErrors(msg) \
do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
                mexPrintf("%s \n",msg);\
                cudaDeviceReset();\
                mexErrMsgIdAndTxt("CBCT:CUDA:POCS_TV",cudaGetErrorString(__err));\
        } \
} while (0)
    
// CUDA kernels
//https://stackoverflow.com/questions/21332040/simple-cuda-kernel-optimization/21340927#21340927
    __global__ void divideArrayScalar(float* vec,float scalar,const size_t n){
        unsigned long long i = (blockIdx.x * blockDim.x) + threadIdx.x;
        for(; i<n; i+=gridDim.x*blockDim.x) {
            vec[i]/=scalar;
        }
    }
    __global__ void multiplyArrayScalar(float* vec,float scalar,const size_t n)
    {
        unsigned long long i = (blockIdx.x * blockDim.x) + threadIdx.x;
        for(; i<n; i+=gridDim.x*blockDim.x) {
            vec[i]*=scalar;
        }
    }
    __global__ void substractArrays(float* vec,float* vec2,const size_t n)
    {
        unsigned long long i = (blockIdx.x * blockDim.x) + threadIdx.x;
        for(; i<n; i+=gridDim.x*blockDim.x) {
            vec[i]-=vec2[i];
        }
    }
    __global__ void copyArrays(float* vec,float* vec2,const size_t n)
    {
        unsigned long long i = (blockIdx.x * blockDim.x) + threadIdx.x;
        for(; i<n; i+=gridDim.x*blockDim.x) {
            vec[i]=vec2[i];
        }
    }
    __global__ void sumArrays(float* vec,float* vec2,const size_t n)
    {
        unsigned long long i = (blockIdx.x * blockDim.x) + threadIdx.x;
        for(; i<n; i+=gridDim.x*blockDim.x) {
            vec[i]+=vec2[i];
        }
    }      
    __global__ void opL11(float* vec,float* vec1, long depth, long rows, long cols){
        unsigned long x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned long y = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned long z = threadIdx.z + blockIdx.z * blockDim.z;
        unsigned long long i = z * rows * cols + y * cols + x; 
        unsigned long long j = z * rows * cols + x; 
        unsigned long size2d = cols;
        unsigned long size3d = depth * rows * cols + rows * cols + cols;
        if ( x >= cols || y >= rows || z >= depth )
            return;
        if ( i+cols+1 >= size3d )
            return;    
        vec[i]=vec1[i];
    }
    __global__ void opL21(float* vec,float* vec1, long depth, long rows, long cols){
        unsigned long x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned long y = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned long z = threadIdx.z + blockIdx.z * blockDim.z;
        unsigned long long i = z * rows * cols + y * cols + x; 
        unsigned long long j = z * rows * cols + x; 
        unsigned long size2d = cols;
        unsigned long size3d = depth * rows * cols + rows * cols + cols;
        if ( x >= cols || y >= rows || z >= depth )
            return;
        if ( i+cols+1 >= size3d )
            return;    
        vec[i+cols]=0.25*(vec1[i+1]+vec1[i]+vec1[i+cols+1]+vec1[i+cols]);
        if ( j+1 >= size2d )
            return;     
        vec[j]=(vec1[j]+vec1[j+1])/4;
    }        
    __global__ void opL12(float* vec,float* vec1, long depth, long rows, long cols){
        unsigned long x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned long y = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned long z = threadIdx.z + blockIdx.z * blockDim.z;
        unsigned long long i = z * rows * cols + y * cols + x; 
        unsigned long long j = z * rows * cols + y * cols; 
        unsigned long size2d = cols;
        unsigned long size3d = depth * rows * cols + rows * cols + cols;
        if ( x >= cols || y >= rows || z >= depth )
            return;
        if ( i+cols+1 >= size3d )
            return;    
        vec[i+1]=0.25*(vec1[i+1]+vec1[i]+vec1[i+cols+1]+vec1[i+cols]);
        if ( j+1 >= size2d )
            return;     
        vec[j]=0.25*(vec1[j]+vec1[j+cols]);
    }
    __global__ void opL23(float* vec,float* vec1, long depth, long rows, long cols){
        unsigned long x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned long y = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned long z = threadIdx.z + blockIdx.z * blockDim.z;
        unsigned long long i = z * rows * cols + y * cols + x; 
        unsigned long long j = z * rows * cols + y * cols; 
        unsigned long size2d = cols;
        unsigned long size3d = depth * rows * cols + rows * cols + cols;
        if ( x >= cols || y >= rows || z >= depth )
            return;
        if ( i+cols+1 >= size3d )
            return;    
        vec[i+cols]=0.5*(vec1[i+cols]+vec1[i]);
        if ( j+1 >= size2d )
            return;     
        vec[j]=0.5*(vec1[j]);
    }
    __global__ void opL21(float* vec,float* vec1, long depth, long rows, long cols){
        unsigned long x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned long y = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned long z = threadIdx.z + blockIdx.z * blockDim.z;
        unsigned long long i = z * rows * cols + y * cols + x; 
        unsigned long long j = z * rows * cols + x; 
        unsigned long size2d = cols;
        unsigned long size3d = depth * rows * cols + rows * cols + cols;
        if ( x >= cols || y >= rows || z >= depth )
            return;
        if ( i+cols+1 >= size3d )
            return;    
        vec[i+cols]=0.25*(vec1[i+1]+vec1[i]+vec1[i+cols+1]+vec1[i+cols]);
        if ( j+1 >= size2d )
            return;     
        vec[j]=(vec1[j]+vec1[j+1])/4;
    }
    __global__ void opL13(float* vec,float* vec1,float* vec2, long depth, long rows, long cols){
        unsigned long x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned long y = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned long z = threadIdx.z + blockIdx.z * blockDim.z;
        unsigned long long i = z * rows * cols + y * cols + x; 
        unsigned long long j = z * rows * cols + y * cols; 
        unsigned long size2d = cols;
        unsigned long size3d = depth * rows * cols + rows * cols + cols;
        if ( x >= cols || y >= rows || z >= depth )
            return;
        if ( i+1 >= size3d )
            return;    
        vec[i+1]=0.5*(vec3[i+1]+vec3[i]);
        if ( j+1 >= size2d )
            return;     
        vec[j]=vec1[j]/2;
    }   
    __global__ void opLadj1(float* vec,float* vec1,float* vec2,float* vec3, long depth, long rows, long cols){
        unsigned long x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned long y = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned long z = threadIdx.z + blockIdx.z * blockDim.z;
        unsigned long long i = z * rows * cols + y * cols + x; 
        unsigned long long j = z * rows * cols + x; 
        unsigned long size2d = cols;
        unsigned long size3d = depth * rows * cols + rows * cols + cols;
        if ( x >= cols || y >= rows || z >= depth )
            return;
        if ( i+cols+1 >= size3d )
            return;    
        vec[i+cols]=vec1[i+cols]+0.25*(vec2[i+cols]+vec2[i]+vec2[i+cols+1]+vec2[i+1])+0.5*(vec3[i+cols]+vec3[i+cols+1]);
        if ( j+1 >= size2d )
            return;     
        vec[j]=vec1[j]+(vec2[j]+vec2[j+1])/4+(vec3[j]+vec3[j+1])/2;
    }
    __global__ void opLadj2(float* vec,float* vec1,float* vec2,float* vec3, long depth, long rows, long cols){
        unsigned long x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned long y = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned long z = threadIdx.z + blockIdx.z * blockDim.z;
        unsigned long long i = z * rows * cols + y * cols + x; 
        unsigned long long j = z * rows * cols + y * cols; 
        unsigned long size2d = z * rows * cols + cols * rows;
        unsigned long size3d = depth * rows * cols + rows * cols + cols;
        if ( x >= cols || y >= rows || z >= depth )
            return;
        if ( i+cols+1 >= size3d )
            return;    
        vec[i+1]=vec1[i+1]+0.25*(vec2[i+1]+vec2[i]+vec2[i+cols+1]+vec2[i+cols])+0.5*(vec3[i+1]+vec3[i+cols+1]);
        if ( j+cols >= size2d )
            return;     
        vec[j]=vec1[j]+(vec2[j]+vec2[j+cols])/4+(vec3[j]+vec3[j+cols])/2;
    }
    __global__ void prox_g(float* vec,float* vec2,const size_t n, const float sigma, const float mu)
    {
        unsigned long long i = (blockIdx.x * blockDim.x) + threadIdx.x;
        for(; i<n; i+=gridDim.x*blockDim.x) {
            if  ( sqrt(vec2[i] * vec2[i])/(sigma * mu) > 1 ) {
                vec[i]=-vec2[i] * (1 - sqrt(vec2[i] * vec2[i])/(sigma * mu));
            }
            else {
                vec[i]=0;
            }
        }
    }
    __global__ void grad_x(const float* u, float* grad,
            long depth, long rows, long cols){
        unsigned long x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned long y = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned long z = threadIdx.z + blockIdx.z * blockDim.z;
        if ( x >= cols || y >= rows || z >= depth )
            return;
        unsigned long size2d = rows*cols;
        unsigned long long idx = z * size2d + y * cols + x;
        
        float uidx = u[idx];
        
        if ( x - 1 >= 0 && x<cols) {
            grad[idx] = (uidx-u[z*size2d + y*cols + (x-1)]);
        }
    }
    __global__ void grad_y(const float* u, float* grad,
            long depth, long rows, long cols){
        unsigned long x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned long y = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned long z = threadIdx.z + blockIdx.z * blockDim.z;
        if ( x >= cols || y >= rows || z >= depth )
            return;
        unsigned long size2d = rows*cols;
        unsigned long long idx = z * size2d + y * cols + x;
        
        float uidx = u[idx];
        
        if ( y - 1 >= 0 && y<rows){
            grad[idx] = (uidx-u[z*size2d + (y-1)*cols + x]) ;
        }
    }
        
    __device__ __inline__
            void gradient(const float* u, float* grad,
            long z, long y, long x,
            long depth, long rows, long cols){
        unsigned long size2d = rows*cols;
        unsigned long long idx = z * size2d + y * cols + x;
        
        float uidx = u[idx];
        
        if ( z - 1 >= 0 && z<depth) {
            grad[0] = (uidx-u[(z-1)*size2d + y*cols + x]) ;
        }
        
        if ( y - 1 >= 0 && y<rows){
            grad[1] = (uidx-u[z*size2d + (y-1)*cols + x]) ;
        }
        
        if ( x - 1 >= 0 && x<cols) {
            grad[2] = (uidx-u[z*size2d + y*cols + (x-1)]);
        }
    }
    
    __global__ void gradientTV(const float* phi1, const float* phi2, float* dftv,
            long depth, long rows, long cols,const float delta){
        unsigned long x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned long y = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned long z = threadIdx.z + blockIdx.z * blockDim.z;
        unsigned long long idx = z * rows * cols + y * cols + x; 

        if ( x >= cols || y >= rows || z >= depth )
            return;
                
        dftv[idx]=-phi1[idx]-phi2[idx]+phi1[idx-1]+phi2[idx-cols][1];    
    
        return;
        
    }
    
    __device__ void warpReduce(volatile float *sdata, size_t tid) {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
    
    __global__ void  reduceNorm2(float *g_idata, float *g_odata, size_t n){
        extern __shared__ volatile float sdata[];
        //http://stackoverflow.com/a/35133396/1485872
        size_t tid = threadIdx.x;
        size_t i = blockIdx.x*blockDim.x + tid;
        size_t gridSize = blockDim.x*gridDim.x;
        float mySum = 0;
        float value=0;
        while (i < n) {
            value=g_idata[i]; //avoid reading twice
            mySum += value*value;
            i += gridSize;
        }
        sdata[tid] = mySum;
        __syncthreads();
        
        if (tid < 512)
            sdata[tid] += sdata[tid + 512];
        __syncthreads();
        if (tid < 256)
            sdata[tid] += sdata[tid + 256];
        __syncthreads();
        
        if (tid < 128)
            sdata[tid] += sdata[tid + 128];
        __syncthreads();
        
        if (tid <  64)
            sdata[tid] += sdata[tid + 64];
        __syncthreads();
        
        
#if (__CUDART_VERSION >= 9000)
        if ( tid < 32 )
        {
            mySum = sdata[tid] + sdata[tid + 32];
            for (int offset = warpSize/2; offset > 0; offset /= 2) {
                mySum += __shfl_down_sync(0xFFFFFFFF, mySum, offset,32);
            }
        }
#else
        if (tid < 32) {
            warpReduce(sdata, tid);
            mySum = sdata[0];
        }
#endif
        if (tid == 0) g_odata[blockIdx.x] = mySum;
    }
    
    __global__ void  reduceSum(float *g_idata, float *g_odata, size_t n){
        extern __shared__ volatile float sdata[];
        //http://stackoverflow.com/a/35133396/1485872
        size_t tid = threadIdx.x;
        size_t i = blockIdx.x*blockDim.x + tid;
        size_t gridSize = blockDim.x*gridDim.x;
        float mySum = 0;
        // float value=0;
        while (i < n) {
            mySum += g_idata[i];
            i += gridSize;
        }
        sdata[tid] = mySum;
        __syncthreads();
        
        if (tid < 512)
            sdata[tid] += sdata[tid + 512];
        __syncthreads();
        if (tid < 256)
            sdata[tid] += sdata[tid + 256];
        __syncthreads();
        
        if (tid < 128)
            sdata[tid] += sdata[tid + 128];
        __syncthreads();
        
        if (tid <  64)
            sdata[tid] += sdata[tid + 64];
        __syncthreads();
        
        
#if (__CUDART_VERSION >= 9000)
        if ( tid < 32 )
        {
            mySum = sdata[tid] + sdata[tid + 32];
            for (int offset = warpSize/2; offset > 0; offset /= 2) {
                mySum += __shfl_down_sync(0xFFFFFFFF, mySum, offset,32);
            }
        }
#else
        if (tid < 32) {
            warpReduce(sdata, tid);
            mySum = sdata[0];
        }
#endif
        if (tid == 0) g_odata[blockIdx.x] = mySum;
    }
    
    
    
    
// main function
void dtv_asd_pocs_tv(float* img,float* dst,float alpha,const long* image_size, int maxIter,const float delta, const GpuIds& gpuids){
        // Prepare for MultiGPU
        int deviceCount = gpuids.GetLength();
        cudaCheckErrors("Device query fail");
        if (deviceCount == 0) {
            mexErrMsgIdAndTxt("minimizeDTV:POCS_DTV:GPUselect","There are no available device(s) that support CUDA\n");
        }
        //
        // CODE assumes
        // 1.-All available devices are usable by this code
        // 2.-All available devices are equal, they are the same machine (warning thrown)
        // Check the available devices, and if they are the same
        if (!gpuids.AreEqualDevices()) {
            mexWarnMsgIdAndTxt("minimizeDTV:POCS_DTV:GPUselect","Detected one (or more) different GPUs.\n This code is not smart enough to separate the memory GPU wise if they have different computational times or memory limits.\n First GPU parameters used. If the code errors you might need to change the way GPU selection is performed.");
        }
        int dev;
        
        // We don't know if the devices are being used. lets check that. and only use the amount of memory we need.
        // check free memory
        size_t mem_GPU_global;
        checkFreeMemory(gpuids, &mem_GPU_global);

        
        
        // %5 of free memory should be enough, we have almost no variables in these kernels
        size_t total_pixels              = image_size[0] * image_size[1]  * image_size[2] ;
        size_t mem_slice_image           = sizeof(float)* image_size[0] * image_size[1]  ;
        size_t mem_size_image            = sizeof(float)* total_pixels;
        size_t mem_auxiliary             = sizeof(float)* (total_pixels + MAXTHREADS - 1) / MAXTHREADS;
        
        // Decide how are we handling the distribution of computation
        size_t mem_img_each_GPU;
        
        unsigned int buffer_length=2;
        //Does everything fit in the GPU?
        unsigned int slices_per_split;
        
        // if it is a thin problem (no need to split), just use one GPU
        if (image_size[2]<4){deviceCount=1;}
        
        unsigned int splits=1; // if the number does not fit in an uint, you have more serious trouble than this.
        if(mem_GPU_global> 3*mem_size_image+3*(deviceCount-1)*mem_slice_image*buffer_length+mem_auxiliary) {
            // We only need to split if we have extra GPUs
            slices_per_split=(image_size[2]+deviceCount-1)/deviceCount;
            mem_img_each_GPU=mem_slice_image*((slices_per_split+buffer_length*2));
        }else{
            // As mem_auxiliary is not expected to be a large value (for a 2000^3 image is around 28Mbytes), lets for now assume we need it all
            size_t mem_free=mem_GPU_global-mem_auxiliary;
            
            splits=(unsigned int)(ceil(((float)(3*mem_size_image)/(float)(deviceCount))/mem_free));
            // Now, there is an overhead here, as each splits should have 2 slices more, to account for overlap of images.
            // lets make sure these 2 slices fit, if they do not, add 1 to splits.
            slices_per_split=(image_size[2]+deviceCount*splits-1)/(deviceCount*splits);
            mem_img_each_GPU=(mem_slice_image*(slices_per_split+buffer_length*2));
            
            // if the new stuff does not fit in the GPU, it means we are in the edge case where adding that extra slice will overflow memory
            if (mem_GPU_global< 3*mem_img_each_GPU+mem_auxiliary){
                // one more split should do the job, as its an edge case.
                splits++;
                //recompute for later
                slices_per_split=(image_size[2]+deviceCount*splits-1)/(deviceCount*splits); // amount of slices that fit on a GPU. Later we add 2 to these, as we need them for overlap
                mem_img_each_GPU=(mem_slice_image*(slices_per_split+buffer_length*2));
            }


            // How many EXTRA buffer slices should be able to fit in here??!?!
            // Only do it if there are splits needed. 
            if(splits>1){
                mem_free=mem_GPU_global-(3*mem_img_each_GPU+mem_auxiliary);
                unsigned int extra_buff=(mem_free/mem_slice_image); 
                buffer_length=(extra_buff/2)/3; // we need double whatever this results in, rounded down.
                buffer_length=max(buffer_length,2);// minimum 2
                buffer_length=min(MAX_BUFFER,buffer_length);

                mem_img_each_GPU=mem_slice_image*(slices_per_split+buffer_length*2);
                
            }else{
                buffer_length=2;
            }

            // Assert
            if (mem_GPU_global< 3*mem_img_each_GPU+mem_auxiliary){
                mexErrMsgIdAndTxt("minimizeDTV:POCS_DTV:GPU","Assertion Failed. Logic behind spliting flawed! Please tell: ander.biguri@gmail.com\n");
            }
        }
        
        
         // Assert
       
        if ((slices_per_split+buffer_length*2)*image_size[0]*image_size[1]* sizeof(float)!= mem_img_each_GPU){
            mexErrMsgIdAndTxt("minimizeDTV:POCS_DTV:GPU","Assertion Failed. Memory needed calculation broken! Please tell: ander.biguri@gmail.com\n");
        }
        
        
        
        
        
        
        float** d_image=    (float**)malloc(deviceCount*sizeof(float*));
        float** phi1    =    (float**)malloc(deviceCount*sizeof(float*));
        float** phi2    =    (float**)malloc(deviceCount*sizeof(float*));
        float** psi1    =    (float**)malloc(deviceCount*sizeof(float*));
        float** psi2    =    (float**)malloc(deviceCount*sizeof(float*));
        float** v11    =    (float**)malloc(deviceCount*sizeof(float*));
        float** v12    =    (float**)malloc(deviceCount*sizeof(float*));
        float** v13    =    (float**)malloc(deviceCount*sizeof(float*));
        float** v21    =    (float**)malloc(deviceCount*sizeof(float*));
        float** v22    =    (float**)malloc(deviceCount*sizeof(float*));
        float** v23    =    (float**)malloc(deviceCount*sizeof(float*));
        float** L11    =    (float**)malloc(deviceCount*sizeof(float*));
        float** L12    =    (float**)malloc(deviceCount*sizeof(float*));
        float** L13    =    (float**)malloc(deviceCount*sizeof(float*));
        float** L21    =    (float**)malloc(deviceCount*sizeof(float*));
        float** L22    =    (float**)malloc(deviceCount*sizeof(float*));
        float** L23    =    (float**)malloc(deviceCount*sizeof(float*));
        float** d_dimgTV=   (float**)malloc(deviceCount*sizeof(float*));
        float** d_norm2aux= (float**)malloc(deviceCount*sizeof(float*));
        float** d_norm2=    (float**)malloc(deviceCount*sizeof(float*));
        float mu1 = 1e5;
        float mu2 = 1e-5;
        float sigma = 0.33;
         
        // allocate memory in each GPU
        for (dev = 0; dev < deviceCount; dev++){
            cudaSetDevice(gpuids[dev]);
            
            cudaMalloc((void**)&d_image[dev]    , mem_img_each_GPU);
            cudaMemset(d_image[dev],0           , mem_img_each_GPU);
            cudaMalloc((void**)&phi1[dev]    , mem_img_each_GPU);
            cudaMemset(phi1[dev],0           , mem_img_each_GPU);
            cudaMalloc((void**)&phi2[dev]    , mem_img_each_GPU);
            cudaMemset(phi2[dev],0           , mem_img_each_GPU);
            cudaMalloc((void**)&psi1[dev]    , mem_img_each_GPU);
            cudaMemset(psi1[dev],0           , mem_img_each_GPU);
            cudaMalloc((void**)&psi2[dev]    , mem_img_each_GPU);
            cudaMemset(psi2[dev],0           , mem_img_each_GPU);
            cudaMalloc((void**)&v11[dev]    , mem_img_each_GPU);
            cudaMemset(v11[dev],0           , mem_img_each_GPU);
            cudaMalloc((void**)&v12[dev]    , mem_img_each_GPU);
            cudaMemset(v12[dev],0           , mem_img_each_GPU);
            cudaMalloc((void**)&v13[dev]    , mem_img_each_GPU);
            cudaMemset(v13[dev],0           , mem_img_each_GPU);
            cudaMalloc((void**)&v21[dev]    , mem_img_each_GPU);
            cudaMemset(v21[dev],0           , mem_img_each_GPU);
            cudaMalloc((void**)&v22[dev]    , mem_img_each_GPU);
            cudaMemset(v22[dev],0           , mem_img_each_GPU);
            cudaMalloc((void**)&v23[dev]    , mem_img_each_GPU);
            cudaMemset(v23[dev],0           , mem_img_each_GPU);
            cudaMalloc((void**)&L11[dev]    , mem_img_each_GPU);
            cudaMemset(L11[dev],0           , mem_img_each_GPU);
            cudaMalloc((void**)&L12[dev]    , mem_img_each_GPU);
            cudaMemset(L12[dev],0           , mem_img_each_GPU);
            cudaMalloc((void**)&L13[dev]    , mem_img_each_GPU);
            cudaMemset(L13[dev],0           , mem_img_each_GPU);
            cudaMalloc((void**)&L21[dev]    , mem_img_each_GPU);
            cudaMemset(L21[dev],0           , mem_img_each_GPU);
            cudaMalloc((void**)&L22[dev]    , mem_img_each_GPU);
            cudaMemset(L22[dev],0           , mem_img_each_GPU);
            cudaMalloc((void**)&L23[dev]    , mem_img_each_GPU);
            cudaMemset(L23[dev],0           , mem_img_each_GPU);
            cudaMalloc((void**)&d_dimgTV[dev]   , mem_img_each_GPU);
            cudaMalloc((void**)&d_norm2[dev]    , slices_per_split*mem_slice_image);
            cudaMalloc((void**)&d_norm2aux[dev] , mem_auxiliary);
            cudaCheckErrors("Malloc  error");
            
            
        }
       unsigned long long buffer_pixels=buffer_length*image_size[0]*image_size[1];
        float* buffer;
        if(splits>1){
            mexWarnMsgIdAndTxt("minimizeDTV:POCS_DTV:Image_split","Your image can not be fully split between the available GPUs. The computation of minTV will be significantly slowed due to the image size.\nApproximated mathematics turned on for computational speed.");
        }else{
            cudaMallocHost((void**)&buffer,buffer_length*image_size[0]*image_size[1]*sizeof(float));
        }
        
        
        
        // Lets try to make the host memory pinned:
        // We laredy queried the GPU and assuemd they are the same, thus should have the same attributes.
        int isHostRegisterSupported = 0;
#if CUDART_VERSION >= 9020
        cudaDeviceGetAttribute(&isHostRegisterSupported,cudaDevAttrHostRegisterSupported,gpuids[0]);
#endif
        // splits>2 is completely empirical observation
        if (isHostRegisterSupported & splits>2){
            cudaHostRegister(img ,image_size[2]*image_size[1]*image_size[0]*sizeof(float),cudaHostRegisterPortable);
            cudaHostRegister(dst ,image_size[2]*image_size[1]*image_size[0]*sizeof(float),cudaHostRegisterPortable);
        }
        cudaCheckErrors("Error pinning memory");

        
        
                // Create streams
        int nStream_device=2;
        int nStreams=deviceCount*nStream_device;
        cudaStream_t* stream=(cudaStream_t*)malloc(nStreams*sizeof(cudaStream_t));
        
        for (dev = 0; dev < deviceCount; dev++){
            cudaSetDevice(gpuids[dev]);
            for (int i = 0; i < nStream_device; ++i){
                cudaStreamCreate(&stream[i+dev*nStream_device]);
            }
        }
        cudaCheckErrors("Stream creation fail");

        
        // For the reduction

        double totalsum_prev;
        double totalsum;
        float sum_curr_spl;
        float * sumnorm2;
        cudaMallocHost((void**)&sumnorm2,deviceCount*sizeof(float));
        
        unsigned int curr_slices;
        unsigned long long curr_pixels;
        size_t linear_idx_start;
        unsigned long long* offset_device=(unsigned long long*)malloc(deviceCount*sizeof(unsigned long long));
        unsigned long long* offset_host  =(unsigned long long*)malloc(deviceCount*sizeof(unsigned long long));
        unsigned long long* bytes_device =(unsigned long long*)malloc(deviceCount*sizeof(unsigned long long));
        bool is_first_chunk;
        bool is_last_chunk;
        for(unsigned int i=0;i<maxIter;i+=(buffer_length-1)){
            if(splits>1){
                totalsum_prev=0;
            }
            for(unsigned int sp=0;sp<splits;sp++){
                
                // For each iteration we need to compute all the image. The ordering of these loops
                // need to be like this due to the bounding layers between splits. If more than 1 split is needed
                // for each GPU then there is no other way that taking the entire memory out of GPU and putting it back.
                // If the memory can be shared between GPUs fully without extra splits, then there is an easy way of synchronizing the memory
                
                // Copy image to memory
                for (dev = 0; dev < deviceCount; dev++){
                    curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                    curr_pixels=curr_slices*image_size[0]*image_size[1];
                    linear_idx_start=image_size[0]*image_size[1]*slices_per_split*(sp*deviceCount+dev);
                    
                    // Check if its the first or last chunck
                    is_last_chunk=!((sp*deviceCount+dev)<deviceCount*splits-1);
                    is_first_chunk=!(sp*deviceCount+dev);
                    
                    // lets compute where we start copyes and how much. This avoids 3 calls to Memcpy
                    offset_device[dev]=buffer_pixels*is_first_chunk;
                    offset_host[dev]=linear_idx_start-buffer_pixels*!is_first_chunk;
                    bytes_device[dev]=curr_pixels+buffer_pixels*!is_first_chunk+buffer_pixels*!is_last_chunk;
                }

                if(i==0){
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        
                        cudaMemcpyAsync(d_image[dev]+offset_device[dev], img+offset_host[dev]  , bytes_device[dev]*sizeof(float), cudaMemcpyHostToDevice,stream[dev*nStream_device+1]);
                        
                        
                    }
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        cudaDeviceSynchronize();
                    }
                }
                // if we need to split and its not the first iteration, then we need to copy from Host memory the previosu result.
                if (splits>1 & i>0){
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        cudaMemcpyAsync(d_image[dev]+offset_device[dev], dst+offset_host[dev]  , bytes_device[dev]*sizeof(float), cudaMemcpyHostToDevice,stream[dev*nStream_device+1]);
                        
                        
                    }
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        cudaDeviceSynchronize();
                    }
                }
                cudaCheckErrors("Memcpy failure on multi split");
                for (dev = 0; dev < deviceCount; dev++){
                    cudaSetDevice(gpuids[dev]);
                    curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                    total_pixels=curr_slices*image_size[0]*image_size[1];
                    grad_x<<<gridGrad, blockGrad,0,stream[dev*nStream_device]>>>(d_image[dev],df1[dev],image_size[2],image_size[1],image_size[0]);
                    grad_y<<<gridGrad, blockGrad,0,stream[dev*nStream_device]>>>(d_image[dev],df2[dev],image_size[2],image_size[1],image_size[0]);
                    grad_x<<<gridGrad, blockGrad,0,stream[dev*nStream_device]>>>(d_image[dev],v11[dev],image_size[2],image_size[1],image_size[0]);
                    grad_y<<<gridGrad, blockGrad,0,stream[dev*nStream_device]>>>(d_image[dev],v22[dev],image_size[2],image_size[1],image_size[0]);
                }                
                
                for(unsigned int ib=0;  (ib<(buffer_length-1)) && ((i+ib)<maxIter);  ib++){
                    
                    // For the gradient
                    dim3 blockGrad(10, 10, 10);
                    dim3 gridGrad((image_size[0]+blockGrad.x-1)/blockGrad.x, (image_size[1]+blockGrad.y-1)/blockGrad.y, (curr_slices+buffer_length*2+blockGrad.z-1)/blockGrad.z);
                    
                    // Compute Condat dual function
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        
                        for (int proj = 0; proj < 270; proj++){}:
                            // Update gradient field v11, v12...
                            opLadj1<<<gridGrad, blockGrad,0,stream[dev*nStream_device]>>>(vadj1[dev], v11[dev],v12[dev],v13[dev],image_size[2],image_size[1],image_size[0]);
                            opLadj2<<<gridGrad, blockGrad,0,stream[dev*nStream_device]>>>(vadj2[dev], v22[dev],v21[dev],v23[dev],image_size[2],image_size[1],image_size[0]);
                            substractArrays<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(vadj1[dev], df1[dev],total_pixels);    
                            substractArrays<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(vadj2[dev], df2[dev],total_pixels);
                            copyArray<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(psi1[dev],phi1[dev],   total_pixels);
                            copyArray<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(psi2[dev],phi2[dev],   total_pixels);
                            multiplyArrayScalar<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(psi1[dev],mu2,   total_pixels);
                            multiplyArrayScalar<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(psi2[dev],mu2,   total_pixels);
                            substractArrays<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(vadj1[dev], psi1[dev],total_pixels);    
                            substractArrays<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(vadj2[dev], psi2[dev],total_pixels);                                
                            opL11<<<gridGrad, blockGrad,0,stream[dev*nStream_device]>>>(L11[dev], phi1[dev],image_size[2],image_size[1],image_size[0]);
                            opL12<<<gridGrad, blockGrad,0,stream[dev*nStream_device]>>>(L12[dev], phi1[dev],image_size[2],image_size[1],image_size[0]);
                            opL13<<<gridGrad, blockGrad,0,stream[dev*nStream_device]>>>(L13[dev], phi1[dev],image_size[2],image_size[1],image_size[0]);
                            opL21<<<gridGrad, blockGrad,0,stream[dev*nStream_device]>>>(L21[dev], phi2[dev],image_size[2],image_size[1],image_size[0]);
                            opL11<<<gridGrad, blockGrad,0,stream[dev*nStream_device]>>>(L22[dev], phi2[dev],image_size[2],image_size[1],image_size[0]);
                            opL23<<<gridGrad, blockGrad,0,stream[dev*nStream_device]>>>(L23[dev], phi2[dev],image_size[2],image_size[1],image_size[0]);
                            multiplyArrayScalar<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(L11[dev],sigma,   total_pixels);
                            multiplyArrayScalar<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(L12[dev],sigma,   total_pixels);
                            multiplyArrayScalar<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(L13[dev],sigma,   total_pixels);
                            multiplyArrayScalar<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(L21[dev],sigma,   total_pixels);
                            multiplyArrayScalar<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(L22[dev],sigma,   total_pixels);
                            multiplyArrayScalar<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(L23[dev],sigma,   total_pixels);
                            substractArrays<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(L11[dev], v11[dev],total_pixels);    
                            substractArrays<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(L12[dev], v12[dev],total_pixels);    
                            substractArrays<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(L13[dev], v13[dev],total_pixels);    
                            substractArrays<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(L21[dev], v21[dev],total_pixels);   
                            substractArrays<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(L22[dev], v22[dev],total_pixels);    
                            substractArrays<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(L23[dev], v23[dev],total_pixels);   
                            prox_g<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(v11[dev],L11[dev],total_pixels,sigma,mu2);
                            prox_g<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(v12[dev],L12[dev],total_pixels,sigma,mu2);
                            prox_g<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(v13[dev],L13[dev],total_pixels,sigma,mu2);
                            prox_g<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(v21[dev],L21[dev],total_pixels,sigma,mu2);
                            prox_g<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(v22[dev],L22[dev],total_pixels,sigma,mu2);
                            prox_g<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(v23[dev],L23[dev],total_pixels,sigma,mu2);                            
                            // Update dual function phi1, phi2
                            opLadj1<<<gridGrad, blockGrad,0,stream[dev*nStream_device]>>>(vadj1[dev], v11[dev],v12[dev],v13[dev],image_size[2],image_size[1],image_size[0]);
                            opLadj2<<<gridGrad, blockGrad,0,stream[dev*nStream_device]>>>(vadj2[dev], v22[dev],v21[dev],v23[dev],image_size[2],image_size[1],image_size[0]);
                            substractArrays<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(vadj1[dev], df1[dev],total_pixels);    
                            substractArrays<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(vadj2[dev], df2[dev],total_pixels);
                            multiplyArrayScalar<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(vadj1[dev],mu1,   total_pixels);
                            multiplyArrayScalar<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(vadj2[dev],mu1,   total_pixels);
                            substractArrays<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(phi1[dev],vadj1[dev],total_pixels);    
                            substractArrays<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(phi2[dev],vadj2[dev],total_pixels);                               
                        }
                    }
                    
                    
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        // Compute the gradient of the Condat (dual) TV
                        gradientTV<<<gridGrad, blockGrad,0,stream[dev*nStream_device]>>>(phi1[dev],phi2[dev],d_dimgTV[dev],(long)(curr_slices+buffer_length*2-1), image_size[1],image_size[0],delta);
                        
                    }
                    
                    
                    
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        // no need to copy the 2 aux slices here
                        cudaStreamSynchronize(stream[dev*nStream_device]);
                        cudaMemcpyAsync(d_norm2[dev], d_dimgTV[dev]+buffer_pixels, image_size[0]*image_size[1]*curr_slices*sizeof(float), cudaMemcpyDeviceToDevice,stream[dev*nStream_device+1]);
                    }
                    
                    
                    // Compute the L2 norm of the gradient. For that, reduction is used.
                    //REDUCE
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        
                        size_t dimblockRed = MAXTHREADS;
                        size_t dimgridRed = (total_pixels + MAXTHREADS - 1) / MAXTHREADS;
                        
                        cudaStreamSynchronize(stream[dev*nStream_device+1]);
                        reduceNorm2 << <dimgridRed, dimblockRed, MAXTHREADS*sizeof(float),stream[dev*nStream_device]>> >(d_norm2[dev], d_norm2aux[dev], total_pixels);
                        
                    }
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        size_t dimblockRed = MAXTHREADS;
                        size_t dimgridRed = (total_pixels + MAXTHREADS - 1) / MAXTHREADS;

                        if (dimgridRed > 1) {
                            reduceSum << <1, dimblockRed, MAXTHREADS*sizeof(float),stream[dev*nStream_device] >> >(d_norm2aux[dev], d_norm2[dev], dimgridRed);
                            cudaStreamSynchronize(stream[dev*nStream_device]);
                            cudaMemcpyAsync(&sumnorm2[dev], d_norm2[dev], sizeof(float), cudaMemcpyDeviceToHost,stream[dev*nStream_device+1]);
                        }
                        else {
                            cudaStreamSynchronize(stream[dev*nStream_device]);
                            cudaMemcpyAsync(&sumnorm2[dev], d_norm2aux[dev], sizeof(float), cudaMemcpyDeviceToHost,stream[dev*nStream_device+1]);
                        }
                    }
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        cudaDeviceSynchronize();
                     }
                    cudaCheckErrors("Reduction error");
                    
                    
                    // Accumulate the norm accross devices
                    sum_curr_spl=0;
                    // this is CPU code
                    for (dev = 0; dev < deviceCount; dev++){
                        sum_curr_spl+=sumnorm2[dev];
                    }
                    sum_curr_spl+=0.0000001f; // avoid division by zero
                    
                    // If we have more than one splits, lets use the result from prior calls
                    if(i>0 && splits>1){
                        // this is already stored:
                        //totalsum=totalsum_prev; 
                    }else{
                        totalsum=sum_curr_spl;
                    }
                    
                    
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        //NORMALIZE
                        //in a Tesla, maximum blocks =15 SM * 4 blocks/SM
                        divideArrayScalar  <<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(d_dimgTV[dev]+buffer_pixels,(float)sqrt(totalsum),total_pixels);
                        //MULTIPLY HYPERPARAMETER
                        multiplyArrayScalar<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(d_dimgTV[dev]+buffer_pixels,alpha,   total_pixels);
                    }
                     for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        cudaDeviceSynchronize();
                     }
                    cudaCheckErrors("Scalar operations error");
                    
                    //SUBSTRACT GRADIENT
                    //////////////////////////////////////////////
                    for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        
                        substractArrays<<<60,MAXTHREADS,0,stream[dev*nStream_device]>>>(d_image[dev]+buffer_pixels,d_dimgTV[dev]+buffer_pixels, total_pixels);
                    }
                }

                // Synchronize mathematics, make sure bounding pixels are correct
                 for (dev = 0; dev < deviceCount; dev++){
                        cudaSetDevice(gpuids[dev]);
                        cudaDeviceSynchronize();
                     }
                
                if(splits==1){
                    for(dev=0; dev<deviceCount;dev++){
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        if (dev<deviceCount-1){
                            cudaSetDevice(gpuids[dev+1]);
                            cudaMemcpy(buffer, d_image[dev+1], buffer_pixels*sizeof(float), cudaMemcpyDeviceToHost);
                            cudaSetDevice(gpuids[dev]);
                            cudaMemcpy(d_image[dev]+total_pixels+buffer_pixels,buffer, buffer_pixels*sizeof(float), cudaMemcpyHostToDevice); 
                        }
                        cudaDeviceSynchronize();
                        if (dev>0){
                            cudaSetDevice(gpuids[dev-1]);
                            cudaMemcpyAsync(buffer, d_image[dev-1]+total_pixels+buffer_pixels, buffer_pixels*sizeof(float), cudaMemcpyDeviceToHost);
                            cudaSetDevice(gpuids[dev]);
                            cudaMemcpyAsync(d_image[dev],buffer, buffer_pixels*sizeof(float), cudaMemcpyHostToDevice);
                        }
                    }
                }else{
                    
                    // We need to take it out :(
                    for(dev=0; dev<deviceCount;dev++){
                        cudaSetDevice(gpuids[dev]);
                        
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        linear_idx_start=image_size[0]*image_size[1]*slices_per_split*(sp*deviceCount+dev);
                        total_pixels=curr_slices*image_size[0]*image_size[1];
                        cudaMemcpyAsync(&dst[linear_idx_start], d_image[dev]+buffer_pixels,total_pixels*sizeof(float), cudaMemcpyDeviceToHost,stream[dev*nStream_device+1]);
                    }
                }
                
                for (dev = 0; dev < deviceCount; dev++){
                    cudaSetDevice(gpuids[dev]);
                    cudaDeviceSynchronize();
                }
                cudaCheckErrors("Memory gather error");

                totalsum_prev+=sum_curr_spl;
            }
            totalsum=totalsum_prev;
        }
        // If there has not been splits, we still have data in memory
        if(splits==1){
            for(dev=0; dev<deviceCount;dev++){
                cudaSetDevice(gpuids[dev]);
                
                curr_slices=((dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*dev;
                total_pixels=curr_slices*image_size[0]*image_size[1];
                cudaMemcpy(dst+slices_per_split*image_size[0]*image_size[1]*dev, d_image[dev]+buffer_pixels,total_pixels*sizeof(float), cudaMemcpyDeviceToHost);
            }
        }
        cudaCheckErrors("Copy result back");
        
        for(dev=0; dev<deviceCount;dev++){
            cudaSetDevice(gpuids[dev]);
            cudaFree(d_image[dev]);
            cudaFree(d_norm2aux[dev]);
            cudaFree(d_dimgTV[dev]);
            cudaFree(d_norm2[dev]);
        }
        if (splits==1){
            cudaFreeHost(buffer);
        }
        
        if (isHostRegisterSupported& splits>2){
            cudaHostUnregister(img);
            cudaHostUnregister(dst);
        }
        for (int i = 0; i < nStreams; ++i)
           cudaStreamDestroy(stream[i]) ;
        cudaCheckErrors("Memory free");
//         cudaDeviceReset();
    }
        
void checkFreeMemory(const GpuIds& gpuids, size_t *mem_GPU_global){
        size_t memfree;
        size_t memtotal;
        const int deviceCount = gpuids.GetLength();
        for (int dev = 0; dev < deviceCount; dev++){
            cudaSetDevice(gpuids[dev]);
            cudaMemGetInfo(&memfree,&memtotal);
            if(dev==0) *mem_GPU_global=memfree;
            if(memfree<memtotal/2){
                mexErrMsgIdAndTxt("tvDenoise:tvdenoising:GPU","One (or more) of your GPUs is being heavily used by another program (possibly graphics-based).\n Free the GPU to run TIGRE\n");
            }
            cudaCheckErrors("Check mem error");
            
            *mem_GPU_global=(memfree<*mem_GPU_global)?memfree:*mem_GPU_global;
        }
        *mem_GPU_global=(size_t)((double)*mem_GPU_global*0.95);
        
        //*mem_GPU_global= insert your known number here, in bytes.
}