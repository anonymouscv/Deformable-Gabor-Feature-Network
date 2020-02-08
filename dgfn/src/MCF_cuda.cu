#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#define CUDA_1D_KERNEL_LOOP(i,n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i<n; \
            i += blockDim.x * gridDim.x)

template <typename T>
__global__ void MCFForward_cuda_kernel(const int nthreads, const T* weight_data, const T* mFilter_data,
        const int nOutputPlane, const int nInputPlane, const int nChannel, const int kH, const int kW, T* output_data){
    CUDA_1D_KERNEL_LOOP(index, nthreads){
    auto w = index % kW;
    auto h = (index / kW) % kH;
    auto c = (index / kW / kH) % nChannel;
    auto in = (index / kW / kH / nChannel) % nInputPlane;
    auto ori = (index / kW / kH / nChannel / nInputPlane) % nChannel;
    auto ou = index / kW / kH / nChannel / nInputPlane / nChannel;
    T val = *(weight_data + (((ou * nInputPlane + in) * nChannel + c) * kH + h) * kW + w);
    T *target = output_data + index;
    T mtmp = *(mFilter_data + ori * (kH  * kW) + h * kW + w);
    * target = val * mtmp;
    }
}

template <typename T>
__global__ void MCFBackward_cuda_kernel(const int nthreads, const T* grad_output_data, const T* mFilter_data,
        const int nOutputPlane, const int nInputPlane, const int nChannel, const int kH, const int kW, T* grad_weight_data){
    auto nEntry = nChannel * kH *kW;
    CUDA_1D_KERNEL_LOOP(index, nthreads){
    auto l = index % nEntry;
    auto j = (index / nEntry) % nInputPlane;
    auto i = index / nEntry / nInputPlane;
    T *val = grad_weight_data + index;
    * val = 0;
    for (int k = 0; k < nChannel; k++){
        T mtmp = *(mFilter_data + k * (kW * kH) + l % (kW * kH));
        T target = *(grad_output_data + i * (nChannel * nInputPlane * nEntry) + 
                k * (nInputPlane * nEntry) + j * nEntry + l);
        *val = *val + target * mtmp;
    }
    }
}

at::Tensor MCF_forward_cuda(const at::Tensor& weight, const at::Tensor& mFilter)
{
    AT_ASSERTM(weight.type().is_cuda(), "weight must be a cuda tensor");
    AT_ASSERTM(mFilter.type().is_cuda(), "mFilter must be a cuda tensor");

    auto nOutputPlane = weight.size(0);
    auto nInputPlane = weight.size(1);
    auto nChannel = weight.size(2);
    auto kH = weight.size(3);
    auto kW = weight.size(4);

    auto output = at::empty ({nOutputPlane * nChannel, nInputPlane * nChannel, kH, kW}, weight.options());
    auto output_size =  nOutputPlane * nChannel * nInputPlane * nChannel * kH * kW;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(std::min(THCCeilDiv(output_size, 512L), 4096L));
    dim3 block(512);

    if (output.numel() == 0){
        THCudaCheck(cudaGetLastError());
        return output;
    }

    AT_DISPATCH_FLOATING_TYPES(weight.type(), "MCF_forward", [&]{
            MCFForward_cuda_kernel<scalar_t><<<grid, block, 0, stream>>>(
                    output_size, weight.data<scalar_t>(), mFilter.data<scalar_t>(), nOutputPlane,
                    nInputPlane, nChannel, kH, kW, output.data<scalar_t>());
            });
    THCudaCheck(cudaGetLastError());
    return  output;
}

at::Tensor MCF_backward_cuda(const at::Tensor& grad_output, const at::Tensor& mFilter)
{
    AT_ASSERTM(grad_output.type().is_cuda(), "grad_output must be a cuda tensor");
    AT_ASSERTM(mFilter.type().is_cuda(), "mFilter must be a cuda tensor");

    auto nChannel = mFilter.size(0);
    auto nOutputPlane = grad_output.size(0) / nChannel;
    auto nInputPlane = grad_output.size(1) / nChannel;
    auto kH = grad_output.size(2);
    auto kW = grad_output.size(3);

    auto grad_weight = at::empty({nOutputPlane, nInputPlane, nChannel, kH, kW}, grad_output.options());
    auto nEntry = nChannel * kH * kW;
    auto grad_weight_size = nOutputPlane * nInputPlane * nEntry;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(std::min(THCCeilDiv(grad_weight_size, 512L), 4096L));
    dim3 block(512);

    if (grad_weight.numel() == 0){
        THCudaCheck(cudaGetLastError());
        return grad_weight;
    }

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "MCF_backward", [&]{
            MCFBackward_cuda_kernel<scalar_t><<<grid, block, 0, stream>>>(
                    grad_weight_size, grad_output.data<scalar_t>(), mFilter.data<scalar_t>(), nOutputPlane,
                    nInputPlane, nChannel, kH, kW, grad_weight.data<scalar_t>());
            });
    THCudaCheck(cudaGetLastError());
    return grad_weight;
}
