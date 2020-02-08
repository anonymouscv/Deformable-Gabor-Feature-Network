#pragma once
#include <torch/extension.h>


at::Tensor GOF_forward_cuda(const at::Tensor& weight, 
                            const at::Tensor& gaborFilterBank);

at::Tensor GOF_backward_cuda(const at::Tensor& grad_output,
                             const at::Tensor& gaborFilterBank);

at::Tensor MCF_forward_cuda(const at::Tensor& weight,
                       const at::Tensor& MFilter);
at::Tensor MCF_backward_cuda(const at::Tensor& grad_output,
                        const at::Tensor& MFilter);

int deform_conv_forward_cuda(at::Tensor input, at::Tensor weight,
                             at::Tensor offset, at::Tensor output,
                             at::Tensor columns, at::Tensor ones, int kW,
                             int kH, int dW, int dH, int padW, int padH,
                             int dilationW, int dilationH, int group,
                             int deformable_group, int im2col_step);

int deform_conv_backward_input_cuda(at::Tensor input, at::Tensor offset,
                                    at::Tensor gradOutput, at::Tensor gradInput,
                                    at::Tensor gradOffset, at::Tensor weight,
                                    at::Tensor columns, int kW, int kH, int dW,
                                    int dH, int padW, int padH, int dilationW,
                                    int dilationH, int group,
                                    int deformable_group, int im2col_step) ;
int deform_conv_backward_parameters_cuda(
    at::Tensor input, at::Tensor offset, at::Tensor gradOutput,
    at::Tensor gradWeight,  // at::Tensor gradBias,
    at::Tensor columns, at::Tensor ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationW, int dilationH, int group,
    int deformable_group, float scale, int im2col_step);

