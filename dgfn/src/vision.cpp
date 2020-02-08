#include "vision.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gof_forward", &GOF_forward_cuda, "GOF forward");
  m.def("gof_backward", &GOF_backward_cuda, "GOF backward");
  m.def("mcf_forward", &MCF_forward_cuda, "MCF forward");
  m.def("mcf_backward", &MCF_backward_cuda, "MCF backward");
  m.def("deform_conv_forward_cuda", &deform_conv_forward_cuda, "deform_conv_forward(cuda)");
  m.def("deform_conv_backward_input_cuda", & deform_conv_backward_input_cuda, "deform_conv_backward_input(cuda)");
  m.def("deform_conv_backward_parameters_cuda", & deform_conv_backward_parameters_cuda, "deform_conv_backward_parameters(cuda)");
}
