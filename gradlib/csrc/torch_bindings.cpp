#include <Python.h>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

// A version of the TORCH_LIBRARY macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

// A version of the TORCH_LIBRARY_IMPL macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
#define TORCH_LIBRARY_IMPL_EXPAND(NAME, DEVICE, MODULE) \
  TORCH_LIBRARY_IMPL(NAME, DEVICE, MODULE)

// REGISTER_EXTENSION allows the shared library to be loaded and initialized
// via python's import statement.
#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }

#include <torch/torch.h>

#include <torch/library.h>

void hipb_create_extension();
void hipb_destroy_extension();
torch::Tensor hipb_mm(const torch::Tensor& mat1, const torch::Tensor& mat2,
                      const int64_t solution_index,
                      at::optional<torch::Tensor> bias = at::nullopt,
                      at::optional<c10::ScalarType> out_dtype = at::nullopt,
                      at::optional<torch::Tensor> scale1 = at::nullopt,
                      at::optional<torch::Tensor> scale2 = at::nullopt,
                      at::optional<torch::Tensor> scaleOut = at::nullopt);

std::vector<int64_t> hipb_findallsols(const torch::Tensor& mat1,
                                      const torch::Tensor& mat2,
                                      at::optional<torch::Tensor> bias,
                                      at::optional<c10::ScalarType> out_dtype);

void rocb_create_extension();
void rocb_destroy_extension();
torch::Tensor RocSolIdxBlas(const torch::Tensor& mat1,
                            const torch::Tensor& mat2,
                            const int64_t solution_index);

std::vector<int64_t> RocFindAllSolIdxBlas(const torch::Tensor& mat1,
                                          const torch::Tensor& mat2);

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, gradlib_ops) {
  // Gradlib custom ops

  gradlib_ops.def("hipb_create_extension", &hipb_create_extension);
  gradlib_ops.def("hipb_destroy_extension", &hipb_destroy_extension);
  gradlib_ops.def("hipb_mm", &hipb_mm);
  gradlib_ops.def("hipb_findallsols", &hipb_findallsols);

  gradlib_ops.def("rocb_create_extension", &rocb_create_extension);
  gradlib_ops.def("rocb_destroy_extension", &rocb_destroy_extension);
  gradlib_ops.def("rocb_mm", &RocSolIdxBlas);
  gradlib_ops.def("rocb_findallsols", &RocFindAllSolIdxBlas);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
