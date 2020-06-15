// Autogenerated file by gen-cpu-ops.py. Do not edit directly!
#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {
namespace cpu {

class AtenIpexCPUSparse {
 public:

  // All Sparse Ops, except for Sparse Attribute Ops
  static at::Tensor add(const at::Tensor & self, const at::Tensor & other, at::Scalar alpha);
  static at::Tensor & add_(at::Tensor & self, const at::Tensor & other, at::Scalar alpha);
  static at::Tensor & add_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other, at::Scalar alpha);
  static at::Tensor div(const at::Tensor & self, const at::Tensor & other);
  static at::Tensor & div_(at::Tensor & self, const at::Tensor & other);
  static at::Tensor & div_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other);
  static at::Tensor empty(at::IntArrayRef size, const at::TensorOptions & options, c10::optional<at::MemoryFormat> memory_format);
  static at::Tensor & log1p_(at::Tensor & self);
  static at::Tensor & log1p_out(at::Tensor & out, const at::Tensor & self);
  static at::Tensor mm(const at::Tensor & self, const at::Tensor & mat2);
  static at::Tensor & mm_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & mat2);
  static at::Tensor mul(const at::Tensor & self, const at::Tensor & other);
  static at::Tensor & mul_(at::Tensor & self, const at::Tensor & other);
  static at::Tensor & mul_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other);
  static at::Tensor narrow_copy(const at::Tensor & self, int64_t dim, int64_t start, int64_t length);
  static at::Tensor & sspaddmm_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, at::Scalar beta, at::Scalar alpha);
  static at::Tensor native_norm(const at::Tensor & self, at::Scalar p);
  static at::Tensor _sparse_sum_backward(const at::Tensor & grad, const at::Tensor & self, at::IntArrayRef dim);
  static at::Tensor clone(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format);
  static at::Tensor & pow_out(at::Tensor & out, const at::Tensor & self, at::Scalar exponent);
  static at::Tensor pow(const at::Tensor & self, at::Scalar exponent);
  static at::Tensor & zero_(at::Tensor & self);
  static at::Tensor & sub_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other, at::Scalar alpha);
  static at::Tensor sub(const at::Tensor & self, const at::Tensor & other, at::Scalar alpha);
  static at::Tensor & sub_(at::Tensor & self, const at::Tensor & other, at::Scalar alpha);
  static at::Tensor & addmm_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, at::Scalar beta, at::Scalar alpha);
  static at::Tensor addmm(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, at::Scalar beta, at::Scalar alpha);
  static at::Tensor & addmm_(at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, at::Scalar beta, at::Scalar alpha);
  static at::Tensor _sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, const at::TensorOptions & options);
  static at::Tensor _sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, const at::Tensor & indices, const at::Tensor & values, const at::TensorOptions & options);
  static at::Tensor & sparse_resize_(at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim);
  static at::Tensor & sparse_resize_and_clear_(at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim);
  static at::Tensor sparse_mask(const at::Tensor & self, const at::Tensor & mask);
  static at::Tensor to_dense(const at::Tensor & self);
  static at::Tensor coalesce(const at::Tensor & self);
  static at::Tensor & hspmm_out(at::Tensor & out, const at::Tensor & mat1, const at::Tensor & mat2);
  static at::Tensor hspmm(const at::Tensor & mat1, const at::Tensor & mat2);
  static at::Tensor & copy_sparse_to_sparse_(at::Tensor & self, const at::Tensor & src, bool non_blocking);
  static at::Tensor index_select(const at::Tensor & self, int64_t dim, const at::Tensor & index);

};

}  // namespace cpu
}  // namespace torch_ipex

