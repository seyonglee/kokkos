/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#include <cinttypes>
#include <desul/atomics/Lock_Array.hpp>
#include <sstream>
#include <string>

namespace desul {
namespace Impl {
int32_t* OPENACC_SPACE_ATOMIC_LOCKS_DEVICE = nullptr;
int32_t* OPENACC_SPACE_ATOMIC_LOCKS_NODE = nullptr;
}  // namespace Impl
}  // namespace desul

namespace desul {

namespace {

void init_lock_arrays_openacc_kernel() {
  const unsigned numItr = (OPENACC_SPACE_ATOMIC_MASK + 1 + 255) / 256;
  #pragma acc parallel loop
  for(unsigned i = 0; i< numItr; i++) {
    if (i < OPENACC_SPACE_ATOMIC_MASK + 1) {
      Impl::OPENACC_SPACE_ATOMIC_LOCKS_DEVICE[i] = 0;
      Impl::OPENACC_SPACE_ATOMIC_LOCKS_NODE[i] = 0;
    }
  }
}

}  // namespace

namespace Impl {

int32_t* OPENACC_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
int32_t* OPENACC_SPACE_ATOMIC_LOCKS_NODE_h = nullptr;

// define functions
template <typename T>
void init_lock_arrays_openacc() {
  if (OPENACC_SPACE_ATOMIC_LOCKS_DEVICE_h != nullptr) return;
  CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h = acc_malloc(sizeof(int32_t) * (CUDA_SPACE_ATOMIC_MASK + 1));
  CUDA_SPACE_ATOMIC_LOCKS_NODE_h = acc_malloc(sizeof(int32_t) * (CUDA_SPACE_ATOMIC_MASK + 1));
  acc_wait_all();
  copy_openacc_lock_arrays_to_device();
  init_lock_arrays_openacc_kernel();
  acc_wait_all();
}

template <typename T>
void finalize_lock_arrays_openacc() {
  if (OPENACC_SPACE_ATOMIC_LOCKS_DEVICE_h == nullptr) return;
  acc_free(OPENACC_SPACE_ATOMIC_LOCKS_DEVICE_h);
  acc_free(OPENACC_SPACE_ATOMIC_LOCKS_NODE_h);
  OPENACC_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
  OPENACC_SPACE_ATOMIC_LOCKS_NODE_h = nullptr;
  copy_openacc_lock_arrays_to_device();
}

// Instantiate functions
template void init_lock_arrays_openacc<int>();
template void finalize_lock_arrays_openacc<int>();

}  // namespace Impl

}  // namespace desul
