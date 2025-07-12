/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_LOCK_ARRAY_OPENACC_HPP_
#define DESUL_ATOMICS_LOCK_ARRAY_OPENACC_HPP_

#include <cstdint>

#include "desul/atomics/Common.hpp"
#include "desul/atomics/Macros.hpp"

namespace desul {
namespace Impl {

/// \brief This global variable in Host space is the central definition
///        of these arrays.
extern int32_t* OPENACC_SPACE_ATOMIC_LOCKS_DEVICE_h;
extern int32_t* OPENACC_SPACE_ATOMIC_LOCKS_NODE_h;

template <typename /*AlwaysInt*/ = int>
void init_lock_arrays_openacc();

template <typename /*AlwaysInt*/ = int>
void finalize_lock_arrays_openacc();

extern int32_t* OPENACC_SPACE_ATOMIC_LOCKS_DEVICE;

extern int32_t* OPENACC_SPACE_ATOMIC_LOCKS_NODE;

#define OPENACC_SPACE_ATOMIC_MASK 0x1FFFF

/// \brief Acquire a lock for the address
///
/// This function tries to acquire the lock for the hash value derived
/// from the provided ptr. If the lock is successfully acquired the
/// function returns true. Otherwise it returns false.
#pragma acc routine seq
inline bool lock_address_openacc(void* ptr, desul::MemoryScopeDevice) {
  size_t offset = size_t(ptr);
  offset = offset >> 2;
  offset = offset & OPENACC_SPACE_ATOMIC_MASK;
  int32_t return_val;
  int32_t *dest = &desul::Impl::OPENACC_SPACE_ATOMIC_LOCKS_DEVICE[offset];
#pragma acc atomic capture
  {   
    return_val = *dest;
    *dest = 1;
  }   
  return (0 == return_val);
}
#pragma acc routine seq
inline bool lock_address_openacc(void* ptr, desul::MemoryScopeNode) {
  size_t offset = size_t(ptr);
  offset = offset >> 2;
  offset = offset & OPENACC_SPACE_ATOMIC_MASK;
  int32_t return_val;
  int32_t *dest = &desul::Impl::OPENACC_SPACE_ATOMIC_LOCKS_NODE[offset];
#pragma acc atomic capture
  {   
    return_val = *dest;
    *dest = 1;
  }   
  return (0 == return_val);
}

/// \brief Release lock for the address
///
/// This function releases the lock for the hash value derived
/// from the provided ptr. This function should only be called
/// after previously successfully acquiring a lock with
/// lock_address.
#pragma acc routine seq
void unlock_address_openacc(void* ptr, desul::MemoryScopeDevice) {
  size_t offset = size_t(ptr);
  offset = offset >> 2;
  offset = offset & OPENACC_SPACE_ATOMIC_MASK;
  atomicExch(&desul::Impl::OPENACC_SPACE_ATOMIC_LOCKS_DEVICE[offset], 0);
}
#pragma acc routine seq
inline void unlock_address_openacc(void* ptr, desul::MemoryScopeNode) {
  size_t offset = size_t(ptr);
  offset = offset >> 2;
  offset = offset & OPENACC_SPACE_ATOMIC_MASK;
  atomicExch(&desul::Impl::OPENACC_SPACE_ATOMIC_LOCKS_NODE[offset], 0);
}

inline void
    copy_openacc_lock_arrays_to_device() {
  static bool once = []() {
    cudaMemcpyToSymbol(OPENACC_SPACE_ATOMIC_LOCKS_DEVICE,
                       &OPENACC_SPACE_ATOMIC_LOCKS_DEVICE_h,
                       sizeof(int32_t*));
    cudaMemcpyToSymbol(OPENACC_SPACE_ATOMIC_LOCKS_NODE,
                       &OPENACC_SPACE_ATOMIC_LOCKS_NODE_h,
                       sizeof(int32_t*));
    return true;
  }();
  (void)once;
}

}  // namespace Impl
}  // namespace desul

#endif /* #ifndef DESUL_ATOMICS_LOCK_ARRAY_OPENACC_HPP_ */
