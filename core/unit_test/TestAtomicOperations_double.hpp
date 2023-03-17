//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <TestAtomicOperations.hpp>

namespace Test {
TEST(TEST_CATEGORY, atomic_operations_double) {
  const int start = 1;  // Avoid zero for division.
  const int end   = 11;
#ifdef KOKKOS_COMPILER_NVHPC
    // FIXME_NVHPC: Old NVHPC compilers (V22.5 or older) fail to compile
	GTEST_SKIP() << "FIXME_NVHPC: Old NVHPC compilers (V22.5 or older) fail to compile atomic_operations_double";
#else
  for (int i = start; i < end; ++i) {
#ifndef KOKKOS_ENABLE_OPENACC
    // FIXME_OPENACC: OpenACC C/C++ does not support atomic min/max operations
    ASSERT_TRUE((TestAtomicOperations::AtomicOperationsTestNonIntegralType<
                 double, TEST_EXECSPACE>(start, end - i, 1)));
    ASSERT_TRUE((TestAtomicOperations::AtomicOperationsTestNonIntegralType<
                 double, TEST_EXECSPACE>(start, end - i, 2)));
#endif
    ASSERT_TRUE((TestAtomicOperations::AtomicOperationsTestNonIntegralType<
                 double, TEST_EXECSPACE>(start, end - i, 3)));
    ASSERT_TRUE((TestAtomicOperations::AtomicOperationsTestNonIntegralType<
                 double, TEST_EXECSPACE>(start, end - i, 4)));
    ASSERT_TRUE((TestAtomicOperations::AtomicOperationsTestNonIntegralType<
                 double, TEST_EXECSPACE>(start, end - i, 5)));
  }
#endif
}
}  // namespace Test
