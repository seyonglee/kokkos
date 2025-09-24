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

#include <Kokkos_Core.hpp>
#include <cstddef>

namespace {
struct Foo {
  KOKKOS_FUNCTION Foo& operator=(double val_) {
    val = val_;
    return *this;
  }
  float val;
  KOKKOS_FUNCTION bool operator==(double val_) {
    return val_ == static_cast<double>(val);
  }
};

template <class D1, class D2, class... Extents>
void test_deep_copy_assignable_types(Extents... exts) {
  Kokkos::View<D1, TEST_EXECSPACE> a("A", exts...);
  Kokkos::deep_copy(a, 1.5);

  Kokkos::View<D2, TEST_EXECSPACE> b("B", exts...);
  Kokkos::deep_copy(b, a);

  auto h_b = Kokkos::create_mirror_view(b);
  Kokkos::deep_copy(h_b, b);
  ASSERT_TRUE(h_b((exts - 1)...) == 1.5);

  // Check that
  Kokkos::deep_copy(TEST_EXECSPACE(), a, 2.5);
  Kokkos::deep_copy(TEST_EXECSPACE(), b, a);
  Kokkos::deep_copy(TEST_EXECSPACE(), h_b, b);
  TEST_EXECSPACE().fence();

  ASSERT_TRUE(h_b((exts - 1)...) == 2.5);

#ifdef KOKKOS_HAS_SHARED_SPACE
  Kokkos::View<D1, Kokkos::SharedSpace> s("S", exts...);
  Kokkos::deep_copy(s, 1.5);

  Kokkos::deep_copy(b, s);
  Kokkos::deep_copy(h_b, b);
  ASSERT_TRUE(h_b((exts - 1)...) == 1.5);
#endif

#ifdef KOKKOS_HAS_SHARED_HOST_PINNED_SPACE
  Kokkos::View<D1, Kokkos::SharedHostPinnedSpace> sp("S", exts...);
  Kokkos::deep_copy(sp, 2.5);

  Kokkos::deep_copy(b, sp);
  Kokkos::deep_copy(h_b, b);
  ASSERT_TRUE(h_b((exts - 1)...) == 2.5);
#endif
}
}  // namespace
