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

#ifndef KOKKOS_OFFSETVIEW_HPP_
#define KOKKOS_OFFSETVIEW_HPP_
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_OFFSETVIEW
#endif

#include <Kokkos_Core.hpp>

#include <Kokkos_View.hpp>

namespace Kokkos {

namespace Experimental {
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <class DataType, class... Properties>
class OffsetView;

template <class>
struct is_offset_view : public std::false_type {};

template <class D, class... P>
struct is_offset_view<OffsetView<D, P...>> : public std::true_type {};

template <class D, class... P>
struct is_offset_view<const OffsetView<D, P...>> : public std::true_type {};

template <class T>
inline constexpr bool is_offset_view_v = is_offset_view<T>::value;

#define KOKKOS_INVALID_OFFSET int64_t(0x7FFFFFFFFFFFFFFFLL)
#define KOKKOS_INVALID_INDEX_RANGE \
  { KOKKOS_INVALID_OFFSET, KOKKOS_INVALID_OFFSET }

template <typename iType,
          std::enable_if_t<std::is_integral_v<iType> && std::is_signed_v<iType>,
                           iType> = 0>
using IndexRange = Kokkos::Array<iType, 2>;

using index_list_type = std::initializer_list<int64_t>;

//  template <typename iType,
//    std::enable_if_t< std::is_integral<iType>::value &&
//      std::is_signed<iType>::value, iType > = 0> using min_index_type =
//      std::initializer_list<iType>;

namespace Impl {

template <class ViewType>
struct GetOffsetViewTypeFromViewType {
  using type =
      OffsetView<typename ViewType::data_type, typename ViewType::array_layout,
                 typename ViewType::device_type,
                 typename ViewType::memory_traits>;
};

template <unsigned, class MapType, class BeginsType>
KOKKOS_INLINE_FUNCTION bool offsetview_verify_operator_bounds(
    const MapType&, const BeginsType&) {
  return true;
}

template <unsigned R, class MapType, class BeginsType, class iType,
          class... Args>
KOKKOS_INLINE_FUNCTION bool offsetview_verify_operator_bounds(
    const MapType& map, const BeginsType& begins, const iType& i,
    Args... args) {
  const bool legalIndex =
      (int64_t(i) >= begins[R]) &&
      (int64_t(i) <= int64_t(begins[R] + map.extent(R) - 1));
  return legalIndex &&
         offsetview_verify_operator_bounds<R + 1>(map, begins, args...);
}
template <unsigned, class MapType, class BeginsType>
inline void offsetview_error_operator_bounds(char*, int, const MapType&,
                                             const BeginsType&) {}

template <unsigned R, class MapType, class BeginsType, class iType,
          class... Args>
inline void offsetview_error_operator_bounds(char* buf, int len,
                                             const MapType& map,
                                             const BeginsType begins,
                                             const iType& i, Args... args) {
  const int64_t b = begins[R];
  const int64_t e = b + map.extent(R) - 1;
  const int n =
      snprintf(buf, len, " %ld <= %ld <= %ld %c", static_cast<unsigned long>(b),
               static_cast<unsigned long>(i), static_cast<unsigned long>(e),
               (sizeof...(Args) ? ',' : ')'));
  offsetview_error_operator_bounds<R + 1>(buf + n, len - n, map, begins,
                                          args...);
}

template <class MemorySpace, class MapType, class BeginsType, class... Args>
KOKKOS_INLINE_FUNCTION void offsetview_verify_operator_bounds(
    Kokkos::Impl::SharedAllocationTracker const& tracker, const MapType& map,
    const BeginsType& begins, Args... args) {
  if (!offsetview_verify_operator_bounds<0>(map, begins, args...)) {
    KOKKOS_IF_ON_HOST(
        (enum {LEN = 1024}; char buffer[LEN];
         const std::string label = tracker.template get_label<MemorySpace>();
         int n                   = snprintf(buffer, LEN,
                                            "OffsetView bounds error of view labeled %s (",
                                            label.c_str());
         offsetview_error_operator_bounds<0>(buffer + n, LEN - n, map, begins,
                                             args...);
         Kokkos::abort(buffer);))

    KOKKOS_IF_ON_DEVICE(
        (Kokkos::abort("OffsetView bounds error"); (void)tracker;))
  }
}

inline void runtime_check_rank_host(const size_t rank_dynamic,
                                    const size_t rank,
                                    const index_list_type minIndices,
                                    const std::string& label) {
  bool isBad = false;
  std::string message =
      "Kokkos::Experimental::OffsetView ERROR: for OffsetView labeled '" +
      label + "':";
  if (rank_dynamic != rank) {
    message +=
        "The full rank must be the same as the dynamic rank. full rank = ";
    message += std::to_string(rank) +
               " dynamic rank = " + std::to_string(rank_dynamic) + "\n";
    isBad = true;
  }

  size_t numOffsets = 0;
  for (size_t i = 0; i < minIndices.size(); ++i) {
    if (minIndices.begin()[i] != KOKKOS_INVALID_OFFSET) numOffsets++;
  }
  if (numOffsets != rank_dynamic) {
    message += "The number of offsets provided ( " +
               std::to_string(numOffsets) +
               " ) must equal the dynamic rank ( " +
               std::to_string(rank_dynamic) + " ).";
    isBad = true;
  }

  if (isBad) Kokkos::abort(message.c_str());
}

KOKKOS_INLINE_FUNCTION
void runtime_check_rank_device(const size_t rank_dynamic, const size_t rank,
                               const index_list_type minIndices) {
  if (rank_dynamic != rank) {
    Kokkos::abort(
        "The full rank of an OffsetView must be the same as the dynamic rank.");
  }
  size_t numOffsets = 0;
  for (size_t i = 0; i < minIndices.size(); ++i) {
    if (minIndices.begin()[i] != KOKKOS_INVALID_OFFSET) numOffsets++;
  }
  if (numOffsets != rank) {
    Kokkos::abort(
        "The number of offsets provided to an OffsetView constructor must "
        "equal the dynamic rank.");
  }
}
}  // namespace Impl

template <class DataType, class... Properties>
class OffsetView : public View<DataType, Properties...> {
 private:
  template <class, class...>
  friend class OffsetView;

  using base_t = View<DataType, Properties...>;

 public:
  // typedefs to reduce typing base_t:: further down
  using traits = typename base_t::traits;
  // FIXME: should be base_t::index_type after refactor
  using index_type   = typename base_t::memory_space::size_type;
  using pointer_type = typename base_t::pointer_type;

  using begins_type = Kokkos::Array<int64_t, base_t::rank()>;

  template <typename iType,
            std::enable_if_t<std::is_integral_v<iType>, iType> = 0>
  KOKKOS_FUNCTION int64_t begin(const iType local_dimension) const {
    return static_cast<size_t>(local_dimension) < base_t::rank()
               ? m_begins[local_dimension]
               : KOKKOS_INVALID_OFFSET;
  }

  KOKKOS_FUNCTION
  begins_type begins() const { return m_begins; }

  template <typename iType,
            std::enable_if_t<std::is_integral_v<iType>, iType> = 0>
  KOKKOS_FUNCTION int64_t end(const iType local_dimension) const {
    return begin(local_dimension) + base_t::extent(local_dimension);
  }

 private:
  begins_type m_begins;

 public:
  //----------------------------------------
  /** \brief  Compatible view of array of scalar types */
  using array_type =
      OffsetView<typename traits::scalar_array_type,
                 typename traits::array_layout, typename traits::device_type,
                 typename traits::memory_traits>;

  /** \brief  Compatible view of const data type */
  using const_type =
      OffsetView<typename traits::const_data_type,
                 typename traits::array_layout, typename traits::device_type,
                 typename traits::memory_traits>;

  /** \brief  Compatible view of non-const data type */
  using non_const_type =
      OffsetView<typename traits::non_const_data_type,
                 typename traits::array_layout, typename traits::device_type,
                 typename traits::memory_traits>;

  /** \brief  Compatible HostMirror view */
  using HostMirror = OffsetView<typename traits::non_const_data_type,
                                typename traits::array_layout,
                                typename traits::host_mirror_space>;

  template <size_t... I, class... OtherIndexTypes>
  KOKKOS_FUNCTION typename base_t::reference_type offset_operator(
      std::integer_sequence<size_t, I...>, OtherIndexTypes... indices) const {
    return base_t::operator()((indices - m_begins[I])...);
  }

  template <class OtherIndexType>
#ifndef KOKKOS_ENABLE_CXX17
    requires(std::is_convertible_v<OtherIndexType, index_type> &&
             std::is_nothrow_constructible_v<index_type, OtherIndexType> &&
             (base_t::rank() == 1))
#endif
  KOKKOS_FUNCTION constexpr typename base_t::reference_type operator[](
      const OtherIndexType& idx) const {
#ifdef KOKKOS_ENABLE_CXX17
    static_assert(std::is_convertible_v<OtherIndexType, index_type> &&
                  std::is_nothrow_constructible_v<index_type, OtherIndexType> &&
                  (base_t::rank() == 1));
#endif
    return base_t::operator[](idx - m_begins[0]);
  }

  template <class... OtherIndexTypes>
#ifndef KOKKOS_ENABLE_CXX17
    requires((std::is_convertible_v<OtherIndexTypes, index_type> && ...) &&
             (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> &&
              ...) &&
             (sizeof...(OtherIndexTypes) == base_t::rank()))
#endif
  KOKKOS_FUNCTION constexpr typename base_t::reference_type operator()(
      OtherIndexTypes... indices) const {
#ifdef KOKKOS_ENABLE_CXX17
    static_assert(
        (std::is_convertible_v<OtherIndexTypes, index_type> && ...) &&
        (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> && ...) &&
        (sizeof...(OtherIndexTypes) == base_t::rank()));
#endif
    return offset_operator(std::make_index_sequence<base_t::rank()>(),
                           indices...);
  }

  template <class... OtherIndexTypes>
  KOKKOS_FUNCTION constexpr typename base_t::reference_type access(
      OtherIndexTypes... args) const = delete;

  //----------------------------------------

  //----------------------------------------
  // Standard destructor, constructors, and assignment operators

  KOKKOS_FUNCTION
  OffsetView() : base_t() {
    for (size_t i = 0; i < base_t::rank(); ++i)
      m_begins[i] = KOKKOS_INVALID_OFFSET;
  }

  // interoperability with View
 private:
  using view_type =
      View<typename traits::scalar_array_type, typename traits::array_layout,
           typename traits::device_type, typename traits::memory_traits>;

 public:
  KOKKOS_FUNCTION
  view_type view() const { return *this; }

  template <class RT, class... RP>
  KOKKOS_FUNCTION OffsetView(const View<RT, RP...>& aview) : base_t(aview) {
    for (size_t i = 0; i < View<RT, RP...>::rank(); ++i) {
      m_begins[i] = 0;
    }
  }

  template <class RT, class... RP>
  KOKKOS_FUNCTION OffsetView(const View<RT, RP...>& aview,
                             const index_list_type& minIndices)
      : base_t(aview) {
    KOKKOS_IF_ON_HOST(
        (Kokkos::Experimental::Impl::runtime_check_rank_host(
             traits::rank_dynamic, base_t::rank(), minIndices, aview.label());))

    KOKKOS_IF_ON_DEVICE(
        (Kokkos::Experimental::Impl::runtime_check_rank_device(
             traits::rank_dynamic, base_t::rank(), minIndices);))
    for (size_t i = 0; i < minIndices.size(); ++i) {
      m_begins[i] = minIndices.begin()[i];
    }
  }
  template <class RT, class... RP>
  KOKKOS_FUNCTION OffsetView(const View<RT, RP...>& aview,
                             const begins_type& beg)
      : base_t(aview), m_begins(beg) {}

  // may assign unmanaged from managed.

  template <class RT, class... RP>
  KOKKOS_FUNCTION OffsetView(const OffsetView<RT, RP...>& rhs)
      : base_t(rhs.view()), m_begins(rhs.m_begins) {}

 private:
  enum class subtraction_failure {
    none,
    negative,
    overflow,
  };

  // Subtraction should return a non-negative number and not overflow
  KOKKOS_FUNCTION static subtraction_failure check_subtraction(int64_t lhs,
                                                               int64_t rhs) {
    if (lhs < rhs) return subtraction_failure::negative;

    if (static_cast<uint64_t>(-1) / static_cast<uint64_t>(2) <
        static_cast<uint64_t>(lhs) - static_cast<uint64_t>(rhs))
      return subtraction_failure::overflow;

    return subtraction_failure::none;
  }

  // Need a way to get at an element from both begins_type (aka Kokkos::Array
  // which doesn't have iterators) and index_list_type (aka
  // std::initializer_list which doesn't have .data() or operator[]).
  // Returns by value
  KOKKOS_FUNCTION
  static int64_t at(const begins_type& a, size_t pos) { return a[pos]; }

  KOKKOS_FUNCTION
  static int64_t at(index_list_type a, size_t pos) {
    return *(a.begin() + pos);
  }

  // Check that begins < ends for all elements
  // B, E can be begins_type and/or index_list_type
  template <typename B, typename E>
  static subtraction_failure runtime_check_begins_ends_host(const B& begins,
                                                            const E& ends) {
    std::string message;
    if (begins.size() != base_t::rank())
      message +=
          "begins.size() "
          "(" +
          std::to_string(begins.size()) +
          ")"
          " != Rank "
          "(" +
          std::to_string(base_t::rank()) +
          ")"
          "\n";

    if (ends.size() != base_t::rank())
      message +=
          "ends.size() "
          "(" +
          std::to_string(ends.size()) +
          ")"
          " != Rank "
          "(" +
          std::to_string(base_t::rank()) +
          ")"
          "\n";

    // If there are no errors so far, then arg_rank == Rank
    // Otherwise, check as much as possible
    size_t arg_rank = begins.size() < ends.size() ? begins.size() : ends.size();
    for (size_t i = 0; i != arg_rank; ++i) {
      subtraction_failure sf = check_subtraction(at(ends, i), at(begins, i));
      if (sf != subtraction_failure::none) {
        message +=
            "("
            "ends[" +
            std::to_string(i) +
            "]"
            " "
            "(" +
            std::to_string(at(ends, i)) +
            ")"
            " - "
            "begins[" +
            std::to_string(i) +
            "]"
            " "
            "(" +
            std::to_string(at(begins, i)) +
            ")"
            ")";
        switch (sf) {
          case subtraction_failure::negative:
            message += " must be non-negative\n";
            break;
          case subtraction_failure::overflow: message += " overflows\n"; break;
          default: break;
        }
      }
    }

    if (!message.empty()) {
      message =
          "Kokkos::Experimental::OffsetView ERROR: for unmanaged OffsetView\n" +
          message;
      Kokkos::abort(message.c_str());
    }

    return subtraction_failure::none;
  }

  // Check the begins < ends for all elements
  template <typename B, typename E>
  KOKKOS_FUNCTION static subtraction_failure runtime_check_begins_ends_device(
      const B& begins, const E& ends) {
    if (begins.size() != base_t::rank())
      Kokkos::abort(
          "Kokkos::Experimental::OffsetView ERROR: for unmanaged "
          "OffsetView: begins has bad Rank");
    if (ends.size() != base_t::rank())
      Kokkos::abort(
          "Kokkos::Experimental::OffsetView ERROR: for unmanaged "
          "OffsetView: ends has bad Rank");

    for (size_t i = 0; i != begins.size(); ++i) {
      switch (check_subtraction(at(ends, i), at(begins, i))) {
        case subtraction_failure::negative:
          Kokkos::abort(
              "Kokkos::Experimental::OffsetView ERROR: for unmanaged "
              "OffsetView: bad range");
          break;
        case subtraction_failure::overflow:
          Kokkos::abort(
              "Kokkos::Experimental::OffsetView ERROR: for unmanaged "
              "OffsetView: range overflows");
          break;
        default: break;
      }
    }

    return subtraction_failure::none;
  }

  template <typename B, typename E>
  KOKKOS_FUNCTION static subtraction_failure runtime_check_begins_ends(
      const B& begins, const E& ends) {
    KOKKOS_IF_ON_HOST((return runtime_check_begins_ends_host(begins, ends);))
    KOKKOS_IF_ON_DEVICE(
        (return runtime_check_begins_ends_device(begins, ends);))
  }

  // Constructor around unmanaged data after checking begins < ends for all
  // elements
  // Each of B, E can be begins_type and/or index_list_type
  // Precondition: begins.size() == ends.size() == m_begins.size() == Rank
  template <typename B, typename E>
  KOKKOS_FUNCTION OffsetView(const pointer_type& p, const B& begins_,
                             const E& ends_, subtraction_failure)
      : base_t(Kokkos::view_wrap(p),
               typename traits::array_layout(
                   base_t::rank() > 0 ? at(ends_, 0) - at(begins_, 0)
                                      : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                   base_t::rank() > 1 ? at(ends_, 1) - at(begins_, 1)
                                      : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                   base_t::rank() > 2 ? at(ends_, 2) - at(begins_, 2)
                                      : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                   base_t::rank() > 3 ? at(ends_, 3) - at(begins_, 3)
                                      : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                   base_t::rank() > 4 ? at(ends_, 4) - at(begins_, 4)
                                      : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                   base_t::rank() > 5 ? at(ends_, 5) - at(begins_, 5)
                                      : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                   base_t::rank() > 6 ? at(ends_, 6) - at(begins_, 6)
                                      : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                   base_t::rank() > 7 ? at(ends_, 7) - at(begins_, 7)
                                      : KOKKOS_IMPL_CTOR_DEFAULT_ARG)) {
    for (size_t i = 0; i != m_begins.size(); ++i) {
      m_begins[i] = at(begins_, i);
    };
  }

 public:
  // Constructor around unmanaged data
  // Four overloads, as both begins and ends can be either
  // begins_type or index_list_type
  KOKKOS_FUNCTION
  OffsetView(const pointer_type& p, const begins_type& begins_,
             const begins_type& ends_)
      : OffsetView(p, begins_, ends_,
                   runtime_check_begins_ends(begins_, ends_)) {}

  KOKKOS_FUNCTION
  OffsetView(const pointer_type& p, const begins_type& begins_,
             index_list_type ends_)
      : OffsetView(p, begins_, ends_,
                   runtime_check_begins_ends(begins_, ends_)) {}

  KOKKOS_FUNCTION
  OffsetView(const pointer_type& p, index_list_type begins_,
             const begins_type& ends_)
      : OffsetView(p, begins_, ends_,
                   runtime_check_begins_ends(begins_, ends_)) {}

  KOKKOS_FUNCTION
  OffsetView(const pointer_type& p, index_list_type begins_,
             index_list_type ends_)
      : OffsetView(p, begins_, ends_,
                   runtime_check_begins_ends(begins_, ends_)) {}

  // Choosing std::pair as type for the arguments allows constructing an
  // OffsetView using list initialization syntax, e.g.,
  //   OffsetView dummy("dummy", {-1, 3}, {-2,2});
  // We could allow arbitrary types RangeType that support
  // std::get<{0,1}>(RangeType const&) with std::tuple_size<RangeType>::value==2
  // but this wouldn't allow using the syntax in the example above.
  template <typename Label>
  explicit OffsetView(
      const Label& arg_label,
      std::enable_if_t<Kokkos::Impl::is_view_label<Label>::value,
                       const std::pair<int64_t, int64_t>>
          range0,
      const std::pair<int64_t, int64_t> range1 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range2 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range3 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range4 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range5 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range6 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range7 = KOKKOS_INVALID_INDEX_RANGE

      )
      : OffsetView(Kokkos::Impl::ViewCtorProp<std::string>(arg_label),
                   typename traits::array_layout(
                       range0.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG - 1
                           : range0.second - range0.first + 1,
                       range1.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range1.second - range1.first + 1,
                       range2.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range2.second - range2.first + 1,
                       range3.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range3.second - range3.first + 1,
                       range4.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range4.second - range4.first + 1,
                       range5.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range5.second - range5.first + 1,
                       range6.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range6.second - range6.first + 1,
                       range7.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range7.second - range7.first + 1),
                   {range0.first, range1.first, range2.first, range3.first,
                    range4.first, range5.first, range6.first, range7.first}) {}

  template <class... P>
  explicit OffsetView(
      const Kokkos::Impl::ViewCtorProp<P...>& arg_prop,
      const std::pair<int64_t, int64_t> range0 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range1 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range2 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range3 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range4 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range5 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range6 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range7 = KOKKOS_INVALID_INDEX_RANGE)
      : OffsetView(arg_prop,
                   typename traits::array_layout(
                       range0.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range0.second - range0.first + 1,
                       range1.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range1.second - range1.first + 1,
                       range2.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range2.second - range2.first + 1,
                       range3.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range3.second - range3.first + 1,
                       range4.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range4.second - range4.first + 1,
                       range5.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range5.second - range5.first + 1,
                       range6.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range6.second - range6.first + 1,
                       range7.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range7.second - range7.first + 1),
                   {range0.first, range1.first, range2.first, range3.first,
                    range4.first, range5.first, range6.first, range7.first}) {}

  template <class... P>
  explicit KOKKOS_FUNCTION OffsetView(
      const Kokkos::Impl::ViewCtorProp<P...>& arg_prop,
      std::enable_if_t<Kokkos::Impl::ViewCtorProp<P...>::has_pointer,
                       typename traits::array_layout> const& arg_layout,
      const index_list_type minIndices)
      : base_t(arg_prop, arg_layout) {
    KOKKOS_IF_ON_HOST((Kokkos::Experimental::Impl::runtime_check_rank_host(
                           traits::rank_dynamic, base_t::rank(), minIndices,
                           base_t::label());))

    KOKKOS_IF_ON_DEVICE(
        (Kokkos::Experimental::Impl::runtime_check_rank_device(
             traits::rank_dynamic, base_t::rank(), minIndices);))
    for (size_t i = 0; i < minIndices.size(); ++i) {
      m_begins[i] = minIndices.begin()[i];
    }
    static_assert(
        std::is_same_v<pointer_type,
                       typename Kokkos::Impl::ViewCtorProp<P...>::pointer_type>,
        "When constructing OffsetView to wrap user memory, you must supply "
        "matching pointer type");
  }

  template <class... P>
  explicit OffsetView(
      const Kokkos::Impl::ViewCtorProp<P...>& arg_prop,
      std::enable_if_t<!Kokkos::Impl::ViewCtorProp<P...>::has_pointer,
                       typename traits::array_layout> const& arg_layout,
      const index_list_type minIndices)
      : base_t(arg_prop, arg_layout) {
    for (size_t i = 0; i < base_t::rank(); ++i)
      m_begins[i] = minIndices.begin()[i];
  }
};

/** \brief Temporary free function rank()
 *         until rank() is implemented
 *         in the View
 */
template <typename D, class... P>
KOKKOS_INLINE_FUNCTION constexpr unsigned rank(const OffsetView<D, P...>& V) {
  return V.rank();
}  // Temporary until added to view

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
namespace Impl {

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_integral_v<T>, T> shift_input(
    const T arg, const int64_t offset) {
  return arg - offset;
}

KOKKOS_INLINE_FUNCTION
Kokkos::ALL_t shift_input(const Kokkos::ALL_t arg, const int64_t /*offset*/) {
  return arg;
}

template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_integral_v<T>, Kokkos::pair<T, T>>
    shift_input(const Kokkos::pair<T, T> arg, const int64_t offset) {
  return Kokkos::make_pair<T, T>(arg.first - offset, arg.second - offset);
}
template <class T>
inline std::enable_if_t<std::is_integral_v<T>, std::pair<T, T>> shift_input(
    const std::pair<T, T> arg, const int64_t offset) {
  return std::make_pair<T, T>(arg.first - offset, arg.second - offset);
}

template <size_t N, class Arg, class A>
KOKKOS_INLINE_FUNCTION void map_arg_to_new_begin(
    const size_t i, Kokkos::Array<int64_t, N>& subviewBegins,
    std::enable_if_t<N != 0, const Arg> shiftedArg, const Arg arg,
    const A viewBegins, size_t& counter) {
  if (!std::is_integral_v<Arg>) {
    subviewBegins[counter] = shiftedArg == arg ? viewBegins[i] : 0;
    counter++;
  }
}

template <size_t N, class Arg, class A>
KOKKOS_INLINE_FUNCTION void map_arg_to_new_begin(
    const size_t /*i*/, Kokkos::Array<int64_t, N>& /*subviewBegins*/,
    std::enable_if_t<N == 0, const Arg> /*shiftedArg*/, const Arg /*arg*/,
    const A /*viewBegins*/, size_t& /*counter*/) {}

template <class D, class... P, class T>
KOKKOS_INLINE_FUNCTION
    typename Kokkos::Experimental::Impl::GetOffsetViewTypeFromViewType<
        typename Kokkos::Impl::ViewMapping<void /* deduce subview type from
                                                   source view traits */
                                           ,
                                           ViewTraits<D, P...>, T>::type>::type
    subview_offset(const OffsetView<D, P...>& src, T arg) {
  auto theView = src.view();
  auto begins  = src.begins();

  T shiftedArg = shift_input(arg, begins[0]);

  constexpr size_t rank =
      Kokkos::Impl::ViewMapping<void /* deduce subview type from source view
                                        traits */
                                ,
                                ViewTraits<D, P...>, T>::type::rank;

  auto theSubview = Kokkos::subview(theView, shiftedArg);

  Kokkos::Array<int64_t, rank> subviewBegins;
  size_t counter = 0;
  Kokkos::Experimental::Impl::map_arg_to_new_begin(0, subviewBegins, shiftedArg,
                                                   arg, begins, counter);

  typename Kokkos::Experimental::Impl::GetOffsetViewTypeFromViewType<
      typename Kokkos::Impl::ViewMapping<void /* deduce subview type from source
                                                 view traits */
                                         ,
                                         ViewTraits<D, P...>, T>::type>::type
      offsetView(theSubview, subviewBegins);

  return offsetView;
}

template <class D, class... P, class T0, class T1>
KOKKOS_INLINE_FUNCTION
    typename Kokkos::Experimental::Impl::GetOffsetViewTypeFromViewType<
        typename Kokkos::Impl::ViewMapping<
            void /* deduce subview type from source view traits */
            ,
            ViewTraits<D, P...>, T0, T1>::type>::type
    subview_offset(const Kokkos::Experimental::OffsetView<D, P...>& src,
                   T0 arg0, T1 arg1) {
  auto theView = src.view();
  auto begins  = src.begins();

  T0 shiftedArg0 = shift_input(arg0, begins[0]);
  T1 shiftedArg1 = shift_input(arg1, begins[1]);

  auto theSubview = Kokkos::subview(theView, shiftedArg0, shiftedArg1);
  constexpr size_t rank =
      Kokkos::Impl::ViewMapping<void /* deduce subview type from source view
                                        traits */
                                ,
                                ViewTraits<D, P...>, T0, T1>::type::rank;

  Kokkos::Array<int64_t, rank> subviewBegins;
  size_t counter = 0;
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      0, subviewBegins, shiftedArg0, arg0, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      1, subviewBegins, shiftedArg1, arg1, begins, counter);

  typename Kokkos::Experimental::Impl::GetOffsetViewTypeFromViewType<
      typename Kokkos::Impl::ViewMapping<
          void /* deduce subview type from source view traits */
          ,
          ViewTraits<D, P...>, T0, T1>::type>::type offsetView(theSubview,
                                                               subviewBegins);

  return offsetView;
}

template <class D, class... P, class T0, class T1, class T2>
KOKKOS_INLINE_FUNCTION
    typename Kokkos::Experimental::Impl::GetOffsetViewTypeFromViewType<
        typename Kokkos::Impl::ViewMapping<
            void /* deduce subview type from source view traits */
            ,
            ViewTraits<D, P...>, T0, T1, T2>::type>::type
    subview_offset(const OffsetView<D, P...>& src, T0 arg0, T1 arg1, T2 arg2) {
  auto theView = src.view();
  auto begins  = src.begins();

  T0 shiftedArg0 = shift_input(arg0, begins[0]);
  T1 shiftedArg1 = shift_input(arg1, begins[1]);
  T2 shiftedArg2 = shift_input(arg2, begins[2]);

  auto theSubview =
      Kokkos::subview(theView, shiftedArg0, shiftedArg1, shiftedArg2);

  constexpr size_t rank =
      Kokkos::Impl::ViewMapping<void /* deduce subview type from source view
                                        traits */
                                ,
                                ViewTraits<D, P...>, T0, T1, T2>::type::rank;

  Kokkos::Array<int64_t, rank> subviewBegins;

  size_t counter = 0;
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      0, subviewBegins, shiftedArg0, arg0, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      1, subviewBegins, shiftedArg1, arg1, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      2, subviewBegins, shiftedArg2, arg2, begins, counter);

  typename Kokkos::Experimental::Impl::GetOffsetViewTypeFromViewType<
      typename Kokkos::Impl::ViewMapping<
          void /* deduce subview type from source view traits */
          ,
          ViewTraits<D, P...>, T0, T1, T2>::type>::type
      offsetView(theSubview, subviewBegins);

  return offsetView;
}

template <class D, class... P, class T0, class T1, class T2, class T3>
KOKKOS_INLINE_FUNCTION
    typename Kokkos::Experimental::Impl::GetOffsetViewTypeFromViewType<
        typename Kokkos::Impl::ViewMapping<
            void /* deduce subview type from source view traits */
            ,
            ViewTraits<D, P...>, T0, T1, T2, T3>::type>::type
    subview_offset(const OffsetView<D, P...>& src, T0 arg0, T1 arg1, T2 arg2,
                   T3 arg3) {
  auto theView = src.view();
  auto begins  = src.begins();

  T0 shiftedArg0 = shift_input(arg0, begins[0]);
  T1 shiftedArg1 = shift_input(arg1, begins[1]);
  T2 shiftedArg2 = shift_input(arg2, begins[2]);
  T3 shiftedArg3 = shift_input(arg3, begins[3]);

  auto theSubview = Kokkos::subview(theView, shiftedArg0, shiftedArg1,
                                    shiftedArg2, shiftedArg3);

  constexpr size_t rank = Kokkos::Impl::ViewMapping<
      void /* deduce subview type from source view traits */
      ,
      ViewTraits<D, P...>, T0, T1, T2, T3>::type::rank;
  Kokkos::Array<int64_t, rank> subviewBegins;

  size_t counter = 0;
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      0, subviewBegins, shiftedArg0, arg0, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      1, subviewBegins, shiftedArg1, arg1, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      2, subviewBegins, shiftedArg2, arg2, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      3, subviewBegins, shiftedArg3, arg3, begins, counter);

  typename Kokkos::Experimental::Impl::GetOffsetViewTypeFromViewType<
      typename Kokkos::Impl::ViewMapping<
          void /* deduce subview type from source view traits */
          ,
          ViewTraits<D, P...>, T0, T1, T2, T3>::type>::type
      offsetView(theSubview, subviewBegins);

  return offsetView;
}

template <class D, class... P, class T0, class T1, class T2, class T3, class T4>
KOKKOS_INLINE_FUNCTION
    typename Kokkos::Experimental::Impl::GetOffsetViewTypeFromViewType<
        typename Kokkos::Impl::ViewMapping<
            void /* deduce subview type from source view traits */
            ,
            ViewTraits<D, P...>, T0, T1, T2, T3, T4>::type>::type
    subview_offset(const OffsetView<D, P...>& src, T0 arg0, T1 arg1, T2 arg2,
                   T3 arg3, T4 arg4) {
  auto theView = src.view();
  auto begins  = src.begins();

  T0 shiftedArg0 = shift_input(arg0, begins[0]);
  T1 shiftedArg1 = shift_input(arg1, begins[1]);
  T2 shiftedArg2 = shift_input(arg2, begins[2]);
  T3 shiftedArg3 = shift_input(arg3, begins[3]);
  T4 shiftedArg4 = shift_input(arg4, begins[4]);

  auto theSubview = Kokkos::subview(theView, shiftedArg0, shiftedArg1,
                                    shiftedArg2, shiftedArg3, shiftedArg4);

  constexpr size_t rank = Kokkos::Impl::ViewMapping<
      void /* deduce subview type from source view traits */
      ,
      ViewTraits<D, P...>, T0, T1, T2, T3, T4>::type::rank;
  Kokkos::Array<int64_t, rank> subviewBegins;

  size_t counter = 0;
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      0, subviewBegins, shiftedArg0, arg0, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      1, subviewBegins, shiftedArg1, arg1, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      2, subviewBegins, shiftedArg2, arg2, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      3, subviewBegins, shiftedArg3, arg3, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      4, subviewBegins, shiftedArg4, arg4, begins, counter);

  typename Kokkos::Experimental::Impl::GetOffsetViewTypeFromViewType<
      typename Kokkos::Impl::ViewMapping<
          void /* deduce subview type from source view traits */
          ,
          ViewTraits<D, P...>, T0, T1, T2, T3, T4>::type>::type
      offsetView(theSubview, subviewBegins);

  return offsetView;
}

template <class D, class... P, class T0, class T1, class T2, class T3, class T4,
          class T5>
KOKKOS_INLINE_FUNCTION
    typename Kokkos::Experimental::Impl::GetOffsetViewTypeFromViewType<
        typename Kokkos::Impl::ViewMapping<
            void /* deduce subview type from source view traits */
            ,
            ViewTraits<D, P...>, T0, T1, T2, T3, T4, T5>::type>::type
    subview_offset(const OffsetView<D, P...>& src, T0 arg0, T1 arg1, T2 arg2,
                   T3 arg3, T4 arg4, T5 arg5) {
  auto theView = src.view();
  auto begins  = src.begins();

  T0 shiftedArg0 = shift_input(arg0, begins[0]);
  T1 shiftedArg1 = shift_input(arg1, begins[1]);
  T2 shiftedArg2 = shift_input(arg2, begins[2]);
  T3 shiftedArg3 = shift_input(arg3, begins[3]);
  T4 shiftedArg4 = shift_input(arg4, begins[4]);
  T5 shiftedArg5 = shift_input(arg5, begins[5]);

  auto theSubview =
      Kokkos::subview(theView, shiftedArg0, shiftedArg1, shiftedArg2,
                      shiftedArg3, shiftedArg4, shiftedArg5);

  constexpr size_t rank = Kokkos::Impl::ViewMapping<
      void /* deduce subview type from source view traits */
      ,
      ViewTraits<D, P...>, T0, T1, T2, T3, T4, T5>::type::rank;

  Kokkos::Array<int64_t, rank> subviewBegins;

  size_t counter = 0;
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      0, subviewBegins, shiftedArg0, arg0, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      1, subviewBegins, shiftedArg1, arg1, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      2, subviewBegins, shiftedArg2, arg2, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      3, subviewBegins, shiftedArg3, arg3, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      4, subviewBegins, shiftedArg4, arg4, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      5, subviewBegins, shiftedArg5, arg5, begins, counter);

  typename Kokkos::Experimental::Impl::GetOffsetViewTypeFromViewType<
      typename Kokkos::Impl::ViewMapping<
          void /* deduce subview type from source view traits */
          ,
          ViewTraits<D, P...>, T0, T1, T2, T3, T4, T5>::type>::type
      offsetView(theSubview, subviewBegins);

  return offsetView;
}
template <class D, class... P, class T0, class T1, class T2, class T3, class T4,
          class T5, class T6>
KOKKOS_INLINE_FUNCTION
    typename Kokkos::Experimental::Impl::GetOffsetViewTypeFromViewType<
        typename Kokkos::Impl::ViewMapping<
            void /* deduce subview type from source view traits */
            ,
            ViewTraits<D, P...>, T0, T1, T2, T3, T4, T5, T6>::type>::type
    subview_offset(const OffsetView<D, P...>& src, T0 arg0, T1 arg1, T2 arg2,
                   T3 arg3, T4 arg4, T5 arg5, T6 arg6) {
  auto theView = src.view();
  auto begins  = src.begins();

  T0 shiftedArg0 = shift_input(arg0, begins[0]);
  T1 shiftedArg1 = shift_input(arg1, begins[1]);
  T2 shiftedArg2 = shift_input(arg2, begins[2]);
  T3 shiftedArg3 = shift_input(arg3, begins[3]);
  T4 shiftedArg4 = shift_input(arg4, begins[4]);
  T5 shiftedArg5 = shift_input(arg5, begins[5]);
  T6 shiftedArg6 = shift_input(arg6, begins[6]);

  auto theSubview =
      Kokkos::subview(theView, shiftedArg0, shiftedArg1, shiftedArg2,
                      shiftedArg3, shiftedArg4, shiftedArg5, shiftedArg6);

  constexpr size_t rank = Kokkos::Impl::ViewMapping<
      void /* deduce subview type from source view traits */
      ,
      ViewTraits<D, P...>, T0, T1, T2, T3, T4, T5, T6>::type::rank;

  Kokkos::Array<int64_t, rank> subviewBegins;

  size_t counter = 0;
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      0, subviewBegins, shiftedArg0, arg0, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      1, subviewBegins, shiftedArg1, arg1, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      2, subviewBegins, shiftedArg2, arg2, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      3, subviewBegins, shiftedArg3, arg3, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      4, subviewBegins, shiftedArg4, arg4, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      5, subviewBegins, shiftedArg5, arg5, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      6, subviewBegins, shiftedArg6, arg6, begins, counter);

  typename Kokkos::Experimental::Impl::GetOffsetViewTypeFromViewType<
      typename Kokkos::Impl::ViewMapping<
          void /* deduce subview type from source view traits */
          ,
          ViewTraits<D, P...>, T0, T1, T2, T3, T4, T5, T6>::type>::type
      offsetView(theSubview, subviewBegins);

  return offsetView;
}

template <class D, class... P, class T0, class T1, class T2, class T3, class T4,
          class T5, class T6, class T7>
KOKKOS_INLINE_FUNCTION
    typename Kokkos::Experimental::Impl::GetOffsetViewTypeFromViewType<
        typename Kokkos::Impl::ViewMapping<
            void /* deduce subview type from source view traits */
            ,
            ViewTraits<D, P...>, T0, T1, T2, T3, T4, T5, T6, T7>::type>::type
    subview_offset(const OffsetView<D, P...>& src, T0 arg0, T1 arg1, T2 arg2,
                   T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) {
  auto theView = src.view();
  auto begins  = src.begins();

  T0 shiftedArg0 = shift_input(arg0, begins[0]);
  T1 shiftedArg1 = shift_input(arg1, begins[1]);
  T2 shiftedArg2 = shift_input(arg2, begins[2]);
  T3 shiftedArg3 = shift_input(arg3, begins[3]);
  T4 shiftedArg4 = shift_input(arg4, begins[4]);
  T5 shiftedArg5 = shift_input(arg5, begins[5]);
  T6 shiftedArg6 = shift_input(arg6, begins[6]);
  T7 shiftedArg7 = shift_input(arg7, begins[7]);

  auto theSubview = Kokkos::subview(theView, shiftedArg0, shiftedArg1,
                                    shiftedArg2, shiftedArg3, shiftedArg4,
                                    shiftedArg5, shiftedArg6, shiftedArg7);

  constexpr size_t rank = Kokkos::Impl::ViewMapping<
      void /* deduce subview type from source view traits */
      ,
      ViewTraits<D, P...>, T0, T1, T2, T3, T4, T5, T6, T7>::type::rank;

  Kokkos::Array<int64_t, rank> subviewBegins;

  size_t counter = 0;
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      0, subviewBegins, shiftedArg0, arg0, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      1, subviewBegins, shiftedArg1, arg1, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      2, subviewBegins, shiftedArg2, arg2, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      3, subviewBegins, shiftedArg3, arg3, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      4, subviewBegins, shiftedArg4, arg4, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      5, subviewBegins, shiftedArg5, arg5, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      6, subviewBegins, shiftedArg6, arg6, begins, counter);
  Kokkos::Experimental::Impl::map_arg_to_new_begin(
      7, subviewBegins, shiftedArg7, arg7, begins, counter);

  typename Kokkos::Experimental::Impl::GetOffsetViewTypeFromViewType<
      typename Kokkos::Impl::ViewMapping<
          void /* deduce subview type from source view traits */
          ,
          ViewTraits<D, P...>, T0, T1, T2, T3, T4, T5, T6, T7>::type>::type
      offsetView(theSubview, subviewBegins);

  return offsetView;
}
}  // namespace Impl

template <class D, class... P, class... Args>
KOKKOS_INLINE_FUNCTION
    typename Kokkos::Experimental::Impl::GetOffsetViewTypeFromViewType<
        typename Kokkos::Impl::ViewMapping<
            void /* deduce subview type from source view traits */
            ,
            ViewTraits<D, P...>, Args...>::type>::type
    subview(const OffsetView<D, P...>& src, Args... args) {
  static_assert(
      OffsetView<D, P...>::rank() == sizeof...(Args),
      "subview requires one argument for each source OffsetView rank");

  return Kokkos::Experimental::Impl::subview_offset(src, args...);
}

}  // namespace Experimental
}  // namespace Kokkos
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {
template <class LT, class... LP, class RT, class... RP>
KOKKOS_INLINE_FUNCTION bool operator==(const OffsetView<LT, LP...>& lhs,
                                       const OffsetView<RT, RP...>& rhs) {
  // Same data, layout, dimensions
  using lhs_traits = ViewTraits<LT, LP...>;
  using rhs_traits = ViewTraits<RT, RP...>;

  return std::is_same_v<typename lhs_traits::const_value_type,
                        typename rhs_traits::const_value_type> &&
         std::is_same_v<typename lhs_traits::array_layout,
                        typename rhs_traits::array_layout> &&
         std::is_same_v<typename lhs_traits::memory_space,
                        typename rhs_traits::memory_space> &&
         unsigned(lhs_traits::rank) == unsigned(rhs_traits::rank) &&
         lhs.data() == rhs.data() && lhs.span() == rhs.span() &&
         lhs.extent(0) == rhs.extent(0) && lhs.extent(1) == rhs.extent(1) &&
         lhs.extent(2) == rhs.extent(2) && lhs.extent(3) == rhs.extent(3) &&
         lhs.extent(4) == rhs.extent(4) && lhs.extent(5) == rhs.extent(5) &&
         lhs.extent(6) == rhs.extent(6) && lhs.extent(7) == rhs.extent(7) &&
         lhs.begin(0) == rhs.begin(0) && lhs.begin(1) == rhs.begin(1) &&
         lhs.begin(2) == rhs.begin(2) && lhs.begin(3) == rhs.begin(3) &&
         lhs.begin(4) == rhs.begin(4) && lhs.begin(5) == rhs.begin(5) &&
         lhs.begin(6) == rhs.begin(6) && lhs.begin(7) == rhs.begin(7);
}

template <class LT, class... LP, class RT, class... RP>
KOKKOS_INLINE_FUNCTION bool operator!=(const OffsetView<LT, LP...>& lhs,
                                       const OffsetView<RT, RP...>& rhs) {
  return !(operator==(lhs, rhs));
}

template <class LT, class... LP, class RT, class... RP>
KOKKOS_INLINE_FUNCTION bool operator==(const View<LT, LP...>& lhs,
                                       const OffsetView<RT, RP...>& rhs) {
  // Same data, layout, dimensions
  using lhs_traits = ViewTraits<LT, LP...>;
  using rhs_traits = ViewTraits<RT, RP...>;

  return std::is_same_v<typename lhs_traits::const_value_type,
                        typename rhs_traits::const_value_type> &&
         std::is_same_v<typename lhs_traits::array_layout,
                        typename rhs_traits::array_layout> &&
         std::is_same_v<typename lhs_traits::memory_space,
                        typename rhs_traits::memory_space> &&
         unsigned(lhs_traits::rank) == unsigned(rhs_traits::rank) &&
         lhs.data() == rhs.data() && lhs.span() == rhs.span() &&
         lhs.extent(0) == rhs.extent(0) && lhs.extent(1) == rhs.extent(1) &&
         lhs.extent(2) == rhs.extent(2) && lhs.extent(3) == rhs.extent(3) &&
         lhs.extent(4) == rhs.extent(4) && lhs.extent(5) == rhs.extent(5) &&
         lhs.extent(6) == rhs.extent(6) && lhs.extent(7) == rhs.extent(7);
}

template <class LT, class... LP, class RT, class... RP>
KOKKOS_INLINE_FUNCTION bool operator==(const OffsetView<LT, LP...>& lhs,
                                       const View<RT, RP...>& rhs) {
  return rhs == lhs;
}

}  // namespace Experimental
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

template <class DT, class... DP>
inline void deep_copy(
    const Experimental::OffsetView<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    std::enable_if_t<std::is_same_v<typename ViewTraits<DT, DP...>::specialize,
                                    void>>* = nullptr) {
  static_assert(
      std::is_same_v<typename ViewTraits<DT, DP...>::non_const_value_type,
                     typename ViewTraits<DT, DP...>::value_type>,
      "deep_copy requires non-const type");

  auto dstView = dst.view();
  Kokkos::deep_copy(dstView, value);
}

template <class DT, class... DP, class ST, class... SP>
inline void deep_copy(
    const Experimental::OffsetView<DT, DP...>& dst,
    const Experimental::OffsetView<ST, SP...>& value,
    std::enable_if_t<std::is_same_v<typename ViewTraits<DT, DP...>::specialize,
                                    void>>* = nullptr) {
  static_assert(
      std::is_same_v<typename ViewTraits<DT, DP...>::value_type,
                     typename ViewTraits<ST, SP...>::non_const_value_type>,
      "deep_copy requires matching non-const destination type");

  auto dstView = dst.view();
  Kokkos::deep_copy(dstView, value.view());
}
template <class DT, class... DP, class ST, class... SP>
inline void deep_copy(
    const Experimental::OffsetView<DT, DP...>& dst,
    const View<ST, SP...>& value,
    std::enable_if_t<std::is_same_v<typename ViewTraits<DT, DP...>::specialize,
                                    void>>* = nullptr) {
  static_assert(
      std::is_same_v<typename ViewTraits<DT, DP...>::value_type,
                     typename ViewTraits<ST, SP...>::non_const_value_type>,
      "deep_copy requires matching non-const destination type");

  auto dstView = dst.view();
  Kokkos::deep_copy(dstView, value);
}

template <class DT, class... DP, class ST, class... SP>
inline void deep_copy(
    const View<DT, DP...>& dst,
    const Experimental::OffsetView<ST, SP...>& value,
    std::enable_if_t<std::is_same_v<typename ViewTraits<DT, DP...>::specialize,
                                    void>>* = nullptr) {
  static_assert(
      std::is_same_v<typename ViewTraits<DT, DP...>::value_type,
                     typename ViewTraits<ST, SP...>::non_const_value_type>,
      "deep_copy requires matching non-const destination type");

  Kokkos::deep_copy(dst, value.view());
}

namespace Impl {

// Deduce Mirror Types
template <class Space, class T, class... P>
struct MirrorOffsetViewType {
  // The incoming view_type
  using src_view_type = typename Kokkos::Experimental::OffsetView<T, P...>;
  // The memory space for the mirror view
  using memory_space = typename Space::memory_space;
  // Check whether it is the same memory space
  enum {
    is_same_memspace =
        std::is_same_v<memory_space, typename src_view_type::memory_space>
  };
  // The array_layout
  using array_layout = typename src_view_type::array_layout;
  // The data type (we probably want it non-const since otherwise we can't even
  // deep_copy to it.)
  using data_type = typename src_view_type::non_const_data_type;
  // The destination view type if it is not the same memory space
  using dest_view_type =
      Kokkos::Experimental::OffsetView<data_type, array_layout, Space>;
  // If it is the same memory_space return the existing view_type
  // This will also keep the unmanaged trait if necessary
  using view_type =
      std::conditional_t<is_same_memspace, src_view_type, dest_view_type>;
};

}  // namespace Impl

namespace Impl {

// create a mirror
// private interface that accepts arbitrary view constructor args passed by a
// view_alloc
template <class T, class... P, class... ViewCtorArgs>
inline auto create_mirror(const Kokkos::Experimental::OffsetView<T, P...>& src,
                          const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  check_view_ctor_args_create_mirror<ViewCtorArgs...>();

  if constexpr (Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space) {
    using Space = typename Impl::ViewCtorProp<ViewCtorArgs...>::memory_space;

    auto prop_copy = Impl::with_properties_if_unset(
        arg_prop, std::string(src.label()).append("_mirror"));

    return typename Kokkos::Impl::MirrorOffsetViewType<
        Space, T, P...>::dest_view_type(prop_copy, src.layout(),
                                        {src.begin(0), src.begin(1),
                                         src.begin(2), src.begin(3),
                                         src.begin(4), src.begin(5),
                                         src.begin(6), src.begin(7)});
  } else {
    return typename Kokkos::Experimental::OffsetView<T, P...>::HostMirror(
        Kokkos::create_mirror(arg_prop, src.view()), src.begins());
  }
#if defined(KOKKOS_COMPILER_NVCC) && KOKKOS_COMPILER_NVCC >= 1130 && \
    !defined(KOKKOS_COMPILER_MSVC)
  __builtin_unreachable();
#endif
}

}  // namespace Impl

// public interface
template <class T, class... P,
          typename = std::enable_if_t<
              std::is_void_v<typename ViewTraits<T, P...>::specialize>>>
inline auto create_mirror(
    const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror(src, Impl::ViewCtorProp<>{});
}

// public interface that accepts a without initializing flag
template <class T, class... P,
          typename = std::enable_if_t<
              std::is_void_v<typename ViewTraits<T, P...>::specialize>>>
inline auto create_mirror(
    Kokkos::Impl::WithoutInitializing_t wi,
    const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror(src, Kokkos::view_alloc(wi));
}

// public interface that accepts a space
template <class Space, class T, class... P,
          typename Enable = std::enable_if_t<
              Kokkos::is_space<Space>::value &&
              std::is_void_v<typename ViewTraits<T, P...>::specialize>>>
inline auto create_mirror(
    const Space&, const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror(
      src, Kokkos::view_alloc(typename Space::memory_space{}));
}

// public interface that accepts a space and a without initializing flag
template <class Space, class T, class... P,
          typename Enable = std::enable_if_t<
              Kokkos::is_space<Space>::value &&
              std::is_void_v<typename ViewTraits<T, P...>::specialize>>>
inline auto create_mirror(
    Kokkos::Impl::WithoutInitializing_t wi, const Space&,
    const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror(
      src, Kokkos::view_alloc(typename Space::memory_space{}, wi));
}

// public interface that accepts arbitrary view constructor args passed by a
// view_alloc
template <class T, class... P, class... ViewCtorArgs,
          typename = std::enable_if_t<
              std::is_void_v<typename ViewTraits<T, P...>::specialize>>>
inline auto create_mirror(
    const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
    const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror(src, arg_prop);
}

namespace Impl {

// create a mirror view
// private interface that accepts arbitrary view constructor args passed by a
// view_alloc
template <class T, class... P, class... ViewCtorArgs>
inline auto create_mirror_view(
    const Kokkos::Experimental::OffsetView<T, P...>& src,
    [[maybe_unused]] const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  if constexpr (!Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space) {
    if constexpr (std::is_same_v<typename Kokkos::Experimental::OffsetView<
                                     T, P...>::memory_space,
                                 typename Kokkos::Experimental::OffsetView<
                                     T, P...>::HostMirror::memory_space> &&
                  std::is_same_v<typename Kokkos::Experimental::OffsetView<
                                     T, P...>::data_type,
                                 typename Kokkos::Experimental::OffsetView<
                                     T, P...>::HostMirror::data_type>) {
      return
          typename Kokkos::Experimental::OffsetView<T, P...>::HostMirror(src);
    } else {
      return Kokkos::Impl::choose_create_mirror(src, arg_prop);
    }
  } else {
    if constexpr (Impl::MirrorOffsetViewType<typename Impl::ViewCtorProp<
                                                 ViewCtorArgs...>::memory_space,
                                             T, P...>::is_same_memspace) {
      return typename Impl::MirrorOffsetViewType<
          typename Impl::ViewCtorProp<ViewCtorArgs...>::memory_space, T,
          P...>::view_type(src);
    } else {
      return Kokkos::Impl::choose_create_mirror(src, arg_prop);
    }
  }
#if defined(KOKKOS_COMPILER_NVCC) && KOKKOS_COMPILER_NVCC >= 1130 && \
    !defined(KOKKOS_COMPILER_MSVC)
  __builtin_unreachable();
#endif
}

}  // namespace Impl

// public interface
template <class T, class... P>
inline auto create_mirror_view(
    const typename Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror_view(src, Impl::ViewCtorProp<>{});
}

// public interface that accepts a without initializing flag
template <class T, class... P>
inline auto create_mirror_view(
    Kokkos::Impl::WithoutInitializing_t wi,
    const typename Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror_view(src, Kokkos::view_alloc(wi));
}

// public interface that accepts a space
template <class Space, class T, class... P,
          typename Enable = std::enable_if_t<Kokkos::is_space<Space>::value>>
inline auto create_mirror_view(
    const Space&, const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror_view(
      src, Kokkos::view_alloc(typename Space::memory_space{}));
}

// public interface that accepts a space and a without initializing flag
template <class Space, class T, class... P,
          typename Enable = std::enable_if_t<Kokkos::is_space<Space>::value>>
inline auto create_mirror_view(
    Kokkos::Impl::WithoutInitializing_t wi, const Space&,
    const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror_view(
      src, Kokkos::view_alloc(typename Space::memory_space{}, wi));
}

// public interface that accepts arbitrary view constructor args passed by a
// view_alloc
template <class T, class... P, class... ViewCtorArgs>
inline auto create_mirror_view(
    const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
    const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror_view(src, arg_prop);
}

// create a mirror view and deep copy it
// public interface that accepts arbitrary view constructor args passed by a
// view_alloc
template <class... ViewCtorArgs, class T, class... P>
typename Kokkos::Impl::MirrorOffsetViewType<
    typename Impl::ViewCtorProp<ViewCtorArgs...>::memory_space, T,
    P...>::view_type
create_mirror_view_and_copy(
    const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
    const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return {create_mirror_view_and_copy(arg_prop, src.view()), src.begins()};
}

template <class Space, class T, class... P>
typename Kokkos::Impl::MirrorOffsetViewType<Space, T, P...>::view_type
create_mirror_view_and_copy(
    const Space& space, const Kokkos::Experimental::OffsetView<T, P...>& src,
    std::string const& name = "") {
  return {create_mirror_view_and_copy(space, src.view(), name), src.begins()};
}
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_OFFSETVIEW
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_OFFSETVIEW
#endif
#endif /* KOKKOS_OFFSETVIEW_HPP_ */
