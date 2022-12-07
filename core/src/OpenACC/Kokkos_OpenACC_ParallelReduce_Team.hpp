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

#ifndef KOKKOS_OPENACC_PARALLEL_REDUCE_TEAM_HPP
#define KOKKOS_OPENACC_PARALLEL_REDUCE_TEAM_HPP

#include <OpenACC/Kokkos_OpenACC_Team.hpp>
#include <OpenACC/Kokkos_OpenACC_FunctorAdapter.hpp>

#ifdef KOKKOS_ENABLE_OPENACC_COLLAPSE_HIERARCHICAL_CONSTRUCTS

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
// Hierarchical Parallelism -> Team level implementation
template <class FunctorType, class ReducerType, class... Properties>
class Kokkos::Impl::ParallelReduce<FunctorType,
                                   Kokkos::TeamPolicy<Properties...>,
                                   ReducerType, Kokkos::Experimental::OpenACC> {
 private:
  using Policy =
      TeamPolicyInternal<Kokkos::Experimental::OpenACC, Properties...>;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;
  using value_type   = typename Analysis::value_type;
  using pointer_type = typename Analysis::pointer_type;

  Kokkos::Experimental::Impl::FunctorAdapter<FunctorType, Policy> m_functor;
  Policy m_policy;
  ReducerType m_reducer;
  pointer_type m_result_ptr;

 public:
  void execute() const {
    auto league_size   = m_policy.league_size();
    auto team_size     = m_policy.team_size();
    auto vector_length = m_policy.impl_vector_length();

    auto const a_functor = m_functor;
    value_type tmp;
    ValueInit::init(a_functor, &tmp);
    if constexpr (!is_reducer_v<ReducerType>) {
#pragma acc parallel loop gang vector reduction(+ : tmp) copyin(a_functor)
      for (int i = 0; i < league_size; i++) {
        int league_id = i;
        typename Policy::member_type team(league_id, league_size, 1,
                                          vector_length);
        a_functor(team, tmp);
      }
      m_result_ptr[0] = tmp;
    } else {
      OpenACCReducerWrapperTeams<ReducerType, FunctorType, Policy,
                                 TagType>::reduce(tmp, m_policy, a_functor);
      m_result_ptr[0] = tmp;
    }
  }

  template <class ViewType>
  ParallelReduce(const FunctorType& arg_functor, const Policy& arg_policy,
                 const ViewType& arg_result_view,
                 std::enable_if_t<Kokkos::is_view_v<ViewType>>* = nullptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result_view.data()) {}

  ParallelReduce(const FunctorType& arg_functor, const Policy& arg_policy,
                 const ReducerType& reducer)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {}
};

namespace Kokkos {

// Hierarchical Parallelism -> Team thread level implementation
#pragma acc routine seq
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!Kokkos::is_reducer<ValueType>::value>
parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                    iType, Impl::OpenACCTeamMember>& loop_boundaries,
                const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();
#pragma acc loop seq
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
    lambda(i, tmp);
  result = tmp;
}

#pragma acc routine seq
template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer<ReducerType>::value>
parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                    iType, Impl::OpenACCTeamMember>& loop_boundaries,
                const Lambda& lambda, ReducerType& result) {
  using ValueType = typename ReducerType::value_type;

  ValueType tmp = ValueType();
#pragma acc loop seq
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
    lambda(i, tmp);
  result = tmp;
}

// FIXME_OPENACC: custom reduction is not implemented.
template <typename iType, class Lambda, typename ValueType, class JoinType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamThreadRangeBoundariesStruct<iType, Impl::OpenACCTeamMember>&
        loop_boundaries,
    const Lambda& lambda, const JoinType& join, ValueType& init_result) {
  static_assert(!Kokkos::Impl::always_true<Lambda>::value,
                "custom reduction is not implemented");
}

// Hierarchical Parallelism -> Thread vector level implementation
#pragma acc routine seq
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::OpenACCTeamMember>& loop_boundaries,
    const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();

#pragma acc loop seq
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    lambda(i, tmp);
  }
  result = tmp;
}

#pragma acc routine seq
template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer<ReducerType>::value>
parallel_reduce(const Impl::ThreadVectorRangeBoundariesStruct<
                    iType, Impl::OpenACCTeamMember>& loop_boundaries,
                const Lambda& lambda, ReducerType const& result) {
  using ValueType = typename ReducerType::value_type;

  ValueType tmp;
  ReducerType::init(tmp);

#pragma acc loop seq
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    lambda(i, tmp);
  }
  result.reference() = tmp;
}

// FIXME_OPENACC: custom reduction is not implemented.
template <typename iType, class Lambda, typename ValueType, class JoinType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::OpenACCTeamMember>& loop_boundaries,
    const Lambda& lambda, const JoinType& join, ValueType& init_result) {
  static_assert(!Kokkos::Impl::always_true<Lambda>::value,
                "custom reduction is not implemented");
}

// Hierarchical Parallelism -> Team vector level implementation
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamVectorRangeBoundariesStruct<iType, Impl::OpenACCTeamMember>&
        loop_boundaries,
    const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();

#pragma acc loop seq
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    lambda(i, tmp);
  }
  result = tmp;
}
}  // namespace Kokkos

#else /* #ifdef KOKKOS_ENABLE_OPENACC_COLLAPSE_HIERARCHICAL_CONSTRUCTS */

// FIXME_OPENACC: below implementation conforms to the OpenACC standard, but
// the NVHPC compiler (V22.11) fails due to the lack of support for lambda
// expressions containing parallel loops.
// Disabled for the time being.
#if 0
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
// Hierarchical Parallelism -> Team level implementation
template <class FunctorType, class ReducerType, class... Properties>
class Kokkos::Impl::ParallelReduce<FunctorType,
                                   Kokkos::TeamPolicy<Properties...>,
                                   ReducerType, Kokkos::Experimental::OpenACC> {
 private:
  using Policy =
      TeamPolicyInternal<Kokkos::Experimental::OpenACC, Properties...>;

  using ReducerTypeFwd =
      std::conditional_t<std::is_same_v<InvalidType, ReducerType>, FunctorType,
                         ReducerType>;
  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;
  using value_type   = typename Analysis::value_type;
  using pointer_type = typename Analysis::pointer_type;

  Kokkos::Experimental::Impl::FunctorAdapter<FunctorType, Policy> m_functor;
  Policy m_policy;
  ReducerType m_reducer;
  pointer_type m_result_ptr;

 public:
  void execute() const {
    auto league_size   = m_policy.league_size();
    auto team_size     = m_policy.team_size();
    auto vector_length = m_policy.impl_vector_length();

    auto const a_functor = m_functor;
    value_type tmp;
    ValueInit::init(a_functor, &tmp);
    if constexpr (!is_reducer_v<ReducerType>) {
#pragma acc parallel loop gang num_gangs(league_size) num_workers(team_size)  vector_length(vector_length) reduction(+:tmp) copyin(a_functor)
      for (int i = 0; i < league_size; i++) {
        int league_id = i;
        typename Policy::member_type team(league_id, league_size, team_size,
                                          vector_length);
        a_functor(team, tmp);
      }
      m_result_ptr[0] = tmp;
    } else {
      OpenACCReducerWrapperTeams<ReducerType, FunctorType, Policy,
                                 TagType>::reduce(tmp, m_policy, a_functor);
      m_result_ptr[0] = tmp;
    }
  }

  template <class ViewType>
  ParallelReduce(const FunctorType& arg_functor, const Policy& arg_policy,
                 const ViewType& arg_result_view,
                 std::enable_if_t<Kokkos::is_view_v<ViewType>>* = nullptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result_view.data()) {}

  ParallelReduce(const FunctorType& arg_functor, const Policy& arg_policy,
                 const ReducerType& reducer)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {}
};

namespace Kokkos {

// Hierarchical Parallelism -> Team thread level implementation
#pragma acc routine worker
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!Kokkos::is_reducer<ValueType>::value>
parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                    iType, Impl::OpenACCTeamMember>& loop_boundaries,
                const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();
#pragma acc loop worker reduction(+ : tmp)
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
    lambda(i, tmp);
  result = tmp;
}

#pragma acc routine worker
template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer<ReducerType>::value>
parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                    iType, Impl::OpenACCTeamMember>& loop_boundaries,
                const Lambda& lambda, ReducerType& result) {
  using ValueType = typename ReducerType::value_type;

  ValueType tmp = ValueType();
#pragma acc loop worker reduction(+ : tmp)
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
    lambda(i, tmp);
  result = tmp;
}

// FIXME_OPENACC: custom reduction is not implemented.
template <typename iType, class Lambda, typename ValueType, class JoinType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamThreadRangeBoundariesStruct<iType, Impl::OpenACCTeamMember>&
        loop_boundaries,
    const Lambda& lambda, const JoinType& join, ValueType& init_result) {
  static_assert(!Kokkos::Impl::always_true<Lambda>::value,
                "custom reduction is not implemented");
}

// Hierarchical Parallelism -> Thread vector level implementation
#pragma acc routine vector
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::OpenACCTeamMember>& loop_boundaries,
    const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();

#pragma acc loop vector reduction(+ : tmp)
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    lambda(i, tmp);
  }
  result = tmp;
}

#pragma acc routine vector
template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer<ReducerType>::value>
parallel_reduce(const Impl::ThreadVectorRangeBoundariesStruct<
                    iType, Impl::OpenACCTeamMember>& loop_boundaries,
                const Lambda& lambda, ReducerType const& result) {
  using ValueType = typename ReducerType::value_type;

  ValueType tmp;
  ReducerType::init(tmp);

#pragma acc loop vector reduction(+ : tmp)
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    lambda(i, tmp);
  }
  result.reference() = tmp;
}

// FIXME_OPENACC: custom reduction is not implemented.
template <typename iType, class Lambda, typename ValueType, class JoinType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::OpenACCTeamMember>& loop_boundaries,
    const Lambda& lambda, const JoinType& join, ValueType& init_result) {
  static_assert(!Kokkos::Impl::always_true<Lambda>::value,
                "custom reduction is not implemented");
}

// Hierarchical Parallelism -> Team vector level implementation
#pragma acc routine vector
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamVectorRangeBoundariesStruct<iType, Impl::OpenACCTeamMember>&
        loop_boundaries,
    const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();

#pragma acc loop vector reduction(+ : tmp)
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    lambda(i, tmp);
  }
  result = tmp;
}
}  // namespace Kokkos
#endif /* #if 0 */

#endif /* #ifdef KOKKOS_ENABLE_OPENACC_COLLAPSE_HIERARCHICAL_CONSTRUCTS */

#endif /* #ifndef KOKKOS_OPENACC_PARALLEL_REDUCE_TEAM_HPP */
