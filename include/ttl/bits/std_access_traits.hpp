#pragma once

namespace ttl
{
namespace internal
{
struct owner;
struct readwrite;
struct readonly;

template <typename A>
struct basic_access_traits;

template <>
struct basic_access_traits<owner> {
    using type = readwrite;
};

template <>
struct basic_access_traits<readwrite> {
    using type = readwrite;
};

template <>
struct basic_access_traits<readonly> {
    using type = readonly;
};
}  // namespace internal
}  // namespace ttl
