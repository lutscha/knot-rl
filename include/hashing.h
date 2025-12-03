#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <sys/types.h>

constexpr uint64_t seed = 42;

template <class F> uint16_t booth_min_rotation(const F &f, uint16_t n) {
  if (n == 0)
    return 0;

  uint16_t i = 0;
  uint16_t j = 1;
  uint16_t k = 0;

  while (i < n && j < n && k < n) {
    uint16_t idx_i= i + k < n ? i + k : i + k - n;
    uint16_t idx_j = j + k < n ? j + k : j + k - n;

    int64_t x = f(idx_i);
    int64_t y = f(idx_j);

    if (x == y) {
      ++k;
    } else if (x < y) {
      j = j + k + 1;
      k = 0;
    } else { // x > y
      i = i + k + 1;
      k = 0;
    }
    if (i == j)
      ++j;
  }
  return std::min(i, j);
}

inline uint64_t mix(uint64_t hash, uint64_t value, uint64_t seed) {
  return hash ^ (value + seed + (hash << 12) + (hash >> 4));
}

template <class F> uint64_t circular_hash(const F &f, uint16_t n) {
  if (n == 0) [[unlikely]]
    return 0;

  uint16_t shift = booth_min_rotation(f, n);

  uint64_t hash = seed;
  for (uint16_t i = 0; i < n; ++i) {
    uint16_t idx = (i + shift < n) ? uint16_t(i + shift) : uint16_t(i + shift - n);
    uint64_t value = static_cast<uint64_t>(f(idx));
    hash = mix(hash, value, seed);
  }
  return hash;
}
