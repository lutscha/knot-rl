#pragma once

#include "knot.h"

inline Knot unknot() {
  static uint16_t comp[1] = {0};
  static Visit visits[1] = {};
  return Knot(0, comp, visits);
}