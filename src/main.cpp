// knot_engine.cpp
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

#include "../include/knot.h"

static constexpr uint16_t MAX_CROSSINGS = 1024;
static int16_t dowker[MAX_CROSSINGS];

void print_dowker(const Knot &K) {
  K.to_dowker(dowker);
  std::cout << "DT: [";
  for (uint16_t i = 0; i < K.n_crossings; i++) {
    std::cout << dowker[i];
    if (i < K.n_crossings - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]\n";
}


void print_state(const Knot &K) {
  std::cout << K.n_crossings << " crossings: ";
  for (uint16_t i = 0; i < 2 * K.n_crossings; i++) {
    if (K.visits[i].type() == VisitType::under)
        continue;
    std::cout << "(" << i << ", " << K.visits[i].mate << ", " << int16_t(K.visits[i].flags) << ") ";
  }
  std::cout << "\n" << std::flush;
}

int main() {
  using std::uint16_t;

  static uint16_t comp[1] = {0};

  // Storage for n_comp_crossings + visits. Adjust MAX_CROSSINGS if needed.

  static constexpr uint16_t N_KNOTS = 10;
  static constexpr uint16_t mem_per_knot = 1 + 2 * MAX_CROSSINGS * (sizeof(Visit) / sizeof(uint16_t) + 1);

  static uint16_t storage[mem_per_knot * N_KNOTS];


  // Initialize as unknot with 0 crossings
  Knot unknot(0, comp, reinterpret_cast<Visit *>(storage));

  Knot twist = unknot.apply_move(0, 
    ReidemeisterMove::R1_pos(Orientation::pos, VisitType::over), 
    reinterpret_cast<uint16_t *>(storage + mem_per_knot));

  Knot snowman = twist.apply_move(
      1, ReidemeisterMove::R1_pos(Orientation::pos, VisitType::over),
      reinterpret_cast<uint16_t *>(storage + 2 * mem_per_knot));

//   Knot snowman_ = twist.apply_move(
//       1, ReidemeisterMove::R1_pos(Orientation::neg, VisitType::under),
//       reinterpret_cast<uint16_t *>(storage + 3 * mem_per_knot));

Knot triforce = snowman.apply_move(
      1, ReidemeisterMove::R1_pos(Orientation::neg, VisitType::under),
      reinterpret_cast<uint16_t *>(storage + 4 * mem_per_knot));

Knot triforce_ = triforce.apply_move(
    3, ReidemeisterMove::R3(Direction::prev, Direction::next),
    reinterpret_cast<uint16_t *>(storage + 5 * mem_per_knot));

Knot amir = triforce_.apply_move(
    1, ReidemeisterMove::R2_pos(Direction::prev, Direction::next),
    reinterpret_cast<uint16_t *>(storage + 6 * mem_per_knot));

Knot kolic = amir.apply_move(
    1, ReidemeisterMove::R2_neg(),
    reinterpret_cast<uint16_t *>(storage + 7 * mem_per_knot));

Knot kms =
    kolic.apply_move(2, ReidemeisterMove::R3(Direction::prev, Direction::next),
                    reinterpret_cast<uint16_t *>(storage + 8 * mem_per_knot));

print_state(snowman);
print_dowker(snowman);
print_state(triforce);
print_dowker(triforce);
print_state(triforce_);
print_dowker(triforce_);
print_state(amir);
print_dowker(amir);
print_state(kolic);
print_dowker(kolic);
print_state(kms);
print_dowker(kms);
return 0;
}
