// knot_engine.cpp
#include "greedy.cpp"
#include <chrono>
#include <cstdint>
#include <iostream>

#include "../include/knot.h"

using Clock = std::chrono::steady_clock;

void test() {
  using std::uint16_t;

  static uint16_t comp[1] = {0};

  static Visit no_visits[0];

  // Initialize as unknot with 0 crossings
  Knot unknot(0, comp, no_visits);

  Knot twist = unknot.apply_move(
      0, ReidemeisterMove::R1_pos(Orientation::pos, VisitType::over));

  Knot snowman = twist.apply_move(
      0, ReidemeisterMove::R1_pos(Orientation::pos, VisitType::over));

  snowman.print_state();
  snowman.print_dowker();

  //   Knot snowman_ = twist.apply_move(
  //       1, ReidemeisterMove::R1_pos(Orientation::neg, VisitType::under),
  //       reinterpret_cast<uint16_t *>(storage + 3 * mem_per_knot));

  Knot triforce = snowman.apply_move(1, ReidemeisterMove::R1_pos(Orientation::neg, VisitType::under));
  
  triforce.print_state();
  triforce.print_dowker();
  Knot triforce_ = triforce.apply_move(
      1, ReidemeisterMove::R3(Direction::next, Direction::prev));
  triforce_.print_state();
  triforce_.print_dowker();

  Knot amir = triforce_.apply_move(1,ReidemeisterMove::R2_pos(VisitType::over, Direction::prev,Direction::next));

  std::cout << "AMIR" << std::endl;
  amir.print_state();
  amir.print_dowker();

  Knot kolic = amir.apply_move(4, ReidemeisterMove::R2_neg());

  kolic.print_state();
  kolic.print_dowker();

  Knot kms = kolic.apply_move(2, ReidemeisterMove::R3(Direction::prev, Direction::next));

  kms.print_state();
  kms.print_dowker();

  std::cout << "SHIFTS" << std::endl;
  static constexpr uint16_t pseudo_n = 5;
  static uint16_t pseudo_comp[1] = {5};
  for (uint16_t shift = 0; shift < 2 * pseudo_n; shift++) {
    static Visit pseudo_visits[2 * pseudo_n];
    for (uint16_t i = 0; i < 2 * pseudo_n; i++) {
      uint16_t idx = ((i - shift) + 2 * pseudo_n) % (2 * pseudo_n);
      pseudo_visits[i].flags = amir.visits[idx].flags;
      pseudo_visits[i].mate = (amir.visits[idx].mate + shift) % (2 * pseudo_n);
    }
    Knot pseudo = Knot(pseudo_n, pseudo_comp, pseudo_visits);

    pseudo.print_state();
    pseudo.print_dowker();
  }
}

Knot amirs_knot() {
  static constexpr uint16_t n_crossings = 4;
  static uint16_t comp[1] = {n_crossings};
  static Visit visits[2 * n_crossings] = {
      Visit(3, Orientation::neg, VisitType::over),
      Visit(6, Orientation::pos, VisitType::under),
      Visit(5, Orientation::pos, VisitType::over),
      Visit(0, Orientation::neg, VisitType::under),
      Visit(7, Orientation::pos, VisitType::under),
      Visit(2, Orientation::pos, VisitType::under),
      Visit(1, Orientation::pos, VisitType::over),
      Visit(4, Orientation::pos, VisitType::over)};
  return Knot(n_crossings, comp, visits);
}

int main() {
  Knot amirs_knot_ = amirs_knot();
  amirs_knot_.print_state();
  amirs_knot_.print_dowker();


  GreedyResult amirs_undone = greedy_minimize_crossings(amirs_knot_, 100000);

  amirs_undone.best.print_state();
  amirs_undone.best.print_dowker();

  for (auto m : amirs_undone.path) {
    std::cout << m.move << " on vertex " << m.v << std::endl;
  }
  return 0;
}
