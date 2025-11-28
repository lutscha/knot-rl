#include <cstdint>
#include "../include/knot.h"


struct Transition {
  uint16_t v;
  ReidemeisterMove move;
  Knot *to;
};


template <uint16_t MAX_CROSSINGS> class Node {
  public:

  Knot knot;
  uint16_t visit_storage[2 * MAX_CROSSINGS];

  Transition children[];



    MCTS(const Knot &knot) : knot(knot) {}

    void run() {
      while (true) {
        Knot knot = knot.apply_move(0, ReidemeisterMove::R1_pos(Orientation::pos, VisitType::over), reinterpret_cast<uint16_t *>(storage + mem_per_knot));
      }
    }
};