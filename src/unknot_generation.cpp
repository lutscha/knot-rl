#pragma once

#include <bit> // std::popcount
#include <random>
#include <stdexcept>

#include "../include/knot.h"

std::mt19937_64 rng = std::mt19937_64(std::random_device()());

Knot random_unknot(Knot start, uint16_t target_n_crossings, std::size_t max_steps) {
  if (start.n_crossings > target_n_crossings) {
    throw std::runtime_error("random_unknot: start must have fewer crossings than target");
  }

  Knot buf[2] = {start, start}; // second buffer
  std::size_t step = 0;
  for (; step < max_steps; ++step) {
    uint16_t cur_idx = step & 1;

    if( buf[cur_idx].n_crossings == target_n_crossings) {
      return buf[cur_idx];
    }

    Knot &cur = buf[cur_idx];
    // Special case: the empty unknot. The only meaningful move is R1+ at v = 0.
    ReidemeisterMove chosen_move = ReidemeisterMove::R1_pos(Orientation::pos, VisitType::over);

    if (cur.n_crossings == 0) {
      buf[cur_idx ^ 1] = cur.apply_move(0, chosen_move);
      cur_idx ^= 1;
      continue;
    }

    // Pick a random vertex v in [0, n_crossings - 1].
    std::uniform_int_distribution<uint16_t> v_dist(0, cur.n_crossings - 1);
    uint16_t v = v_dist(rng);

    // Get the move mask for this vertex (only over-visit moves).
    uint16_t mask = cur.move_mask(v);
    unsigned n_moves = std::popcount(mask); // number of bits set in mask

    // If we are exactly one crossing away from the target, we MUST do R1+.
    bool force_R1_pos = (cur.n_crossings + 1 == target_n_crossings);

    // Total options = all bits in mask + 1 (for R1+).
    std::uniform_int_distribution<unsigned> idx_dist(0, n_moves);
    unsigned idx = idx_dist(rng);

    if (force_R1_pos || idx == n_moves) {
      buf[cur_idx ^ 1] = cur.apply_move(v, chosen_move);
      continue;
    }

    // Choose the idx-th '1' bit in mask (0-based).
    unsigned ones = 0;
    for (uint16_t bit = 0; bit < 16; ++bit) {
      if ((mask & (uint16_t(1) << bit)) == 0) {
        continue;
      }
      if (ones == idx) {
        chosen_move = Visit::GET_DIRECT_MOVE(bit);
        break;
      }
      ++ones;
    }

    buf[cur_idx ^ 1] = cur.apply_move(v, chosen_move);
    continue;
  }
  return buf[step & 1];
}
