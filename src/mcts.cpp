#include "../include/knot.h"
#include "../include/node.h"
#include "../include/visit.h"
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <chrono>
#include "../include/arena.h"

// static InferenceServer inference_server;

constexpr uint16_t TO_RESERVE = 256;


static float** get_probs(const Knot &knot){
  return nullptr;
};

AvailableMove mcts_select_move(const Knot root_knot, std::size_t n_simulations) {
  Node root(std::move(root_knot));

  // Expand root once at the start.

  root.expand();

  if (root.children.empty()) {
    // No moves available; return some dummy move.
    throw std::runtime_error("No moves available");
  }


  for (std::size_t sim = 0; sim < n_simulations; ++sim) {
    Node *cur = &root;
    std::vector<Node *> path;
    path.reserve(TO_RESERVE);
    path.push_back(cur);

    // SELECTION
    while (cur->is_expanded && !cur->children.empty()) {
      // for (auto &child : cur->children) {
      //   if (!child.is_computed) {
      //     child.compute();
      //   }
      // }

      Child *child = cur->select_best_child();
      if (!child) {
        break;
      }
      // child->compute() is guaranteed to have been called in select_best_child
      // if needed, so node is non-null here.
      cur = child->node.get();
      path.push_back(cur);
    }

    if (cur->is_expanded) {
      std::cout << "Node already expanded" << std::endl;
      continue;
    }

    // EXPANSION
    const double value = cur->expand();

    // BACKPROPAGATION
    for (Node *n : path) {
      n->n_visits += 1;
      n->value += value;
    }
  }

  // Choose the root child with maximum visit count.
  Child *best_child = nullptr;
  uint32_t best_visits = 0;

  for (auto &child : root.children) {
    uint32_t v = child.n_visits();
    if (v > best_visits) {
      best_visits = v;
      best_child = &child;
    }
  }

  if (!best_child) {
    // Fallback, should not happen if root.children was non-empty.
    std::cerr << "No best child found" << std::endl;
  }

  return best_child->move;
}

Knot mcts_run(const Knot &knot, std::size_t n_simulations) {
  Knot cur = knot;
  for (std::size_t sim = 0; sim < n_simulations; ++sim) {
    AvailableMove move = mcts_select_move(cur, knot.n_crossings * 10);
    cur = cur.apply_move(move.v, move.move);
    std::cout << "Simulation: " << sim << " " << move.move << " on " << move.v << ", n_crossings: " << cur.n_crossings << std::endl;
  }
  return cur;
}