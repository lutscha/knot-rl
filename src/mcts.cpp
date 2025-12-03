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

static double evaluate_knot(const Knot &knot) noexcept { //TODO FIXME
  return -static_cast<double>(knot.n_crossings);
}

static ProbDistribution get_probs(const Knot &knot){
  throw std::runtime_error("Not implemented");
};

AvailableMove mcts_select_move(const Knot &root_knot, std::size_t n_simulations) {
  Node root(std::move(root_knot));

  // Expand root once at the start.
  root.expand(get_probs(root.knot));

  if (root.children.empty()) {
    // No moves available; return some dummy move.
    throw std::runtime_error("No moves available");
  }

  for (std::size_t sim = 0; sim < n_simulations; ++sim) {
    Node *cur = &root;
    std::vector<Node *> path;
    path.reserve(64);
    path.push_back(cur);

    // SELECTION
    while (cur->is_expanded && !cur->children.empty()) {
      Child *child = cur->select_best_child();
      if (!child) {
        break;
      }
      // child->compute() is guaranteed to have been called in select_best_child
      // if needed, so node is non-null here.
      cur = child->node.get();
      path.push_back(cur);
    }

    // EXPANSION
    if (!cur->is_expanded) {
      ProbDistribution prob_distribution = get_probs(cur->knot);
      cur->expand(prob_distribution);
    }

    // EVALUATION
    const double value = evaluate_knot(cur->knot);

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