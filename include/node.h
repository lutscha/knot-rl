#pragma once

#include "../include/knot.h"
#include "../include/visit.h"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

constexpr double c = 0.1;


struct Node;

struct Child {
  AvailableMove move;
  Node *parent;
  std::unique_ptr<Node> node;
  bool is_computed = false;
  double p = 0.0;

  Child(Node *parent, AvailableMove move, double p)
      : move(move), parent(parent), p(p) {};

  uint32_t n_visits() const noexcept;

  // Q-Value: The average value of this node (defined after Node)
  double q_value() const noexcept;

  double puct(double sqrt_parent_visits) const {
    return q_value() + c * p * sqrt_parent_visits / (1.0 + n_visits());
  }

  void compute();
};

struct SharedArena;

struct Node {
  Knot knot;
  std::vector<Child> children;

  uint32_t n_visits = 0;
  bool is_expanded = false;
  double value = 0.0;

  Node(Knot knot) : knot(std::move(knot)) {}

  Child *select_best_child() noexcept {
    if (children.empty()) {
      std::cerr << "No children to select from" << std::endl;
      return nullptr;
    }

    double best_puct = -std::numeric_limits<double>::infinity();
    double sqrt_n_visits = std::sqrt(n_visits);
    Child *best_child = nullptr;
    for (auto &child : children) {
      double puct = child.puct(sqrt_n_visits);
      if (puct > best_puct) {
        best_puct = puct;
        best_child = &child;
      }
    }
    if (!best_child->is_computed) {
      best_child->compute();
    }
    return best_child;
  }

  double expand(SharedArena &arena);
};

uint32_t Child::n_visits() const noexcept {
  if (!is_computed)
    return 0;
  return node->n_visits;
}

double Child::q_value() const noexcept {
  if (!is_computed || node->n_visits == 0)
    return 0.0;
  return node->value / node->n_visits;
}

void Child::compute() {
  node = std::make_unique<Node>(parent->knot.apply_move(move.v, move.move));
  is_computed = true;
}