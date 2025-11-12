#include <array>
#include <stdexcept>
#include <string>

#define uint unsigned int

#define BUMP_UP(a, b) ((a) < (b) ? (a) : ((a) + 1u))
#define BUMP_DOWN(a, b) ((a) <= (b) ? (a) : ((a) - 1u))

enum class Orientation { pos = 1, neg = -1 };

enum class ReidemeisterMove { R1_neg = -1, R1_pos = 1, R2_neg = -2, R2_pos = 2, R3_neg = -3, R3_pos = 3 };

class Crossing {
public:
  static constexpr uint deg = 4;
  static constexpr uint OVER_OUT = 0, OVER_IN = 2, UNDER_OUT = 1, UNDER_IN = 3;
  uint adj[deg];
  Orientation orientation;

  Crossing(uint adj[deg], Orientation orientation) : orientation(orientation) {
    for (uint i = 0; i < deg; i++)
      this->adj[i] = adj[i];
  }

  bool is_loop() const noexcept { return adj[OVER_OUT] == adj[UNDER_IN] || adj[OVER_IN] == adj[UNDER_OUT]; }

  Crossing bump_up(uint k) const noexcept {
    uint new_adj[deg];
    for (uint i = 0; i < deg; i++)
      new_adj[i] = BUMP_UP(adj[i], k);
    return Crossing(new_adj, orientation);
  }

  Crossing bump_down(uint k) {
    uint new_adj[deg];
    for (uint i = 0; i < deg; i++) {
      new_adj[i] = BUMP_DOWN(adj[i], k);
    }
    return Crossing(new_adj, orientation);
  }
};

class Loop : public Crossing {
public:
  using Crossing::Crossing;

  Loop(uint self, uint over, uint under, Orientation orientation) : 
    Crossing(make_adj(self, over, under, orientation).data(), orientation) {}

  std::pair<uint, uint> neighbors() const noexcept {
    if (orientation == Orientation::pos)
      return {adj[OVER_IN], adj[UNDER_OUT]};
    else
      return {adj[OVER_OUT], adj[UNDER_IN]};
  }

private:
  static std::array<uint, Crossing::deg> make_adj(uint self, uint over, uint under, Orientation orientation) noexcept {
    std::array<uint, Crossing::deg> adj{};
    if (orientation == Orientation::pos) {
      adj[OVER_OUT] = adj[UNDER_IN] = self;
      adj[OVER_IN] = over;
      adj[UNDER_OUT] = under;
    } else {
      adj[OVER_IN] = adj[UNDER_OUT] = self;
      adj[OVER_OUT] = over;
      adj[UNDER_IN] = under;
    }
    return adj;
  }
};

template <uint max_crossings> class Link {
  uint n_crossings;
  Crossing crossings[max_crossings];

  inline uint edge_ind(uint v, uint w) const noexcept {
    for (uint i = 0; i < Crossing::deg; i++) {
      if (crossings[v].adj[i] == w)
        return i;
    }
    return n_crossings; // EXCEPTION
  }

  std::array<Crossing, max_crossings> insert_crossing(uint crossing) const {
    Crossing new_crossings[max_crossings];
    for (uint i = 0; i < crossing; i++) {
      new_crossings[i] = bump_up(crossings[i], crossing);
    }
    for (uint i = crossing; i < n_crossings; i++) {
      new_crossings[i + 1] = bump_up(crossings[i], crossing);
    }
    return new_crossings;
  }

  std::array<Crossing, max_crossings>  remove_crossing(uint crossing) const {
    Crossing new_crossings[max_crossings];
    for (uint i = 0; i < crossing; i++) {
      new_crossings[i] = bump_down(crossings[i], crossing);
    }
    for (uint i = crossing + 1; i < n_crossings; i++) {
      new_crossings[i - 1] = bump_down(crossings[i], crossing);
    }
    return new_crossings;
  }

  Link<max_crossings> R1_pos(uint crossing, uint edge, Orientation orientation) const {
    if (this->n_crossings >= max_crossings)
      throw std::runtime_error("Max number of crossings reached");
    if (crossing >= n_crossings)
      throw std::runtime_error("Crossing not found");

    uint v = crossing, w = crossings[v].adj[edge];
    uint u = n_crossings + 1; // TODO: actual crossing number in Dowker notation
    uint v_ = BUMP_UP(v, u), w_ = BUMP_UP(w, u);

    uint back_edge = edge_ind(w, v);

    Crossing new_crossings[max_crossings] = insert_crossing(crossing);

    new_crossings[v_].adj[edge] = u;
    new_crossings[w_].adj[back_edge] = u;
    new_crossings[u] = Loop(u, v_, w_, orientation);
    return Link<max_crossings>(new_crossings, n_crossings + 1);
  }

  Link<max_crossings> R1_neg(uint crossing) const{
    if (!crossings[crossing].is_loop())
      throw std::runtime_error("R1_neg failed: crossing is not a loop");

    uint u = crossing;
    Loop &loop = reinterpret_cast<Loop &>(crossings[u]);
    auto [v, w] = loop.neighbors();
    uint edge_v = edge_ind(v, u), edge_w = edge_ind(w, u);
    uint v_ = BUMP_DOWN(v, u), w_ = BUMP_DOWN(w, u);

    uint new_crossings[max_crossings] = remove_crossing(crossing);

    new_crossings[v_].adj[edge_v] = w;
    new_crossings[w_].adj[edge_w] = v;

    return Link<max_crossings>(new_crossings, n_crossings - 1);
  }

  Link<max_crossings> R2_pos(uint crossing, uint edge, Orientation orientation) const {
    if (this->n_crossings >= max_crossings)
      throw std::runtime_error("Max number of crossings reached");
    if (crossing >= n_crossings)
      throw std::runtime_error("Crossing not found");

    uint v = crossing, w = crossings[v].adj[edge];
    uint u = n_crossings + 1; // TODO: actual crossing number in Dowker notation
  }
};