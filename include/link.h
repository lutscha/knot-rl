#pragma once

#include "hashing.h"
#include "visit.h"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <type_traits>
#include <utility>

#if defined(__GNUC__) && !defined(__clang__)
#define KNOT_PRAGMA_IVDEP _Pragma("GCC ivdep")
#else
#define KNOT_PRAGMA_IVDEP
#endif

static constexpr uint16_t MAX_CROSSINGS = 100;
static constexpr uint16_t MAX_DYNAMIC_COMPONENTS = 100;

struct Vertex {
  uint16_t under[2];
  uint16_t over[2];
  uint16_t flags;

  inline Vertex() : under{0, 0}, over{0, 0}, flags(0) {}

  Vertex(uint16_t under[2], uint16_t over[2], uint16_t flags) : flags(flags) {
    for (uint16_t i = 0; i < 2; i++) {
      this->under[i] = under[i];
      this->over[i] = over[i];
    }
  }
};

template <uint16_t static_n_components> class MoveIterator;

template <uint16_t static_n_components> class Link {
public:
  static constexpr bool dynamic = (static_n_components == 0);

  uint16_t n_crossings;
  uint64_t hash_;
  uint16_t static_n_conn_crossings[static_n_components > 0 ? static_n_components: 1];
  uint16_t dynamic_components_cnt = 0;
  uint16_t *dynamic_n_conn_crossings = nullptr;
  Visit visits[2 * MAX_CROSSINGS];

#define DYNAMIC template <bool D = dynamic, typename = std::enable_if_t<D>>
#define STATIC template <bool D = dynamic, typename = std::enable_if_t<!D>>

  inline uint16_t comp_cnt() const noexcept {
    if constexpr (dynamic)
      return dynamic_components_cnt;
    else
      return static_n_components;
  }

  inline uint16_t *n_conn_crossings() noexcept {
    if constexpr (dynamic)
      return dynamic_n_conn_crossings;
    else
      return static_n_conn_crossings;
  }

  inline const uint16_t *n_conn_crossings() const noexcept {
    if constexpr (dynamic)
      return dynamic_n_conn_crossings;
    else
      return static_n_conn_crossings;
  }

  inline uint16_t &n_conn_crossings(uint16_t c) noexcept {
    if constexpr (dynamic)
      return dynamic_n_conn_crossings[c];
    else
      return static_n_conn_crossings[c];
  }

  inline const uint16_t &n_conn_crossings(uint16_t c) const noexcept {
    if constexpr (dynamic)
      return dynamic_n_conn_crossings[c];
    else
      return static_n_conn_crossings[c];
  }

private:
  static inline int16_t N_CROSSING_CHANGE(ReidemeisterKind kind) noexcept {
    switch (kind) {
    case ReidemeisterKind::R1_neg:
      return -1;
    case ReidemeisterKind::R1_pos:
      return 1;
    case ReidemeisterKind::R2_neg:
      return -2;
    case ReidemeisterKind::R2_pos:
      return 2;
    case ReidemeisterKind::R3:
      return 0;
    case ReidemeisterKind::R4_pos:
      return 2;
    }
    return 0;
  }

  inline uint16_t new_n_crossings(ReidemeisterKind kind) const {
    int32_t result = int32_t(n_crossings) + N_CROSSING_CHANGE(kind);
    return uint16_t(result);
  }

  inline uint16_t mate(uint16_t a) const noexcept { return visits[a].mate; }

  inline uint16_t flags(uint16_t a) const noexcept { return visits[a].flags; }

  inline bool are_adjacent(uint16_t a, uint16_t b) const noexcept {
    return next(a) == b || prev(a) == b;
  }

  inline std::pair<uint16_t, uint16_t> get_comp(uint16_t a) const noexcept {
    if (comp_cnt() == 1)
      return {0, a};

    uint16_t pref = 0;
    for (uint16_t j = 0; j < comp_cnt(); j++) {
      if (a < 2 * (pref + n_conn_crossings(j)))
        return {j, a - 2 * pref};
      pref += n_conn_crossings(j);
    }
    return {0, 0}; // unknot
  }

  inline uint16_t next(uint16_t a) const noexcept {
    auto [comp, comp_ind] = get_comp(a);
    return comp_ind + 1 < 2 * n_conn_crossings(comp)
               ? a + 1
               : a + 1 - 2 * n_conn_crossings(comp);
  }

  inline uint16_t prev(uint16_t a) const noexcept {
    auto [comp, comp_ind] = get_comp(a);
    return comp_ind > 0 ? a - 1 : a + 2 * n_conn_crossings(comp) - 1;
  }

  inline uint16_t iter(uint16_t a, Direction dir) const noexcept {
    return dir == Direction::next ? next(a) : prev(a);
  }

  static inline uint16_t looping_index(uint16_t added, uint16_t b) noexcept {
    const uint16_t shift = 2 * (b >= added);
    return b + shift;
  }

  static inline uint16_t delooping_index(const uint16_t removed1,
                                         const uint16_t removed2,
                                         const uint16_t a) noexcept {
    uint16_t shift = (a > removed1) + (a > removed2);
    return a - shift;
  }

  static inline uint16_t poking_index(const uint16_t added_min, const uint16_t added_max,const uint16_t c) noexcept {
    const uint16_t shift = 2 * (c >= added_min) + 2 * (c + 2 >= added_max);
    return c + shift;
  }

  template <uint16_t n_excluded>
  static inline uint16_t depoking_index(const uint16_t *excluded,
                                        const uint16_t c) noexcept {
    uint16_t shift = 0;
    for (uint16_t i = 0; i < n_excluded; i++)
      if (c > excluded[i]) [[unlikely]]
        shift++;
    return c - shift;
  }

  inline void defrag_comps() noexcept {
    uint16_t shift = 0, comp_cnt_ = comp_cnt();
    for (uint16_t i = 0; i < comp_cnt_; i++) {
      if (n_conn_crossings(i) == 0) [[unlikely]] {
        shift++;
      } else if (shift > 0)
        n_conn_crossings(i - shift) = n_conn_crossings(i);
    }
    for (uint16_t i = comp_cnt() - shift; i < comp_cnt(); i++)
      n_conn_crossings(i) = 0;
  }

  inline void inc_comp(uint16_t a) noexcept {
    auto [comp, comp_ind] = get_comp(a);
    n_conn_crossings(comp)++;
  }

  inline bool inc_comps(uint16_t a, uint16_t b) noexcept {
    auto [comp_a, comp_ind_a] = get_comp(a);
    auto [comp_b, comp_ind_b] = get_comp(b);
    n_conn_crossings(comp_a)++;
    n_conn_crossings(comp_b)++;
    return comp_a == comp_b;
  }

  inline void dec_comp(uint16_t a) noexcept {
    auto [comp, comp_ind] = get_comp(a);
    n_conn_crossings(comp)--;

    if (n_conn_crossings(comp) == 0)
      defrag_comps();
  }

  inline bool dec_comps(uint16_t a, uint16_t b) noexcept {
    auto [comp_a, comp_ind_a] = get_comp(a);
    auto [comp_b, comp_ind_b] = get_comp(b);
    n_conn_crossings(comp_a)--;
    n_conn_crossings(comp_b)--;

    if (n_conn_crossings(comp_a) == 0 || n_conn_crossings(comp_b) == 0)
      defrag_comps();

    return comp_a == comp_b;
  }

  inline bool is_loop(uint16_t a) const noexcept {
    return are_adjacent(a, mate(a));
  }

  inline bool is_bigon(uint16_t a1) const noexcept {
    uint16_t a2 = next(a1);
    const Visit &A1 = visits[a1];
    const Visit &A2 = visits[a2];
    bool res = (A1.type() == VisitType::over && A2.type() == VisitType::over) &&
               a1 != a2 && are_adjacent(mate(a1), mate(a2));

    return res;
  }

  inline bool is_triangle(uint16_t a_c, Direction dir_a, Direction dir_c) const noexcept {
    const uint16_t flag_ac = visits[a_c].crossing_flags();
    const uint16_t c_a = visits[a_c].mate; // flag_b: A -> C
    const uint16_t a_b = iter(a_c, dir_a);
    const uint16_t c_b = iter(c_a, dir_c);
    const uint16_t b_c = visits[c_b].mate;
    const uint16_t flag_cb = visits[c_b].crossing_flags(); // flag_a: C -> B
    const uint16_t flag_ab = visits[a_b].crossing_flags(); // flag_c: A -> B
    const uint16_t b_a = visits[a_b].mate;

    return Visit::GET_TYPE(flag_cb) == VisitType::under &&
           Visit::GET_TYPE(flag_ac) == VisitType::over &&
           Visit::GET_TYPE(flag_ab) == VisitType::over &&
           are_adjacent(b_a, b_c) && a_c != a_b && a_c != b_c && a_b != b_c;
  }

  inline bool is_R4(uint16_t a, Direction dir, VisitType type) const noexcept {
    if (type == VisitType::over) return true;

    const uint16_t a_b = mate(a);
    const uint16_t a_c = iter(a_b, dir);
    return visits[a_c].type() == VisitType::under;
  }

  uint16_t compute_moves(uint16_t a) const noexcept {
    if (visits[a].type() == VisitType::under)
      return 0;

    uint16_t res = is_loop(a) << Visit::R1_NEG_SHIFT | is_bigon(a) << Visit::R2_NEG_SHIFT;
    for (auto dir_a : {Direction::next, Direction::prev}) {
      for (auto dir_c : {Direction::next, Direction::prev}) {
        res |= is_triangle(a, dir_a, dir_c) << (Visit::R3_ARG_SHIFT(dir_a, dir_c));
      }
    }

    for (auto dir : {Direction::next, Direction::prev}) {
      for (auto type : {VisitType::over, VisitType::under}) {
        res |= is_R4(a, dir, type) << (Visit::R4_ARG_SHIFT(dir, type));
      }
    }
    return res;
  }

  void bind(uint16_t a, uint16_t b, uint16_t crossing_flags) {
    visits[a] = Visit(b, crossing_flags);
    visits[b] = Visit(a, Visit::FLIP(crossing_flags));
  }

  bool is_valid_move(uint16_t a, ReidemeisterMove move) const noexcept {
    if (a >= 2 * n_crossings) [[unlikely]] {
      if (n_crossings == 0) {
        if (move.kind != ReidemeisterKind::R1_pos) {
          std::cerr << "Invalid move: only positive R1 is allowed on unknots, "
                       "received "
                    << move << std::endl;
          return false;
        }
        if (a != 0) {
          std::cerr
              << "Invalid move: R1+ on an unknot should have index 0, received "
              << a << std::endl;
          return false;
        }
        return true;
      }
      std::cerr << "Invalid move: " << a << " is out of bounds for "
                << 2 * n_crossings << " crossings" << std::endl;
      return false;
    }

    switch (move.kind) {
    case ReidemeisterKind::R1_neg:
      return visits[a].is_loop();
    case ReidemeisterKind::R1_pos:
      return true;
    case ReidemeisterKind::R2_neg:
      return visits[a].is_bigon();
    case ReidemeisterKind::R2_pos:
      return true;
    case ReidemeisterKind::R3:
      return visits[a].is_triangle(move.dir_over(), move.dir_under());
    case ReidemeisterKind::R4_pos:
      return visits[a].is_R4(move.dir(), move.type());
    default:
      return false;
    }
  }

  bool
  is_valid_R2(uint16_t a, uint16_t b, const ReidemeisterMove &move) const noexcept { // non_local R2+
    std::cerr << "Not implemented: is_valid_R2 for " << a << " and " << b << " with args " << move << std::endl;
    return get_comp(a).second > 0 && get_comp(b).second > 0 &&
           false; // TODO this is not enough
  }

  bool is_valid_R2(uint16_t a, const ReidemeisterMove &move) const noexcept {
    return true; // TODO: compute actual value
  }

  void look_for_bigon(uint16_t a, uint16_t b) {
    std::cerr << "Not implemented: look_for_bigon for " << a << " and " << b
              << std::endl;
  }

  void audit_bigon(uint16_t a, uint16_t b) {
    std::cerr << "Not implemented: audit_bigon for " << a << " and " << b
              << std::endl;
  }

  void look_for_triangles(uint16_t a, uint16_t b) {
    std::cerr << "Not implemented: look_for_triangles for " << a << " and " << b
              << std::endl;
  }

  void audit_triangles(uint16_t a, uint16_t b) {
    std::cerr << "Not implemented: audit_triangles for " << a << " and " << b
              << std::endl;
  }

  void compute_all_moves() noexcept {
    for (uint16_t a = 0; a < 2 * n_crossings; a++) {
      visits[a].flags = (visits[a].flags & ~Visit::MOVES_MASK) | compute_moves(a);
    }
  }

public: // constructors
  STATIC Link(const uint16_t n_crossings, const uint16_t *n_conn_crossings, Visit *visits_)
      : n_crossings(n_crossings) {
    std::memcpy(static_n_conn_crossings, n_conn_crossings,comp_cnt() * sizeof(uint16_t));
    std::memcpy(visits, visits_, 2 * n_crossings * sizeof(Visit));
    compute_all_moves();
    set_hash();
  }

  DYNAMIC Link(const uint16_t n_crossings, uint16_t components_cnt, const uint16_t *n_conn_crossings, Visit *visits_)
      : n_crossings(n_crossings), dynamic_components_cnt(components_cnt),
        dynamic_n_conn_crossings(
            reinterpret_cast<uint16_t *>(0)) { // TODO: fix this

    std::memcpy(dynamic_n_conn_crossings, n_conn_crossings,dynamic_components_cnt * sizeof(uint16_t));
    std::memcpy(visits, visits_, 2 * n_crossings * sizeof(Visit));
    compute_all_moves();
    set_hash();
  }

  Link<static_n_components> apply_move(uint16_t v,  ReidemeisterMove move) const {
    if (n_crossings == 0) [[unlikely]] {
      if (move.kind != ReidemeisterKind::R1_pos || v != 0) {
        std::cerr << "Invalid move: only positive R1 with v=0 is allowed on "
                     "unknots, received "
                  << move << " on " << v << std::endl;
        return *this;
      }
      if constexpr (dynamic) {
        return Link<static_n_components>(*this, 0, move, nullptr);
      } else {
        return Link<static_n_components>(*this, 0, move);
      }
    }

    bool under = (move.type() == VisitType::under) &&
                 (move.kind == ReidemeisterKind::R2_pos);
    const uint16_t a =
        (visits[2 * v].type() == VisitType::over) ^ under ? 2 * v : mate(2 * v);

#ifdef DEBUG
    if (!is_valid_move(a, move)) [[unlikely]] {
      std::cerr << "Invalid move: " << move << " at " << a << std::endl;
      throw std::runtime_error("Invalid move");
    }
#endif

    if constexpr (dynamic) {
      return Link<static_n_components>(*this, a, move, nullptr);
    } else {
      return Link<static_n_components>(*this, a, move);
    }
  }

  STATIC Link(const Link<static_n_components> &link, uint16_t a, const ReidemeisterMove &move)
      : n_crossings(link.new_n_crossings(move.kind)), visits() {

    std::memcpy(static_n_conn_crossings, link.static_n_conn_crossings,
                static_n_components * sizeof(uint16_t));

    switch (move.kind) {
    case ReidemeisterKind::R1_neg:
      R1_neg(link, a);
      break;
    case ReidemeisterKind::R1_pos:
      R1_pos(link, a, move.sign(), move.type());
      break;
    case ReidemeisterKind::R2_neg:
      R2_neg(link, a);
      break;
    case ReidemeisterKind::R2_pos:
      R2_pos(link, a, move.type(), move.dir_over(), move.dir_under());
      break;
    case ReidemeisterKind::R3:
      R3(link, a, move.dir_over(), move.dir_under());
      break;
    case ReidemeisterKind::R4_pos:
      R4_pos(link, a, move.dir(), move.type());
      break;
    }
    set_hash();
  }

  DYNAMIC Link(const Link<static_n_components> &link, uint16_t a,const ReidemeisterMove &move, uint16_t *storage)
      : n_crossings(link.new_n_crossings(move.kind)),
        dynamic_components_cnt(link.dynamic_components_cnt),
        dynamic_n_conn_crossings(storage), visits() {

    std::memcpy(dynamic_n_conn_crossings, link.dynamic_n_conn_crossings,
                dynamic_components_cnt * sizeof(uint16_t));

    switch (move.kind) {
    case ReidemeisterKind::R1_neg:
      R1_neg(link, a);
      break;
    case ReidemeisterKind::R1_pos:
      R1_pos(link, a, move.sign(), move.type());
      break;
    case ReidemeisterKind::R2_neg:
      R2_neg(link, a);
      break;
    case ReidemeisterKind::R2_pos:
      R2_pos(link, a, move.type(), move.dir_over(), move.dir_under());
      break;
    case ReidemeisterKind::R3:
      R3(link, a, move.dir_over(), move.dir_under());
      break;
    case ReidemeisterKind::R4_pos:
      R4_pos(link, a, move.dir(), move.type());
      break;
    }
    set_hash();
  }


  // given that you know the flags of a-b and a-c, return the sign of a-c 
  static Orientation third_sign(uint16_t flag_ab, uint16_t flag_ac) noexcept {

  }

  void R1_neg(const Link<static_n_components> &link,
              const uint16_t a) noexcept {

    const uint16_t a0 = link.mate(link.prev(a)) == a ? link.prev(a) : a;
    const uint16_t a1 = link.next(a0);

    const bool wrapping = a1 != a0 + 1;
    auto deloop = [a0, a1, &link](uint16_t i) {
      return Visit(delooping_index(a0, a1, link.visits[i].mate), link.visits[i].flags);
    };

    uint16_t prev_ = delooping_index(a0, a1, link.prev(a0));
    uint16_t next_ = delooping_index(a0, a1, link.next(a1));

    dec_comp(a0);

    KNOT_PRAGMA_IVDEP
    if (!wrapping) {
      for (uint16_t i = 0; i < a0; i++)
        visits[i] = deloop(i);
      for (uint16_t i = a0 + 2; i < 2 * link.n_crossings; i++)
        visits[i - 2] = deloop(i);
    } else {
      for (uint16_t i = 0; i < a1; i++)
        visits[i] = deloop(i);
      for (uint16_t i = a1 + 1; i < a0; i++)
        visits[i - 1] = deloop(i);
      for (uint16_t i = a0 + 1; i < 2 * link.n_crossings; i++)
        visits[i - 2] = deloop(i);
    }

    if (mate(prev_) == next_) [[unlikely]] {
      uint16_t over_ = visits[prev_].type() == VisitType::over ? prev_ : next_;
      visits[over_].flags |= Visit::R1_NEG;
      if (prev_ == over_)
        visits[prev_].flags &= ~Visit::R2_NEG;
    } else {
      // look_for_bigon(prev_, next_);
      // look_for_triangles(prev_, next_);
    }

    compute_all_moves(); // TODO: replace with a more efficient implementation

#ifdef DEBUG
    verify_invariant();
#endif
  }

  void R1_pos(const Link<static_n_components> &link, const uint16_t a,
              Orientation sign, VisitType type) noexcept { // loop (a, a+1)
    auto loop = [a, &link](uint16_t i) {
      return Visit(looping_index(a, link.visits[i].mate), link.visits[i].flags);
    };

    inc_comp(a);

    KNOT_PRAGMA_IVDEP
    for (uint16_t i = 0; i < a; i++)
      visits[i] = loop(i);

    for (uint16_t i = a; i < 2 * link.n_crossings; i++)
      visits[i + 2] = loop(i);

    bind(a, a + 1, Visit::FLAG(sign, type));

    // const uint16_t prev_ = prev(a), next_ = next(a + 1);

    // audit_bigon(prev_, next_);
    // audit_triangles(prev_, next_);
    // look_for_bigon(prev_, a);

    // const uint16_t over_visit = Visit::GET_TYPE(crossing_flags) ==
    // VisitType::over ? a : a + 1; visits[over_visit].flags |= Visit::R1_NEG;

    compute_all_moves(); // TODO: replace with a more efficient implementation

#ifdef DEBUG
    verify_invariant();
#endif
  }

  void R2_neg(const Link<static_n_components> &link, uint16_t a1) noexcept {
    uint16_t b1 = link.mate(a1);
    uint16_t a2 = link.next(a1);
    uint16_t b2 = link.mate(a2);

    uint16_t excluded[4] = {a1, b1, a2, b2};

    dec_comps(a1, a2);

    auto depoke = [&link, &excluded](uint16_t i) {
      return Visit(depoking_index<4>(excluded, link.visits[i].mate), link.visits[i].flags);
    };

    std::sort(excluded, excluded + 4);

    KNOT_PRAGMA_IVDEP
    uint16_t shift = 0;
    for (uint16_t i = 0; i < 2 * link.n_crossings; i++) {
      if (shift < 4 && i == excluded[shift]) [[unlikely]] {
        shift++;
      } else [[likely]]
        visits[i - shift] = depoke(i);
    }

    compute_all_moves(); // TODO: replace with a more efficient implementation
  }

  void R2_pos(const Link<static_n_components> &link, uint16_t a, uint16_t b, bool collinear, Orientation sign) noexcept { // a is over
    const uint16_t min_ = std::min(a, b);
    const uint16_t max_ = std::max(a, b);

    auto poke = [min_, max_, &link](uint16_t i) {
      return Visit(poking_index(min_, max_, link.visits[i].mate), link.visits[i].flags);
    };

    uint16_t b1 = b, b2 = b + 1;
    if (!collinear)
      std::swap(b1, b2);

    inc_comps(std::min(a, b), std::max(a, b) - 2);

    KNOT_PRAGMA_IVDEP

    for (uint16_t i = 0; i < min_; i++)
      visits[i] = poke(i);

    for (uint16_t i = min_; i < max_ - 2; i++)
      visits[i + 2] = poke(i);

    for (uint16_t i = max_ - 2; i < 2 * link.n_crossings; i++)
      visits[i + 4] = poke(i);

    const uint16_t flag = Visit::FLAG(sign, VisitType::over);

    bind(a, b1, flag);
    bind(a + 1, b2, Visit::MIRROR(flag));

    compute_all_moves(); // TODO: replace with a more efficient implementation

#ifdef DEBUG
    verify_invariant();
#endif
  }

  std::pair<uint16_t, uint16_t>
  R2_insertion_indices(const uint16_t a0, const Direction dir) const noexcept {
    if (dir == Direction::next) {
      return {a0 + 1, a0 + 2};
    } else {
      const uint16_t a3 = prev(a0);
      return {a3 + 2, a3 + 1};
    }
  }

  void R2_pos(const Link<static_n_components> &link, uint16_t a0,  VisitType type, Direction dir_over, Direction dir_under) noexcept { // local R2+

    if (type == VisitType::under) {
      a0 = mate(a0);
    }

    uint16_t b0 = link.mate(a0);
    auto [a1, a2] = link.R2_insertion_indices(a0, dir_over);
    auto [b1, b2] = link.R2_insertion_indices(b0, dir_under);

    if (a1 > b1) {
      a1 += 2, a2 += 2;
    } else {
      b1 += 2, b2 += 2;
    }

    bool collinear = (dir_over == dir_under);
    Orientation sign = ((link.visits[a0].sign() == Orientation::pos) ^
                        (link.visits[a0].type() == VisitType::over) ^
                        (dir_over == Direction::prev)) == 0
                           ? Orientation::pos
                           : Orientation::neg;

    R2_pos(link, std::min(a1, a2), std::min(b1, b2), collinear, sign);
  }

  void R3(const Link<static_n_components> &link, const uint16_t a_c,  Direction dir_a, Direction dir_c) noexcept { // a_b, a_c and b_a are OVER
    std::memcpy(visits, link.visits, 2 * n_crossings * sizeof(Visit));

    uint16_t flag_b = visits[a_c].crossing_flags();
    uint16_t c_a = mate(a_c); // flag_b: A -> C
    uint16_t a_b = iter(a_c, dir_a);
    uint16_t c_b = iter(c_a, dir_c);
    uint16_t flag_a = visits[c_b].crossing_flags(); // flag_a: C -> B
    uint16_t b_c = mate(c_b);
    uint16_t flag_c = visits[a_b].crossing_flags(); // flag_c: A -> B
    uint16_t b_a = mate(a_b);

    bind(c_a, b_a, flag_a);
    bind(a_b, c_b, flag_b);
    bind(a_c, b_c, flag_c);

    compute_all_moves(); // TODO: replace with a more efficient implementation

#ifdef DEBUG
    verify_invariant();
#endif
  }

  void R4_pos(const Link<static_n_components> &link, const uint16_t a, Direction dir, VisitType type) noexcept {
    const uint16_t a_b = type == VisitType::over ? a : mate(a);
    const uint16_t a_c = iter(a_b, dir);
    const uint16_t b = mate(a_b);
    const uint16_t c = mate(a_c);

    const uint16_t min_ = std::min(b, c) + 1, max_ = std::max(b, c) + 1;

    auto poke_index = [min_, max_](uint16_t i) {return poking_index(min_, max_, i);};

    auto poke_visit = [min_, max_, poke_index, &link](uint16_t i) {
      return Visit(poke_index(i), link.visits[i].flags);
    };

    const uint16_t new_b = poking_index(min_, max_, b);
    const uint16_t new_c = poking_index(min_, max_, c);

    inc_comps(min_, max_ - 2);

    KNOT_PRAGMA_IVDEP

    for (uint16_t i = 0; i < min_; i++)
      visits[i] = poke_visit(i);

    for (uint16_t i = min_; i < max_ - 2; i++)
      visits[i + 2] = poke_visit(i);

    for (uint16_t i = max_ - 2; i < 2 * link.n_crossings; i++)
      visits[i + 4] = poke_visit(i);

    uint16_t ba_flag = link.visits[b].crossing_flags();
    uint16_t ca_flag = link.visits[c].crossing_flags();

    const uint16_t b_up_shift = 0;
    const uint16_t c_up_shift = 0;
    uint16_t b_down_shift = 2 - b_up_shift;
    uint16_t c_down_shift = 2 - c_up_shift;

    uint16_t up_flag = Visit::FLAG(third_sign(Visit::FLIP(ba_flag), Visit::FLIP(ba_flag)),VisitType::over);
    uint16_t down_flag = Visit::MIRROR(up_flag);

    bind(new_b + b_up_shift, new_c + c_up_shift, up_flag);
    bind(new_b + b_down_shift, new_c + c_down_shift, down_flag);

    bind(poke_index(a_c), new_b + 1, Visit::FLIP(ba_flag));
    bind(poke_index(a_b), new_c + 1, Visit::FLIP(ca_flag));

#ifdef DEBUG
    verify_invariant();
#endif

    //FIXME: finish dealing with the signs
  }

  void to_dowker(int16_t *out) const noexcept {
    for (uint16_t vertex = 0; vertex < n_crossings; vertex++) {
      int16_t sgn = -pm(visits[2 * vertex].type());
      out[vertex] = sgn * (mate(2 * vertex) + 1);
    }
  }

  inline uint16_t vertex_index(uint16_t a) const noexcept {
    return a % 2 == 0 ? a / 2 : mate(a) / 2;
  }

  inline uint16_t vertex_flags(uint16_t a) const noexcept {
    return visits[a].type() == VisitType::over ? visits[a].flags
                                               : visits[mate(a)].flags;
  }

  void to_graph(Vertex *out) const noexcept {

    for (uint16_t a = 0; a < 2 * n_crossings; a += 2) {
      Vertex &v = out[vertex_index(a)];

      v.over[0] = vertex_index(next(a));
      v.over[1] = vertex_index(prev(a));
      v.under[0] = vertex_index(next(mate(a)));
      v.under[1] = vertex_index(prev(mate(a)));
      v.flags = vertex_flags(a);
    }
  }

  uint16_t visits_until_mate(uint16_t a, uint16_t comp_size) const noexcept {
    uint16_t b = mate(a);
    return b >= a ? (b - a + 1) / 2 : (2 * comp_size + b - a + 1) / 2;
  }

  uint64_t hash() const noexcept {
    uint64_t hash = 0;
    uint16_t pref = 0;
    for (uint16_t i = 0; i < comp_cnt(); i++) {
      uint16_t comp_size = n_conn_crossings(i);
      auto fun = [this, pref, comp_size](uint16_t k) {
        uint16_t a = k + 2 * pref;
        uint16_t n_visits = visits_until_mate(a, comp_size);
        return static_cast<int64_t>(pm(visits[a].type())) *
               static_cast<int64_t>(n_visits);
      };

      hash ^= circular_hash<decltype(fun)>(fun, 2 * comp_size);
      pref += comp_size;
    }
    return hash;
  }

  uint64_t set_hash() noexcept { return hash_ = hash(); }

  using move_iterator = MoveIterator<static_n_components>;

  inline move_iterator moves_begin() const noexcept {
    return move_iterator::begin(*this);
  }

  inline move_iterator moves_end() const noexcept {
    return move_iterator::end(*this);
  }

  // ice range for `for (auto m : link.moves())`
  struct MoveRange {
    const Link &link;
    move_iterator begin() const noexcept { return move_iterator::begin(link); }
    move_iterator end() const noexcept { return move_iterator::end(link); }
  };

  MoveRange moves() const noexcept { return MoveRange{*this}; }

std::pair<bool, std::string> verify_invariant() const noexcept {
    std::ostringstream res;
    bool violated = false;
    for (uint16_t a = 0; a < 2 * n_crossings; a++) {
      const uint16_t b = mate(a);
      uint16_t flags_a = flags(a), flags_b = flags(b);
      if (visits[a].type() == VisitType::under && visits[a].has_move()) {
        res << "Under visit " << a << " has moves flags " << int(flags_a)
            << " (mate " << int(flags_b) << ")\n";
        violated = true;
      }

      if ((a + b) % 2 == 0) {
        res << "Jordan's lemma violation: " << a << " and " << b
            << " are mates\n";
        violated = true;
      }

      if ((visits[a].crossing_flags() ^ visits[b].crossing_flags()) !=
          Visit::TYPE) {
        res << "Crossing flags of " << a << " and " << b
            << " do not match: " << int(visits[a].crossing_flags()) << " ^ "
            << int(visits[b].crossing_flags()) << "\n";
        violated = true;
      }

      if (mate(b) != a) {
        res << "Not an involution " << a << " -> " << b << " -> " << mate(b)
            << "\n";
        violated = true;
      }

      if (visits[a].moves_flags() != compute_moves(a)) {
        res << "Wrong moves for " << a << ": " << int(visits[a].moves_flags())
            << " != " << int(compute_moves(a)) << "\n";
        violated = true;
      }
    }

    uint16_t sum_comp = 0;
    for (uint16_t j = 0; j < comp_cnt(); j++) {
      sum_comp += n_conn_crossings(j);
    }
    if (sum_comp != n_crossings) {
      res << "Sum of components is not equal to n_crossings: " << sum_comp
          << " != " << n_crossings << "\n";
      violated = true;
    }

    if (violated) [[unlikely]] {
      std::cerr << res.str();
      throw std::runtime_error(res.str());
    }

    return {violated, res.str()};
  }

  void print_dowker() const noexcept {
    static int16_t dowker[MAX_CROSSINGS];
    this->to_dowker(dowker);
    std::cout << "DT: [";
    for (uint16_t i = 0; i < this->n_crossings; i++) {
      std::cout << dowker[i];
      if (i < this->n_crossings - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]\n";
  }

  void print_state() const noexcept {
    std::cout << this->n_crossings << " crossings: ";
    for (uint16_t i = 0; i < 2 * this->n_crossings; i++) {
      if (this->visits[i].type() == VisitType::under)
        continue;
      std::cout << "(" << i << ", " << this->visits[i].mate << ", "
                << int16_t(this->visits[i].flags) << ") ";
    }
    std::cout << "hash: " << this->hash_ << "\n" << std::flush;
  }
};

#undef DYNAMIC
#undef STATIC

template <uint16_t static_n_components> class MoveIterator {

  MoveIterator(const Link<static_n_components> &link, uint16_t a, uint16_t v, uint16_t bit, uint16_t moves_flags)
      : link(link), a(a), v(v), bit(bit), moves_flags(moves_flags) {}

public:
  const Link<static_n_components> &link; // which Link we’re iterating
  uint16_t a;                            // visit index [0 .. 2*n_crossings)
  uint16_t v;
  uint16_t bit;
  uint16_t moves_flags;

  AvailableMove operator*() const noexcept {
    return AvailableMove(v, Visit::GET_DIRECT_MOVE(bit));
  }

  MoveIterator &operator++() noexcept {
    if (moves_flags >> (bit + 1)) {
      auto casted = static_cast<unsigned>(moves_flags >> (bit + 1));
      bit = std::countr_zero(casted) + bit + 1;
    } else {
      do {
        ++a;
      } while (a < 2 * link.n_crossings && !link.visits[a].has_move());

      if (a < 2 * link.n_crossings) [[likely]] {
        moves_flags = link.visits[a].moves_flags();
        bit = std::countr_zero(moves_flags);
        v = link.vertex_index(a);
      } else {
        moves_flags = 0;
        v = link.n_crossings;
        bit = Visit::MOVES_LAST_BIT;
      }
    }
    return *this;
  }

  bool operator==(const MoveIterator &other) const noexcept {
    return a == other.a && bit == other.bit && &link == &other.link;
  }

  bool operator!=(const MoveIterator &other) const noexcept {
    return !(*this == other);
  }

  static MoveIterator begin(const Link<static_n_components> &link) noexcept {
    if (link.n_crossings == 0) [[unlikely]]
      return end(link);

    MoveIterator it(link, 0, link.vertex_index(0), 0, link.visits[0].moves_flags());
    if ((it.moves_flags & (uint16_t(1) << it.bit)) == 0) [[likely]]
      ++it;
    return it;
  }

  static MoveIterator end(const Link<static_n_components> &link) noexcept {
    return MoveIterator(link, 2 * link.n_crossings, link.n_crossings,
                        Visit::MOVES_LAST_BIT, 0);
  }
};

using Link_ = Link<0>;
