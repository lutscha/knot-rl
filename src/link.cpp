// optimize for knots

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

typedef uint16_t uint;

enum class Orientation {
    pos = 0,
    neg = 1
};

enum class CrossingType {
    over = 0,
    under = 1
};

constexpr CrossingType operator!(CrossingType b) noexcept {
  return (b == CrossingType::under) ? CrossingType::over : CrossingType::under;
}

struct Vertex {
    uint under[2];
    uint over[2];

    Vertex(uint under[2], uint over[2]) {
        for (uint i = 0; i < 2; i++) {
            this->under[i] = under[i];
            this->over[i] = over[i];
        }
    }
};


template <uint static_n_components> class Link {
public:
  const uint dynamic_n_components = 0;
  uint n_crossings;
  uint *mate;
  uint *n_comp_crossings;

  static constexpr bool dynamic = (static_n_components == 0);

  template <bool D = dynamic, typename = std::enable_if_t<D>>
  Link(uint dynamic_n_components) : dynamic_n_components(dynamic_n_components) {}

  template <bool D = dynamic, typename = std::enable_if_t<!D>> 
  Link(){};

private:
  static constexpr uint16_t TYPE_SHIFT = (sizeof(uint) * 8 - 1);
  static constexpr uint16_t SIGN_SHIFT = (sizeof(uint) * 8 - 2);
  static constexpr uint TYPE = uint(1) << TYPE_SHIFT;
  static constexpr uint SIGN = uint(1) << SIGN_SHIFT;
  static constexpr uint FLAG_MASK = TYPE | SIGN;
  static constexpr uint INDEX_MASK = FLAG_MASK; //TODO it should the bit flip of FLAG

  static inline uint FLAG(Orientation sign, CrossingType type) noexcept {
    return (uint) sign << SIGN_SHIFT | (uint) type << TYPE_SHIFT;
  }


  static inline uint FLAGGED(uint a, uint flag) noexcept {return a | flag;}

  static inline Orientation GET_SIGN(uint a) noexcept { return Orientation ((a >> SIGN_SHIFT) & 1); }
  static inline CrossingType GET_TYPE(uint a) noexcept { return CrossingType ((a >> TYPE_SHIFT) & 1);}

  static inline uint FLIP(uint a) noexcept { return a ^ TYPE; }
  static inline uint MIRROR(uint a) noexcept { return a ^ SIGN; }

  static inline uint ABS(uint a) noexcept { return a & ~FLAG_MASK; }
  inline uint abs_mate(uint a) const noexcept { return ABS(mate[a]); }

  inline uint LOOP_FLAG(Orientation sign){ return FLAG(sign, (CrossingType) sign);}

  inline uint n_comp() const noexcept {
    return dynamic ? dynamic_n_components : static_n_components;
  }


  inline std::pair<uint, uint> unflag(uint a) const noexcept {
    return {GET_FLAG(mate[a]), ABS(mate[a])};
  }

  inline bool are_adjacent(uint a, uint b) const noexcept {
    return next(a) == b || prev(a) == b;
  }

  inline std::pair<uint, uint> get_comp(uint a) const noexcept {
    if (n_comp() == 1){
        return {0, a};
    }

    uint pref = 0;
    for (uint j = 0; j < n_comp(); j++) {
      if (a < 2 * (pref + n_comp_crossings[j]))
        return {j, a - 2 * pref};
      pref += n_comp_crossings[j];
    }
    return {n_comp(), n_comp()}; // EXCEPTION
  }

  inline uint next(uint a) const noexcept {
    auto [comp, comp_ind] = get_comp(a);
    return comp_ind + 1 < 2 * n_comp_crossings[comp]
               ? a + 1
               : a + 1 - 2 * n_comp_crossings[comp];
  }

  inline uint prev(uint a) const noexcept { 
    auto [comp, comp_ind] = get_comp(a);
    return comp_ind > 0
               ? a - 1
               : a + 2 * n_comp_crossings[comp] - 1;
  }

  inline uint looping_index(uint a, uint b_) const noexcept { // a is unsigned, b_ is signed
    return ABS(b_) < a ? b_ : b_ + 2;
  } 

  inline uint delooping_index(uint a, uint b_) const noexcept { // a is unsigned, b_ is signed
    return ABS(b_) < a ? b_ : b_ - 2;
  } 

  inline uint poking_index(uint min_, uint max_, uint c_)
      const noexcept { // min_ and max_ are unsigned, c_ is signed
    if (ABS(c_) < min_)
      return c_;
    if (ABS(c_) > max_)
      return c_ + 4;
    return c_ + 2;
  }

  inline uint depoking_index(uint min_, uint max_, uint c_)
      const noexcept { // min_ and max_ are unsigned, c_ is signed
    if (ABS(c_) < min_)
      return c_;
    if (ABS(c_) > max_ + 2)
      return c_ - 4;
    return c_ - 2;
  }

  inline void inc_comp(uint a) noexcept {
    auto [comp, comp_ind] = get_comp(a);
    n_comp_crossings[comp]++;
  }

  inline void inc_comps(uint a, uint b) noexcept {
    auto [comp_a, comp_ind_a] = get_comp(a);
    auto [comp_b, comp_ind_b] = get_comp(b);
    n_comp_crossings[comp_a]++;
    n_comp_crossings[comp_b]++;
  }

  inline void dec_comp(uint a) noexcept {
    auto [comp, comp_ind] = get_comp(a);
    n_comp_crossings[comp]--;
  }

  inline void dec_comps(uint a, uint b) noexcept {
    auto [comp_a, comp_ind_a] = get_comp(a);
    auto [comp_b, comp_ind_b] = get_comp(b);
    n_comp_crossings[comp_a]--;
    n_comp_crossings[comp_b]--;
  }

  inline std::pair<uint, uint> comp_boundaries(uint a) const noexcept {
    return {-1, -1}; //TODO: implement
  }

  inline bool is_loop(uint a) const noexcept {
    return a < 2 * n_crossings && abs_mate(a) == next(a);
  }

  inline bool is_bigon(uint a) const noexcept {
    if (a >= 2 * n_crossings)
      return false;
    uint diff = mate[a] - mate[next(a)];
    return diff == 1 || -diff == 1;
  }

  public:

  int R1_neg(uint a, Link<static_n_components> &result) const noexcept {
    if (!is_loop(a))
      return -1;
    result.n_crossings = n_crossings - 1;
    std::memcpy(result.n_comp_crossings, n_comp_crossings,n_comp() * sizeof(uint));
    result.dec_comp(a);
    auto [comp_start_a, comp_end_a] = comp_boundaries(a);
    
#pragma GCC ivdep
    if (a != comp_end_a) {
        for (uint i = comp_start_a; i < a; i++)
            result.mate[i] = delooping_index(a, mate[i]);
        for (uint i = a + 2; i < comp_end_a; i++)
          result.mate[i - 2] = delooping_index(a, mate[i]);
    } else {
        for (uint i = 0; i < 2 * comp_start_a; i++)
          result.mate[i] = delooping_index(a, mate[i]);
        for (uint i = 2 *comp_end_a + 1; i < 2 * a; i++)
          result.mate[i - 1] = delooping_index(a, mate[i]);
        for (uint i = 2 * a + 1; i < 2 * n_crossings; i++)
          result.mate[i - 2] = delooping_index(a, mate[i]);
    }
  }


  int R1_pos(uint a, Orientation sign, Link<static_n_components> &result) const noexcept { // loop a
    result.n_crossings = n_crossings + 1;
    std::memcpy(result.n_comp_crossings, n_comp_crossings,n_comp() * sizeof(uint));
    result.inc_comp(a);

#pragma GCC ivdep
    for (uint i = 0; i < a; i++)
      result.mate[i] = looping_index(a, mate[i]);

    for (uint i = a; i < 2 * n_crossings; i++)
      result.mate[i + 2] = looping_index(a, mate[i]);

    result.bind(a, a + 1, LOOP_FLAG(sign));
    return 0;
  }

  int R2_neg(uint a, Link<static_n_components> &result) const noexcept {
    if (!is_bigon(a)) { //should only be in debug
      return -1;
    }

    uint b = abs_mate(a);
    auto [min_, max_] = std::minmax(a, b);

    result.n_crossings = n_crossings - 2;
    std::memcpy(result.n_comp_crossings, n_comp_crossings,n_comp() * sizeof(uint));
    result.dec_comps(a, b);

#pragma GCC ivdep // FIX THE LOOPS BELOW: they don't account for wrapping around the component
    for (uint i = 0; i < min_; i++)
      result.mate[i] = depoking_index(min_, max_, mate[i]);

    for (uint i = min_ + 2; i < max_; i++)
      result.mate[i - 2] = depoking_index(min_, max_, mate[i]);

    for (uint i = max_ + 2; i < 2 * n_crossings; i++)
      result.mate[i - 4] = depoking_index(min_, max_, mate[i]);

    return 0;
  }


  int R2_pos(uint a, uint b, bool collinear, Link<static_n_components> &result) const noexcept { //a is over and positive
    result.n_crossings = n_crossings + 2;
    std::memcpy(result.n_comp_crossings, n_comp_crossings,n_comp() * sizeof(uint));
    result.inc_comps(a, b);

    uint a2 = next(a);
    uint b2 =  collinear ? next(b) : prev(b);

    auto [min_, max_] = std::minmax(a, b);

#pragma GCC ivdep // 
    uint shift = 0;
    for (uint i = 0; i < 2 * n_crossings; i++){
        if (i == a || i == a2 || i == b || i == b2)
            shift++;
        result.mate[i + shift] = poking_index(min_, max_, mate[i]);
    }

    const uint flag = FLAG(Orientation::pos, CrossingType::over); //fix orientation
    result.bind(a, b, flag);
    result.bind(a2, b2, MIRROR(flag));
    return 0;
  }

  int R3(uint a, bool dir_a, bool dir_b, Link<static_n_components> &result) const noexcept {
    const uint a1 = a;
    auto [flag_pivot, b1] = unflag(mate[a1]);
    uint a2 = dir_a ? next(a1) : prev(a1);
    uint b2 = dir_b ? next(b1) : prev(b1);
    auto [flag_a, c_a] = unflag(mate[a2]);
    auto [flag_b, c_b] = unflag(mate[b2]);

    if (!(are_adjacent(c_a, c_b) && GET_TYPE(flag_a) == GET_TYPE(flag_b))) { //should only be in debug
      return -1;
    }

    result.n_crossings = n_crossings;
    std::memcpy(result.n_comp_crossings, n_comp_crossings,n_comp() * sizeof(uint));
    std::memcpy(result.mate, mate, 2 * n_crossings * sizeof(uint));

    result.bind(a1, c_b, flag_a);
    result.bind(b1, c_a, flag_b);
    result.bind(a2, b2, flag_pivot);
    return 0;
  }

  void to_graph(Vertex *v0) const noexcept {
    uint vertex[2 * n_crossings]; //TODO: fill in
    uint cur = 0; //unsigned
    uint cur_comp_start = 0;
    uint cur_comp_ind = 0;
    for (uint k = 0; k < n_crossings; k++){
        while (GET_SIGN(mate[cur]) == Orientation::pos)
            cur = ABS(next(cur));
        vertex[cur] = k;
        vertex[abs_mate(cur)] = k;
        uint next_ = abs(next(cur));
        if (next_ != cur_comp_start){
            cur = next_;
        } else {
            cur = next_ + 2 * n_comp_crossings[cur_comp_ind];
            cur_comp_ind++;
        }
    }

    for (uint a = 0; a < 2 * n_crossings; a++){
        if (GET_SIGN(mate[a]) == CrossingType::over) 
            continue;

        uint k = vertex[a];
        uint b = abs_mate(a);
        v0[k].over[0] = vertex[next(a)];
        v0[k].over[1] = vertex[prev(a)];
        if (true) {
            v0[k].under[0] = vertex[next(b)];
            v0[k].under[1] = vertex[prev(b)];
        } else {
            v0[k].under[0] = vertex[prev(b)];
            v0[k].under[1] = vertex[next(b)];
        }
    }

  }

  private:
  void bind(uint a, uint b, uint flag) {
    mate[a] = FLAGGED(b, flag);
    mate[b] = FLAGGED(a, FLIP(flag));
  }
};

using Knot = Link<1>;
