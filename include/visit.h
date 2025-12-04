#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include <sys/types.h>

static constexpr uint16_t MAX_CROSSINGS = 1200;
static constexpr uint16_t N_MOVES = 10;

enum class Orientation { pos = 0, neg = 1 };

inline Orientation operator!(Orientation orient) noexcept {
  return static_cast<Orientation>(1 - static_cast<int>(orient));
}

inline Orientation select_orientation(Orientation sign, bool same) noexcept {
  return same ? sign : !sign;
}

enum class VisitType { over = 0, under = 1 };

enum class Direction { next = 0, prev = 1 };

inline Direction operator!(Direction dir) noexcept {
  return static_cast<Direction>(1 - static_cast<int>(dir));
}

inline Direction select_direction(Direction dir, bool same) noexcept {
  return same ? dir : !dir;
}

enum class ReidemeisterKind {
  R1_neg = -1,
  R1_pos = 1,
  R2_neg = -2,
  R2_pos = 2,
  R3 = 3,
  R4_pos = 4
};

inline int16_t pm(const VisitType type) noexcept {
  return type == VisitType::over ? 1 : -1;
}

struct ReidemeisterMove {
  static constexpr uint8_t DIR_OVER_SHIFT = 2;
  static constexpr uint8_t DIR_UNDER_SHIFT = 3;
  static constexpr uint8_t DIR_SHIFT = 4;
  static constexpr uint8_t TYPE_SHIFT = 7;
  static constexpr uint8_t SIGN_SHIFT = 0;
  static constexpr uint8_t COLLINEAR_SHIFT = 1;

  ReidemeisterKind kind;
  uint8_t args;

  ReidemeisterMove(ReidemeisterKind kind, uint8_t args)
      : kind(kind), args(args) {};

  ReidemeisterMove(ReidemeisterKind kind) : kind(kind), args(0) {};

  inline VisitType type() const noexcept {
    return VisitType((args >> TYPE_SHIFT) & 1);
  }
  inline Orientation sign() const noexcept {
    return Orientation((args >> SIGN_SHIFT) & 1);
  }
  inline Direction dir_over() const noexcept {
    return Direction((args >> DIR_OVER_SHIFT) & 1);
  }
  inline Direction dir_under() const noexcept {
    return Direction((args >> DIR_UNDER_SHIFT) & 1);
  }
  inline Direction dir() const noexcept {
    return Direction((args >> DIR_SHIFT) & 1);
  }
  static inline ReidemeisterMove R1_neg() noexcept {
    return ReidemeisterMove(ReidemeisterKind::R1_neg);
  }

  static inline ReidemeisterMove R1_pos(Orientation sign,
                                        VisitType type) noexcept {
    return ReidemeisterMove(ReidemeisterKind::R1_pos,
                            (uint8_t)sign << SIGN_SHIFT | (uint8_t)type
                                                              << TYPE_SHIFT);
  }

  static inline ReidemeisterMove R2_neg() noexcept {
    return ReidemeisterMove(ReidemeisterKind::R2_neg);
  }

  static inline ReidemeisterMove R2_pos(VisitType type, Direction dir_over,
                                        Direction dir_under) noexcept {
    return ReidemeisterMove(ReidemeisterKind::R2_pos,
                            (uint8_t)type << TYPE_SHIFT |
                                (uint8_t)dir_over << DIR_OVER_SHIFT |
                                (uint8_t)dir_under << DIR_UNDER_SHIFT);
  }

  static inline ReidemeisterMove R2_pos(Orientation sign,
                                        bool collinear) noexcept {
    return ReidemeisterMove(ReidemeisterKind::R2_pos,
                            (uint8_t)sign << SIGN_SHIFT |
                                (uint8_t)collinear << COLLINEAR_SHIFT);
  }

  static inline ReidemeisterMove R3(Direction dir_over,
                                    Direction dir_under) noexcept {
    return ReidemeisterMove(ReidemeisterKind::R3,
                            (uint8_t)dir_over << DIR_OVER_SHIFT |
                                (uint8_t)dir_under << DIR_UNDER_SHIFT);
  }

  static inline ReidemeisterMove R4_pos(Direction dir,
                                        VisitType type) noexcept {
    return ReidemeisterMove(ReidemeisterKind::R4_pos,
                            (uint8_t)dir << DIR_SHIFT | (uint8_t)type
                                                            << TYPE_SHIFT);
  }
};

inline std::ostream &operator<<(std::ostream &os, const VisitType &arg) {
  os << (arg == VisitType::over ? "over" : "undr");
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const Orientation &arg) {
  os << (arg == Orientation::pos ? "anticlock" : "clockwise");
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const Direction &arg) {
  os << (arg == Direction::next ? "next" : "prev");
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const ReidemeisterKind &arg) {
  int8_t int_val = static_cast<int8_t>(arg);
  char sgn = int_val < 0 ? '-' : '+';
  os << "R" << abs(int_val) << sgn;
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const ReidemeisterMove &arg) {
  switch (arg.kind) {
  case ReidemeisterKind::R1_neg:
    os << "R1-";
    break;
  case ReidemeisterKind::R1_pos:
    os << "R1+ " << arg.sign() << " " << arg.type();
    break;
  case ReidemeisterKind::R2_neg:
    os << "R2-";
    break;
  case ReidemeisterKind::R2_pos:
    os << "R2+ over=" << arg.dir_over() << " under=" << arg.dir_under();
    break;
  case ReidemeisterKind::R3:
    os << "R3  " << arg.dir_over() << " " << arg.dir_under();
    break;
  case ReidemeisterKind::R4_pos:
    os << "R4+ " << arg.dir() << " " << arg.type();
    break;
  }
  return os;
}

inline constexpr VisitType operator!(VisitType b) noexcept {
  return (b == VisitType::under) ? VisitType::over : VisitType::under;
}

struct Visit {
  // the shifts are chosen in such a way that positive crossings correspond to
  // positive flags, and over visits correspond to even flags
  static constexpr uint8_t N_R3_MOVES = 4;
  static constexpr uint8_t N_R4_MOVES = 4;

  static constexpr uint8_t SIGN_SHIFT = 0;
  static constexpr uint8_t TYPE_SHIFT = 1;

  static constexpr uint8_t R1_NEG_SHIFT = 2;
  static constexpr uint8_t R2_NEG_SHIFT = 3;
  static constexpr uint8_t R3_SHIFT = 4;
  static constexpr uint8_t R4_SHIFT = R3_SHIFT + N_R3_MOVES;

  static constexpr uint8_t MOVES_FIRST_BIT = R1_NEG_SHIFT;
  static constexpr uint8_t MOVES_LAST_BIT = R4_SHIFT + N_R4_MOVES;

  static constexpr uint8_t R3_ARG_SHIFT(const Direction dir_over, const Direction dir_under) noexcept {
    return R3_SHIFT + uint8_t(dir_over) + 2 * uint8_t(dir_under);
  }

  static constexpr uint8_t R4_ARG_SHIFT(const Direction dir, const VisitType type) noexcept {
    return R4_SHIFT + uint8_t(dir) + 2 * uint8_t(type);
  }

  static constexpr uint16_t TYPE = (uint16_t)(1 << TYPE_SHIFT);
  static constexpr uint16_t SIGN = (uint16_t)(1 << SIGN_SHIFT);
  static constexpr uint16_t R1_NEG = (uint16_t)(1 << R1_NEG_SHIFT);
  static constexpr uint16_t R2_NEG = (uint16_t)(1 << R2_NEG_SHIFT);
  static constexpr uint16_t CROSSING_MASK = TYPE | SIGN;
  static constexpr uint16_t R3_MASK = 0b1111 << R3_SHIFT;
  static constexpr uint16_t R4_MASK = 0b1111 << R4_SHIFT;
  static constexpr uint16_t MOVES_MASK = R1_NEG | R2_NEG | R3_MASK | R4_MASK;

  static inline ReidemeisterMove BIT_TO_MOVE(uint8_t bit) noexcept {
    if (bit < MOVES_FIRST_BIT || bit >= MOVES_LAST_BIT) [[unlikely]] {
      std::cerr << "Invalid move bit: " << bit << std::endl;
    }

    if (bit == R1_NEG_SHIFT) {
      return ReidemeisterMove::R1_neg();
    }
    if (bit == R2_NEG_SHIFT) {
      return ReidemeisterMove::R2_neg();
    }

    if (R3_SHIFT <= bit && bit < R3_SHIFT + N_R3_MOVES) {
      uint8_t r3_arg_shift = bit - R3_SHIFT;
      Direction dir_under = Direction((r3_arg_shift >> 1) & 1);
      Direction dir_over = Direction(r3_arg_shift & 1);
      return ReidemeisterMove::R3(dir_over, dir_under);
    }

    uint8_t r4_arg_shift = bit - R4_SHIFT;
    Direction dir = Direction(r4_arg_shift & 1);
    VisitType type = VisitType((r4_arg_shift >> 1) & 1);
    return ReidemeisterMove::R4_pos(dir, type);
  }

  static inline uint16_t MOVE_TO_BIT(ReidemeisterMove move) {
    switch (move.kind) {
    case ReidemeisterKind::R1_neg:
      return R1_NEG_SHIFT;
    case ReidemeisterKind::R1_pos:
      std::cerr << "R1_pos has no bit" << std::endl;
      exit(1);
    case ReidemeisterKind::R2_neg:
      return R2_NEG_SHIFT;
    case ReidemeisterKind::R2_pos:
      std::cerr << "R2_pos has no bit" << std::endl;
      exit(1);
    case ReidemeisterKind::R3:
      return R3_SHIFT + Visit::R3_ARG_SHIFT(move.dir_over(), move.dir_under());
    case ReidemeisterKind::R4_pos:
      return R4_SHIFT + move.args;
    }
  }

  uint16_t mate;
  uint16_t flags;

  inline Visit() : mate(0), flags(0) {};

  static inline uint16_t FLIP(uint16_t flags) noexcept { return flags ^ TYPE; }

  static inline uint16_t MIRROR(uint16_t flags) noexcept {
    return flags ^ SIGN;
  }

  static inline uint16_t FLAG(Orientation sign, VisitType type) noexcept {
    return (uint16_t)sign << SIGN_SHIFT | (uint16_t)type << TYPE_SHIFT;
  }

  Visit(uint16_t mate, uint16_t flags) : mate(mate), flags(flags) {};

  Visit(uint16_t mate, Orientation sign, VisitType type)
      : mate(mate), flags(FLAG(sign, type)) {};

  inline bool is_loop() const noexcept { return flags & R1_NEG; }

  inline bool is_bigon() const noexcept { return flags & R2_NEG; }

  inline bool is_triangle(Direction dir_over, Direction dir_under) const noexcept {
    return (flags >> R3_ARG_SHIFT(dir_over, dir_under)) & 1;
  }

  inline bool is_R4(Direction dir, VisitType type) const noexcept {
    return (flags >> R4_ARG_SHIFT(dir, type)) & 1;
  }

  inline uint16_t crossing_flags() const noexcept {
    return flags & CROSSING_MASK;
  }

  inline uint16_t moves_flags() const noexcept { return flags & MOVES_MASK; }

  static inline VisitType GET_TYPE(uint16_t flags) noexcept {
    return VisitType((flags >> TYPE_SHIFT) & 1);
  }

  static inline Orientation GET_SIGN(uint16_t flags) noexcept {
    return Orientation((flags >> SIGN_SHIFT) & 1);
  }

  inline VisitType type() const noexcept { return GET_TYPE(crossing_flags()); }

  inline Orientation sign() const noexcept {
    return Orientation((flags >> SIGN_SHIFT) & 1);
  }

  inline bool has_move() const noexcept { return moves_flags() != 0; }
};

struct AvailableMove {
  uint16_t v;
  ReidemeisterMove move;
  AvailableMove(uint16_t v, ReidemeisterMove move) : v(v), move(move) {}
};