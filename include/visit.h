#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include <sys/types.h>


enum class Orientation { pos = 0, neg = 1 };

enum class VisitType { over = 0, under = 1 };

enum class Direction { next = 0, prev = 1 };

enum class ReidemeisterKind {
  R1_neg = -1,
  R1_pos = 1,
  R2_neg = -2,
  R2_pos = 2,
  R3 = 3
};

inline int16_t pm(const VisitType type) noexcept {
  return type == VisitType::over ? 1 : -1;
}

struct ReidemeisterMove {
  static constexpr uint8_t DIR_OVER_SHIFT = 2;
  static constexpr uint8_t DIR_UNDER_SHIFT = 3;
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

  static inline ReidemeisterMove R1_neg() noexcept {
    return ReidemeisterMove(ReidemeisterKind::R1_neg);
  }

  static inline ReidemeisterMove R1_pos(Orientation sign, VisitType type) noexcept {
    return ReidemeisterMove(ReidemeisterKind::R1_pos,(uint8_t)sign << SIGN_SHIFT | (uint8_t)type << TYPE_SHIFT);
  }

  static inline ReidemeisterMove R2_neg() noexcept {
    return ReidemeisterMove(ReidemeisterKind::R2_neg);
  }

  static inline ReidemeisterMove R2_pos(Direction dir_over, Direction dir_under) noexcept {
    return ReidemeisterMove(ReidemeisterKind::R2_pos,
                            (uint8_t)dir_over << DIR_OVER_SHIFT | (uint8_t)dir_under << DIR_UNDER_SHIFT);
  }

  static inline ReidemeisterMove R2_pos(Orientation sign, bool collinear) noexcept {
    return ReidemeisterMove(ReidemeisterKind::R2_pos,
(uint8_t)sign << SIGN_SHIFT |  (uint8_t)collinear << COLLINEAR_SHIFT);
  }

  static inline ReidemeisterMove R3(Direction dir_over, Direction dir_under) noexcept {
    return ReidemeisterMove(ReidemeisterKind::R3,(uint8_t)dir_over << DIR_OVER_SHIFT | (uint8_t)dir_under << DIR_UNDER_SHIFT);
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
    os << "R3 " << arg.dir_over() << " " << arg.dir_under();
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
  static constexpr uint8_t SIGN_SHIFT = 0;
  static constexpr uint8_t R1_NEG_SHIFT = 1;
  static constexpr uint8_t R2_NEG_SHIFT = 2;
  static constexpr uint8_t R3_SHIFT = 3;
  static constexpr uint8_t TYPE_SHIFT = 7;

  static constexpr uint8_t DIR_SHIFT(const Direction dir_a, const Direction dir_c) noexcept {
    return R3_SHIFT + uint8_t(dir_a) + 2 * uint8_t(dir_c);
  }

  static constexpr uint8_t TYPE = (uint8_t)(1 << TYPE_SHIFT);
  static constexpr uint8_t SIGN = (uint8_t)(1 << SIGN_SHIFT);
  static constexpr uint8_t R1_NEG = (uint8_t)(1 << R1_NEG_SHIFT);
  static constexpr uint8_t R2_NEG = (uint8_t)(1 << R2_NEG_SHIFT);
  static constexpr uint8_t CROSSING_MASK = TYPE | SIGN;
  static constexpr uint8_t R3_MASK = 0b1111 << R3_SHIFT;
  static constexpr uint8_t MOVES_MASK = R1_NEG | R2_NEG | R3_MASK;

  uint16_t mate;
  uint8_t flags;

  inline Visit() : mate(0), flags(0) {};

  static inline uint8_t FLIP(uint8_t flags) noexcept { return flags ^ TYPE; }

  static inline uint8_t MIRROR(uint8_t flags) noexcept { return flags ^ SIGN; }

  static inline uint8_t FLAG(Orientation sign, VisitType type) noexcept {
    return (uint8_t)sign << SIGN_SHIFT | (uint8_t)type << TYPE_SHIFT;
  }

  Visit(uint16_t mate, uint8_t flags) : mate(mate), flags(flags) {};

  inline bool is_loop() const noexcept { return flags & R1_NEG; }

  inline bool is_bigon() const noexcept { return flags & R2_NEG; }

  inline uint8_t crossing_flags() const noexcept {
    return flags & CROSSING_MASK;
  }

  inline uint8_t moves_flags() const noexcept { return flags & MOVES_MASK; }

  static inline VisitType GET_TYPE(uint8_t flags) noexcept {
    return VisitType((flags >> TYPE_SHIFT) & 1);
  }
  inline VisitType type() const noexcept { return GET_TYPE(crossing_flags()); }

  inline Orientation sign() const noexcept {
    return Orientation((flags >> SIGN_SHIFT) & 1);
  }
};
