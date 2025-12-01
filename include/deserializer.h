#include "link.h"
#include "visit.h"
#include "nlohmann/json.hpp"
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

inline VisitType parse_visit_type(const std::string &s) {
  if (s == "OVER")
    return VisitType::over;
  if (s == "UNDER")
    return VisitType::under;
  throw std::runtime_error("Unknown VisitType: " + s);
}

inline Orientation parse_orientation(const std::string &s) {
  if (s == "POS")
    return Orientation::pos;
  if (s == "NEG")
    return Orientation::neg;
  throw std::runtime_error("Unknown Orientation: " + s);
}

template <uint16_t static_n_components>
Link<static_n_components> link_from_json(const json &j) {
  const uint16_t n_components = j.at("n_components").get<uint16_t>();
  const uint16_t n_crossings = j.at("n_crossings").get<uint16_t>();

  if (n_components != static_n_components) {
    throw std::runtime_error("JSON n_components != static_n_components");
  }
  if (n_crossings > MAX_CROSSINGS) {
    throw std::runtime_error("Too many crossings for MAX_CROSSINGS");
  }

  // JSON stores #visits per component; C++ expects #crossings per component.
  std::vector<uint16_t> n_conn_visits =
      j.at("n_conn_visits").get<std::vector<uint16_t>>();

  uint16_t n_conn_crossings[static_n_components] = {0};
  for (uint16_t c = 0; c < n_components; ++c) {
    if (n_conn_visits[c] % 2 != 0) {
      throw std::runtime_error("n_conn_visits must be even");
    }
    n_conn_crossings[c] = n_conn_visits[c] / 2;
  }

  Visit visits[2 * MAX_CROSSINGS];
  const auto &vj = j.at("visits");
  if (vj.size() != 2u * n_crossings) {
    throw std::runtime_error("visits.size() != 2 * n_crossings");
  }

  for (size_t k = 0; k < vj.size(); ++k) {
    const auto &e = vj[k];
    uint16_t i = e.at("i").get<uint16_t>();
    uint16_t mate = e.at("mate").get<uint16_t>();
    std::string t = e.at("type").get<std::string>();
    std::string ori = e.at("orientation").get<std::string>();

    if (i != k) {
      throw std::runtime_error("visit index mismatch");
    }

    VisitType type = parse_visit_type(t);
    Orientation sign = parse_orientation(ori);
    uint16_t crossingFlag = Visit::FLAG(sign, type);

    visits[i] = Visit(mate, crossingFlag);
  }

  return Link<static_n_components>(n_crossings, n_conn_crossings, visits);
}

// Example for your JSON (1 component):
// Link<1> L = link_from_json<1>(j);
