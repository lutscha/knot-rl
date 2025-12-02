#include "nlohmann/json.hpp"
#include <stdexcept>
#include <vector>

#include "link.h"
#include "visit.h"

using json = nlohmann::json;

template <uint16_t static_n_components>
Link<static_n_components> deserialize_link(const std::string &json_str) {
  json j = json::parse(json_str);

  // 1. Components
  uint16_t n_components = j.at("n_components").get<uint16_t>();
  if (n_components != static_n_components) {
    throw std::runtime_error(
        "n_components in JSON does not match static_n_components");
  }

  // 2. Crossings (number of vertices)
  uint16_t n_crossings = j.at("n_crossings").get<uint16_t>();

  // 3. Component sizes: visits (not crossings!)
  auto n_conn_visits = j.at("n_conn_visits").get<std::vector<uint16_t>>();
  if (n_conn_visits.size() != static_n_components) {
    throw std::runtime_error("n_conn_visits length != static_n_components");
  }

  uint16_t sum_visits = 0;
  for (uint16_t c = 0; c < static_n_components; ++c) {
    if (n_conn_visits[c] % 2 != 0) {
      throw std::runtime_error("n_conn_visits entry is not even");
    }
    sum_visits = static_cast<uint16_t>(sum_visits + n_conn_visits[c]);
  }

  if (sum_visits != 2 * n_crossings) {
    throw std::runtime_error("sum(n_conn_visits) != 2 * n_crossings");
  }

  // 4. Visits: [mate, flags]
  const auto &visits_json = j.at("visits");
  if (visits_json.size() != 2 * n_crossings) {
    throw std::runtime_error("visits size mismatch");
  }

  std::vector<Visit> visits(2 * n_crossings);
  for (uint16_t i = 0; i < visits_json.size(); ++i) {
    const auto &vj = visits_json[i];
    if (!vj.is_array() || vj.size() != 2) {
      throw std::runtime_error("visit entry must be [mate, flags]");
    }

    uint16_t mate = vj[0].get<uint16_t>();
    uint16_t flags = vj[1].get<uint16_t>();

    visits[i] = Visit(mate, flags);
  }

  // 5. Construct static link with n_conn_visits (visits, not crossings)
  return Link<static_n_components>(n_crossings, n_conn_visits.data(),
                                   visits.data());
}
