#pragma once

#include <cstddef>
#include <cstdint>
#include <new>     // placement new
#include <utility> // std::forward

#include "link.h" // your Link<static_n_components> and Visit

template <uint16_t static_n_components, uint16_t max_crossings, uint32_t max_links>
class LinkArena {
public:
  static_assert(static_n_components > 0,
                "LinkArena assumes static (non-dynamic) number of components.");

  using LinkType = Link<static_n_components>;


private:
  struct Slot {
    alignas(LinkType) unsigned char link_buf[sizeof(LinkType)];
    Visit visits[2 * max_crossings];
  };

  Slot slots_[max_links];
  uint16_t next_ = 0; // number of constructed links

public:
  LinkArena() noexcept = default;

  ~LinkArena() noexcept { clear_all(); }

  /// Construct a Link in-place in the arena.
  ///
  /// The Link constructors must have the form
  ///   LinkType(..., Visit* storage)
  /// where `storage` is the last parameter.
  template <class... Args> LinkType *allocate(Args &&...args) noexcept {
    if (next_ >= max_links) [[unlikely]]
      return nullptr;

    Slot &slot = slots_[next_++];

    return new (slot.link_buf)
        LinkType(std::forward<Args>(args)..., slot.visits);
  }

  /// Destroy all Links and reset arena.
  void clear_all() noexcept {
    for (uint16_t i = 0; i < next_; ++i) {
      LinkType *p = reinterpret_cast<LinkType *>(slots_[i].link_buf);
      p->~LinkType();
    }
    next_ = 0;
  }
};
