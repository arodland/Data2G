#include "arq/burst.hpp"

#include <algorithm>

namespace data2g::arq {

bool dup_ctl(const TxBurst& b) {
    return std::any_of(b.slots.begin(), b.slots.end(), [](const Slot& s) { return s.rv && s.mask_id.seq >= SEQ_MOD; });
}

}  // namespace data2g::arq
