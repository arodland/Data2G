// data2g/modem.py's burst timing: header layout, frames on air, burst
// length. Pure arithmetic on config; the rest of modem.py builds on it.
#pragma once

#include <optional>
#include <string_view>
#include <vector>

#include "generated/config.hpp"

namespace data2g::modem {

// Throws std::out_of_range for an unknown band.
const config::Band& band(std::string_view name);
int preamble_samples(const config::Band& b);

bool hosts(std::string_view band);                          // _hosts: other bands' frames follow its header
std::vector<bool> header_layout(std::string_view band);     // per header symbol: true = pilot
int header_samples(std::string_view band);
std::optional<int> copy_frame(std::string_view band, int n_f);  // nullopt: no header copy
int frames_on_air(const config::Submode& spec, int n_cw);
long burst_end(long p0, const config::Submode& spec, int n_cw);
int head_samples(std::string_view band);
double burst_seconds(const config::Submode& spec, int n_cw);

}  // namespace data2g::modem
