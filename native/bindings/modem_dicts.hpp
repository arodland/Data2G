// bind_modem.cpp's dict conversions, for the other bindings that return modem results.
#pragma once

#include <optional>

#include "convert.hpp"
#include "modem/modem.hpp"

namespace data2g::bind {

py::dict modem_lock_dict(const modem::Lock& l);
py::dict modem_received_dict(const modem::Received& r);
std::optional<modem::Accept> modem_accept_of(const py::object& a);

}  // namespace data2g::bind
