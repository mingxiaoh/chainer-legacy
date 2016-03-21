#ifndef _DNET_UTILS_HPP_
#define _DNET_UTILS_HPP_

#include <mkldnn.hpp>
using namespace mkldnn;

memory::format get_desired_format(int channel);

#endif
