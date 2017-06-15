#ifndef _DNET_UTILS_HPP_
#define _DNET_UTILS_HPP_

#include <mkldnn.hpp>
using namespace mkldnn;

int cpu_support_avx512_p(void);
int cpu_support_avx2_p(void);

memory::format get_desired_format(int channel);

#endif
