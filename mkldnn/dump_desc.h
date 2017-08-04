/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef _DUMP_DESC_H_
#define _DUMP_DESC_H_

#include <mkldnn.hpp>

namespace mkldnn {

enum operation_kind {
    cdump_op_invalid = 0,
    cdump_op_lrn_forward,
    cdump_op_lrn_backward,
    cdump_op_max
};

enum parm_kind {
    cdump_memory_invalid = 0,
    cdump_src_memory,
    cdump_ws_memory,
    cdump_diff_dst_memory,
    cdump_lrn_local_size,
    cdump_lrn_doulbe_parms,
    cdump_memory_max
};

#define CDUMP_ID_NUM  0xA0A05050

struct DumpHeader {
    int              idnum;
    operation_kind   ok;
}__attribute__ ((packed));

struct DumpDesc {
    parm_kind   pk;
    int         desc_size;
    int         data_size;
    union {
        int     iparms[TENSOR_MAX_DIMS];
        double  dparms[TENSOR_MAX_DIMS];
    };
}__attribute__ ((packed));

};

#endif
