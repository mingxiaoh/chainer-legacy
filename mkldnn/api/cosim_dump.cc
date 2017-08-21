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

#include "cosim_dump.h"
#include <assert.h>
#include <glog/logging.h>

namespace mkldnn {

cosim_dump::cosim_dump(operation_kind aop_kind) {
    header.id_num = CDUMP_ID_NUM;
    header.mkldnn_ver = 0;
    header.op_kind = aop_kind;

    const char* dname = NULL;
    switch(header.op_kind) {
        case cdump_op_conv_forward:
            dname = "Conv_forward.cdump";
            break;
        case cdump_op_conv_backward:
            dname = "Conv_backward.cdump";
            break;
        case cdump_op_lrn_forward:
            dname = "Lrn_forward.cdump";
            break;
        case cdump_op_lrn_backward:
            dname = "Lrn_backward.cdump";
            break;
        case cdump_op_max_pooling_forward:
            dname = "MaxPooling_forward.cdump";
            break;
        case cdump_op_max_pooling_backward:
            dname = "MaxPooling_backward.cdump";
            break;
        case cdump_op_avg_pooling_forward:
            dname = "AvgPooling_forward.cdump";
            break;
        case cdump_op_avg_pooling_backward:
            dname = "AvgPooling_backward.cdump";
            break;
        default:
            dname =  "Cosim_dump.cdump";
            break;
    }

    dfile.open(dname, std::ios::binary | std::ios::trunc | std::ios::out);
    if (!dfile.is_open() || !dfile.good()) {
        printf("Failed to open dump file %s\n", dname);
        return;
    }

    dfile.write((const char*)&header, sizeof(DumpHeader));
    if (!dfile.good()) {
        printf("Failed to write file header to dump file %s\n", dname);
        dfile.close();
        return;
    }

    dfile.write((const char*)&dummy_mdesc, sizeof(mkldnn_memory_desc_t));
    if (!dfile.good()) {
        printf("Failed to write dummy_mdesc to dump file %s\n", dname);
        dfile.close();
        return;
    }
}

cosim_dump::~cosim_dump() {
    if (dfile.is_open()) {
        dfile.close();
    }
}

void cosim_dump::dump_memory(parm_kind aparm_kind, const memory &mem) {
    if (!dfile.is_open()) {
        printf("FATAL: the dump file is unavailable!\n");
        return;
    }

    auto mp = mem.get_primitive_desc();

    DumpDesc dd;
    dd.pa_kind = aparm_kind;
    dd.desc_size = sizeof(mkldnn_memory_desc_t);
    dd.data_size = mp.get_size();
    dfile.write((const char*)&dd, sizeof(DumpDesc));
    if (!dfile.good()) {
        printf("Failed to write memory DumpDesc to dump file!\n");
        return;
    }

    auto md = mp.desc();
    dfile.write(reinterpret_cast<const char*>(&md.data), dd.desc_size);
    if (!dfile.good()) {
        printf("Failed to write memory desc to dump file!\n");
        return;
    }

    void* data = mem.get_data_handle();
    dfile.write(reinterpret_cast<const char*>(data), dd.data_size);
    if (!dfile.good()) {
        printf("Failed to write memory data to dump file!\n");
        return;
    }
}

void cosim_dump::dump_int_parms(parm_kind aparm_kind, int nargs, ...) {
    assert(nargs <= TENSOR_MAX_DIMS);

    if (!dfile.is_open()) {
        printf("FATAL: the dump file is unavailable!\n");
        return;
    }

    DumpDesc dd;
    dd.pa_kind = aparm_kind;
    dd.desc_size = 0;
    dd.data_size = 0;

    int i = 0;
    va_list vl;
    va_start(vl, nargs);
    for (i = 0; i < nargs; i++) {
        dd.iparms[i] = va_arg(vl, int);
    }
    va_end(vl);

    dfile.write(reinterpret_cast<const char*>(&dd), sizeof(DumpDesc));
    if (!dfile.good()) {
        printf("Failed to write int DumpDesc to dump file!\n");
        return;
    }
}

void cosim_dump::dump_double_parms(parm_kind aparm_kind, int nargs, ...) {
    assert(nargs <= TENSOR_MAX_DIMS);

    if (!dfile.is_open()) {
        printf("FATAL: the dump file is unavailable!\n");
        return;
    }

    DumpDesc dd;
    dd.pa_kind = aparm_kind;
    dd.desc_size = 0;
    dd.data_size = 0;

    int i = 0;
    va_list vl;
    va_start(vl, nargs);
    for (i = 0; i < nargs; i++) {
        dd.dparms[i] = va_arg(vl, double);
    }
    va_end(vl);

    dfile.write(reinterpret_cast<const char*>(&dd), sizeof(DumpDesc));
    if (!dfile.good()) {
        printf("Failed to write double DumpDesc to dump file!\n");
        return;
    }
}

};
