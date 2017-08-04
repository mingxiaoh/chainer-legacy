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
    header.idnum = CDUMP_ID_NUM;
    header.ok = aop_kind;

    const char* dname = NULL;
    switch(header.ok) {
        case cdump_op_lrn_forward:
            dname = "Lrn_forward.cdump";
            break;
        case cdump_op_lrn_backward:
            dname = "Lrn_backward.cdump";
            break;
        default:
            dname =  "Cosim_dump.cdump";
            break;
    }

    dfile.open(dname, std::ios::binary | std::ios::trunc | std::ios::out);
    if (!dfile.is_open() || !dfile.good()) {
        LOG(ERROR) << "Failed to open dump file " << dname;
        return;
    }

    dfile.write((const char*)&header, sizeof(DumpHeader));
    if (!dfile.good()) {
        LOG(ERROR) << "Failed to write file header to dump file " << dname;
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
        LOG(ERROR) << "FATAL: the dump file is unavailable!";
        return;
    }

    auto mp = mem.get_primitive_desc();

    DumpDesc dd;
    dd.pk = aparm_kind;
    dd.desc_size = sizeof(mkldnn_memory_desc_t);
    dd.data_size = mp.get_size();
    dfile.write((const char*)&dd, sizeof(DumpDesc));
    if (!dfile.good()) {
        LOG(ERROR) << "Failed to write memory DumpDesc to dump file!";
        return;
    }

    auto md = mp.desc();
    dfile.write(reinterpret_cast<const char*>(&md.data), dd.desc_size);
    if (!dfile.good()) {
        LOG(ERROR) << "Failed to write memory desc to dump file!";
        return;
    }

    void* data = mem.get_data_handle();
    dfile.write(reinterpret_cast<const char*>(data), dd.data_size);
    if (!dfile.good()) {
        LOG(ERROR) << "Failed to write memory data to dump file!";
        return;
    }
}

void cosim_dump::dump_int_parms(parm_kind aparm_kind, int i1, int i2, int i3,
        int i4, int i5, int i6) {
    if (!dfile.is_open()) {
        LOG(ERROR) << "FATAL: the dump file is unavailable!";
        return;
    }

    DumpDesc dd;
    dd.pk = aparm_kind;
    dd.desc_size = 0;
    dd.data_size = 0;

    assert(TENSOR_MAX_DIMS >= 6);
    dd.iparms[0] = i1;
    dd.iparms[1] = i2;
    dd.iparms[2] = i3;
    dd.iparms[3] = i4;
    dd.iparms[4] = i5;
    dd.iparms[5] = i6;

    dfile.write(reinterpret_cast<const char*>(&dd), sizeof(DumpDesc));
    if (!dfile.good()) {
        LOG(ERROR) << "Failed to write int DumpDesc to dump file!";
        return;
    }
}

void cosim_dump::dump_double_parms(parm_kind aparm_kind, double d1, double d2,
        double d3, double d4, double d5, double d6) {
    if (!dfile.is_open()) {
        LOG(ERROR) << "FATAL: the dump file is unavailable!";
        return;
    }

    DumpDesc dd;
    dd.pk = aparm_kind;
    dd.desc_size = 0;
    dd.data_size = 0;

    assert(TENSOR_MAX_DIMS >= 6);
    dd.dparms[0] = d1;
    dd.dparms[1] = d2;
    dd.dparms[2] = d3;
    dd.dparms[3] = d4;
    dd.dparms[4] = d5;
    dd.dparms[5] = d6;

    dfile.write(reinterpret_cast<const char*>(&dd), sizeof(DumpDesc));
    if (!dfile.good()) {
        LOG(ERROR) << "Failed to write double DumpDesc to dump file!";
        return;
    }
}

};
