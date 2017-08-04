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

%module (package="mkldnn") cosim_dump
%{
  #define SWIG_FILE_WITH_INIT
  #include <mkldnn.hpp>
  #include <fstream>
  #include "mdarray.h"
  #include "dump_desc.h"
  #include "cosim_dump.h"
%}

%init %{
  import_array();
%}

%include stl.i
%include exception.i

%feature("flatnested");
%feature("nodefaultctor");

%import support.i
%import memory.i
%import mdarray.i

namespace mkldnn {

namespace c_api {
  %include c_api.i
}

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

class cosim_dump {
public:
    cosim_dump(operation_kind aop_kind);

    void dump_memory(parm_kind aparm_kind, const memory &mem);

    void dump_int_parms(parm_kind aparm_kind, int i1, int i2 = 0, int i3 = 0,
            int i4 = 0, int i5 = 0, int i6 = 0);

    void dump_double_parms(parm_kind aparm_kind, double d1, double d2 = 0.0,
            double d3 = 0.0, double d4 = 0.0, double d5 = 0.0, double d6 = 0.0);

    virtual ~cosim_dump();

private:
    cosim_dump() {}

    std::fstream dfile;

    DumpHeader header;
};

} // namespace mkldnn
