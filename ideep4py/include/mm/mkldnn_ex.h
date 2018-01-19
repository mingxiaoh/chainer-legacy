#pragma once
#include "mkldnn.hpp"
#include "reorder.h"

inline static mkldnn::memory reorder_if_must(mkldnn::memory user
        , mkldnn::memory::primitive_desc expect
        , std::unique_ptr<mkldnn::memory> &mreorder
        , std::vector<mkldnn::primitive> *dag) {

    if (user.get_primitive_desc() != expect) {
        mkldnn::memory interm(expect);
#if 0
        auto user_mpd = user.get_primitive_desc();
        mkldnn::memory::format user_fmt = static_cast<mkldnn::memory::format>(
                user_mpd.desc().data.format);
        mkldnn::memory::format mkl_fmt = static_cast<mkldnn::memory::format>(
                expect.desc().data.format);
        mkldnn::memory::data_type dtype = static_cast<mkldnn::memory::data_type>(
                expect.desc().data.data_type);

        if ((user_fmt == mkldnn::memory::format::nChw16c &&
                    mkl_fmt == mkldnn::memory::format::nChw8c) ||
                (mkl_fmt == mkldnn::memory::format::nChw16c &&
                 user_fmt == mkldnn::memory::format::nChw8c)) {
            auto m = expect.desc().data;
            int n = m.dims[0], c = m.dims[1], h = m.dims[2], w = m.dims[3];
            mkldnn::memory::dims tz = {n, c, h, w};
            mreorder.reset(new mkldnn::memory({{{ tz }, dtype, mkldnn::memory::format::nchw }, expect.get_engine()}));
            //auto mreorder = new mkldnn::memory({{{ tz }, dtype, mkldnn::memory::format::nchw }, expect.get_engine()});
            auto rep1 = mkldnn::reorder(user, *mreorder);
            auto rep2 = mkldnn::reorder(*mreorder, interm);
            dag->push_back(rep1);
            dag->push_back(rep2);
            //static int spl_nr = 0;
            //printf("\n   %d *Reorder(split) iutput from:%d, to:%d\n", spl_nr++, user_fmt, mkl_fmt);
        } else {
            dag->push_back(mkldnn::reorder(user, interm));
        }
#else
        dag->push_back(mkldnn::reorder(user, interm));
#endif
        return interm;
    }

    return user;
}

template<typename T>
inline static void axpby(Tensor *dst, T a, Tensor *x, T b, Tensor *y) {
    std::vector<mkldnn::primitive> prims;
    std::unique_ptr<mkldnn::memory> mreorder;

    /// Reorder to x's format
    auto mid = reorder_if_must(y->mkldnn_memory(), x->mkldnn_memory().get_primitive_desc()
            , mreorder, &prims);

    mkldnn::sum::primitive_desc sum_pd(std::vector<float>({(float)a, (float)b})
            , {x->mkldnn_memory().get_primitive_desc(), mid.get_primitive_desc()});

    std::vector<mkldnn::memory::primitive::at> inputs_at {x->mkldnn_memory(), mid};

    mkldnn::sum sum_prim(sum_pd, inputs_at, dst->mkldnn_memory());
    prims.push_back(sum_prim);

    mkldnn::stream s(mkldnn::stream::kind::eager);
    s.submit(prims).wait();
}

