struct conv_param_t {
    int src_d1, src_d2, src_d3, src_d4; // input shape
    int weights_d1, weights_d2, weights_d3, weights_d4; //weight shape
    int dst_d1, dst_d2, dst_d3, dst_d4; // output shape
    int bias_d1; // bias shape
    int kh, kw; // kernel size
    int dilate_y = 0, dilate_x = 0; // in MKL-DNN, common conv is treated as 0 dilate
    int sy, sx; // stride
    int pad_lh, pad_lw, pad_rh, pad_rw; //padding
    bool with_bias;
    bool with_weights_opt = false; // whether pass back optimized weight
};

struct pooling_param_t {
    int src_d1, src_d2, src_d3, src_d4; // input shape
    int dst_d1, dst_d2, dst_d3, dst_d4; // output shape
    int kh, kw; // kernel size
    int sy, sx; // stride
    int pad_lh, pad_lw, pad_rh, pad_rw; //padding

    enum algorithm {
        pooling_max,
        pooling_avg,
        pooling_avg_include_padding,
        pooling_avg_exclude_padding,
    } algo_kind;
};

struct linear_param_t {
    int src_d1, src_d2, src_d3, src_d4;
    int src_ndims;
    bool with_bias;
    bool with_weights_opt = false; // whether pass back optimized weight
};

struct lrn_param_t {
    int src_d1, src_d2, src_d3, src_d4; // input shape
    int dst_d1, dst_d2, dst_d3, dst_d4; // output shape
    int n; // local size
    double k;
    double alpha;
    double beta;

    enum algorithm {
        lrn_across_channels,
        lrn_within_channel,
    } algo_kind;
};

