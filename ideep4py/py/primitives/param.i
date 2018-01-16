%rename (convolution2DParam) conv_param_t;
struct conv_param_t {
    std::vector<int> out_dims;
    int kh, kw; // kernel size
    int dilate_y = 0, dilate_x = 0; // in MKL-DNN, common conv is treated as 0 dilate
    int sy, sx; // stride
    int pad_lh, pad_lw, pad_rh, pad_rw; //padding
};

%rename (pooling2DParam) pooling_param_t;
struct pooling_param_t {
    std::vector<int> out_dims;
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

%rename (localResponseNormalizationParam) lrn_param_t;
struct lrn_param_t {
    int n; // local size
    double k;
    double alpha;
    double beta;

    enum algorithm {
        lrn_across_channels,
        lrn_within_channel,
    } algo_kind;
};
