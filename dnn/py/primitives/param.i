struct conv_param_t {
    int src_d1, src_d2, src_d3, src_d4; // input shape
    int weights_d1, weights_d2, weights_d3, weights_d4; //weight shape
    int dst_d1, dst_d2, dst_d3, dst_d4; // output shape
    int bias_d1; // bias shape
    int kh, kw; // kernel size
    int sy, sx; // stride
    int pad_lh, pad_lw, pad_rh, pad_rw; //padding
    bool with_bias;
};
