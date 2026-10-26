extern "C" __global__
__launch_bounds__(256) void k_mean_6e1b56(const float* mean_count, const float* x, float* mean) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 4) {
        int a0 = _gid;
        float acc0 = 0.0f;
        for (int a1 = 0; a1 < 1024; a1++) {
            float in1 = x[a0 * 1024 + a1];
            acc0 += in1;
        }
        float in0 = mean_count[0];
        float v1 = acc0 * (1.0f / in0);
        mean[a0] = v1;
    }
}
