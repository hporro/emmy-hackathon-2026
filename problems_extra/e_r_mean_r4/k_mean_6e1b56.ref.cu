extern "C" __global__
__launch_bounds__(256) void k_mean_6e1b56(const float* mean_count, const float* x, float* mean) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 4) {
        int a0 = _gid;
        float acc0 = 0.0f;
        float acc0__r1 = 0.0f;
        float acc0__r2 = 0.0f;
        float acc0__r3 = 0.0f;
        for (int a1 = 0; a1 < 1024; a1 += 4) {
            float in1 = x[a0 * 1024 + a1];
            acc0 += in1;
            float in1__r1 = x[a0 * 1024 + (a1 + 1)];
            acc0__r1 += in1__r1;
            float in1__r2 = x[a0 * 1024 + (a1 + 2)];
            acc0__r2 += in1__r2;
            float in1__r3 = x[a0 * 1024 + (a1 + 3)];
            acc0__r3 += in1__r3;
        }
        acc0 += acc0__r1;
        acc0 += acc0__r2;
        acc0 += acc0__r3;
        float in0 = mean_count[0];
        float v1 = acc0 * (1.0f / in0);
        mean[a0] = v1;
    }
}
