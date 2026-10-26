extern "C" __global__
__launch_bounds__(256) void k_amax_reduce(const float* x, float* amax) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 4) {
        int a0 = _gid;
        float acc0 = -1e+30f;
        float acc0__r1 = -1e+30f;
        float acc0__r2 = -1e+30f;
        float acc0__r3 = -1e+30f;
        for (int a1 = 0; a1 < 1024; a1 += 4) {
            float in0 = x[a0 * 1024 + a1];
            acc0 = fmaxf(acc0, in0);
            float in0__r1 = x[a0 * 1024 + (a1 + 1)];
            acc0__r1 = fmaxf(acc0__r1, in0__r1);
            float in0__r2 = x[a0 * 1024 + (a1 + 2)];
            acc0__r2 = fmaxf(acc0__r2, in0__r2);
            float in0__r3 = x[a0 * 1024 + (a1 + 3)];
            acc0__r3 = fmaxf(acc0__r3, in0__r3);
        }
        acc0 = fmaxf(acc0, acc0__r1);
        acc0 = fmaxf(acc0, acc0__r2);
        acc0 = fmaxf(acc0, acc0__r3);
        amax[a0] = acc0;
    }
}
