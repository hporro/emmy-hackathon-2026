extern "C" __global__
__launch_bounds__(256) void k_amax_reduce(const float* x, float* amax) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 4) {
        int a0 = _gid;
        float acc0 = -1e+30f;
        for (int a1 = 0; a1 < 1024; a1++) {
            float in0 = x[a0 * 1024 + a1];
            acc0 = fmaxf(acc0, in0);
        }
        amax[a0] = acc0;
    }
}
