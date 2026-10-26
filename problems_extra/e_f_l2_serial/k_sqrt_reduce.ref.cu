extern "C" __global__
__launch_bounds__(256) void k_sqrt_reduce(const float* x, float* sqrt) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 4) {
        int a0 = _gid;
        float acc0 = 0.0f;
        for (int a1 = 0; a1 < 1024; a1++) {
            float in0 = x[a0 * 1024 + a1];
            float v0 = in0 * in0;
            acc0 += v0;
        }
        float v1 = sqrtf(acc0);
        sqrt[a0] = v1;
    }
}
