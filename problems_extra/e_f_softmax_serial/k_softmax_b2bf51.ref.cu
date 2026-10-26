extern "C" __global__
__launch_bounds__(256) void k_softmax_b2bf51(const float* x, float* softmax) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 4) {
        int a0 = _gid;
        float acc1 = 0.0f;
        float acc0 = -1e+30f;
        for (int a1 = 0; a1 < 1024; a1++) {
            float one = 1.0f;
            float in0 = x[a0 * 1024 + a1];
            float in0__gn = fmaxf(acc0, in0);
            float in0__acc1_sa = acc1 * expf((acc0 - in0__gn));
            float in0__acc1_sb = expf((in0 - in0__gn)) * one;
            acc1 = in0__acc1_sa + in0__acc1_sb;
            acc0 = fmaxf(acc0, in0);
        }
        float v2 = 1.0f / acc1;
        for (int a3 = 0; a3 < 1024; a3++) {
            float in2 = x[a0 * 1024 + a3];
            float v5 = v2 * expf((in2 - acc0));
            softmax[a0 * 1024 + a3] = v5;
        }
    }
}
