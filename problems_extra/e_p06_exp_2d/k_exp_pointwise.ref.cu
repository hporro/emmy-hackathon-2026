extern "C" __global__
__launch_bounds__(256) void k_exp_pointwise(const float* x, float* exp) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 32) {
        int a0 = _gid;
        float in0__u0 = x[a0 * 2];
        float v0__u0 = expf(in0__u0);
        float in0__u1 = x[a0 * 2 + 1];
        float v0__u1 = expf(in0__u1);
        float2 _vs_v0__u0_59fa3876 = make_float2(v0__u0, v0__u1);
        *reinterpret_cast<float2*>(&exp[a0 * 2]) = _vs_v0__u0_59fa3876;
    }
}
