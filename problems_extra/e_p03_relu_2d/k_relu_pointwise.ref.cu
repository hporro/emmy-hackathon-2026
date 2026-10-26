extern "C" __global__
__launch_bounds__(256) void k_relu_pointwise(const float* x, float* relu) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 128) {
        int a0 = _gid;
        float in0__u0 = x[a0 * 4];
        float v0__u0 = fmaxf(0.0f, in0__u0);
        float in0__u1 = x[a0 * 4 + 1];
        float v0__u1 = fmaxf(0.0f, in0__u1);
        float in0__u2 = x[a0 * 4 + 2];
        float v0__u2 = fmaxf(0.0f, in0__u2);
        float in0__u3 = x[a0 * 4 + 3];
        float v0__u3 = fmaxf(0.0f, in0__u3);
        float4 _vs_v0__u0_bb46b48b = make_float4(v0__u0, v0__u1, v0__u2, v0__u3);
        *reinterpret_cast<float4*>(&relu[a0 * 4]) = _vs_v0__u0_bb46b48b;
    }
}
