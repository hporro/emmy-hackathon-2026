extern "C" __global__
__launch_bounds__(256) void k_relu_pointwise(const float* x, float* relu) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 2) {
        int a0 = _gid;
        float2 _v_in0__u0 = *reinterpret_cast<const float2*>(&x[a0 * 2]);
        float in0__u0 = _v_in0__u0.x;
        float in0__u1 = _v_in0__u0.y;
        float v0__u0 = fmaxf(0.0f, in0__u0);
        float v0__u1 = fmaxf(0.0f, in0__u1);
        float2 _vs_v0__u0_c398b459 = make_float2(v0__u0, v0__u1);
        *reinterpret_cast<float2*>(&relu[a0 * 2]) = _vs_v0__u0_c398b459;
    }
}
