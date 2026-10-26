extern "C" __global__
__launch_bounds__(256) void k_silu_pointwise(const float* silu_one, const float* x, float* silu) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in1__u0 = x[a0 * 4];
        float in0__u0 = silu_one[0];
        float v4__u0 = in1__u0 * (1.0f / (in0__u0 + expf((0.0f - in1__u0))));
        float in1__u1 = x[a0 * 4 + 1];
        float in0__u1 = silu_one[0];
        float v4__u1 = in1__u1 * (1.0f / (in0__u1 + expf((0.0f - in1__u1))));
        float in1__u2 = x[a0 * 4 + 2];
        float in0__u2 = silu_one[0];
        float v4__u2 = in1__u2 * (1.0f / (in0__u2 + expf((0.0f - in1__u2))));
        float in1__u3 = x[a0 * 4 + 3];
        float in0__u3 = silu_one[0];
        float v4__u3 = in1__u3 * (1.0f / (in0__u3 + expf((0.0f - in1__u3))));
        float4 _vs_v4__u0_5366d9e3 = make_float4(v4__u0, v4__u1, v4__u2, v4__u3);
        *reinterpret_cast<float4*>(&silu[a0 * 4]) = _vs_v4__u0_5366d9e3;
}
